#!/usr/bin/env python3
"""
CHoCH-AI  -  self-learning bullish CHoCH scanner
MEXC perpetuals + top CEX perpetuals (via ccxt), Weekly / 1D / 4H.

What it does on every run (all from live data, nothing hard-coded to a past market):
  1. UNIVERSE   - loads perpetual markets of up to 15 exchanges, keeps crypto USDT perps,
                  dedupes by coin (MEXC first), liquidity filter, analysis summary.
  2. DETECT     - strict bullish CHoCH state machine (close above the lower-high that made
                  the last lower-low, while the trend state is bearish).
  3. LEARN      - every historical CHoCH in the fetched history is replayed as a trade
                  (entry / SL / TP1) -> labelled win/loss. A logistic model per timeframe
                  is trained on those outcomes with recency weighting (newest market
                  behaviour counts most). Swing length, SL buffer and TP multiple are
                  auto-tuned per timeframe on the same data (grid, t-stat ranked).
  4. VALIDATE   - walk-forward hold-out (train on old, test on newest 30%) is reported so you
                  can see whether the model really beats the raw CHoCH hit-rate.
  5. SELF-ADJUST- live signals are tracked to TP/SL; the EV threshold tightens when live
                  expectancy is negative and relaxes when it is strong; bearish market
                  breadth raises the bar automatically.
  6. ALERT      - Telegram: Weekly + 1D + 4H live chart (CHoCH marked) with entry, SL, TP1-3,
                  position sizing, HTF bias, listing info.
"""
import os, re, sys, json, time, math
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# =============================== CONFIG ===============================
EXCHANGES = [e.strip() for e in os.getenv(
    "EXCHANGES",
    "mexc,binanceusdm,bybit,okx,bitget,gate,kucoinfutures,htx,bingx,phemex,bitmex,coinex,krakenfutures,cryptocom,lbank"
).split(",") if e.strip()]

TFS = ["4h", "1d", "1w"]
TF_MS = {"4h": 4 * 3600_000, "1d": 86400_000, "1w": 7 * 86400_000}
LIMITS = {"4h": 500, "1d": 400, "1w": 200}
MIN_BARS = {"4h": 150, "1d": 80, "1w": 30}
HTF1 = {"4h": "1d", "1d": "1w", "1w": None}
HTF2 = {"4h": "1w", "1d": None, "1w": None}

SWS = [2, 3, 4]                 # swing (pivot) lengths tried by the auto-tuner
SL_BUFS = [0.15, 0.40]          # SL buffer below structure low, in ATR
TP1S = [1.5, 2.0]               # TP1 in R multiples
DEFAULTS = {"4h": dict(sw=3, sl_buf=0.30, tp1=1.5),
            "1d": dict(sw=3, sl_buf=0.30, tp1=1.5),
            "1w": dict(sw=2, sl_buf=0.30, tp1=1.5)}
HORIZON = {"4h": 30, "1d": 20, "1w": 10}          # bars a trade may stay open
FRESH = {"4h": 2, "1d": 1, "1w": 1}               # CHoCH must be within last N closed candles
MAX_RISK = {"4h": 0.10, "1d": 0.18, "1w": 0.35}   # reject setups with stop wider than this
HALF_LIFE = {"4h": 180, "1d": 150, "1w": 100}     # recency half-life (bars) for learning
BARS_SHOWN = {"4h": 120, "1d": 100, "1w": 70}

FEE_PCT = 0.001                 # round-trip fee+slippage assumption (0.1%)
MIN_GRID_N = 40                 # min trades for a grid combo to be eligible
MIN_TRAIN = 120                 # min resolved events to train a model for a timeframe
EV_MIN_START = 0.12             # starting minimum expected value (R) to alert
MIN_P = 0.38                    # absolute floor for win probability
MAX_SIGNALS = int(os.getenv("MAX_SIGNALS", "10"))
MIN_VOL_USD = float(os.getenv("MIN_VOL_USD", "200000"))
MEXC_ONLY = os.getenv("MEXC_ONLY", "0") == "1"      # 1 = only alert coins that exist on MEXC
WORKERS_PER_EX = int(os.getenv("WORKERS_PER_EX", "3"))
FETCH_BUDGET_MIN = float(os.getenv("FETCH_BUDGET_MIN", "35"))
WATERMARK = os.getenv("WATERMARK", "RANJHA")

STATE_DIR, OUT_DIR = "state", "out"
TG_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TG_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")

FEATS = ["break_atr", "body_ratio", "vol_ratio", "leg_atr", "log_bars_down", "rsi", "dist_ema50",
         "ll_count", "htf1", "htf2", "coin_prior_wr", "atr_rel", "risk_atr"]

# Non-crypto contracts that MEXC & others list (stocks / ETFs / metals / energy / FX / indices).
# Extend with the EXTRA_BLOCK env var (comma separated) if one slips through.
NON_CRYPTO = set("""
XAU XAG XPT XPD GOLD SILVER COPPER PLATINUM PALLADIUM WTI BRENT USOIL UKOIL OIL NATGAS NGAS
EUR GBP JPY AUD CAD CHF NZD CNH EURUSD GBPUSD USDJPY SPX NDX DJI US500 US30 NAS100 SP500 VIX DXY
SPY QQQ IWM DIA GLD SLV TLT ARKK VOO VTI EEM XLK XLF XLE SMH SOXX USO UNG TQQQ SQQQ SOXL
AAPL MSFT GOOGL GOOG AMZN META NVDA TSLA NFLX AMD INTC COIN MSTR HOOD PLTR CRCL BABA BIDU
PDD NIO UBER ABNB SHOP PYPL SNAP DIS PEP MCD NKE WMT COST JPM BAC GS XOM CVX PFE JNJ LLY ORCL
CRM ADBE AVGO QCOM TSM ASML SMCI RBLX ROKU GME AMC SOFI RIVN LCID CRWV IBM CSCO BA GM MRVL ANET
DELL NOW SNOW PANW CRWD ZM DKNG MARA RIOT CLSK BMNR SBET OKLO RKLB IONQ RGTI APP TTD
""".split()) | set(x.strip().upper() for x in os.getenv("EXTRA_BLOCK", "").split(",") if x.strip())


def log(*a):
    print(*a, flush=True)


# ============================ INDICATORS ==============================
def ema(x, n):
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / (dn + 1e-12)
    return (100 - 100 / (1 + rs)).values


def atr(h, l, c, n=14):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values


def prep(raw, tf):
    t = raw[:, 0]
    o, h, l, c, v = raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4], raw[:, 5]
    a = atr(h, l, c)
    return dict(
        t=t, tc=t + TF_MS[tf], o=o, h=h, l=l, c=c, v=v, atr=a, rsi=rsi(c), ema50=ema(c, 50),
        vma=pd.Series(v).rolling(20, min_periods=5).mean().shift(1).values,
        amed=pd.Series(a).rolling(100, min_periods=20).median().values, n=len(c))


# ====================== MARKET STRUCTURE / CHoCH ======================
def structure(h, l, c, sw):
    """
    Causal market-structure state machine (pivots are confirmed `sw` bars late, closes decide breaks).
    trend:  +1 bullish, -1 bearish, 0 unknown.
    Bullish CHoCH = trend is bearish and a candle CLOSES above the swing high that preceded the
    latest confirmed swing low (the lower-high that created the last lower-low).
    Returns (events, trend_array).
    """
    n = len(c)
    trend = np.zeros(n, dtype=np.int8)
    ph = []                       # confirmed pivot highs [(idx, price)]
    pl_idx, lower, lower_active = -1, None, False
    t, flip_i, ll, prev_pl = 0, 0, 0, None
    last_choch_low = -1
    events = []
    for i in range(n):
        p = i - sw
        if p >= sw:
            if h[p] >= max(h[p - sw:p]) and h[p] > max(h[p + 1:i + 1]):
                ph.append((p, h[p]))
            if l[p] <= min(l[p - sw:p]) and l[p] < min(l[p + 1:i + 1]):
                pl_idx, lower, lower_active = p, l[p], True
                if t == -1:
                    if prev_pl is not None and lower < prev_pl:
                        ll += 1
                    prev_pl = lower
        if t != -1:
            if lower_active and c[i] < lower:
                t, flip_i, lower_active, ll, prev_pl = -1, i, False, 0, None
            elif t == 0 and ph and c[i] > ph[-1][1]:
                t, flip_i = 1, i
        else:
            if lower_active and c[i] < lower:
                lower_active = False
            if pl_idx >= 0 and pl_idx != last_choch_low:
                lvl = None
                for (pi, pv) in reversed(ph):
                    if pi < pl_idx:
                        lvl = (pi, pv)
                        break
                if lvl is not None and c[i] > lvl[1]:
                    events.append(dict(i=i, lvl=lvl[1], lvl_i=lvl[0], low=min(l[lvl[0]:i + 1]),
                                       bars=i - flip_i, ll=ll))
                    last_choch_low = pl_idx
                    t, flip_i, ll, prev_pl = 1, i, 0, None
                    lower_active = (lower is not None and c[i] > lower)
        trend[i] = t
    return events, trend


def trend_at(Ph, ts_close):
    if Ph is None:
        return 0
    k = int(np.searchsorted(Ph["tc"], ts_close, side="right")) - 1
    return int(Ph["trend"][k]) if k >= 0 else 0


# ========================== TRADE SIMULATION ==========================
def simulate(h, l, c, i, entry, sl, tp, H, n):
    """Market entry at close of bar i. SL checked before TP inside a bar (pessimistic)."""
    risk = entry - sl
    last = min(n - 1, i + H)
    for j in range(i + 1, last + 1):
        if l[j] <= sl:
            return True, 0, -1.0, j
        if h[j] >= tp:
            return True, 1, (tp - entry) / risk, j
    if i + H <= n - 1:
        return True, 0, (c[i + H] - entry) / risk, i + H
    return False, 0, 0.0, None


def tune(tf, keys, preps, cache):
    """Grid-search swing length / SL buffer / TP1 on all historical CHoCH trades of this timeframe."""
    table, best = [], None
    for sw in SWS:
        evs = [(k, ev) for k in keys if (k, tf, sw) in cache for ev in cache[(k, tf, sw)][0]]
        for buf in SL_BUFS:
            for tp1 in TP1S:
                Rs = []
                for k, ev in evs:
                    P = preps[(k, tf)]
                    i = ev["i"]
                    if i < 30:
                        continue
                    a = P["atr"][i]
                    if not a > 0:
                        continue
                    entry = P["c"][i]
                    sl = ev["low"] - buf * a
                    risk = entry - sl
                    if risk <= 0 or risk / entry > MAX_RISK[tf]:
                        continue
                    ok, y, R, j = simulate(P["h"], P["l"], P["c"], i, entry, sl, entry + tp1 * risk, HORIZON[tf], P["n"])
                    if ok:
                        Rs.append(R - FEE_PCT * entry / risk)
                n = len(Rs)
                if n >= MIN_GRID_N:
                    m, sd = float(np.mean(Rs)), float(np.std(Rs))
                    ts = m / (sd / math.sqrt(n) + 1e-9)
                    row = dict(sw=sw, sl_buf=buf, tp1=tp1, n=n, exp=round(m, 3), t=round(ts, 2))
                    table.append(row)
                    if best is None or ts > best["t"]:
                        best = row
    if best is None:
        d = dict(DEFAULTS[tf])
        d.update(n=0, exp=0.0, t=0.0, tuned=False)
        return d
    best = dict(best)
    best["tuned"] = True
    return best


# ============================ LEARNING ================================
def fit_logit(X, y, w, l2=3.0):
    w = w / (w.mean() + 1e-12)
    mu = np.average(X, axis=0, weights=w)
    sd = np.sqrt(np.average((X - mu) ** 2, axis=0, weights=w)) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    k = Z.shape[1]
    beta = np.zeros(k)
    reg = np.r_[np.ones(k - 1), 0.0]
    for _ in range(30):
        p = 1 / (1 + np.exp(-np.clip(Z @ beta, -30, 30)))
        g = Z.T @ (w * (p - y)) + l2 * reg * beta
        Hm = (Z.T * (w * p * (1 - p) + 1e-6)) @ Z + l2 * np.diag(reg) + 1e-6 * np.eye(k)
        step = np.linalg.solve(Hm, g)
        beta -= step
        if np.abs(step).max() < 1e-6:
            break
    return dict(mu=mu, sd=sd, beta=beta)


def predict(m, X):
    X = np.asarray(X, dtype=float)
    Z = np.hstack([(X - m["mu"]) / m["sd"], np.ones((len(X), 1))])
    return 1 / (1 + np.exp(-np.clip(Z @ m["beta"], -30, 30)))


def auc_score(y, p):
    y = np.asarray(y)
    n1, n0 = int(y.sum()), int(len(y) - y.sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(p).rank().values
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build_rows(key, tf, P, events, prm, P1, P2):
    rows = []
    for ev in events:
        i = ev["i"]
        if i < 30:
            continue
        a = P["atr"][i]
        if not a > 0:
            continue
        entry = P["c"][i]
        sl = ev["low"] - prm["sl_buf"] * a
        risk = entry - sl
        if risk <= 0 or risk / entry > MAX_RISK[tf]:
            continue
        tp = entry + prm["tp1"] * risk
        ok, y, R, j = simulate(P["h"], P["l"], P["c"], i, entry, sl, tp, HORIZON[tf], P["n"])
        tcl = int(P["tc"][i])
        cost = FEE_PCT * entry / risk
        rows.append(dict(key=key, tf=tf, i=i, ts=tcl, ev=ev, entry=entry, sl=sl, tp=tp, risk=risk, atr=a,
                         resolved=ok, y=y, R=(R - cost) if ok else 0.0, cost=cost,
                         res_ts=int(P["tc"][j]) if ok else None,
                         h1=trend_at(P1, tcl), h2=trend_at(P2, tcl)))
    for k, r in enumerate(rows):                       # expanding, time-safe coin history
        past = [q for q in rows[:k] if q["resolved"] and q["res_ts"] <= r["ts"]]
        r["prior"] = (sum(q["y"] for q in past) + 0.4 * 3) / (len(past) + 3)
        i, ev = r["i"], r["ev"]
        a, c = r["atr"], P["c"][i]
        rng = P["h"][i] - P["l"][i]
        vma = P["vma"][i]
        amed = P["amed"][i]
        r["x"] = [
            float(np.clip((c - ev["lvl"]) / a, 0, 5)),
            float(np.clip((c - P["o"][i]) / rng, -1, 1)) if rng > 0 else 0.0,
            float(np.clip(P["v"][i] / vma, 0, 6)) if vma and vma > 0 else 1.0,
            float(np.clip((ev["lvl"] - ev["low"]) / a, 0, 15)),
            float(np.log1p(min(ev["bars"], 200))),
            float(P["rsi"][i] / 100),
            float(np.clip((c - P["ema50"][i]) / a, -10, 10)),
            float(min(ev["ll"], 6)),
            float(r["h1"]), float(r["h2"]), float(r["prior"]),
            float(np.clip(a / amed, 0, 4)) if amed and amed > 0 else 1.0,
            float(np.clip((r["entry"] - r["sl"]) / a, 0, 10)),
        ]
    return rows


# =========================== STATE / UTILS ============================
def jload(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def jsave(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def fmt(x):
    return f"{x:.6g}"


def arrow(v):
    return "↑" if v > 0 else ("↓" if v < 0 else "–")


def norm_base(b):
    return re.sub(r"^(1000000|100000|10000|1000|1M)", "", b.upper())


# =============================== ANALYSIS =============================
def analyze(data, meta, state, now_ms):
    keys = list(data.keys())
    preps = {}
    for k in keys:
        for tf, raw in data[k].items():
            cl = raw[:-1] if raw[-1, 0] + TF_MS[tf] > now_ms else raw
            if len(cl) >= MIN_BARS[tf]:
                preps[(k, tf)] = prep(cl, tf)
    cache = {}
    for (k, tf), P in preps.items():
        hl, ll_, cl_ = P["h"].tolist(), P["l"].tolist(), P["c"].tolist()
        for sw in SWS:
            cache[(k, tf, sw)] = structure(hl, ll_, cl_, sw)

    params = {tf: tune(tf, keys, preps, cache) for tf in TFS}
    for (k, tf), P in preps.items():
        P["trend"] = cache[(k, tf, params[tf]["sw"])][1]

    rows = {tf: [] for tf in TFS}
    for (k, tf), P in preps.items():
        prm = params[tf]
        evs = cache[(k, tf, prm["sw"])][0]
        P1 = preps.get((k, HTF1[tf])) if HTF1[tf] else None
        P2 = preps.get((k, HTF2[tf])) if HTF2[tf] else None
        rows[tf] += build_rows(k, tf, P, evs, prm, P1, P2)

    ev_min = state["params"].get("ev_min", EV_MIN_START)
    d1 = [preps[(k, "1d")]["trend"][-1] for k in keys if (k, "1d") in preps]
    d4 = [preps[(k, "4h")]["trend"][-1] for k in keys if (k, "4h") in preps]
    breadth1d = float(np.mean([1 if t == 1 else 0 for t in d1])) if d1 else 0.5
    breadth4h = float(np.mean([1 if t == 1 else 0 for t in d4])) if d4 else 0.5
    ev_thr = ev_min + (0.08 if breadth1d < 0.30 else 0.0)

    models, report = {}, {}
    for tf in TFS:
        R = sorted([r for r in rows[tf] if r["resolved"]], key=lambda r: r["ts"])
        info = dict(events=len(rows[tf]), resolved=len(R), base_wr=None, auc=None, hold_n=0,
                    hold_base=None, hold_sel_wr=None, hold_sel_n=0, hold_sel_R=None, hold_all_R=None)
        if R:
            info["base_wr"] = float(np.mean([r["y"] for r in R]))
        if len(R) >= MIN_TRAIN:
            X = np.array([r["x"] for r in R])
            y = np.array([r["y"] for r in R], dtype=float)
            w = np.array([0.5 ** (max(0, now_ms - r["ts"]) / TF_MS[tf] / HALF_LIFE[tf]) for r in R])
            cut = int(len(R) * 0.7)
            split_ts = R[cut]["ts"]
            tr_idx = [j for j in range(cut) if R[j]["res_ts"] <= split_ts]
            te_idx = list(range(cut, len(R)))
            if len(tr_idx) >= 60 and len(te_idx) >= 30:
                m_ho = fit_logit(X[tr_idx], y[tr_idx], w[tr_idx])
                p_te = predict(m_ho, X[te_idx])
                Rte = np.array([R[j]["R"] for j in te_idx])
                tp1 = params[tf]["tp1"]
                ev_te = p_te * tp1 - (1 - p_te) - np.array([R[j]["cost"] for j in te_idx])
                sel = ev_te >= ev_thr
                info.update(auc=auc_score(y[te_idx], p_te), hold_n=len(te_idx), hold_base=float(y[te_idx].mean()),
                            hold_all_R=float(Rte.mean()), hold_sel_n=int(sel.sum()))
                if sel.sum() > 0:
                    info["hold_sel_wr"] = float(y[te_idx][sel].mean())
                    info["hold_sel_R"] = float(Rte[sel].mean())
            models[tf] = fit_logit(X, y, w)
            jsave(f"{STATE_DIR}/model_{tf}.json", dict(feats=FEATS, mu=models[tf]["mu"], sd=models[tf]["sd"],
                                                       beta=models[tf]["beta"], params=params[tf],
                                                       trained_on=len(R)))
        report[tf] = info

    # ------------ live signal tracking + self-adjust ------------
    sigs = state["signals"]
    for s in sigs:
        if s["status"] != "open":
            continue
        P = preps.get((s["key"], s["tf"]))
        if P is None:
            continue
        idx = int(np.searchsorted(P["tc"], s["ts"], side="left"))
        i = idx if (idx < P["n"] and P["tc"][idx] == s["ts"]) else idx - 1
        if i < 0 or i >= P["n"] - 1:
            continue
        ok, y, R, j = simulate(P["h"], P["l"], P["c"], i, s["entry"], s["sl"], s["tp1"], HORIZON[s["tf"]], P["n"])
        if ok:
            risk = s["entry"] - s["sl"]
            s["R"] = round(R - FEE_PCT * s["entry"] / risk, 3)
            s["status"] = "win" if y == 1 else ("loss" if R <= -0.99 else "timeout")
    done = [s for s in sigs if s["status"] != "open"][-30:]
    live = dict(n=len(done), wr=None, avg_R=None)
    if done:
        live["wr"] = float(np.mean([1 if s["status"] == "win" else 0 for s in done]))
        live["avg_R"] = float(np.mean([s["R"] for s in done]))
        if len(done) >= 12:
            if live["avg_R"] < 0:
                ev_min = min(0.6, ev_min + 0.05)
            elif live["avg_R"] > 0.30:
                ev_min = max(0.05, ev_min - 0.02)
    state["params"]["ev_min"] = round(ev_min, 3)
    ev_thr = ev_min + (0.08 if breadth1d < 0.30 else 0.0)

    # ------------------------- fresh candidates -------------------------
    sent_ids = {s["id"] for s in sigs}
    open_keys = {(s["key"], s["tf"]) for s in sigs if s["status"] == "open"}
    cands = []
    for tf in TFS:
        m = models.get(tf)
        prm = params[tf]
        base = report[tf]["base_wr"] if report[tf]["base_wr"] else 0.4
        for r in rows[tf]:
            k = r["key"]
            P = preps[(k, tf)]
            if r["i"] < P["n"] - FRESH[tf] or r["resolved"]:
                continue
            sid = f"{k}|{tf}|{r['ts']}"
            if sid in sent_ids or (k, tf) in open_keys:
                continue
            price = float(data[k][tf][-1, 4])
            if price <= r["sl"] or price >= r["tp"] or price > r["entry"] + 0.6 * r["risk"]:
                continue
            if MEXC_ONLY and not meta[k]["mexc"]:
                continue
            p = float(predict(m, [r["x"]])[0]) if m else float(base)
            ev = p * prm["tp1"] - (1 - p) - r["cost"]
            if p < MIN_P or ev < ev_thr:
                continue
            cands.append(dict(r=r, p=p, ev=ev, price=price, sid=sid, model=bool(m)))
    best = {}
    for c in sorted(cands, key=lambda c: -c["ev"]):
        k = c["r"]["key"]
        if k not in best:
            best[k] = c
            best[k]["also"] = []
        else:
            best[k]["also"].append(c["r"]["tf"].upper())
    chosen = sorted(best.values(), key=lambda c: -c["ev"])[:MAX_SIGNALS]

    ctx = dict(preps=preps, cache=cache, params=params, report=report, breadth1d=breadth1d,
               breadth4h=breadth4h, ev_thr=ev_thr, live=live, n_cands=len(cands))
    return chosen, ctx


# ============================== CHARTS ================================
def draw_panel(ax, raw, tf, title, sig, cache, params, P):
    nb = BARS_SHOWN[tf]
    start = max(0, len(raw) - nb)
    d = raw[start:]
    x = np.arange(len(d))
    up = d[:, 4] >= d[:, 1]
    ax.vlines(x, d[:, 3], d[:, 2], colors=np.where(up, "#26a69a", "#ef5350"), linewidth=1)
    span = max(d[:, 2].max() - d[:, 3].min(), 1e-12)
    for k in range(len(d)):
        o, c = d[k, 1], d[k, 4]
        ax.add_patch(Rectangle((k - 0.35, min(o, c)), 0.7, max(abs(c - o), span * 0.002),
                               color="#26a69a" if c >= o else "#ef5350"))
    full = raw[:, 4]
    ax.plot(x, ema(full, 20)[start:], color="#ffb300", lw=1.1, label="EMA20")
    ax.plot(x, ema(full, 50)[start:], color="#42a5f5", lw=1.1, label="EMA50")
    r = sig["r"]
    lo = min(d[:, 3].min(), r["sl"])
    hi = max(d[:, 2].max(), r["tp"])
    pad = (hi - lo) * 0.06
    ax.set_ylim(lo - pad, hi + pad)
    # CHoCH marking for this timeframe (last bullish CHoCH inside the window)
    prm_sw = params[tf]["sw"]
    evs = cache.get((sig["r"]["key"], tf, prm_sw), ([], None))[0]
    evs = [e for e in evs if e["i"] >= start]
    if evs:
        e = evs[-1]
        ax.hlines(e["lvl"], e["lvl_i"] - start, e["i"] - start, colors="#ffeb3b", linestyles="--", lw=1.2)
        ax.text((e["lvl_i"] + e["i"]) / 2 - start, e["lvl"], "CHoCH", color="#ffeb3b", fontsize=9,
                ha="center", va="bottom", fontweight="bold")
        ax.scatter([e["i"] - start], [P["c"][e["i"]]], marker="^", color="#ffeb3b", s=60, zorder=5)
    # trade plan lines
    lines = [("SL", r["sl"], "#ef5350"), ("ENTRY", r["entry"], "#90caf9"), ("TP1", r["tp"], "#66bb6a"),
             ("TP2", sig["tp2"], "#43a047"), ("TP3", sig["tp3"], "#2e7d32")]
    cur_lo, cur_hi = ax.get_ylim()
    for nm, val, col in lines:
        if cur_lo <= val <= cur_hi:
            ax.axhline(val, color=col, lw=0.9, ls=":")
            ax.text(len(d) + 0.5, val, f" {nm} {fmt(val)}", color=col, fontsize=7.5, va="center")
    last = d[-1, 4]
    ax.axhline(last, color="#cfd8dc", lw=0.6, ls="--")
    ax.text(len(d) - 0.5, last, f"{fmt(last)} ", color="#cfd8dc", fontsize=7.5, ha="right", va="bottom")
    step = max(1, len(d) // 6)
    ticks = list(range(0, len(d), step))
    fm = "%d %b %y" if tf == "1w" else ("%d %b" if tf == "1d" else "%d %b %H:%M")
    ax.set_xticks(ticks)
    ax.set_xticklabels([datetime.fromtimestamp(d[t, 0] / 1000, tz=timezone.utc).strftime(fm) for t in ticks],
                       fontsize=8, color="#bbbbbb")
    ax.set_xlim(-1, len(d) + 11)
    ax.set_title(title, loc="left", fontsize=10, color="white")
    ax.grid(alpha=0.12)
    ax.set_facecolor("#111418")
    ax.tick_params(colors="#bbbbbb", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.legend(loc="upper left", fontsize=7, facecolor="#111418", labelcolor="white", framealpha=0.5)


def make_chart(sig, data, ctx, meta):
    k = sig["r"]["key"]
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), facecolor="#111418")
    for ax, tf, nm in zip(axes, ["1w", "1d", "4h"], ["WEEKLY", "DAILY (1D)", "4 HOUR (4H)"]):
        if tf not in data[k] or (k, tf) not in ctx["preps"]:
            ax.set_facecolor("#111418")
            ax.text(0.5, 0.5, f"{nm}: not enough history", color="white", ha="center", transform=ax.transAxes)
            continue
        draw_panel(ax, data[k][tf], tf, f"{meta[k]['base']}/USDT  {meta[k]['eid'].upper()}  {nm}", sig,
                   ctx["cache"], ctx["params"], ctx["preps"][(k, tf)])
    now_s = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    fig.suptitle(f"BULLISH CHoCH {sig['r']['tf'].upper()}  |  P(win TP1) {sig['p']:.0%}  EV {sig['ev']:+.2f}R  |  live {now_s}",
                 color="white", fontsize=11, y=0.995)
    if WATERMARK:
        fig.text(0.5, 0.5, WATERMARK, fontsize=90, color="white", alpha=0.05, ha="center", va="center",
                 rotation=30, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = f"{OUT_DIR}/{meta[k]['base']}_{sig['r']['tf']}.png"
    fig.savefig(path, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def plan_signal(c, data, ctx, meta):
    r, k, tf = c["r"], c["r"]["key"], c["r"]["tf"]
    prm = ctx["params"][tf]
    P = ctx["preps"][(k, tf)]
    entry, sl, risk = r["entry"], r["sl"], r["risk"]
    c["tp2"] = entry + 2 * prm["tp1"] * risk
    rng_hi = float(P["h"][max(0, r["i"] - 100):r["i"] + 1].max())
    c["tp3"] = rng_hi if rng_hi > c["tp2"] * 1.02 else entry + 3 * prm["tp1"] * risk
    c["retest"] = r["ev"]["lvl"] + 0.1 * r["atr"]
    c["rr_retest"] = (r["tp"] - c["retest"]) / max(c["retest"] - sl, 1e-12) if c["retest"] > sl else float("nan")
    return c


def caption(c, meta):
    r, k, tf = c["r"], c["r"]["key"], c["r"]["tf"]
    m = meta[k]
    risk_pct = r["risk"] / r["entry"] * 100
    notional = 0.01 / (r["risk"] / r["entry"])
    grade = "A" if c["p"] >= 0.60 else ("B" if c["p"] >= 0.50 else "C")
    tp1r = (r["tp"] - r["entry"]) / r["risk"]
    L = [f"🟢 BULLISH CHoCH · {tf.upper()} · {m['base']}/USDT  [Grade {grade}]",
         f"Data: {m['eid'].upper()} | MEXC perp: {'YES' if m['mexc'] else 'NO'} | Listed on {len(m['listed'])}: {', '.join(x.upper() for x in m['listed'][:6])}",
         f"AI win-prob (TP1 before SL): {c['p']:.0%} | EV {c['ev']:+.2f}R" + ("" if c["model"] else " (low-data)"),
         f"ENTRY (market): {fmt(r['entry'])}   | Retest limit: {fmt(c['retest'])} (RR {c['rr_retest']:.1f})",
         f"SL: {fmt(r['sl'])}  (-{risk_pct:.1f}%)",
         f"TP1: {fmt(r['tp'])} (+{(r['tp']/r['entry']-1)*100:.1f}% | {tp1r:.1f}R)",
         f"TP2: {fmt(c['tp2'])} (+{(c['tp2']/r['entry']-1)*100:.1f}%)",
         f"TP3: {fmt(c['tp3'])} (+{(c['tp3']/r['entry']-1)*100:.1f}%)",
         f"Size: risk 1% = {notional:.0%} of equity notional (≈{notional:.1f}x lev, keep ≤ {max(1, int(50/risk_pct))}x)",
         f"HTF: 1D {arrow(r['h1'] if tf == '4h' else 0)} 1W {arrow(r['h2'] if tf == '4h' else (r['h1'] if tf == '1d' else 0))} | RSI {r['x'][5]*100:.0f} | Vol x{r['x'][2]:.1f} | Break +{r['x'][0]:.1f} ATR | Leg {r['x'][3]:.1f} ATR",
         f"Price now: {fmt(c['price'])}"]
    if c["also"]:
        L.append("Also CHoCH on: " + ", ".join(c["also"]))
    L.append("Not financial advice. Use the SL.")
    return "\n".join(L)[:1020]


# ============================== TELEGRAM ==============================
def tg_text(text):
    if not (TG_TOKEN and TG_CHAT):
        log("[telegram off]\n" + text)
        return
    try:
        requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
                      data={"chat_id": TG_CHAT, "text": text[:4000]}, timeout=30)
    except Exception as e:
        log("tg text error", e)


def tg_photo(path, cap):
    if not (TG_TOKEN and TG_CHAT):
        log("[telegram off] photo", path, "\n" + cap)
        return
    try:
        with open(path, "rb") as f:
            requests.post(f"https://api.telegram.org/bot{TG_TOKEN}/sendPhoto",
                          data={"chat_id": TG_CHAT, "caption": cap}, files={"photo": f}, timeout=60)
    except Exception as e:
        log("tg photo error", e)


def summary_text(umeta, ctx, nsig, state):
    L = [f"🧠 CHoCH-AI | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
         f"Exchanges OK {len(umeta['ok'])}/{len(EXCHANGES)}" + (f" | failed: {', '.join(umeta['failed'])}" if umeta['failed'] else ""),
         f"MEXC perps {umeta['mexc_total']} (crypto {umeta['mexc_crypto']}) | unique coins {umeta['unique']} | scanned {umeta['scanned']} | not on MEXC {umeta['not_mexc']}",
         f"Filtered: non-crypto {umeta['nc']}, low-vol {umeta['lowvol']}, no data {umeta['nodata']}, stock-like {umeta['stocklike']}",
         f"Market breadth (coins in bullish structure): 1D {ctx['breadth1d']:.0%} | 4H {ctx['breadth4h']:.0%}"]
    for tf in TFS:
        p, rp = ctx["params"][tf], ctx["report"][tf]
        s = f"[{tf.upper()}] events {rp['events']} resolved {rp['resolved']}"
        if rp["base_wr"] is not None:
            s += f" | raw CHoCH win {rp['base_wr']:.0%}"
        s += f" | tuned sw{p['sw']} SLbuf {p['sl_buf']} TP1 {p['tp1']}R (exp {p['exp']:+.2f}R)"
        if rp["auc"] is not None and not math.isnan(rp["auc"]):
            s += f" | holdout AUC {rp['auc']:.2f}"
            if rp["hold_sel_wr"] is not None:
                s += f", picks win {rp['hold_sel_wr']:.0%} vs {rp['hold_base']:.0%} base (n={rp['hold_sel_n']}, avg {rp['hold_sel_R']:+.2f}R)"
        L.append(s)
    lv = ctx["live"]
    L.append(f"Live record: {lv['n']} closed" + (f" | win {lv['wr']:.0%} | avg {lv['avg_R']:+.2f}R" if lv["n"] else "") +
             f" | EV threshold {ctx['ev_thr']:.2f}R")
    L.append(f"New signals this run: {nsig} (candidates {ctx['n_cands']})")
    return "\n".join(L)


# ============================ DATA / EXCHANGES ========================
def make_ex(eid):
    import ccxt
    cls = getattr(ccxt, eid, None)
    if cls is None and eid == "gate":
        cls = getattr(ccxt, "gateio", None)
    if cls is None:
        raise RuntimeError(f"ccxt has no exchange {eid}")
    return cls({"enableRateLimit": True, "timeout": 20000, "options": {"defaultType": "swap"}})


def load_exchange(eid):
    ex = make_ex(eid)
    mk = ex.load_markets()
    out = {}
    for s, m in mk.items():
        if not m.get("swap") or m.get("linear") is False or m.get("active") is False:
            continue
        if m.get("quote") not in ("USDT", "USDC", "USD") or not m.get("base"):
            continue
        key = norm_base(m["base"])
        if key in out and out[key]["quote"] == "USDT":
            continue
        out[key] = dict(symbol=s, base=m["base"], quote=m["quote"])
    vol = {}
    try:
        tk = ex.fetch_tickers()
        for s, t in tk.items():
            qv = t.get("quoteVolume") or ((t.get("baseVolume") or 0) * (t.get("last") or 0))
            vol[s] = float(qv or 0)
    except Exception as e:
        log(f"  {eid}: tickers unavailable ({str(e)[:60]})")
    return ex, out, vol


def fetch_tf(ex, sym, tf, now_ms):
    rows = None
    for a in range(3):
        try:
            rows = ex.fetch_ohlcv(sym, tf, limit=LIMITS[tf])
            break
        except Exception:
            time.sleep(1.5 * (a + 1))
    if not rows or len(rows) < MIN_BARS[tf]:
        return None
    arr = np.array(rows, dtype=float)[:, :6]
    arr = arr[~np.isnan(arr).any(axis=1)]
    if len(arr) < MIN_BARS[tf]:
        return None
    arr = arr[np.argsort(arr[:, 0])]
    arr = arr[np.r_[True, np.diff(arr[:, 0]) > 0]]
    if now_ms - arr[-1, 0] > 3 * TF_MS[tf]:
        return None
    return arr


def fetch_all(assign, exs, deadline, now_ms):
    by_ex = {}
    for key, (eid, sym) in assign.items():
        by_ex.setdefault(eid, []).append((key, sym))
    tasks = []
    for eid, items in by_ex.items():
        n = max(1, min(WORKERS_PER_EX, len(items)))
        for w in range(n):
            tasks.append((eid, items[w::n]))

    def work(eid, items):
        ex = make_ex(eid)
        try:
            ex.set_markets(exs[eid].markets)
        except Exception:
            ex.load_markets()
        out = {}
        for key, sym in items:
            if time.time() > deadline:
                break
            d = {}
            for tf in TFS:
                raw = fetch_tf(ex, sym, tf, now_ms)
                if raw is not None:
                    d[tf] = raw
            if "4h" in d and "1d" in d:
                out[key] = d
        return out

    data = {}
    with ThreadPoolExecutor(max_workers=min(48, len(tasks))) as pool:
        futs = [pool.submit(work, e, it) for e, it in tasks]
        for f in as_completed(futs):
            try:
                data.update(f.result())
            except Exception as e:
                log("worker error", str(e)[:80])
    return data


def weekend_flat_ratio(raw4h):
    d = raw4h[-300:]
    dow = ((d[:, 0] // 86400_000) + 3) % 7          # 0=Mon .. 6=Sun (epoch day 0 was a Thursday)
    wk = d[dow >= 5]
    if len(wk) < 6:
        return 0.0
    flat = (wk[:, 2] == wk[:, 3]) & (wk[:, 5] == 0)
    return float(flat.mean())


def build_universe():
    exs, per, vols, failed = {}, {}, {}, []
    with ThreadPoolExecutor(max_workers=len(EXCHANGES)) as pool:
        futs = {pool.submit(load_exchange, e): e for e in EXCHANGES}
        for f in as_completed(futs):
            e = futs[f]
            try:
                ex, mk, vol = f.result()
                if not mk:
                    raise RuntimeError("no swap markets")
                exs[e], per[e], vols[e] = ex, mk, vol
                log(f"  {e}: {len(mk)} perps")
            except Exception as ex_:
                failed.append(e)
                log(f"  {e}: FAILED ({str(ex_)[:90]})")
    order = [e for e in EXCHANGES if e in per]
    meta, assign = {}, {}
    nc = lowvol = 0
    mexc_total = len(per.get("mexc", {}))
    mexc_crypto = sum(1 for k in per.get("mexc", {}) if k not in NON_CRYPTO)
    all_keys = []
    for e in order:
        for k in per[e]:
            if k not in all_keys:
                all_keys.append(k)
    for k in all_keys:
        if k in NON_CRYPTO:
            nc += 1
            continue
        listed = [e for e in order if k in per[e]]
        eid = listed[0]
        sym = per[eid][k]["symbol"]
        v = vols[eid].get(sym, 0)
        if v and v < MIN_VOL_USD:
            lowvol += 1
            continue
        meta[k] = dict(eid=eid, symbol=sym, base=per[eid][k]["base"], listed=listed, mexc=("mexc" in listed), vol=v)
        assign[k] = (eid, sym)
    um = dict(ok=order, failed=failed, mexc_total=mexc_total, mexc_crypto=mexc_crypto, unique=len(all_keys),
              nc=nc, lowvol=lowvol, not_mexc=sum(1 for m in meta.values() if not m["mexc"]))
    return exs, meta, assign, um


# ================================ MAIN ================================
def main():
    t0 = time.time()
    now_ms = int(time.time() * 1000)
    log("== CHoCH-AI start ==")
    state = dict(params=jload(f"{STATE_DIR}/params.json", {}), signals=jload(f"{STATE_DIR}/signals.json", []))
    log("1) universe")
    exs, meta, assign, um = build_universe()
    if not assign:
        tg_text("CHoCH-AI: no exchange could be loaded (check network / geo-blocks).")
        return
    log(f"   coins to scan: {len(assign)}")
    log("2) fetching candles (4h/1d/1w)")
    data = fetch_all(assign, exs, t0 + FETCH_BUDGET_MIN * 60, now_ms)
    stocklike = [k for k, d in data.items() if weekend_flat_ratio(d["4h"]) >= 0.6]
    for k in stocklike:
        data.pop(k, None)
    um.update(scanned=len(data), nodata=len(assign) - len(data) - len(stocklike), stocklike=len(stocklike))
    log(f"   fetched {len(data)} coins in {time.time()-t0:.0f}s")
    if len(data) < 30:
        tg_text("CHoCH-AI: too little data fetched, skipping run.")
        return
    log("3) learn + detect")
    chosen, ctx = analyze(data, meta, state, now_ms)
    log(f"   signals: {len(chosen)}")
    sent = 0
    for c in chosen:
        c = plan_signal(c, data, ctx, meta)
        try:
            path = make_chart(c, data, ctx, meta)
            tg_photo(path, caption(c, meta))
        except Exception as e:
            log("chart error", c["r"]["key"], str(e)[:100])
            tg_text(caption(c, meta))
        r = c["r"]
        state["signals"].append(dict(id=c["sid"], key=r["key"], tf=r["tf"], eid=meta[r["key"]]["eid"],
                                     base=meta[r["key"]]["base"], ts=r["ts"], entry=r["entry"], sl=r["sl"],
                                     tp1=r["tp"], p=round(c["p"], 3), ev=round(c["ev"], 3),
                                     sent=now_ms, status="open", R=0.0))
        sent += 1
    state["signals"] = state["signals"][-500:]
    state["params"].update(updated=datetime.now(timezone.utc).isoformat(), tf=ctx["params"],
                           runs=state["params"].get("runs", 0) + 1)
    jsave(f"{STATE_DIR}/params.json", state["params"])
    jsave(f"{STATE_DIR}/signals.json", state["signals"])
    jsave(f"{STATE_DIR}/universe.json", dict(um, breadth1d=ctx["breadth1d"], breadth4h=ctx["breadth4h"],
                                              top_coins=sorted(meta, key=lambda k: -meta[k]["vol"])[:50]))
    tg_text(summary_text(um, ctx, sent, state))
    log(f"== done in {time.time()-t0:.0f}s ==")


if __name__ == "__main__":
    main()
