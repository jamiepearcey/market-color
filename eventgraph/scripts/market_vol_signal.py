# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Second-moment test: does PREDICTION-MARKET ATTENTION predict forward REALISED VOLATILITY
of the linked instrument?

Thesis (user, the hypothesis to prove FIRST): price moves happen when attention arrives,
not when a distribution is realised — and prediction-market repricing is the cleanest
attention/disagreement observable news timestamps can't give. News EXPLAINS direction
contemporaneously but doesn't PREDICT (efficient). The second moment (vol) may be the
leading, less-efficient signal. Prediction markets on BTC/ETH/GLD/USO/^TNX give a dense
implied-probability PATH we can test against the underlying's forward realised vol.

Design (out-of-sample forward, within-instrument so vol LEVEL is controlled):
  For each market with an implied-prob path P and a linked symbol s, at non-overlapping
  anchor dates T:
    feature   probchurn = std of daily Δprob over [T-W, T]   (repricing intensity)
              absmove   = |p_T - p_{T-W}|                     (directional repricing)
    target    log( RV(T, T+H] / RV[T-H, T] )                  (forward vol EXPANSION,
              i.e. does vol RISE after the market churns; ratio controls symbol vol level)
  RV = std of daily log returns from Yahoo. Anchors spaced >= H apart (non-overlapping
  forward windows) and kept >= H from resolution (drop mechanical settlement churn).
  Report Spearman IC(feature, target) pooled + per-symbol + a t-stat (flagged for the
  residual overlap from shared underlyings).

Usage:
  uv run eventgraph/scripts/market_vol_signal.py --graph-dir eventgraph/data/eg_live
  ... --w 10 --h 10        # churn window / forward vol horizon (trading days)
"""
import argparse, json, math, time, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np


def yahoo_logret(sym, cache, p1, p2):
    p = cache / f"{sym.replace('/', '_')}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}"
                          f"?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            txt = r.text if r.status_code == 200 else ""
        except Exception:
            txt = ""
        p.write_text(txt); time.sleep(0.1)
    cl = {}
    try:
        res = json.loads(txt)["chart"]["result"][0]; ind = res["indicators"]
        px = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], px):
            if c is not None and c > 0:
                cl[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception:
        pass
    ds = sorted(cl)
    lr = {ds[i]: math.log(cl[ds[i]] / cl[ds[i - 1]]) for i in range(1, len(ds))}
    return ds, lr


def rv(ds, lr, d0, d1):
    """realised vol (std of daily log returns) over trading days in (d0, d1]."""
    xs = [lr[d] for d in ds if d0 < d <= d1 and d in lr]
    return (np.std(xs), len(xs)) if len(xs) >= 3 else (None, len(xs))


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def spearman(x, y):
    if len(x) < 5:
        return 0.0, 0
    rx, ry = _rank(x), _rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0, 0
    return np.corrcoef(rx, ry)[0, 1], len(x)


def partial_spearman(x, y, z):
    """rank correlation of x,y after linearly removing control z (all in rank space).

    Isolates whether feature x predicts y BEYOND the control z (recent realised vol)."""
    if len(x) < 8:
        return 0.0, 0
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    if rx.std() == 0 or ry.std() == 0 or rz.std() == 0:
        return 0.0, 0
    Z = np.column_stack([np.ones(len(rz)), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    if ex.std() == 0 or ey.std() == 0:
        return 0.0, 0
    return np.corrcoef(ex, ey)[0, 1], len(x)


def tstat(ic, n):
    return ic * math.sqrt(max(n - 2, 1)) / math.sqrt(max(1 - ic * ic, 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--w", type=int, default=10, help="prob-churn lookback (calendar days on path)")
    ap.add_argument("--h", type=int, default=10, help="forward realised-vol horizon (trading days)")
    ap.add_argument("--min-path", type=int, default=25, help="min path points to use a market")
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)

    props = {json.loads(l)["prop_id"]: json.loads(l)
             for l in open(lake / "proposition.jsonl") if l.strip()}
    paths = collections.defaultdict(dict)
    for l in open(lake / "proposition_price.jsonl"):
        j = json.loads(l); paths[j["prop_id"]][j["d"]] = j["p"]
    print(f"{len(paths)} markets with implied-prob paths")

    # price histories for the linked symbols
    syms = sorted({s for pid in paths for s in (props.get(pid, {}).get("symbols") or [])})
    p1 = int(dt.datetime(2023, 1, 1, tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime(2026, 12, 31, tzinfo=dt.UTC).timestamp())
    hist = {s: yahoo_logret(s, cache, p1, p2) for s in syms}
    print(f"pulled histories for {sum(1 for s in hist if hist[s][1])}/{len(syms)} symbols")

    rows = []                                   # (symbol, probchurn, absmove, log_vol_ratio)
    for pid, path in paths.items():
        p = props.get(pid)
        if not p or not p.get("symbols") or len(path) < a.min_path:
            continue
        s = p["symbols"][0]
        ds, lr = hist.get(s, ([], {}))
        if not ds:
            continue
        pdays = sorted(path)
        resd = p.get("resolution_date") or pdays[-1]
        # non-overlapping anchors spaced ~H days apart
        anchors = pdays[a.w::max(a.h, 5)]
        for T in anchors:
            # need forward window strictly inside available price history and >= H from resolution
            if T >= ds[-1] or T <= ds[a.h]:
                continue
            if resd and (dt.date.fromisoformat(min(resd, ds[-1])) - dt.date.fromisoformat(T)).days < a.h:
                continue
            Td = dt.date.fromisoformat(T)
            w0 = (Td - dt.timedelta(days=a.w)).isoformat()
            # prob churn over [T-W, T]
            seg = [(d, path[d]) for d in pdays if w0 <= d <= T]
            if len(seg) < 4:
                continue
            dp = np.diff([v for _, v in seg])
            probchurn = float(np.std(dp)); absmove = abs(seg[-1][1] - seg[0][1])
            # forward vs baseline realised vol (trading-day windows around T)
            fwd_end = ds[min(ds.index(min(ds, key=lambda d: abs((dt.date.fromisoformat(d) - Td).days))) + a.h, len(ds) - 1)]
            base_start = ds[max(ds.index(min(ds, key=lambda d: abs((dt.date.fromisoformat(d) - Td).days))) - a.h, 0)]
            fv, nf = rv(ds, lr, T, fwd_end); bv, nb = rv(ds, lr, base_start, T)
            if fv is None or bv is None or bv == 0 or fv == 0:
                continue
            rows.append((s, probchurn, absmove, math.log(fv), math.log(bv)))

    if len(rows) < 30:
        print(f"only {len(rows)} usable (market,anchor) obs — insufficient"); return
    S = [r[0] for r in rows]; CH = [r[1] for r in rows]; AM = [r[2] for r in rows]
    FV = [r[3] for r in rows]; BV = [r[4] for r in rows]
    print(f"\n{len(rows)} (market,anchor) obs across {len(set(S))} instruments; "
          f"target = log forward RV, H={a.h}d, churn window W={a.w}d\n")

    # (1) RAW: churn vs forward vol level (confounded by recent vol)
    ic_raw, n = spearman(CH, FV)
    # (2) baseline itself (HAR): recent vol vs forward vol — the bar to beat
    ic_base, _ = spearman(BV, FV)
    # (3) INCREMENTAL: churn vs forward vol AFTER controlling recent vol (the real question)
    ic_inc, _ = partial_spearman(CH, FV, BV)
    ic_inc_am, _ = partial_spearman(AM, FV, BV)
    print(f"  recent-vol -> forward-vol (HAR baseline)      IC {ic_base:+.3f}  t≈{tstat(ic_base,n):+.2f}")
    print(f"  prob-churn -> forward-vol (RAW, confounded)   IC {ic_raw:+.3f}  t≈{tstat(ic_raw,n):+.2f}")
    print(f"  prob-churn -> forward-vol | recent-vol (NET)  IC {ic_inc:+.3f}  t≈{tstat(ic_inc,n):+.2f}   <-- the test")
    print(f"  abs-move   -> forward-vol | recent-vol (NET)  IC {ic_inc_am:+.3f}  t≈{tstat(ic_inc_am,n):+.2f}")
    print("    [t inflated by overlapping underlyings; read IC sign/size, not t precisely]")

    # per-symbol incremental (within-instrument, cleanest)
    print("\n  per-instrument NET prob-churn -> forward-vol | recent-vol:")
    bys = collections.defaultdict(list)
    for s, ch, am, fv, bv in rows:
        bys[s].append((ch, fv, bv))
    for s in sorted(bys, key=lambda s: -len(bys[s])):
        v = bys[s]
        if len(v) < 20:
            continue
        ic, nn = partial_spearman([c for c, _, _ in v], [f for _, f, _ in v], [b for _, _, b in v])
        print(f"    {s:9} NET IC {ic:+.3f}  t≈{tstat(ic, nn):+.2f}  (n={nn})")


if __name__ == "__main__":
    main()
