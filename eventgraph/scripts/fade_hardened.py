# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Hardened early-bias fade — honest, cluster-robust, and the decisive question: does the
edge BEAT SLIPPAGE?

Fixes over backtest_fade.py:
  * FIXED NOTIONAL ($1 per short-yes position) — kills the (1-p0)-collateral leverage
    that inflated returns. Per-trade PnL = (p0 - outcome) dollars.
  * CLUSTER-ROBUST significance: collapse correlated trades to ONE mean return per
    underlying, then t across underlyings + block-bootstrap CI (crypto co-movement no
    longer masquerades as independent bets).
  * COST SWEEP + BREAKEVEN: net edge = raw edge - round-trip cost; report the round-trip
    slippage c* at which the edge dies.
  * LIVE SPREAD: pull current best-bid/ask on comparable ACTIVE markets (Gamma) to see
    whether the real round-trip spread is above or below breakeven. THIS answers 'beats
    slippage?'.
  * OUT-OF-SAMPLE temporal split (fit vs holdout by resolution date).

Usage:
  uv run eventgraph/scripts/fade_hardened.py --graph-dir eventgraph/data/eg_live --hype-only
"""
import argparse, json, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at
from early_bias import hype_flags

GAMMA = "https://gamma-api.polymarket.com"


def build_trades(res, cache, sess, age, max_hold):
    trades = []
    for r in res:
        try:
            sd = dt.datetime.fromisoformat(r["start_date"] + "T00:00:00+00:00")
            rd = dt.date.fromisoformat(r["resolution_date"])
        except Exception:
            continue
        h = hist(r["clob_token_yes"], cache, sess)
        if not h:
            continue
        entry = (sd + dt.timedelta(days=age)).date()
        if entry >= rd or (rd - entry).days > max_hold:
            continue
        p0 = prob_at(h, int(dt.datetime.combine(entry, dt.time(), dt.timezone.utc).timestamp()))
        if p0 is None or p0 <= 0.02 or p0 >= 0.98:
            continue
        y = 1 if r["resolved_outcome"] == "yes" else 0
        und = r["symbols"][0] if r.get("symbols") else (r.get("event_title") or r["prop_id"])
        trades.append({"und": und, "p0": p0, "y": y, "entry": entry, "res": rd,
                       "pnl": p0 - y, "hold": (rd - entry).days, "token": r["clob_token_yes"], "h": h})
    return trades


def cluster_edge(trades):
    """mean per-trade edge, but significance from ONE obs per underlying (cluster-robust)."""
    byu = collections.defaultdict(list)
    for t in trades:
        byu[t["und"]].append(t["pnl"])
    cl = np.array([np.mean(v) for v in byu.values()])
    m = cl.mean(); se = cl.std(ddof=1) / math.sqrt(len(cl)) if len(cl) > 1 else float("nan")
    return m, se, len(cl), cl


def block_boot(trades, nb=2000, seed=1):
    """block-bootstrap by underlying -> CI on the cluster-mean edge + a cluster Sharpe."""
    byu = collections.defaultdict(list)
    for t in trades:
        byu[t["und"]].append(t["pnl"])
    clusters = [np.mean(v) for v in byu.values()]
    u = list(byu.values()); rng = np.random.default_rng(seed)
    means, sharpes = [], []
    for _ in range(nb):
        idx = rng.integers(0, len(u), len(u))
        samp = np.array([np.mean(u[i]) for i in idx])
        means.append(samp.mean())
        sharpes.append(samp.mean() / samp.std() if samp.std() > 0 else 0)
    return (np.percentile(means, 5), np.percentile(means, 95),
            np.percentile(sharpes, 5), np.percentile(sharpes, 95), np.mean(clusters))


def live_spread(sess, want_hype, n=400):
    """current best-bid/ask round-trip spread on comparable ACTIVE markets."""
    spreads = []
    off = 0
    while len(spreads) < n and off < 2000:
        try:
            page = sess.get(f"{GAMMA}/markets", params={"closed": "false", "active": "true", "limit": 100,
                            "offset": off, "order": "volumeNum", "ascending": "false"}, timeout=30).json()
        except Exception:
            break
        if not page:
            break
        off += 100
        for m in page:
            q = m.get("question") or ""
            if want_hype and "target/ATH" not in hype_flags(q):
                continue
            bb, ba = m.get("bestBid"), m.get("bestAsk")
            try:
                bb = float(bb); ba = float(ba)
            except (TypeError, ValueError):
                continue
            if 0 < bb < ba < 1:
                spreads.append(ba - bb)      # one-way top-of-book spread
    return np.array(spreads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--age", type=int, default=3)
    ap.add_argument("--min-volume", type=float, default=10000.0)
    ap.add_argument("--max-markets", type=int, default=5000)
    ap.add_argument("--max-hold", type=int, default=120)
    ap.add_argument("--hype-only", action="store_true")
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume and r.get("start_date")]
    if a.hype_only:
        res = [r for r in res if "target/ATH" in hype_flags(r.get("question"))]
    res = res[:a.max_markets]
    trades = build_trades(res, cache, sess, a.age, a.max_hold)
    if len(trades) < 40:
        print(f"only {len(trades)} trades"); return
    tag = "HYPE-ONLY" if a.hype_only else "ALL"
    print(f"=== {tag}: {len(trades)} trades (fixed $1 notional, short yes, entry age {a.age}d) ===")

    # 1) raw edge, per-trade + cluster-robust
    pnl = np.array([t["pnl"] for t in trades])
    print(f"1) RAW EDGE (fixed notional):")
    print(f"   per-trade    mean {pnl.mean():+.4f}  std {pnl.std():.3f}  per-trade Sharpe {pnl.mean()/pnl.std():+.3f}  n={len(pnl)}")
    m, se, nu, cl = cluster_edge(trades)
    print(f"   cluster-robust (1 obs / underlying, n={nu}): mean {m:+.4f}  ±{2*se:.4f}  "
          f"t={m/se:+.2f}  {'SIG' if abs(m)>2*se else 'n.s.'}")
    lo, hi, slo, shi, cm = block_boot(trades)
    print(f"   block-bootstrap edge 90%CI [{lo:+.4f},{hi:+.4f}] | cluster-Sharpe 90%CI [{slo:+.2f},{shi:+.2f}]")

    # 2) COST SWEEP + BREAKEVEN
    print(f"2) COST SWEEP — net cluster-edge vs round-trip cost (breakeven = edge dies):")
    for c in [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06]:
        net = m - c
        print(f"   round-trip {c*100:2.0f}c: net edge {net:+.4f}  {'+' if net>0 else 'UNDERWATER'}")
    print(f"   -> BREAKEVEN round-trip cost c* = raw edge = {m*100:.1f}c  ({m*50:.1f}c per side)")

    # 3) LIVE SPREAD on comparable active markets -> beats slippage?
    sp = live_spread(sess, a.hype_only)
    if len(sp) >= 10:
        med = np.median(sp); q25, q75 = np.percentile(sp, [25, 75])
        rt = med * 2
        print(f"3) LIVE SPREAD ({len(sp)} comparable active markets): one-way median {med*100:.1f}c "
              f"[IQR {q25*100:.1f}-{q75*100:.1f}c]; round-trip ~{rt*100:.1f}c")
        verdict = "BEATS slippage" if m > rt else ("MARGINAL" if m > med else "DOES NOT beat slippage")
        print(f"   raw edge {m*100:.1f}c vs round-trip spread {rt*100:.1f}c  ->  {verdict}")
    else:
        print("3) LIVE SPREAD: too few comparable active markets to estimate")

    # 4) OUT-OF-SAMPLE temporal split by resolution date
    trades.sort(key=lambda t: t["res"])
    mid = len(trades) // 2
    for lbl, sub in [("in-sample (early)", trades[:mid]), ("out-of-sample (late)", trades[mid:])]:
        mm, sse, nnu, _ = cluster_edge(sub)
        print(f"4) {lbl:22} cluster-edge {mm:+.4f} ±{2*sse:.4f}  ({nnu} underlyings)  "
              f"{'SIG' if abs(mm)>2*sse else 'n.s.'}")


if __name__ == "__main__":
    main()
