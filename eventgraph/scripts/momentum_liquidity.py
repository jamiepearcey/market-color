# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Is the prediction-market MOMENTUM real underreaction, or a CAN'T-REACT / slippage
artifact of illiquidity? (user hypothesis)

If books are thin, price cannot jump to fair value — it crawls there through slippage,
producing positive autocorrelation of price changes that is NOT tradeable (to capture
the drift you pay the very slippage that made it). The behavioural-underreaction story
and the microstructure story make OPPOSITE predictions across liquidity:

  * slippage / can't-react  => momentum CONCENTRATES in illiquid (low-volume/liquidity)
    markets and VANISHES in liquid ones (where price can move freely). Monotone decline
    of momentum-IC with liquidity.
  * genuine underreaction    => momentum PERSISTS (or even strengthens) in liquid markets
    where you could actually trade it.

Test: momentum IC = Spearman(Δp[14->7d], Δp[3->1d]) (non-overlapping, distinct endpoints)
bucketed by market liquidity, with per-bucket bootstrap CI + monotonicity — the
conditional_signal discipline. The liquid bucket is the 'could you actually trade it' read.

Usage:
  uv run eventgraph/scripts/momentum_liquidity.py --graph-dir eventgraph/data/eg_live
  ... --by liquidity --q 5 --min-volume 5000
"""
import argparse, json, math, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at, spearman, tstat


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def boot_ic(x, y, seed, nb=400):
    x, y = np.asarray(x), np.asarray(y)
    n = len(x)
    if n < 12:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed); ics = []
    for _ in range(nb):
        idx = rng.integers(0, n, n)
        ic, _ = spearman(x[idx], y[idx]); ics.append(ic)
    return (float(np.percentile(ics, 5)), float(np.percentile(ics, 95)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--by", default="volume", choices=["volume", "liquidity"])
    ap.add_argument("--q", type=int, default=5)
    ap.add_argument("--min-volume", type=float, default=5000.0)
    ap.add_argument("--max-markets", type=int, default=3000)
    ap.add_argument("--leads", default="14,7,3,1")
    a = ap.parse_args()
    leads = [int(x) for x in a.leads.split(",")]
    La, Lb, Lc, Ld = leads
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume]
    res.sort(key=lambda r: -(r.get("volume") or 0)); res = res[:a.max_markets]

    obs = []                                    # (d_prev, d_next, cond)
    for r in res:
        try:
            rd = dt.datetime.fromisoformat(r["resolution_date"] + "T00:00:00+00:00")
        except Exception:
            continue
        h = hist(r["clob_token_yes"], cache, sess)
        if not h:
            continue
        p = {L: prob_at(h, int((rd - dt.timedelta(days=L)).timestamp())) for L in leads}
        if any(p[L] is None for L in leads):
            continue
        cond = math.log((r.get(a.by) or 0) + 1)
        obs.append((p[Lb] - p[La], p[Ld] - p[Lc], cond))
    print(f"{len(obs)} resolved markets with all {leads} leads present\n")
    if len(obs) < 60:
        print("insufficient"); return

    DP = [o[0] for o in obs]; DN = [o[1] for o in obs]; C = [o[2] for o in obs]
    ic_all, n = spearman(DP, DN)
    lo, hi = boot_ic(np.array(DP), np.array(DN), 1)
    print(f"POOLED momentum IC  Spearman(Δp[{La}->{Lb}], Δp[{Lc}->{Ld}]) = {ic_all:+.3f}  "
          f"t≈{tstat(ic_all,n):+.2f}  90%CI[{lo:+.3f},{hi:+.3f}]  (n={n})\n")

    # bucket by liquidity, low -> high
    cv = np.array(C); edges = np.unique(np.quantile(cv, np.linspace(0, 1, a.q + 1)))
    bidx = np.clip(np.digitize(cv, edges[1:-1]), 0, len(edges) - 2)
    print(f"momentum IC conditioned on {a.by} (log), low→high:")
    bics = []
    for b in range(len(edges) - 1):
        sel = [o for o, bi in zip(obs, bidx) if bi == b]
        if len(sel) < 20:
            bics.append(np.nan); print(f"  b{b}: n={len(sel):4} (too few)"); continue
        dp = [o[0] for o in sel]; dn = [o[1] for o in sel]
        ic, nn = spearman(dp, dn); blo, bhi = boot_ic(np.array(dp), np.array(dn), 100 + b)
        bics.append(ic)
        vol_rng = f"[{math.exp(min(o[2] for o in sel)):,.0f},{math.exp(max(o[2] for o in sel)):,.0f}]"
        excl = "  <-- excl 0" if (blo > 0 or bhi < 0) else ""
        print(f"  b{b}: n={nn:4}  {a.by}∈{vol_rng:20}  IC {ic:+.3f}  90%CI[{blo:+.3f},{bhi:+.3f}]{excl}")
    valid = [(i, v) for i, v in enumerate(bics) if not math.isnan(v)]
    if len(valid) >= 3:
        gi, gv = zip(*valid)
        mono = float(np.corrcoef(_rank(gi), _rank(gv))[0, 1])
        print(f"\n  MONOTONICITY (liquidity-bucket vs IC): {mono:+.2f}")
        print("  → strong NEGATIVE = momentum concentrated in illiquid = CAN'T-REACT/slippage artifact")
        print("  → flat/positive   = momentum persists where you CAN trade = genuine underreaction")
        # the decisive cell: does momentum survive in the MOST-liquid bucket?
        top = valid[-1]
        print(f"\n  most-liquid bucket IC = {top[1]:+.3f}  "
              f"({'survives -> potentially tradeable' if top[1] > 0.05 else 'vanishes -> consistent with slippage artifact'})")


if __name__ == "__main__":
    main()
