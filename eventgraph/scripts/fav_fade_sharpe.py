# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Sharpe of the EARLY-FAVORITE fade — fade outcomes priced >= T one day after open (where
the 1d calibration showed 50-90% legs over-priced by 10-25 pts), held to resolution.

Honest treatment (the calibration that motivates this is IN-SAMPLE, so guard hard):
  * fixed $1 notional, short yes, PnL = p0 - outcome;
  * entry price at age 1d from the HOURLY opening-window cache (pm_hist_hr), MTM daily to
    resolution off the DAILY cache (pm_hist);
  * per-trade + CLUSTER-ROBUST (by underlying) edge & t; daily-MTM annualised Sharpe;
  * COST sweep; OUT-OF-SAMPLE temporal split (fit period vs holdout);
  * LIQUIDITY reality check: median total volume of the selected markets (can you even
    fill a fade at day 1 before it corrects?).

Usage:
  uv run eventgraph/scripts/fav_fade_sharpe.py --graph-dir eventgraph/data/eg_live --thresh 0.5
"""
import argparse, json, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at   # daily cache reader

HR = None


def hr_price_at(cache, token, t0, age):
    f = cache / f"{token}.json"
    if not f.exists():
        return None
    try:
        h = json.loads(f.read_text())
    except Exception:
        return None
    tgt = t0 + age * 86400; best = None
    for pt in h:
        if pt["t"] <= tgt + 1800 and (best is None or pt["t"] > best["t"]):
            best = pt
    return best["p"] if best else None


def _rank(x):
    return np.argsort(np.argsort(np.asarray(x, float))).astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--thresh", type=float, default=0.5, help="fade outcomes priced >= this at 1d")
    ap.add_argument("--age", type=float, default=1.0)
    ap.add_argument("--cost", type=float, default=0.02)
    ap.add_argument("--min-volume", type=float, default=20000.0)
    ap.add_argument("--max-hold", type=int, default=120)
    a = ap.parse_args()
    gd = Path(a.graph_dir); hrc = gd / "pm_hist_hr"; daily = gd / "prices"; daily.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})
    cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume and r.get("start_date")]

    trades = []
    for r in res:
        try:
            sd = dt.datetime.fromisoformat(r["start_date"] + "T00:00:00+00:00")
            rd = dt.date.fromisoformat(r["resolution_date"])
        except Exception:
            continue
        t0 = int(sd.timestamp())
        p0 = hr_price_at(hrc, r["clob_token_yes"], t0, a.age)
        if p0 is None or p0 < a.thresh or p0 >= 0.98:
            continue
        entry = (sd + dt.timedelta(days=a.age)).date()
        if entry >= rd or (rd - entry).days > a.max_hold:
            continue
        y = 1 if r["resolved_outcome"] == "yes" else 0
        und = r["symbols"][0] if r.get("symbols") else (r.get("event_title") or r["prop_id"])
        # daily marks entry..resolution off the daily path (fallback: flat p0 then outcome)
        h = hist(r["clob_token_yes"], cache, sess)
        days = [entry + dt.timedelta(days=i) for i in range((rd - entry).days + 1)]
        prices = []
        for d in days:
            if d == rd:
                prices.append(float(y))
            else:
                pv = prob_at(h, int(dt.datetime.combine(d, dt.time(), dt.timezone.utc).timestamp())) if h else None
                prices.append(pv if pv is not None else (prices[-1] if prices else p0))
        cap = max(1 - p0, 0.05)
        daily_ret = {}
        for i in range(1, len(days)):
            daily_ret[days[i]] = -(prices[i] - prices[i - 1])          # fixed $1 notional short-yes
        if len(days) > 1:
            daily_ret[days[1]] = daily_ret.get(days[1], 0.0) - a.cost
        trades.append({"und": und, "p0": p0, "y": y, "pnl": p0 - y - a.cost,
                       "daily": daily_ret, "res": rd, "vol": r.get("volume") or 0,
                       "hold": (rd - entry).days})
    if len(trades) < 25:
        print(f"only {len(trades)} trades priced>={a.thresh} at {a.age}d"); return

    pnl = np.array([t["pnl"] for t in trades])
    print(f"=== fade outcomes priced >= {a.thresh} at {a.age}d after open (net {a.cost:.0%} cost) ===")
    print(f"  {len(trades)} trades | avg entry price {np.mean([t['p0'] for t in trades]):.2f} | "
          f"resolve-no {np.mean([1-t['y'] for t in trades]):.0%} | avg hold {np.mean([t['hold'] for t in trades]):.0f}d")
    print(f"  per-trade edge {pnl.mean():+.3f}  std {pnl.std():.3f}  PER-TRADE SHARPE {pnl.mean()/pnl.std():+.3f}")

    # cluster-robust by underlying
    byu = collections.defaultdict(list)
    for t in trades:
        byu[t["und"]].append(t["pnl"])
    cl = np.array([np.mean(v) for v in byu.values()])
    se = cl.std(ddof=1) / math.sqrt(len(cl))
    print(f"  cluster-robust ({len(cl)} underlyings): edge {cl.mean():+.3f} ±{2*se:.3f}  "
          f"t={cl.mean()/se:+.2f}  {'SIG' if abs(cl.mean())>2*se else 'n.s.'}")

    # daily-MTM annualised Sharpe
    alld = sorted({d for t in trades for d in t["daily"]})
    port = np.array([np.mean([t["daily"][d] for t in trades if d in t["daily"]]) for d in alld
                     if any(d in t["daily"] for t in trades)])
    if port.std() > 0:
        print(f"  daily-MTM portfolio: ANNUALISED SHARPE {port.mean()/port.std()*math.sqrt(365):+.2f}  "
              f"(ann.return {port.mean()*365:+.0%}, {len(port)} days)")

    # OUT-OF-SAMPLE temporal split
    trades.sort(key=lambda t: t["res"]); mid = len(trades) // 2
    for lbl, sub in [("in-sample (early)", trades[:mid]), ("OUT-OF-SAMPLE (late)", trades[mid:])]:
        bu = collections.defaultdict(list)
        for t in sub:
            bu[t["und"]].append(t["pnl"])
        c = np.array([np.mean(v) for v in bu.values()]); s = c.std(ddof=1) / math.sqrt(len(c))
        print(f"  {lbl:22} edge {c.mean():+.3f} ±{2*s:.3f} ({len(c)} und)  {'SIG' if abs(c.mean())>2*s else 'n.s.'}")

    # liquidity reality: total volume of selected markets (capacity at day 1 is a FRACTION of this)
    v = np.array([t["vol"] for t in trades])
    print(f"  LIQUIDITY: selected-market lifetime volume median ${np.median(v):,.0f} "
          f"[IQR ${np.percentile(v,25):,.0f}-${np.percentile(v,75):,.0f}] — day-1 fillable size is a small fraction")


if __name__ == "__main__":
    main()
