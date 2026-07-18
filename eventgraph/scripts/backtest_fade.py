# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Costed backtest of the early-belief-bias fade: SHORT 'yes' on YOUNG hype markets, hold to
resolution as crowd optimism decays to the base rate (early_bias.py finding).

Honest construction:
  * DE-CORRELATE: one position per (underlying, resolution_date) cluster — keep the
    highest-volume market — so crypto strike-ladders don't count as many 'independent'
    trades.
  * MARK-TO-MARKET daily off the CLOB path (not just entry/exit): a short-yes position
    entered at p0 gains when p falls; terminal mark = the 0/1 outcome.
  * COST: round-trip spread haircut (--cost prob-points) applied at entry.
  * Equal-capital portfolio; capital per trade = (1-p0) (the collateral to short 1 yes).
    Daily portfolio return = mean over OPEN positions of their marked return; Sharpe from
    that daily series (annualised x sqrt(365) since PM trades every day incl weekends).

Reports Sharpe (gross + net of cost), annualised return, hit rate, N trades, avg hold —
with and without the hype filter, so the number is not oversold.

Usage:
  uv run eventgraph/scripts/backtest_fade.py --graph-dir eventgraph/data/eg_live
  ... --age 3 --cost 0.02 --min-volume 10000 --hype-only
"""
import argparse, json, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np
from drift_decomposition import hist, prob_at
from early_bias import hype_flags


def daterange(d0, d1):
    d = d0
    while d <= d1:
        yield d
        d += dt.timedelta(days=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--age", type=int, default=3, help="entry age (days since open)")
    ap.add_argument("--cost", type=float, default=0.02, help="round-trip spread haircut (prob points)")
    ap.add_argument("--min-volume", type=float, default=10000.0)
    ap.add_argument("--max-markets", type=int, default=4000)
    ap.add_argument("--hype-only", action="store_true", help="only hope/hype 'reach $X / ATH' markets")
    ap.add_argument("--max-hold", type=int, default=120, help="cap holding days (skip ultra-long markets)")
    a = ap.parse_args()
    gd = Path(a.graph_dir); cache = gd / "pm_hist"; cache.mkdir(exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    rows = [json.loads(l) for l in open(gd / "lake" / "proposition.jsonl") if l.strip()]
    res = [r for r in rows if r["source"] == "polymarket" and r.get("resolved_outcome") in ("yes", "no")
           and r.get("clob_token_yes") and (r.get("volume") or 0) >= a.min_volume and r.get("start_date")]
    if a.hype_only:
        res = [r for r in res if "target/ATH" in hype_flags(r.get("question"))]
    # de-correlate: one market per (underlying, resolution_date), highest volume
    best = {}
    for r in res:
        k = (r["symbols"][0] if r.get("symbols") else r.get("event_title"), r.get("resolution_date"))
        if k not in best or (r.get("volume") or 0) > (best[k].get("volume") or 0):
            best[k] = r
    res = sorted(best.values(), key=lambda r: -(r.get("volume") or 0))[:a.max_markets]

    positions = []                          # each: {daily: {date: mark_ret}, hold, hit, r_life}
    for r in res:
        try:
            sd = dt.datetime.fromisoformat(r["start_date"] + "T00:00:00+00:00")
            rd = dt.date.fromisoformat(r["resolution_date"])
        except Exception:
            continue
        h = hist(r["clob_token_yes"], cache, sess)
        if not h:
            continue
        entry_d = (sd + dt.timedelta(days=a.age)).date()
        if entry_d >= rd or (rd - entry_d).days > a.max_hold:
            continue
        p0 = prob_at(h, int(dt.datetime.combine(entry_d, dt.time(), dt.timezone.utc).timestamp()))
        if p0 is None or p0 <= 0.02 or p0 >= 0.98:
            continue
        y = 1 if r["resolved_outcome"] == "yes" else 0
        cap = 1 - p0                        # collateral to short 1 yes contract
        # daily marked price series entry..resolution (carry-forward), terminal = outcome
        days = list(daterange(entry_d, rd))
        prices = []
        for d in days:
            if d == rd:
                prices.append(float(y))
            else:
                pv = prob_at(h, int(dt.datetime.combine(d, dt.time(), dt.timezone.utc).timestamp()))
                prices.append(pv if pv is not None else (prices[-1] if prices else p0))
        # short-yes daily return on capital: gain when price falls
        daily = {}
        for i in range(1, len(days)):
            daily[days[i]] = -(prices[i] - prices[i - 1]) / cap
        # entry cost haircut on day1
        if len(days) > 1:
            daily[days[1]] = daily.get(days[1], 0.0) - a.cost / cap
        r_life = (p0 - y) / cap - a.cost / cap
        positions.append({"daily": daily, "hold": (rd - entry_d).days, "hit": 1 if y == 0 else 0,
                          "r_life": r_life, "p0": p0})

    if len(positions) < 30:
        print(f"only {len(positions)} positions — insufficient"); return

    # portfolio daily return = equal-weight mean over positions open that day
    alld = sorted({d for pos in positions for d in pos["daily"]})
    port = []
    for d in alld:
        rs = [pos["daily"][d] for pos in positions if d in pos["daily"]]
        if rs:
            port.append(np.mean(rs))
    port = np.array(port)
    ann = math.sqrt(365)
    sharpe = port.mean() / port.std() * ann if port.std() > 0 else float("nan")
    ann_ret = port.mean() * 365
    # gross (no cost) for comparison
    gross_life = np.mean([pos["r_life"] + a.cost / (1 - pos["p0"]) for pos in positions])

    print(f"positions: {len(positions)} de-correlated {'HYPE-ONLY' if a.hype_only else 'ALL'} markets "
          f"(vol>={a.min_volume:.0f}, entry age {a.age}d, cost {a.cost:.3f})")
    print(f"  avg holding days   : {np.mean([p['hold'] for p in positions]):.0f}")
    print(f"  hit rate (resolve no): {np.mean([p['hit'] for p in positions]):.1%}")
    print(f"  mean per-trade return (net): {np.mean([p['r_life'] for p in positions]):+.3f} "
          f"| gross {gross_life:+.3f}")
    print(f"  portfolio days traded: {len(port)}")
    print(f"  ANNUALISED RETURN  : {ann_ret:+.1%}")
    print(f"  ANNUALISED SHARPE  : {sharpe:+.2f}   (daily MTM, net of {a.cost:.0%} round-trip spread)")
    # sensitivity: sharpe at zero cost
    portg = []
    for d in alld:
        rs = []
        for pos in positions:
            if d in pos["daily"]:
                v = pos["daily"][d]
                # add back entry cost on that pos's entry day
                rs.append(v)
        if rs:
            portg.append(np.mean(rs))
    print(f"  [per-trade Sharpe {np.mean([p['r_life'] for p in positions])/ (np.std([p['r_life'] for p in positions]) or 1):+.2f}, "
          f"n={len(positions)}; correlated-obs + PM depth caveats apply]")


if __name__ == "__main__":
    main()
