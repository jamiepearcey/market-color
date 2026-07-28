# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
MULTI-REGIME replication, on Polymarket, of the Kalshi macro finding in
data/eg_live/pm_calibration_oos.json: "mid-to-high-priced (~50-90c) macro/economic
YES-contracts are systematically overpriced; buying NO nets positive after costs."
Kalshi's public API has a hard ~2.5-month rolling retention wall (one regime only).
Polymarket's on-chain history (data/eg_live/polymarket_macro/markets.jsonl, built by
scripts/polymarket_macro_pull.py -- genuine resolvedBy+clean-outcomePrices settlement,
never price-derived) spans 2021-2025, several distinct macro regimes (2021-22 rate-hike
cycle, 2023 disinflation, 2024-25 cut cycle). This script:

  1. overall calibration curve (decile bins) + favorite-longshot read.
  2. applies the SAME pre-registered rule as the Kalshi OOS study, UNCHANGED:
     buy NO on macro YES priced in [0.5, 0.9) (data/eg_live/pm_calibration_oos.json
     ->preregistered_rule.rule_bin). No re-fitting on Polymarket data -- this is a
     pure stress test of an already-fixed rule on a new venue/regime set.
  3. THE KEY CUT: same rule, split by RESOLUTION YEAR (2021-2025) as separate
     regimes. Sign/magnitude stability across years is the multi-regime validation
     a single 72-day Kalshi window structurally cannot give.
  4. sub-category (cpi_inflation/fed_rates/jobs/gdp/other_macro) sign check.

COSTS: Polymarket charged ~0% explicit taker fees historically (unlike Kalshi's
0.07*p*(1-p) schedule) -- so GROSS here is not "the real venue cost" the way it was
on Kalshi. We instead model a SPREAD/slippage haircut for a taker crossing the book:
TAKER_HAIRCUT cents flat per contract (stated below), applied symmetrically to entry
price, on top of $0 explicit fee. This is a stated assumption, not measured from a
Polymarket order-book sample (none is on disk here) -- flagged explicitly, same
spirit as the Kalshi OOS study's own spread-proxy caveat.

Usage:
  uv run scripts/pm_calibration_polymarket.py --graph-dir data/eg_live
"""
import argparse, json, math, statistics
from pathlib import Path

RULE_LO, RULE_HI = 0.5, 0.9  # fixed, imported from the Kalshi OOS pre-registration; not re-fit here
TAKER_HAIRCUT = 0.02  # 2c/contract one-way slippage/spread-crossing proxy (Polymarket has ~0% explicit fee)


def decile_bins(rows, n_bins=10):
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        idx = min(n_bins - 1, max(0, int(r["price_yes"] * n_bins)))
        bins[idx].append(r)
    out = []
    for i, b in enumerate(bins):
        n = len(b)
        if n == 0:
            out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": 0,
                        "mean_price": None, "realized_freq": None, "edge": None})
            continue
        mp = statistics.mean(r["price_yes"] for r in b)
        fr = statistics.mean(1 if r["outcome"] == "yes" else 0 for r in b)
        out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": n,
                    "mean_price": round(mp, 4), "realized_freq": round(fr, 4),
                    "edge": round(fr - mp, 4)})
    return out


def stats(xs):
    if not xs:
        return {"n": 0}
    m = statistics.mean(xs)
    se = statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
    t = (m / se) if se else None
    return {"n": len(xs), "mean": round(m, 4), "total": round(sum(xs), 2),
            "se": round(se, 4) if se else None, "t_stat": round(t, 3) if t else None}


def apply_rule(rows, lo, hi):
    """buy NO at (1-price_yes); payoff = 1 if outcome==no else 0."""
    sub = [r for r in rows if lo <= r["price_yes"] < hi]
    gross, net = [], []
    for r in sub:
        entry = 1 - r["price_yes"]
        payoff = 1 if r["outcome"] == "no" else 0
        g = payoff - entry
        gross.append(g)
        net.append(g - TAKER_HAIRCUT)
    out = {"n": len(sub), "gross": stats(gross), "net_taker": stats(net)}
    if sub:
        out["calibration_in_band"] = {
            "mean_price": round(statistics.mean(r["price_yes"] for r in sub), 4),
            "realized_freq": round(statistics.mean(1 if r["outcome"] == "yes" else 0 for r in sub), 4),
        }
    return out


def sign(x):
    if x is None:
        return "n/a"
    return "+" if x > 0 else ("-" if x < 0 else "0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/eg_live")
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    inpath = gd / "polymarket_macro" / "markets.jsonl"
    rows = [json.loads(l) for l in open(inpath) if l.strip()]
    print(f"loaded {len(rows)} genuine-settlement macro markets from {inpath}")

    years = sorted({r["resolution_year"] for r in rows})
    print(f"year span: {years[0]}..{years[-1]} ({years})")

    calib = decile_bins(rows)
    print("\noverall calibration (price bin -> realized YES freq, n):")
    for b in calib:
        print(f"  {b}")
    hi_bins = [b for b in calib if b["bin"][0] >= 0.5 and b["n"]]
    lo_bins = [b for b in calib if b["bin"][1] <= 0.5 and b["n"]]
    hi_edge = statistics.mean(b["edge"] for b in hi_bins) if hi_bins else None
    lo_edge = statistics.mean(b["edge"] for b in lo_bins) if lo_bins else None
    flb_verdict = (
        f"high-price bins mean edge={round(hi_edge,4) if hi_edge is not None else None} "
        f"(sign {sign(hi_edge)} = {'overpriced YES' if (hi_edge or 0) < 0 else 'underpriced YES' if hi_edge else 'n/a'}); "
        f"low-price bins mean edge={round(lo_edge,4) if lo_edge is not None else None} (sign {sign(lo_edge)})"
    )
    print(f"\nfavorite-longshot read: {flb_verdict}")

    overall_rule = apply_rule(rows, RULE_LO, RULE_HI)
    print(f"\nbuy-NO rule [{RULE_LO},{RULE_HI}) OVERALL: n={overall_rule['n']}")
    print(f"  gross: {overall_rule.get('gross')}")
    print(f"  net (taker, {TAKER_HAIRCUT} haircut): {overall_rule.get('net_taker')}")

    per_year = {}
    for y in years:
        yr_rows = [r for r in rows if r["resolution_year"] == y]
        res = apply_rule(yr_rows, RULE_LO, RULE_HI)
        per_year[y] = res
        g = res.get("gross", {})
        nt = res.get("net_taker", {})
        print(f"  YEAR {y}: n_in_band={res['n']}  gross_net/contract={g.get('mean')} "
              f"net_taker/contract={nt.get('mean')} sign={sign(nt.get('mean'))}")

    signs = [sign(per_year[y].get("net_taker", {}).get("mean")) for y in years if per_year[y]["n"] > 0]
    n_pos = signs.count("+"); n_neg = signs.count("-")
    if not signs:
        verdict = "no year has any markets in the rule band -- inconclusive, sample too thin."
    elif n_neg == 0:
        verdict = f"net-taker sign is POSITIVE in all {n_pos} years with data -- replicates across regimes."
    elif n_pos == 0:
        verdict = f"net-taker sign is NEGATIVE in all {n_neg} years with data -- does NOT replicate (rule fails on Polymarket)."
    else:
        verdict = f"net-taker sign FLIPS across years ({n_pos} positive / {n_neg} negative) -- not regime-stable."

    subcat = {}
    for name in ["cpi_inflation", "fed_rates", "jobs", "gdp", "other_macro"]:
        cell = [r for r in rows if r["subcat"] == name and RULE_LO <= r["price_yes"] < RULE_HI]
        if not cell:
            subcat[name] = {"n": 0}
            continue
        net = [((1 - r["price_yes"]) * -1 + (1 if r["outcome"] == "no" else 0)) - TAKER_HAIRCUT for r in cell]
        # equivalent: payoff - entry - haircut
        net = [(1 if r["outcome"] == "no" else 0) - (1 - r["price_yes"]) - TAKER_HAIRCUT for r in cell]
        subcat[name] = {"n": len(cell), "net_taker_mean": round(statistics.mean(net), 4), "sign": sign(statistics.mean(net))}
    print("\nsub-category sign check (within rule band, net taker):")
    for k, v in subcat.items():
        print(f"  {k}: {v}")

    out = {
        "data_source": str(inpath),
        "n_markets": len(rows), "year_span": [years[0], years[-1]] if years else None,
        "by_year_n": {y: sum(1 for r in rows if r["resolution_year"] == y) for y in years},
        "overall_calibration": calib,
        "favorite_longshot_read": flb_verdict,
        "rule": {"lo": RULE_LO, "hi": RULE_HI, "source": "data/eg_live/pm_calibration_oos.json preregistered_rule.rule_bin (fixed, not re-fit)"},
        "cost_model": {
            "explicit_fee": 0.0,
            "note": "Polymarket historically ~0% explicit taker fee",
            "taker_haircut_per_contract": TAKER_HAIRCUT,
            "haircut_note": "flat cents proxy for spread-crossing/slippage; not measured from an on-disk Polymarket order-book sample -- stated assumption, flagged explicitly",
        },
        "rule_overall": overall_rule,
        "rule_by_year": per_year,
        "rule_by_subcategory": subcat,
        "verdict": verdict,
    }
    outpath = gd / "pm_calibration_polymarket.json"
    outpath.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
