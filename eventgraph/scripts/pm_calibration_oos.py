# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
PRE-REGISTERED, temporally-held-out stress test of the pm_calibration.py macro
finding: "mid-to-high-priced (~55-90c) macro/economic Kalshi YES-contracts are
systematically overpriced; buying NO nets positive after fees."

That finding (data/eg_live/pm_calibration.json) was underpowered: ~2.5mo window,
thin bins, no real out-of-sample split, no multiple-comparison discipline, and
NET P&L that only paid the per-contract fee at the last traded price (an
optimistic maker-fill assumption).

DATA: scripts/kalshi_macro_backfill.py paginated Kalshi's public /markets
status=settled endpoint to EXHAUSTION for every macro-titled series (280 series
across all categories, same MACRO_KW regex pm_calibration.py uses). Result:
settled macro history is available ONLY 2026-05-13..2026-07-24 (~72 days) no
matter how hard we paginate -- cursors go empty immediately, well inside the
per-series page size. This is a hard rolling-retention wall on Kalshi's public
API, not a fetch limitation on our side: the window barely grew vs the original
study. This alone matters and is reported plainly.

PROTOCOL (fixed before touching TEST):
  1. Split macro rows by RESOLUTION (close_time) date: first 65% of the
     macro-row timeline = TRAIN, remaining 35% = TEST (strictly later,
     non-overlapping -- a real temporal holdout, not adjacent same-regime
     halves; short window means H1/H2 straddle different FOMC/CPI/jobs prints
     by construction).
  2. On TRAIN ONLY: decile-bin macro calibration (mean price vs realized
     frequency). Scan the mid-to-high price deciles the original finding was
     about -- [0.5,0.6) [0.6,0.7) [0.7,0.8) [0.8,0.9) -- and keep the maximal
     contiguous run (starting at 0.5) where n >= MIN_N and edge (freq-price) <
     0, i.e. YES still overpriced. That run's [lo,hi) IS the single
     pre-registered rule: buy NO on macro YES-contracts priced in [lo,hi).
     No other rule is tried; nothing here depends on TEST.
  3. On TEST: apply that ONE rule unchanged. Report gross P&L/contract,
     NET-MAKER (fee only, the optimistic assumption the original study used)
     and NET-TAKER (fee + spread-crossing haircut, the pessimistic assumption
     this stress test adds) -- with N, SE, t-stat, and the TEST-period
     calibration inside the rule band.
  4. Robustness: same rule, same TEST window, split by macro sub-category
     (CPI/inflation, Fed/rates, jobs/payrolls, GDP, other) -- sign check only,
     tiny N per cell, reported as such.

FEE: Kalshi real taker fee, fee = ceil(0.07*P*(1-P)*100)/100 (same formula as
pm_calibration.py / kalshi_drift.py).
TAKER HAIRCUT: settlement-snapshot bid/ask on kalshi_markets/*.json degenerates
to 0/1 post-resolution (useless for pre-resolution spread), and the
kalshi_candles/ cache (322 tickers with real daily bid/ask) has ZERO overlap
with the macro ticker set fetched here -- so there is no macro-specific spread
sample on disk. We use the best available real proxy: the observed spread
distribution on ALL cached tickers in the same 0.55-0.90 price band (n=4012
candle-days, median full spread 0.05 = 5c) and charge HALF of that median (2.5c)
as a one-way taker haircut on top of the real fee. This is a cross-category
proxy, not macro-specific -- flagged explicitly, and is intentionally on the
pessimistic side (crossing meaningful book width, not a maker fill).

Usage:
  uv run scripts/pm_calibration_oos.py --graph-dir data/eg_live
"""
import argparse, glob, json, math, re, statistics, datetime as dt
from pathlib import Path

MACRO_KW = re.compile(
    r"\b(cpi|core cpi|pce|core pce|gdp|fomc|fed(?:eral reserve)?|interest rate|fed funds|"
    r"payroll|nonfarm|unemployment|jobless claims|treasury|10-?year|yield|recession|"
    r"inflation|ism|pmi|retail sales|housing starts)\b", re.I)

SUBCAT_PATTERNS = [
    ("cpi_inflation", re.compile(r"\b(cpi|pce|inflation)\b", re.I)),
    ("fed_rates", re.compile(r"\b(fomc|fed(?:eral reserve)?|interest rate|fed funds)\b", re.I)),
    ("jobs", re.compile(r"\b(payroll|nonfarm|unemployment|jobless claims)\b", re.I)),
    ("gdp", re.compile(r"\bgdp\b", re.I)),
]

TRAIN_FRAC = 0.65
SCAN_LO, SCAN_HI, SCAN_STEP = 0.5, 0.9, 0.1  # mid-to-high price zone the finding is about
MIN_N = 25  # matches pm_calibration.py's own macro min_bin_n
TAKER_HALF_SPREAD = 0.025  # see module docstring: half of observed median spread, cross-category proxy


def fee(p):
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def to_epoch(s):
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def load_category_map(gd):
    smap = {}
    p = gd / "lake" / "proposition.jsonl"
    if not p.exists():
        return smap
    for line in open(p):
        try:
            j = json.loads(line)
        except Exception:
            continue
        if j.get("source") == "kalshi" and j.get("series"):
            smap[j["series"]] = j.get("category")
    return smap


def load_macro_rows(gd):
    smap = load_category_map(gd)
    rows = []
    n_finalized = n_result = 0
    for fn in glob.glob(str(gd / "kalshi_markets" / "*.json")):
        try:
            markets = json.load(open(fn))
        except Exception:
            continue
        for m in markets:
            if m.get("status") != "finalized":
                continue
            n_finalized += 1
            res = (m.get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            n_result += 1
            try:
                vol = float(m.get("volume_fp") or 0)
            except Exception:
                vol = 0.0
            if vol <= 0:
                continue
            try:
                price = float(m.get("last_price_dollars"))
            except (TypeError, ValueError):
                continue
            if not (0.0 <= price <= 1.0):
                continue
            close_ep = to_epoch(m.get("close_time"))
            if close_ep is None:
                continue
            ticker = m.get("ticker") or ""
            series = ticker.split("-")[0] if ticker else ""
            text = " ".join(x for x in [m.get("title"), m.get("yes_sub_title"), m.get("rules_primary")] if x)
            cat = smap.get(series)
            is_macro = (cat == "macro") if cat else bool(MACRO_KW.search(text or ""))
            if not is_macro:
                continue
            subcat = "other_macro"
            for name, pat in SUBCAT_PATTERNS:
                if pat.search(text or ""):
                    subcat = name
                    break
            rows.append({
                "ticker": ticker, "series": series, "price": price,
                "y": 1 if res == "yes" else 0, "close_time": m.get("close_time"),
                "close_ep": close_ep, "text": text, "subcat": subcat,
            })
    rows.sort(key=lambda r: r["close_ep"])
    return rows, n_finalized, n_result


def decile_bins(rows, n_bins=10):
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        idx = min(n_bins - 1, int(r["price"] * n_bins))
        bins[idx].append(r)
    out = []
    for i, b in enumerate(bins):
        n = len(b)
        if n == 0:
            out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": 0,
                        "mean_price": None, "realized_freq": None, "edge": None})
            continue
        mp = statistics.mean(r["price"] for r in b)
        fr = statistics.mean(r["y"] for r in b)
        out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": n,
                    "mean_price": round(mp, 4), "realized_freq": round(fr, 4),
                    "edge": round(fr - mp, 4)})
    return out


def preregister_rule(train_rows, min_n, scan_lo, scan_hi, scan_step):
    """Mechanical, TRAIN-only selection: maximal contiguous run of deciles in
    [scan_lo, scan_hi) with n>=min_n and edge<0, starting from scan_lo. No TEST data used."""
    bins = decile_bins(train_rows)
    scan_bins = [b for b in bins if scan_lo - 1e-9 <= b["bin"][0] < scan_hi - 1e-9]
    lo = None
    hi = None
    trace = []
    for b in scan_bins:
        qualifies = b["n"] >= min_n and b["edge"] is not None and b["edge"] < 0
        trace.append({**b, "qualifies": qualifies})
        if qualifies:
            if lo is None:
                lo = b["bin"][0]
            hi = b["bin"][1]
        else:
            break  # contiguous run stops at first non-qualifying decile
    return {"rule_bin": [lo, hi] if lo is not None else None, "scan_trace": trace}


def apply_rule(rows, lo, hi):
    """buy NO at (1-price) on rows with price in [lo,hi); payoff = 1-y."""
    sub = [r for r in rows if lo <= r["price"] < hi]
    n = len(sub)
    if n == 0:
        return {"n": 0}
    gross, net_maker, net_taker = [], [], []
    for r in sub:
        entry = 1 - r["price"]
        payoff = 1 - r["y"]
        f = fee(entry)
        g = payoff - entry
        gross.append(g)
        net_maker.append(g - f)
        net_taker.append(g - f - TAKER_HALF_SPREAD)
    def stats(xs):
        m = statistics.mean(xs)
        se = statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else None
        t = (m / se) if se else None
        return {"mean": round(m, 4), "total": round(sum(xs), 2), "se": round(se, 4) if se else None,
                "t_stat": round(t, 3) if t else None}
    return {
        "n": n, "bin": [round(lo, 2), round(hi, 2)],
        "gross": stats(gross), "net_maker_fee_only": stats(net_maker), "net_taker_fee_plus_haircut": stats(net_taker),
        "calibration_in_band": {
            "mean_price": round(statistics.mean(r["price"] for r in sub), 4),
            "realized_freq": round(statistics.mean(r["y"] for r in sub), 4),
        },
    }


def cluster_robust(rows, lo, hi, key_fn):
    """Cluster-robust mean/SE/t of net-TAKER P&L, resampled at the cluster level
    (e.g. by series, or by series+release-date) -- because many strike-ladder
    contracts on the SAME print (e.g. 8 different CPI thresholds on one release)
    are highly correlated draws, not independent trials; naive per-contract SE
    overstates significance if effective N is really "number of releases", not
    "number of strikes"."""
    sub = [r for r in rows if lo <= r["price"] < hi]
    by = {}
    for r in sub:
        entry = 1 - r["price"]; payoff = 1 - r["y"]
        net = payoff - entry - fee(entry) - TAKER_HALF_SPREAD
        by.setdefault(key_fn(r), []).append(net)
    means = [statistics.mean(v) for v in by.values()]
    if len(means) < 2:
        return {"n_clusters": len(means), "mean": None, "se": None, "t_stat": None}
    m = statistics.mean(means)
    se = statistics.stdev(means) / math.sqrt(len(means))
    return {"n_clusters": len(means), "mean": round(m, 4), "se": round(se, 4),
            "t_stat": round(m / se, 3) if se else None}


def subcategory_check(rows, lo, hi):
    sub = [r for r in rows if lo <= r["price"] < hi]
    out = {}
    for name in ["cpi_inflation", "fed_rates", "jobs", "gdp", "other_macro"]:
        cell = [r for r in sub if r["subcat"] == name]
        if not cell:
            out[name] = {"n": 0}
            continue
        net_taker = []
        for r in cell:
            entry = 1 - r["price"]; payoff = 1 - r["y"]
            net_taker.append(payoff - entry - fee(entry) - TAKER_HALF_SPREAD)
        out[name] = {"n": len(cell), "net_taker_mean": round(statistics.mean(net_taker), 4),
                     "sign": "+" if statistics.mean(net_taker) > 0 else "-"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/eg_live")
    ap.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    a = ap.parse_args()
    gd = Path(a.graph_dir)

    rows, n_finalized, n_result = load_macro_rows(gd)
    n = len(rows)
    cut = int(n * a.train_frac)
    train, test = rows[:cut], rows[cut:]

    print(f"macro rows (finalized, traded, valid price, close_time present): {n}")
    print(f"date span: {rows[0]['close_time'][:10]} .. {rows[-1]['close_time'][:10]}" if rows else "no rows")
    print(f"TRAIN n={len(train)} [{train[0]['close_time'][:10]}..{train[-1]['close_time'][:10]}]")
    print(f"TEST  n={len(test)} [{test[0]['close_time'][:10]}..{test[-1]['close_time'][:10]}]")

    reg = preregister_rule(train, MIN_N, SCAN_LO, SCAN_HI, SCAN_STEP)
    print("\nTRAIN scan (pre-registration, mid-to-high price deciles):")
    for t in reg["scan_trace"]:
        print(f"  {t}")
    rule = reg["rule_bin"]
    print(f"\nPRE-REGISTERED RULE (from TRAIN only): buy NO on macro YES priced in {rule}")

    out = {
        "step1_data": {
            "source": "Kalshi public API /markets status=settled, paginated to cursor-exhaustion "
                       "(scripts/kalshi_macro_backfill.py), 280 macro-titled series across all categories",
            "n_macro_finalized_labeled": n,
            "date_span": [rows[0]["close_time"][:10], rows[-1]["close_time"][:10]] if rows else None,
            "api_reach_finding": "settled-market history for EVERY macro series tested terminates at "
                                  "2026-05-13..2026-05-21 regardless of full cursor pagination (cursor "
                                  "goes empty well under page size) -- this is Kalshi's public API rolling "
                                  "retention wall, not a pagination bug on our side. The window could not "
                                  "be meaningfully extended beyond what the original pm_calibration.py study "
                                  "already had (~2.5 months); N grew (1913->2431 macro rows) via fuller "
                                  "series coverage + exhaustive pagination, but the DATE SPAN did not.",
        },
        "protocol": {
            "train_frac": a.train_frac,
            "train_n": len(train), "test_n": len(test),
            "train_range": [train[0]["close_time"][:10], train[-1]["close_time"][:10]],
            "test_range": [test[0]["close_time"][:10], test[-1]["close_time"][:10]],
            "scan_zone": [SCAN_LO, SCAN_HI], "min_n": MIN_N,
        },
        "preregistered_rule": {
            "train_scan_trace": reg["scan_trace"],
            "rule_bin": rule,
            "rule_text": f"buy NO on macro YES-contracts priced in [{rule[0]},{rule[1]})" if rule else "NO QUALIFYING RULE ON TRAIN",
        },
        "fee_formula": "ceil(0.07*p*(1-p)*100)/100 per contract (Kalshi real taker fee schedule)",
        "taker_haircut_assumption": f"{TAKER_HALF_SPREAD} (half of observed median full spread 0.05 in "
                                     f"[0.55,0.90] price band, n=4012 candle-days, ALL cached tickers -- "
                                     f"cross-category proxy, zero macro-ticker overlap in the candle cache, "
                                     f"flagged explicitly) charged in addition to the real fee",
    }

    if rule:
        lo, hi = rule
        train_result = apply_rule(train, lo, hi)
        test_result = apply_rule(test, lo, hi)
        test_calib = decile_bins(test)
        subcat = subcategory_check(test, lo, hi)
        cluster_series = cluster_robust(test, lo, hi, lambda r: r["series"])
        cluster_event = cluster_robust(test, lo, hi, lambda r: r["series"] + "|" + r["close_time"][:10])
        out["train_result_in_sample_reference_only"] = train_result
        out["test_result_out_of_sample"] = test_result
        out["test_period_full_calibration"] = test_calib
        out["subcategory_sign_check_test"] = subcat
        out["cluster_robust_net_taker"] = {
            "note": "resamples at cluster level since many strike-ladder contracts share one underlying "
                    "print (correlated, not independent trials); naive per-contract SE above may overstate significance.",
            "by_series": cluster_series, "by_series_and_release_date": cluster_event,
        }

        print(f"\nTEST result, rule {rule}: n={test_result.get('n')}")
        if test_result.get("n"):
            print(f"  gross:      {test_result['gross']}")
            print(f"  net MAKER (fee only, optimistic):    {test_result['net_maker_fee_only']}")
            print(f"  net TAKER (fee+haircut, pessimistic): {test_result['net_taker_fee_plus_haircut']}")
            print(f"  calibration in band: {test_result['calibration_in_band']}")
            print(f"  cluster-robust (by series):            {cluster_series}")
            print(f"  cluster-robust (by series+release-date): {cluster_event}")
        print("\nsub-category sign check (TEST, within rule band, net TAKER):")
        for k, v in subcat.items():
            print(f"  {k}: {v}")
    else:
        out["test_result_out_of_sample"] = None
        print("\nNo TRAIN-qualifying rule -> nothing to test out-of-sample (null result).")

    outpath = gd / "pm_calibration_oos.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}")


if __name__ == "__main__":
    main()
