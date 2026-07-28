# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
Prediction-market CALIBRATION study (Kalshi): are contract prices systematically
miscalibrated vs realized outcome (favorite-longshot bias / drift-into-resolution),
and does the bias survive Kalshi's real per-contract fee?

GROUND TRUTH: Kalshi's on-disk settlement field (`result` yes/no) on FINALIZED
markets in data/eg_live/kalshi_markets/<series>.json — the raw /markets response,
one file per series, already fetched to disk (see scripts/kalshi_ingest.py /
kalshi_drift.py for the same schema/conventions). This is the cleanest possible
label: it is Kalshi's own settlement, not an inferred/derived one.

Polymarket is DELIBERATELY EXCLUDED from the outcome-labelled study: proposition.jsonl's
`resolved_outcome` for polymarket is *derived* from outcomePrices>=0.5 at close, not a
real settlement field on disk — using it as ground truth would be circular (it bakes
"price==outcome" into the label itself, guaranteeing fake calibration). No FRED_KEY /
FRED_API_KEY is set in this environment, so per the task's own rule we restrict to the
on-disk-settled contracts (Kalshi) and say so, rather than guess an external outcome.

Two price horizons:
  (a) NEAR-RESOLUTION, full sample: `last_price_dollars` on every finalized, traded
      (volume_fp>0) market — the last observed trade before settlement. N in the
      thousands.
  (b) FIXED HORIZON >=24h before close: only possible where we have an actual price
      PATH, which is the much smaller kalshi_candles/<ticker>.json cache (322 tickers,
      daily candles, {t,p,sp}; same cache kalshi_drift.py reads/writes). For this
      subset we compare price ~24h out vs price at the last available candle
      (near-resolution) to see calibration bias and its drift into resolution.

FEE: Kalshi's real per-contract taker fee, fee = ceil(0.07 * P * (1-P) * 100)/100
dollars (round UP to the cent), same formula scripts/kalshi_drift.py uses (there
left unrounded per-contract; here rounded to match live fee schedule). We report
GROSS (no fee) vs NET (buy at the recorded last price, pay only the fee — a
maker/best-case fill, no assumed spread-crossing, noted explicitly).

WALK-FORWARD: split finalized markets by close_time date at the sample median into
H1 (earlier) / H2 (later); the strategy's bin selection is fixed from the FULL
sample and re-evaluated unchanged on each half, so H2 is a genuine out-of-sample
check on H1-discovered bins.

Usage:
  uv run scripts/pm_calibration.py --graph-dir data/eg_live
"""
import argparse, glob, json, math, re, statistics, datetime as dt
from pathlib import Path
import numpy as np

MACRO_KW = re.compile(
    r"\b(cpi|core cpi|pce|core pce|gdp|fomc|fed(?:eral reserve)?|interest rate|fed funds|"
    r"payroll|nonfarm|unemployment|jobless claims|treasury|10-?year|yield|recession|"
    r"inflation|ism|pmi|retail sales|housing starts)\b", re.I)


def fee(p):
    """Kalshi real per-contract taker fee: ceil(0.07*P*(1-P)*100)/100, P in dollars."""
    return math.ceil(0.07 * p * (1 - p) * 100) / 100.0


def to_epoch(s):
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def price_at_or_before(cs, tgt, grace=43200):
    """nearest candle at or before tgt+grace (12h grace, matches kalshi_drift.py)."""
    best = None
    for c in cs:
        if c["t"] <= tgt + grace and (best is None or c["t"] > best["t"]):
            best = c
    return best["p"] if best else None


def load_finalized(gd):
    """all finalized, result-bearing, traded Kalshi markets across kalshi_markets/*.json."""
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
            ticker = m.get("ticker") or ""
            series = ticker.split("-")[0] if ticker else ""
            close_ep = to_epoch(m.get("close_time"))
            text = " ".join(x for x in [m.get("title"), m.get("yes_sub_title"),
                                         m.get("rules_primary")] if x)
            rows.append({
                "ticker": ticker, "series": series, "price": price,
                "y": 1 if res == "yes" else 0, "close_time": m.get("close_time"),
                "close_ep": close_ep, "volume": vol, "text": text,
            })
    return rows, n_finalized, n_result


def load_category_map(gd):
    """series -> category, from lake/proposition.jsonl (kalshi rows only); partial coverage."""
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


def categorize(row, smap):
    cat = smap.get(row["series"])
    if cat:
        return cat
    return "macro" if MACRO_KW.search(row["text"] or "") else "other"


def decile_bins(rows, price_key="price", y_key="y", n_bins=10):
    edges = [i / n_bins for i in range(n_bins + 1)]
    bins = [[] for _ in range(n_bins)]
    for r in rows:
        p = r[price_key]
        idx = min(n_bins - 1, int(p * n_bins))
        bins[idx].append(r)
    out = []
    for i, b in enumerate(bins):
        n = len(b)
        if n == 0:
            out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": 0,
                        "mean_price": None, "realized_freq": None, "se": None, "edge": None})
            continue
        mp = statistics.mean(r[price_key] for r in b)
        freq = statistics.mean(r[y_key] for r in b)
        se = math.sqrt(freq * (1 - freq) / n) if n > 0 else None
        out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)], "n": n,
                    "mean_price": round(mp, 4), "realized_freq": round(freq, 4),
                    "se": round(se, 4) if se else None,
                    "edge": round(freq - mp, 4)})
    return out


def select_bins(bins, min_n=30):
    """Two independent selections, each split by edge sign (buy YES needs edge>0,
    buy NO needs edge<0) so we never report a "most underpriced" bin that is in fact
    still overpriced (just less extreme) -- return None on a side with no qualifying bin.
      * best_z:   largest |edge/se| -- most STATISTICALLY significant.
      * best_mag: largest |edge|    -- most ECONOMICALLY significant (may be thin n).
    """
    cands = [b for b in bins if b["n"] >= min_n and b["se"]]
    under = [b for b in cands if b["edge"] > 0]   # buy YES candidates
    over = [b for b in cands if b["edge"] < 0]    # buy NO candidates
    out = {"buy_yes_best_z": None, "buy_yes_best_mag": None,
           "buy_no_best_z": None, "buy_no_best_mag": None}
    if under:
        out["buy_yes_best_z"] = max(under, key=lambda b: b["edge"] / b["se"])
        out["buy_yes_best_mag"] = max(under, key=lambda b: b["edge"])
    if over:
        out["buy_no_best_z"] = min(over, key=lambda b: b["edge"] / b["se"])
        out["buy_no_best_mag"] = min(over, key=lambda b: b["edge"])
    return out


def strategy_pnl(rows, bin_range, side):
    """side='yes' buy YES at price; side='no' buy NO at (1-price). 1 contract, hold to settle."""
    lo, hi = bin_range
    sub = [r for r in rows if lo <= r["price"] < hi or (hi == 1.0 and r["price"] == 1.0)]
    if not sub:
        return {"n": 0}
    gross, net = [], []
    for r in sub:
        if side == "yes":
            entry = r["price"]; payoff = r["y"]
        else:
            entry = 1 - r["price"]; payoff = 1 - r["y"]
        f = fee(entry)
        gross.append(payoff - entry)
        net.append(payoff - entry - f)
    n = len(sub)
    return {
        "n": n, "side": side, "bin": [round(lo, 2), round(hi, 2)],
        "gross_mean": round(statistics.mean(gross), 4),
        "gross_total": round(sum(gross), 2),
        "net_mean": round(statistics.mean(net), 4),
        "net_total": round(sum(net), 2),
        "net_se": round(statistics.pstdev(net) / math.sqrt(n), 4) if n > 1 else None,
    }


def summarize_halves(rows):
    dated = [r for r in rows if r["close_ep"]]
    dated.sort(key=lambda r: r["close_ep"])
    mid = len(dated) // 2
    return dated[:mid], dated[mid:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/eg_live")
    ap.add_argument("--min-bin-n", type=int, default=30)
    a = ap.parse_args()
    gd = Path(a.graph_dir)

    rows, n_finalized, n_result = load_finalized(gd)
    print(f"finalized markets: {n_finalized}, with yes/no result: {n_result}, "
          f"traded+usable (volume>0, valid price): {len(rows)}")

    dated = [r for r in rows if r["close_time"]]
    date_range = [min(r["close_time"] for r in dated)[:10], max(r["close_time"] for r in dated)[:10]] if dated else None

    # 1. full-sample near-resolution calibration
    full_bins = decile_bins(rows)

    # 2. category segmentation
    smap = load_category_map(gd)
    for r in rows:
        r["category"] = categorize(r, smap)
    macro_rows = [r for r in rows if r["category"] == "macro"]
    other_rows = [r for r in rows if r["category"] != "macro"]
    macro_bins = decile_bins(macro_rows)
    other_bins = decile_bins(other_rows)

    # 3. bin selection (by statistical z AND by economic magnitude, buy-YES / buy-NO
    #    kept separate so we never call a still-net-negative-edge bin "underpriced")
    sel_full = select_bins(full_bins, a.min_bin_n)
    sel_macro = select_bins(macro_bins, min(a.min_bin_n, 25))

    def run_strategy(sel, rows_):
        out = {}
        for key, side in (("buy_yes_best_z", "yes"), ("buy_yes_best_mag", "yes"),
                          ("buy_no_best_z", "no"), ("buy_no_best_mag", "no")):
            b = sel.get(key)
            if not b:
                continue
            r = strategy_pnl(rows_, b["bin"], side)
            r["source_bin_edge"] = b["edge"]; r["source_bin_se"] = b["se"]
            out[key] = r
        return out

    strat_full = run_strategy(sel_full, rows)
    strat_macro = run_strategy(sel_macro, macro_rows)

    # 4. walk-forward: bins SELECTED ON FULL SAMPLE, evaluated unchanged on each half
    h1, h2 = summarize_halves(rows)
    mh1, mh2 = summarize_halves(macro_rows)
    walk = {"h1_n": len(h1), "h2_n": len(h2),
            "h1_range": [h1[0]["close_time"][:10], h1[-1]["close_time"][:10]] if h1 else None,
            "h2_range": [h2[0]["close_time"][:10], h2[-1]["close_time"][:10]] if h2 else None,
            "h1_bins": decile_bins(h1), "h2_bins": decile_bins(h2),
            "macro_h1_n": len(mh1), "macro_h2_n": len(mh2)}
    for key, side in (("buy_yes_best_z", "yes"), ("buy_yes_best_mag", "yes"),
                      ("buy_no_best_z", "no"), ("buy_no_best_mag", "no")):
        b = sel_full.get(key)
        if not b:
            continue
        walk[f"{key}_h1"] = strategy_pnl(h1, b["bin"], side)
        walk[f"{key}_h2"] = strategy_pnl(h2, b["bin"], side)
        bm = sel_macro.get(key)
        if bm:
            walk[f"macro_{key}_h1"] = strategy_pnl(mh1, bm["bin"], side)
            walk[f"macro_{key}_h2"] = strategy_pnl(mh2, bm["bin"], side)

    # 5. horizon/drift subset via kalshi_candles cache (small N, real price PATH)
    mkt_by_ticker = {r["ticker"]: r for r in rows}
    horizon_rows = []
    for fn in glob.glob(str(gd / "kalshi_candles" / "*.json")):
        ticker = Path(fn).stem
        base = mkt_by_ticker.get(ticker)
        if not base or not base["close_ep"]:
            continue
        try:
            cs = json.load(open(fn))
        except Exception:
            continue
        if not isinstance(cs, list) or len(cs) < 2:
            continue
        p24 = price_at_or_before(cs, base["close_ep"] - 86400)
        pnear = price_at_or_before(cs, base["close_ep"])
        if p24 is None or pnear is None:
            continue
        horizon_rows.append({"ticker": ticker, "y": base["y"], "p24h": p24, "pnear": pnear})
    hz_24h_bins = decile_bins(horizon_rows, price_key="p24h") if horizon_rows else []
    hz_near_bins = decile_bins(horizon_rows, price_key="pnear") if horizon_rows else []

    out = {
        "data_source": "kalshi settlement field on disk (kalshi_markets/*.json, status=='finalized', "
                        "result in {yes,no}); polymarket EXCLUDED (no disk settlement field, only "
                        "price-derived resolved_outcome -> would be circular); FRED not used (no "
                        "FRED_KEY/FRED_API_KEY in env).",
        "n_finalized_markets": n_finalized,
        "n_with_result": n_result,
        "n_usable_near_resolution": len(rows),
        "date_range": date_range,
        "fee_formula": "ceil(0.07 * price * (1-price) * 100) / 100, per contract, round up to cent "
                        "(Kalshi taker schedule; same 0.07*p*(1-p) core as scripts/kalshi_drift.py)",
        "fee_assumption": "NET P&L pays only the fee at the recorded last traded price (maker/best-case "
                           "fill); no bid-ask spread crossing modeled (spread not reliably available "
                           "across the full settled-market snapshot).",
        "calibration_full_near_resolution": full_bins,
        "category_segmentation": {
            "macro_n": len(macro_rows), "other_n": len(other_rows),
            "macro_bins": macro_bins, "other_bins": other_bins,
            "category_source": f"{len(smap)} series categorized via lake/proposition.jsonl kalshi rows; "
                                "remainder via macro-keyword regex on title/rules text (fallback).",
        },
        "selected_bins_full_sample": sel_full,
        "selected_bins_macro": sel_macro,
        "min_bin_n": a.min_bin_n,
        "strategy_full_sample": strat_full,
        "strategy_macro": strat_macro,
        "walk_forward": walk,
        "horizon_drift_subset": {
            "n": len(horizon_rows),
            "note": "kalshi_candles/*.json cache is a 322-ticker subset (daily candles); this is the "
                    "only place a genuine >=24h-pre-resolution price PATH exists on disk, vs. the full "
                    "sample which only has the last traded price.",
            "price_24h_before_bins": hz_24h_bins,
            "price_near_resolution_bins": hz_near_bins,
        },
    }

    outpath = gd / "pm_calibration.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"wrote {outpath}")

    # console digest for convenience
    print("\nfull-sample near-resolution calibration (decile: mean_price -> realized_freq, N):")
    for b in full_bins:
        if b["n"]:
            print(f"  [{b['bin'][0]:.1f},{b['bin'][1]:.1f}) price={b['mean_price']:.3f} "
                  f"freq={b['realized_freq']:.3f} edge={b['edge']:+.3f} n={b['n']}")
    print("\nselected bins (full sample):", json.dumps(sel_full, indent=2))
    print("\nselected bins (macro):", json.dumps(sel_macro, indent=2))
    print("\nstrategy (full sample):", json.dumps(strat_full, indent=2))
    print("\nstrategy (macro):", json.dumps(strat_macro, indent=2))
    print("\nwalk-forward H1/H2 N:", walk["h1_n"], walk["h2_n"], walk["h1_range"], walk["h2_range"])


if __name__ == "__main__":
    main()
