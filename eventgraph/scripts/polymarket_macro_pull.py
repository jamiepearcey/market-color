# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Polymarket RESOLVED macro/economics market pull -- multi-regime replication data
for the Kalshi finding in data/eg_live/pm_calibration_oos.json ("mid-to-high-priced
macro YES-contracts are systematically overpriced; buying NO nets positive after
fees"). Kalshi's public /markets API has a hard ~2.5-month rolling retention wall
(one regime). Polymarket is on-chain and reachable much further back -- this script
gets the real settlement + a real pre-resolution price for as many resolved macro
markets as the public APIs will actually give up, 2021-2025.

GENUINE SETTLEMENT, NOT PRICE-DERIVED: scripts/polymarket_ingest.py's
proposition.jsonl writes `resolved_outcome` for polymarket as `"yes" if prob_yes >=
0.5 else "no"` at whatever price was last recorded -- pm_calibration.py explicitly
EXCLUDES polymarket from its outcome-labelled study because that would be circular
(bakes "price==outcome" into the label). This script uses a different, real signal
instead: Gamma's `resolvedBy` (the address -- UMA optimistic-oracle assertion or
Polymarket's admin resolver -- that actually settled the conditional-token market)
combined with a CLEANLY settled `outcomePrices` (>0.999 / <0.001; CTF conditional
tokens redeem to exactly 0 or 1 once a condition resolves, so a clean pair is the
terminal on-chain state, not a mid-market snapshot). Markets that are `closed` but
lack `resolvedBy` or never settle cleanly (still near 0.5, void/50-50 markets, data
glitches) are DROPPED and counted -- they have no genuine label.

DISCOVERY: Gamma has no working full-text search over historical closed markets, so
two complementary sweeps are unioned (dedup by conditionId):
  1. Month-windowed brute-force scan of /markets?closed=true&end_date_min=..&end_date_max=..
     back to 2021-01, filtered client-side by MACRO_KW (same regex as
     pm_calibration_oos.py, for comparability with the Kalshi study). This is
     exhaustive as long as a month's total closed-market count stays under Gamma's
     pagination depth cap (empirically ~2100-2200 offset -> 422); recon showed 2021-2023
     comfortably under that (max ~750/month), 2024+ growing past it in the
     highest-volume months -- logged explicitly per month when the cap is hit.
  2. /events?closed=true&tag_id=<id> for the macro/econ tag ids (economy, inflation,
     fed-rates, interest-rates, recession, jobs, cpi, economics, gdp, macro-indicators,
     macro-graph, macro-single, federal-government). Tags only started being applied
     ~2024, so this sweep is a supplementary safety net for the high-volume 2024-2025
     months the brute-force scan may cap out on; markets are still gated through the
     SAME MACRO_KW filter, so tags never smuggle in a non-macro market on their own.

PRICE: for each surviving market, CLOB /prices-history (daily fidelity) on the YES
token. Take the point closest to (resolution - 7d) within a 2-day tolerance ("7d");
else the last available point strictly before resolution ("last_available", with the
actual gap recorded); a market with NO pre-resolution price point at all (the CLOB
order book has no history before roughly when Polymarket's book launched -- older
markets traded on the prior AMM) is DROPPED and counted.

Usage:
  uv run scripts/polymarket_macro_pull.py --graph-dir data/eg_live
"""
import argparse, json, re, time, math, hashlib, datetime as dt
from pathlib import Path
import httpx

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

MACRO_KW = re.compile(
    r"\b(cpi|core cpi|pce|core pce|gdp|fomc|fed(?:eral reserve)?|interest rate|fed funds|"
    r"rate cut|rate hike|payroll|nonfarm|jobs report|unemployment|jobless(?: claims)?|"
    r"recession|inflation|economic)\b", re.I)

SUBCAT_PATTERNS = [
    ("cpi_inflation", re.compile(r"\b(cpi|pce|inflation)\b", re.I)),
    ("fed_rates", re.compile(r"\b(fomc|fed(?:eral reserve)?|interest rate|fed funds|rate cut|rate hike)\b", re.I)),
    ("jobs", re.compile(r"\b(payroll|nonfarm|unemployment|jobless)\b", re.I)),
    ("gdp", re.compile(r"\bgdp\b", re.I)),
]

MACRO_TAG_IDS = [100328, 702, 100196, 131, 100201, 993, 101701, 225, 370, 102000, 101247, 101250, 933]

HORIZON_DAYS = 7
HORIZON_TOL_DAYS = 2


def get_json(sess, url, params, max_retries=6):
    """GET with retry/backoff on 429 and transient errors; returns None on hard failure."""
    delay = 1.0
    for attempt in range(max_retries):
        try:
            r = sess.get(url, params=params, timeout=30)
        except Exception:
            time.sleep(delay); delay = min(delay * 1.7, 20); continue
        if r.status_code == 200:
            return r.json()
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", delay))
            time.sleep(max(wait, delay)); delay = min(delay * 1.7, 20); continue
        if r.status_code == 422:
            return None  # pagination-depth cap or bad params; not retryable
        time.sleep(delay); delay = min(delay * 1.7, 20)
    return None


def month_windows(y0, y1):
    y, m = y0, 1
    while y <= y1:
        lo = f"{y:04d}-{m:02d}-01"
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        hi = f"{ny:04d}-{nm:02d}-01"
        yield y, m, lo, hi
        y, m = ny, nm


def sweep_month(sess, lo, hi, sleep_s):
    """exhaustive offset-paginate one month window; returns (markets, capped)."""
    acc, off, capped = [], 0, False
    while True:
        page = get_json(sess, f"{GAMMA}/markets",
                         {"closed": "true", "limit": 100, "offset": off,
                          "end_date_min": lo, "end_date_max": hi})
        time.sleep(sleep_s)
        if page is None:
            capped = True
            break
        if not page:
            break
        acc.extend(page)
        off += 100
        if off > 4000:  # sanity backstop
            capped = True
            break
    return acc, capped


def sweep_tags(sess, sleep_s):
    acc = []
    for tid in MACRO_TAG_IDS:
        off = 0
        while True:
            page = get_json(sess, f"{GAMMA}/events",
                             {"closed": "true", "tag_id": tid, "limit": 100, "offset": off})
            time.sleep(sleep_s)
            if not page:
                break
            for e in page:
                acc.extend(e.get("markets") or [])
            off += 100
            if off > 3000:
                break
    return acc


def genuine_outcome(m):
    """real settlement label from resolvedBy + cleanly-settled outcomePrices, or None."""
    if not m.get("closed") or not m.get("resolvedBy"):
        return None
    try:
        outs = json.loads(m.get("outcomes") or "[]")
        prices = [float(x) for x in json.loads(m.get("outcomePrices") or "[]")]
    except Exception:
        return None
    if len(outs) != 2 or len(prices) != 2:
        return None
    yi = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "up", "above")), 0)
    ni = 1 - yi
    if prices[yi] > 0.999 and prices[ni] < 0.001:
        return "yes"
    if prices[ni] > 0.999 and prices[yi] < 0.001:
        return "no"
    return None


def subcat_of(text):
    for name, pat in SUBCAT_PATTERNS:
        if pat.search(text or ""):
            return name
    return "other_macro"


def to_epoch(s):
    if not s:
        return None
    try:
        return int(dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())
    except Exception:
        return None


def price_history(sess, token, sleep_s):
    j = get_json(sess, f"{CLOB}/prices-history",
                 {"market": token, "interval": "max", "fidelity": 1440})
    time.sleep(sleep_s)
    if not j:
        return []
    return sorted(j.get("history", []), key=lambda p: p["t"])


def pick_price(hist, resolution_ep):
    """price ~HORIZON_DAYS before resolution, else last available pre-resolution point."""
    pre = [p for p in hist if p["t"] <= resolution_ep]
    if not pre:
        return None
    target = resolution_ep - HORIZON_DAYS * 86400
    tol = HORIZON_TOL_DAYS * 86400
    near = [p for p in pre if abs(p["t"] - target) <= tol]
    if near:
        best = min(near, key=lambda p: abs(p["t"] - target))
        return {"p": best["p"], "t": best["t"], "source": "7d",
                "actual_days_before": round((resolution_ep - best["t"]) / 86400, 2)}
    best = pre[-1]
    return {"p": best["p"], "t": best["t"], "source": "last_available",
            "actual_days_before": round((resolution_ep - best["t"]) / 86400, 2)}


def prop_id(cond):
    return "pm_macro_" + hashlib.sha1((cond or "").encode()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/eg_live")
    ap.add_argument("--start-year", type=int, default=2021)
    ap.add_argument("--end-year", type=int, default=2025)
    ap.add_argument("--min-volume", type=float, default=1.0)
    ap.add_argument("--sleep", type=float, default=0.25, help="delay between HTTP calls (rate-limit friendly)")
    a = ap.parse_args()

    gd = Path(a.graph_dir)
    outdir = gd / "polymarket_macro"
    outdir.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    # 1. discovery: month brute-force + tag sweep, unioned by conditionId
    by_cond = {}
    capped_months = []
    for y, m, lo, hi in month_windows(a.start_year, a.end_year):
        markets, capped = sweep_month(sess, lo, hi, a.sleep)
        if capped:
            capped_months.append(f"{y:04d}-{m:02d}")
        for mk in markets:
            cid = mk.get("conditionId")
            if cid:
                by_cond[cid] = mk
        print(f"  scan {y:04d}-{m:02d}: {len(markets)} closed markets"
              f"{' (CAPPED at pagination depth)' if capped else ''}", flush=True)

    tag_markets = sweep_tags(sess, a.sleep)
    for mk in tag_markets:
        cid = mk.get("conditionId")
        if cid and cid not in by_cond:
            by_cond[cid] = mk
    print(f"discovery: {len(by_cond)} unique closed markets "
          f"(month-scan + {len(tag_markets)} tag-swept)", flush=True)

    # 2. filter to macro/econ by question text (same MACRO_KW as Kalshi study)
    hits = [mk for mk in by_cond.values() if MACRO_KW.search(mk.get("question") or "")]
    print(f"macro/econ keyword hits: {len(hits)}", flush=True)

    # 3. genuine settlement + volume gate
    n_no_settle = n_low_vol = n_no_price = 0
    rows = []
    outpath = outdir / "markets.jsonl"
    outf = open(outpath, "w")
    for mk in hits:
        try:
            vol = float(mk.get("volumeNum") or mk.get("volume") or 0)
        except Exception:
            vol = 0.0
        if vol < a.min_volume:
            n_low_vol += 1
            continue
        outcome = genuine_outcome(mk)
        if outcome is None:
            n_no_settle += 1
            continue
        end_ep = to_epoch(mk.get("endDate"))
        if end_ep is None:
            n_no_settle += 1
            continue
        try:
            outs = json.loads(mk.get("outcomes") or "[]")
            clobs = json.loads(mk.get("clobTokenIds") or "[]")
        except Exception:
            outs, clobs = [], []
        yi = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "up", "above")), 0)
        yes_token = clobs[yi] if len(clobs) > yi else None
        if not yes_token:
            n_no_price += 1
            continue
        hist = price_history(sess, yes_token, a.sleep)
        pick = pick_price(hist, end_ep)
        if pick is None:
            n_no_price += 1
            continue
        text = mk.get("question") or ""
        row = {
            "prop_id": prop_id(mk["conditionId"]), "source": "polymarket",
            "contract_ref": mk["conditionId"], "question": text,
            "resolution_date": (mk.get("endDate") or "")[:10],
            "resolution_year": int((mk.get("endDate") or "0000")[:4]),
            "subcat": subcat_of(text),
            "price_yes": round(pick["p"], 4), "price_source": pick["source"],
            "actual_days_before_resolution": pick["actual_days_before"],
            "volume": vol,
            "outcome": outcome,
            "resolved_by": mk.get("resolvedBy"),
            "settlement_check": "resolvedBy set + outcomePrices clean (>0.999/<0.001)",
        }
        rows.append(row)
        outf.write(json.dumps(row) + "\n"); outf.flush()
        print(f"  + {text[:70]!r:72s} {mk.get('endDate','')[:10]} price={pick['p']:.3f}"
              f"({pick['source']}) outcome={outcome}", flush=True)
    outf.close()

    by_year = {}
    for r in rows:
        by_year[r["resolution_year"]] = by_year.get(r["resolution_year"], 0) + 1

    print("\n=== SUMMARY ===")
    print(f"unique closed markets discovered: {len(by_cond)}")
    print(f"macro keyword hits: {len(hits)}")
    print(f"dropped (volume < {a.min_volume}): {n_low_vol}")
    print(f"dropped (no genuine settlement label): {n_no_settle}")
    print(f"dropped (no pre-resolution price data): {n_no_price}")
    print(f"KEPT (genuine outcome + real price): {len(rows)}")
    print(f"year span: {min(by_year) if by_year else None}..{max(by_year) if by_year else None}")
    print(f"by year: {dict(sorted(by_year.items()))}")
    if capped_months:
        print(f"months where brute-force scan hit the pagination-depth cap "
              f"(tag-sweep may still have caught some macro markets in these): {capped_months}")
    print(f"wrote {len(rows)} -> {outpath}")


if __name__ == "__main__":
    main()
