# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
MAXIMAL macro/economic settled-history backfill from Kalshi's public API, for the
pm_calibration.py OOS stress-test (scripts/pm_calibration_oos.py).

Finding under test: pm_calibration.py found (on-disk snapshot, ~2.5mo window) that
mid-to-high-priced macro YES-contracts are systematically overpriced (buy-NO edge
survives fees). That study was underpowered / no real OOS. Step 1 of the stress test
is to get as much settled macro history as the API will give us.

Strategy: scan ALL series across kalshi_drift.py's category list, keep any series
whose TITLE matches the same MACRO_KW regex pm_calibration.py uses to categorize
markets (CPI/PCE/GDP/FOMC/Fed/payrolls/unemployment/jobless/inflation/ISM/PMI/retail
sales/housing starts/rate-decision-adjacent terms), then paginate status=settled to
EXHAUSTION (cursor until empty) for each -- not the per_series=40 cap kalshi_drift.py
uses -- and overwrite data/eg_live/kalshi_markets/<ticker>.json with the full raw
/markets rows (same schema pm_calibration.py already reads directly).

Usage:
  uv run scripts/kalshi_macro_backfill.py --graph-dir data/eg_live
"""
import argparse, json, re, time
from pathlib import Path
import httpx

BASE = "https://api.elections.kalshi.com/trade-api/v2"
CATS = ["Companies", "Financials", "Economics", "Politics", "Science and Technology", "Health",
        "World", "Mentions", "Entertainment", "Commodities", "Transportation", "Climate and Weather"]
# SAME regex pm_calibration.py uses to categorize a market as "macro" (kept identical
# so series selected here match what the calibration/OOS scripts will call macro).
MACRO_KW = re.compile(
    r"\b(cpi|core cpi|pce|core pce|gdp|fomc|fed(?:eral reserve)?|interest rate|fed funds|"
    r"payroll|nonfarm|unemployment|jobless claims|treasury|10-?year|yield|recession|"
    r"inflation|ism|pmi|retail sales|housing starts)\b", re.I)


def kget(sess, path, **q):
    for att in range(5):
        try:
            r = sess.get(f"{BASE}{path}", params={k: v for k, v in q.items() if v is not None}, timeout=30)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429:
                time.sleep(1.5 * (att + 1)); continue
            return {}
        except Exception:
            time.sleep(0.6 * (att + 1))
    return {}


def fetch_all_settled(sess, ticker, per_page=200):
    """paginate status=settled to exhaustion (cursor empty), no cap."""
    out, cursor, pages = [], None, 0
    while True:
        j = kget(sess, "/markets", series_ticker=ticker, status="settled", limit=per_page, cursor=cursor)
        mk = j.get("markets", []) if isinstance(j, dict) else []
        out.extend(mk)
        cursor = j.get("cursor") if isinstance(j, dict) else None
        pages += 1
        if not cursor or not mk or pages > 50:
            break
        time.sleep(0.08)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="data/eg_live")
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    mkcache = gd / "kalshi_markets"; mkcache.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})

    # 1. discover series across all categories, keep macro-titled ones
    series = {}
    for cat in CATS:
        j = kget(sess, "/series", category=cat)
        for s in j.get("series", []):
            t = s.get("ticker"); title = s.get("title") or ""
            if t and MACRO_KW.search(title):
                series[t] = title
        time.sleep(0.05)
    print(f"{len(series)} macro-titled series across {len(CATS)} categories")

    # 2. paginate settled history to exhaustion for each, overwrite cache
    all_close = []
    n_total = n_labeled = 0
    per_series_n = {}
    for i, (t, title) in enumerate(sorted(series.items())):
        mk = fetch_all_settled(sess, t)
        (mkcache / f"{t}.json").write_text(json.dumps(mk))
        labeled = sum(1 for m in mk if (m.get("result") or "").lower() in ("yes", "no"))
        n_total += len(mk); n_labeled += labeled
        per_series_n[t] = labeled
        for m in mk:
            ct = m.get("close_time")
            if ct and (m.get("result") or "").lower() in ("yes", "no"):
                all_close.append(ct)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(series)} series scanned, {n_labeled} labeled settled markets so far")
        time.sleep(0.05)

    all_close.sort()
    print(f"\nDONE. {len(series)} macro series scanned.")
    print(f"total settled rows written: {n_total}; with yes/no result: {n_labeled}")
    if all_close:
        print(f"date span of settled+labeled macro markets: {all_close[0][:10]} .. {all_close[-1][:10]}")
    top = sorted(per_series_n.items(), key=lambda kv: -kv[1])[:15]
    print("top series by labeled-N:", top)


if __name__ == "__main__":
    main()
