# /// script
# requires-python = ">=3.10"
# dependencies = ["curl_cffi"]
# ///
"""
BACKFILL PRICE HISTORY for a graph's resolved tickers.

WHY. The live capture's price files hold ~135 daily bars, because they were pulled
alongside a 30-day news window. Every measurement in this project needs far more:
a 250-day rolling market beta plus a 60-day trailing volatility window, so roughly
600 bars before a single standardised move can be computed. With 135 bars the
pipeline yields zero usable names — not a marginal result, a hard stop.

The news window and the price window are independent. News can be 30 days old and
prices can still reach back years, which is what makes a recent capture analysable
at all: betas and volatility come from long price history, and only the EVENTS need
to be recent.

Fetches the full available daily history (range=max) for every resolved ticker and
writes it in the same chart-API envelope the rest of the pipeline reads.

Usage:
    uv run scripts/backfill_prices.py --graph eg_live2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from curl_cffi import requests as cf

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
CHART = "https://query2.finance.yahoo.com/v8/finance/chart/{sym}"
MIN_BARS = 600
# ~12 years of daily bars: enough for the 250-day beta window plus the 60-day
# volatility window, with margin, on any name listed through the period.
PERIOD1 = 1388534400   # 2014-01-01
PERIOD2 = 1785196800   # 2026-07-27


def resolved_tickers(gdir: Path) -> list[str]:
    out = set()
    p = gdir / "classification" / "entity_class.jsonl"
    if p.exists():
        for l in open(p):
            j = json.loads(l)
            if j.get("resolution_status") in ("resolved_security", "resolved_proxy") \
                    and j.get("resolved_ticker"):
                out.add(j["resolved_ticker"].upper())
    # also anything already on disk, so existing thin files get repaired
    pd = gdir / "prices"
    if pd.exists():
        for f in pd.iterdir():
            if f.suffix == ".json":
                out.add(f.stem.upper())
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph", default="eg_live2")
    ap.add_argument("--sleep", type=float, default=0.35)
    ap.add_argument("--force", action="store_true",
                    help="refetch even if the existing file already has enough history")
    a = ap.parse_args()
    gdir = ROOT / a.graph
    outdir = gdir / "prices"
    outdir.mkdir(parents=True, exist_ok=True)
    tks = resolved_tickers(gdir)
    print(f"{a.graph}: {len(tks)} tickers to consider")

    ok = skipped = failed = thin = 0
    s = cf.Session(impersonate="chrome")
    for n, tk in enumerate(tks, 1):
        fp = outdir / f"{tk.replace('/', '-')}.json"
        if fp.exists() and not a.force:
            try:
                cur = json.loads(fp.read_text())["chart"]["result"][0]["timestamp"]
                if len(cur) >= MIN_BARS:
                    skipped += 1
                    continue
            except Exception:
                pass
        try:
            # EXPLICIT period1/period2, never range=max. Yahoo SILENTLY DOWNGRADES
            # the granularity when range=max is requested: it returns interval=3mo
            # regardless of interval=1d, so a request that looks like 40 years of
            # daily data yields ~168 QUARTERLY bars and no error. Bounded epoch
            # timestamps keep the daily interval.
            r = s.get(CHART.format(sym=tk), timeout=30,
                      params={"period1": PERIOD1, "period2": PERIOD2, "interval": "1d",
                              "events": "div,split", "includeAdjustedClose": "true"})
            if r.status_code != 200:
                failed += 1
                continue
            j = r.json()
            res = (j.get("chart") or {}).get("result")
            if not res or not res[0].get("timestamp"):
                failed += 1
                continue
            gran = (res[0].get("meta") or {}).get("dataGranularity")
            if gran and gran != "1d":
                failed += 1          # not daily data; do not write it
                continue
            nb = len(res[0]["timestamp"])
            fp.write_text(json.dumps(j))
            ok += 1
            if nb < MIN_BARS:
                thin += 1
        except Exception:
            failed += 1
        time.sleep(a.sleep)
        if n % 50 == 0:
            print(f"  {n}/{len(tks)}  fetched={ok} skipped={skipped} failed={failed}")

    print(f"\nfetched {ok}, already sufficient {skipped}, failed {failed}")
    print(f"  of the fetched, {thin} still have < {MIN_BARS} bars "
          f"(genuinely short listing history)")
    lens = []
    for f in outdir.glob("*.json"):
        try:
            lens.append(len(json.loads(f.read_text())["chart"]["result"][0]["timestamp"]))
        except Exception:
            pass
    if lens:
        lens.sort()
        print(f"  history now: median {lens[len(lens)//2]} bars, "
              f"{sum(1 for x in lens if x >= MIN_BARS)}/{len(lens)} usable")


if __name__ == "__main__":
    main()
