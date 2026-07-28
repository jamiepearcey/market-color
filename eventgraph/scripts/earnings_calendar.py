# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
EARNINGS CALENDAR FROM SEC 8-K ITEM 2.02 — an exogenous, exactly-dated event set.

WHY A CALENDAR BEATS THE NEWS GRAPH FOR THIS. The event-window profile found risk
elevated at t-1 (1.12x a clean control) as well as t=0, but t+1 was elevated by
almost exactly the same amount (1.14x). Leakage should be ASYMMETRIC — build up,
then resolve. Near-perfect symmetry instead points at timestamp smearing: our news
dates are `published_at`, so a story filed at 6pm Tuesday is dated Tuesday but
moves the stock Wednesday.

A calendar fixes this by construction:
  - EXACT. Item 2.02 of an 8-K IS the earnings release. No inference from text.
  - EXOGENOUS. Nobody schedules earnings because the stock moved, so it cannot be
    reverse-caused the way "Bloomberg wrote about a name that was already moving"
    can be.
  - EX ANTE. The date is known in advance, which is the only thing that makes a
    PRE-event window tradeable rather than merely observable.
  - TIMESTAMPED. `acceptanceDateTime` gives the filing moment, so a release
    accepted after the close can be assigned to the NEXT session instead of
    smearing across two days.

SOURCE. data.sec.gov/submissions/CIK##########.json — the `items` field lists 8-K
item codes. `recent` covers roughly the last decade; older filings live in the
files listed under filings.files, which is why both are fetched. Free, public, and
authoritative: this IS the filing.

CAVEAT worth stating. Item 2.02 is the results release, but firms occasionally
file 2.02 for non-quarterly results, and a few report via press release with the
8-K following. Expect ~4/year/firm; anything far from that deserves a look.

Usage:
    uv run scripts/earnings_calendar.py --fetch
    uv run scripts/earnings_calendar.py --report
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
SIC_MAP = ROOT / "sic_map.json"
OUT = ROOT / "earnings_calendar.json"
UA = "market-color research (jamiep.tigereye@gmail.com)"
YEARS = (2008, 2016)          # generous around the 2010-2014 sample
CLOSE_HOUR_ET = 16            # US equity close


def _rows(block: dict) -> list[dict]:
    n = len(block.get("form", []))
    out = []
    for i in range(n):
        out.append({"form": block["form"][i],
                    "items": block.get("items", [""] * n)[i] or "",
                    "filed": block["filingDate"][i],
                    "report": (block.get("reportDate") or [""] * n)[i] or "",
                    "accepted": (block.get("acceptanceDateTime") or [""] * n)[i] or ""})
    return out


def fetch() -> None:
    import httpx
    sic = json.loads(SIC_MAP.read_text())
    universe = {t: v["cik"] for t, v in sic.items() if v.get("cik")}
    print(f"{len(universe)} tickers with a CIK")

    cal: dict[str, list] = {}
    h = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    with httpx.Client(headers=h, timeout=40, follow_redirects=True) as c:
        for n, (tk, cik) in enumerate(sorted(universe.items()), 1):
            try:
                r = c.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")
                if r.status_code != 200:
                    continue
                j = r.json()
                blocks = [j["filings"]["recent"]]
                time.sleep(0.11)
                for extra in j["filings"].get("files", []):
                    rr = c.get(f"https://data.sec.gov/submissions/{extra['name']}")
                    if rr.status_code == 200:
                        blocks.append(rr.json())
                    time.sleep(0.11)
            except Exception as exc:
                print(f"  {tk}: {exc}")
                continue
            hits = []
            for b in blocks:
                for row in _rows(b):
                    if row["form"] != "8-K" or "2.02" not in row["items"]:
                        continue
                    y = int(row["filed"][:4])
                    if not (YEARS[0] <= y <= YEARS[1]):
                        continue
                    hits.append(row)
            if hits:
                seen, uniq = set(), []
                for x in sorted(hits, key=lambda r: r["filed"]):
                    if x["filed"] not in seen:
                        seen.add(x["filed"])
                        uniq.append(x)
                cal[tk] = uniq
            if n % 50 == 0:
                print(f"  {n}/{len(universe)}  ({sum(len(v) for v in cal.values())} releases)")
                OUT.write_text(json.dumps(cal))
    OUT.write_text(json.dumps(cal))
    tot = sum(len(v) for v in cal.values())
    print(f"\nwrote {OUT}: {len(cal)} tickers, {tot} Item 2.02 releases "
          f"{YEARS[0]}-{YEARS[1]}")


def report() -> None:
    cal = json.loads(OUT.read_text())
    tot = sum(len(v) for v in cal.values())
    per_year = collections.Counter()
    after_close = timed = 0
    for tk, rows in cal.items():
        for r in rows:
            per_year[r["filed"][:4]] += 1
            acc = r.get("accepted") or ""
            if len(acc) >= 16:
                timed += 1
                try:
                    hh = int(acc[11:13])
                    # acceptanceDateTime is US Eastern
                    if hh >= CLOSE_HOUR_ET:
                        after_close += 1
                except ValueError:
                    pass
    print(f"{len(cal)} tickers, {tot} Item 2.02 releases")
    print("by year: " + "  ".join(f"{y}:{n}" for y, n in sorted(per_year.items())))
    counts = [len(v) for v in cal.values()]
    counts.sort()
    print(f"releases per ticker: median {counts[len(counts)//2]}, "
          f"min {counts[0]}, max {counts[-1]}")
    if timed:
        print(f"\nfiling time known for {timed}/{tot} ({timed/tot:.0%})")
        print(f"  filed AT OR AFTER {CLOSE_HOUR_ET}:00 ET: {after_close} "
              f"({after_close/timed:.0%})")
        print("  -> those releases move the NEXT session, not the filing date. This is")
        print("     exactly the smearing the news-graph dates could not resolve.")
    # sample
    tk = "AAPL" if "AAPL" in cal else sorted(cal)[0]
    print(f"\nsample — {tk}:")
    for r in cal[tk][:5]:
        print(f"  filed {r['filed']}  accepted {r['accepted'][:16]}  "
              f"period {r['report']}  items {r['items']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    elif a.report:
        report()
    else:
        ap.print_help()
