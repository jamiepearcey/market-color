# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "httpx"]
# ///
"""
EARNINGS SURPRISE (SUE) — the variable this project never had.

THE GAP. Every "surprise" measured here so far has been a REACTION surprise: was
the move large for its news class? That can never separate "the news moved the
price" from "the results moved the price and the news merely described them".
What was missing is an INFORMATION surprise — what the numbers were versus what
was expected. That is the variable the earnings-drift literature is defined on,
and its absence is the most likely reason nothing predictive survived here.

NO ANALYST DATA NEEDED. Standardised Unexpected Earnings in the Foster /
Bernard-Thomas form uses a SEASONAL RANDOM WALK as the expectation:

    surprise_q = EPS_q - EPS_{q-4}
    SUE_q      = surprise_q / stdev(surprise over the trailing K quarters)

Firms' earnings are strongly seasonal, so "same quarter last year" is a decent
naive forecast, and standardising by the firm's own surprise volatility makes SUE
comparable across names. This is the pre-analyst-consensus version of the measure
and it is entirely free.

DATA. SEC XBRL frames API: one call returns EarningsPerShareDiluted for EVERY
filer in a calendar quarter (~4,000 companies), so the whole panel is ~40 calls.
The `frame`/ccp scoping is what makes the values properly QUARTERLY rather than
year-to-date, which is the usual trap with XBRL EPS.

WHAT IT ENABLES — the question actually being asked:
  1. does the size of the price reaction scale with the size of the surprise?
     (if not, SUE is not being priced and nothing downstream is meaningful)
  2. does the SIGN of the move follow the sign of the surprise?
  3. THE DISCREPANCY CASE: where the news narrative and the numbers DISAGREE,
     which one does the price follow?

Usage:
    uv run scripts/earnings_surprise.py --fetch
    uv run scripts/earnings_surprise.py --build
"""
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
RAW = ROOT / "xbrl_eps_frames.json"
OUT = ROOT / "earnings_sue.json"
UA = "market-color research (jamiep.tigereye@gmail.com)"
CONCEPT = "us-gaap/EarningsPerShareDiluted/USD-per-shares"
YEARS = range(2007, 2017)
MIN_HIST = 6          # trailing seasonal differences needed to standardise


def fetch() -> None:
    import httpx
    frames = {}
    h = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    with httpx.Client(headers=h, timeout=60, follow_redirects=True) as c:
        for y in YEARS:
            for q in (1, 2, 3, 4):
                key = f"CY{y}Q{q}"
                try:
                    r = c.get(f"https://data.sec.gov/api/xbrl/frames/{CONCEPT}/{key}.json")
                    if r.status_code != 200:
                        print(f"  {key}: HTTP {r.status_code}")
                        time.sleep(0.15)
                        continue
                    d = r.json()
                    frames[key] = {str(x["cik"]): x["val"] for x in d["data"]}
                    print(f"  {key}: {len(frames[key])} filers")
                except Exception as exc:
                    print(f"  {key}: {exc}")
                time.sleep(0.15)
    RAW.write_text(json.dumps(frames))
    print(f"\nwrote {RAW}: {len(frames)} quarters, "
          f"{sum(len(v) for v in frames.values())} firm-quarters")


def build() -> None:
    frames = json.loads(RAW.read_text())
    sic = json.loads((ROOT / "sic_map.json").read_text())
    cik_to_tk = {}
    for tk, v in sic.items():
        if v.get("cik"):
            cik_to_tk.setdefault(str(int(v["cik"])), tk)

    keys = sorted(frames, key=lambda k: (int(k[2:6]), int(k[-1])))
    # cik -> ordered [(quarter_key, eps)]
    series = collections.defaultdict(dict)
    for k in keys:
        for cik, val in frames[k].items():
            series[cik][k] = float(val)

    def qidx(k: str) -> int:
        return int(k[2:6]) * 4 + int(k[-1]) - 1

    out = {}
    for cik, s in series.items():
        tk = cik_to_tk.get(cik)
        if not tk:
            continue
        idx = {qidx(k): (k, v) for k, v in s.items()}
        diffs = {}
        for qi, (k, v) in sorted(idx.items()):
            prev = idx.get(qi - 4)
            if prev:
                diffs[qi] = (k, v - prev[1], v, prev[1])
        rows = []
        ordered = sorted(diffs)
        for pos, qi in enumerate(ordered):
            k, d, v, vprev = diffs[qi]
            hist = [diffs[q][1] for q in ordered[:pos]][-8:]
            if len(hist) < MIN_HIST:
                continue
            sd = float(np.std(hist))
            if sd <= 1e-9:
                continue
            rows.append({"quarter": k, "eps": v, "eps_year_ago": vprev,
                         "surprise": round(d, 4), "sue": round(d / sd, 3),
                         "sd": round(sd, 4)})
        if rows:
            out[tk] = rows
    OUT.write_text(json.dumps(out))
    n = sum(len(v) for v in out.values())
    allsue = np.array([r["sue"] for v in out.values() for r in v])
    print(f"wrote {OUT}: {len(out)} tickers, {n} firm-quarters with a SUE")
    print(f"  SUE percentiles: p5={np.percentile(allsue,5):.2f} "
          f"p25={np.percentile(allsue,25):.2f} p50={np.percentile(allsue,50):.2f} "
          f"p75={np.percentile(allsue,75):.2f} p95={np.percentile(allsue,95):.2f}")
    print(f"  |SUE| > 1: {(np.abs(allsue)>1).mean():.1%}   "
          f"positive surprises: {(allsue>0).mean():.1%}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch()
    elif a.build:
        build()
    else:
        ap.print_help()
