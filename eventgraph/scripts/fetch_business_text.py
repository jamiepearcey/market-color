# /// script
# requires-python = ">=3.10"
# dependencies = ["curl_cffi"]
# ///
"""
Fetch STABLE business-description text (10-K Item 1) for the live universe.

WHY. F53 found recency decay HURTS the exposure signal — profiles behave like a
stable characteristic, not a news flow. F55 found news vocabulary is period topic
that does not transfer across time. Put together: we were estimating a STABLE
CHARACTERISTIC using a TIME-VARYING EVENT STREAM. A 10-K Item 1 says what a firm
IS; a news article says what happened to it last Tuesday.

This is the input for the correctly-specified version of the test, and it is what
the Hoberg-Phillips text-based industry work uses (10-K product descriptions),
which is the literature strand that reports text similarity beating SIC/GICS.

TWO TRAPS HANDLED.
  - company_tickers.json can point at a SUCCESSOR registrant: XOM maps to CIK
    2115436, which has only S-8s and an 8-K12B, because the historical 10-K filer
    is a different CIK. Falls back to EDGAR company search when no 10-K is found.
  - The first "Item 1. Business" hit is almost always the TABLE OF CONTENTS. We
    score every candidate on prose-density and take the best, rather than the first.

Range-fetches the first 600KB of each filing — Item 1 sits near the front, and the
documents run to 5MB.

Usage:
    uv run scripts/fetch_business_text.py --out data/eg_runs/eg_live2/business_text.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

from curl_cffi import requests as cf

UA = {"User-Agent": "market-color research jamiep.tigereye@gmail.com"}
ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
HDR = re.compile(r"Item\s*1\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}Business"
                 # 20-F (foreign private issuers) puts the business description in
                 # Item 4 "Information on the Company" / "Business Overview".
                 r"|Item\s*4\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}Information\s+on\s+the\s+Compan"
                 r"|Business\s+Overview", re.I)
# Where Item 1 ENDS. Extracting to the section boundary rather than a fixed 2,500
# chars is the difference between a business description and a business
# description plus whatever followed it.
END = re.compile(r"Item\s*1A\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}Risk\s*Factors"
                 r"|Item\s*1B\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}Unresolved"
                 r"|Item\s*2\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}Propert"
                 # safe-harbour / risk sections. AMD~DE paired on nothing but
                 # "cautionary, forward-looking, uncertainties, differ" at 6k chars,
                 # for two firms with correlation -0.12.
                 r"|(?:Special\s+Note|Cautionary\s+(?:Note|Statement|Statements))"
                 r"|Forward[- ]Looking\s+Statements"
                 r"|Item\s*3\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}(?:Key\s+Information|Legal)"
                 r"|Item\s*5\s*[\.\:\-–—]?\s*(?:&#\d+;|&nbsp;|\W){0,8}Operating", re.I)


def prose_score(s: str) -> float:
    """TOC fragments are dense in digits and repeated 'Item'; prose is not."""
    if len(s) < 300:
        return -1.0
    digits = sum(c.isdigit() for c in s) / len(s)
    items = len(re.findall(r"\bItem\s*\d", s, re.I))
    sents = s.count(". ")
    return sents - 8 * digits * 10 - 3 * items


def extract_item1(html: str, maxlen: int = 2500) -> str | None:
    t = re.sub(r"<[^>]+>", " ", html)
    t = re.sub(r"&#\d+;|&[a-z]+;", " ", t)
    t = re.sub(r"\s+", " ", t)
    best, bestscore = None, -1e9
    for m in HDR.finditer(t):
        seg = t[m.end():m.end() + maxlen]
        # Score the OPENING of the segment, not the whole thing: with maxlen at
        # 20k a table of contents followed by 19k of prose still scores well, so
        # scoring the full window silently reintroduced the TOC bug.
        sc = prose_score(seg[:1500])
        if sc > bestscore:
            best, bestscore = seg, sc
    if best is None or bestscore < 0:
        return None
    # cut at the next section header so the text is Item 1 and nothing else
    e = END.search(best)
    if e and e.start() > 400:
        best = best[:e.start()]
    return best.strip()[:maxlen]


def cik_map():
    j = cf.get("https://www.sec.gov/files/company_tickers.json", headers=UA,
               impersonate="chrome", timeout=40).json()
    return {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in j.values()}


def tenk_url(cik: str):
    s = cf.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=UA,
               impersonate="chrome", timeout=40)
    if s.status_code != 200:
        return None, None
    r = s.json().get("filings", {}).get("recent", {})
    for i, f in enumerate(r.get("form", [])):
        if f in ("10-K", "10-K405", "20-F"):
            acc = r["accessionNumber"][i].replace("-", "")
            return (f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
                    f"{r['primaryDocument'][i]}"), r["filingDate"][i]
    return None, None


def fallback_cik(ticker: str):
    """EDGAR company search resolves to the filer that actually files 10-Ks."""
    u = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&ticker="
         f"{ticker}&type=&dateb=&owner=include&count=10&output=atom")
    r = cf.get(u, headers=UA, impersonate="chrome", timeout=40)
    m = re.search(r"CIK=(\d{10})", r.text) or re.search(r"/data/(\d+)/", r.text)
    return m.group(1).zfill(10) if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="eg_live2")
    ap.add_argument("--out", default=None)
    ap.add_argument("--maxlen", type=int, default=2500,
                    help="chars of Item 1 to keep; full sections run to tens of thousands")
    ap.add_argument("--range-kb", type=int, default=600)
    a = ap.parse_args()
    G = ROOT / a.graph
    out = Path(a.out) if a.out else G / "business_text.json"
    have = json.load(open(out)) if out.exists() else {}

    tickers = sorted(f[:-5] for f in os.listdir(G / "prices") if f.endswith(".json"))
    cm = cik_map()
    print(f"{len(tickers)} tickers · ticker->CIK map {len(cm)} · already have {len(have)}")

    ok = miss = 0
    for i, t in enumerate(tickers):
        if t in have:
            ok += 1
            continue
        try:
            c = cm.get(t)
            url = date = None
            if c:
                url, date = tenk_url(c)
            if not url:
                c2 = fallback_cik(t)
                if c2 and c2 != c:
                    url, date = tenk_url(c2)
            if not url:
                miss += 1
                continue
            h = dict(UA); h["Range"] = f"bytes=0-{a.range_kb*1000}"
            x = cf.get(url, headers=h, impersonate="chrome", timeout=60)
            body = extract_item1(x.text, a.maxlen)
            if body:
                have[t] = {"date": date, "text": body}
                ok += 1
            else:
                miss += 1
        except Exception:
            miss += 1
        time.sleep(0.12)                      # SEC fair-access
        if (i + 1) % 40 == 0:
            json.dump(have, open(out, "w"))
            print(f"  {i+1}/{len(tickers)}  ok={ok} miss={miss}", flush=True)

    json.dump(have, open(out, "w"))
    print(f"\nwrote {out}: {len(have)} business descriptions (missing {miss})")
    for t in list(have)[:3]:
        print(f"\n  {t} [{have[t]['date']}]: {have[t]['text'][:200]}...")


if __name__ == "__main__":
    main()
