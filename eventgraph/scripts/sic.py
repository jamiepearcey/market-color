# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
SIC CLASSIFICATION FROM SEC EDGAR — a real, public-domain, hierarchical taxonomy.

WHY REPLACE gics.py. GICS is proprietary (MSCI/S&P); it cannot be redistributed
and there is no free authoritative mapping, so gics.py was ~140 tickers assigned
BY HAND. That covered a quarter of the 547-name universe, was unverifiable, and
had only one level (11 flat sectors) — no way to drill from a sector into a
subsector.

SIC is published by the SEC, is public domain, and is assigned PER FILER by the
SEC itself, so it is authoritative rather than guessed. It is also genuinely
hierarchical, which is what a drill-down needs:

    division      A-J        e.g. D = Manufacturing
    major group   2-digit    e.g. 28 = Chemicals & Allied Products
    industry grp  3-digit    e.g. 283 = Drugs
    industry      4-digit    e.g. 2834 = Pharmaceutical Preparations

CAVEATS, stated plainly:
  - SIC is old (last revised 1987) and lumps modern software/internet businesses
    into 7372 "Prepackaged Software" and 7370 "Computer Services". It separates
    industries the market treats as one, and merges ones it treats as distinct.
    It is a WORSE risk taxonomy than GICS. It is used here because it is real,
    complete, and checkable, and those beat a better-shaped taxonomy that was
    invented by hand.
  - It is the filer's self-reported code on its most recent filing, so it is
    point-in-TODAY, not point-in-time. For a 2010-2013 sample a company that has
    since repositioned carries its modern code.
  - Foreign issuers that file 20-F/40-F are present; those that do not file with
    the SEC at all (many OTC ADRs) have no CIK and no SIC.

Usage:
    uv run scripts/sic.py --fetch     # build the cache (respects SEC rate limits)
    uv run scripts/sic.py --coverage  # coverage against the eventgraph universe
    from sic import sector, subsector, industry
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "sic_map.json"
UA = "market-color research (jamiep.tigereye@gmail.com)"

# SIC divisions — the top of the official hierarchy.
DIVISIONS = [
    ((1, 9), "A", "Agriculture, Forestry & Fishing"),
    ((10, 14), "B", "Mining"),
    ((15, 17), "C", "Construction"),
    ((20, 39), "D", "Manufacturing"),
    ((40, 49), "E", "Transportation & Public Utilities"),
    ((50, 51), "F", "Wholesale Trade"),
    ((52, 59), "G", "Retail Trade"),
    ((60, 67), "H", "Finance, Insurance & Real Estate"),
    ((70, 89), "I", "Services"),
    ((91, 99), "J", "Public Administration"),
]

# Major groups actually present in a large-cap universe. Not the full 2-digit
# table — only what we need to LABEL; anything unlisted falls back to "SIC nn".
MAJOR = {
    "10": "Metal Mining", "12": "Coal Mining", "13": "Oil & Gas Extraction",
    "14": "Nonmetallic Minerals", "15": "Building Construction",
    "16": "Heavy Construction", "17": "Special Trade Contractors",
    "20": "Food & Kindred Products", "21": "Tobacco Products",
    "22": "Textile Mill Products", "23": "Apparel", "24": "Lumber & Wood",
    "25": "Furniture & Fixtures", "26": "Paper & Allied Products",
    "27": "Printing & Publishing", "28": "Chemicals & Allied Products",
    "29": "Petroleum Refining", "30": "Rubber & Plastics", "31": "Leather",
    "32": "Stone, Clay & Glass", "33": "Primary Metal Industries",
    "34": "Fabricated Metal Products", "35": "Industrial Machinery & Computers",
    "36": "Electronic & Electrical Equipment", "37": "Transportation Equipment",
    "38": "Instruments & Related Products", "39": "Misc Manufacturing",
    "40": "Railroad Transportation", "41": "Local Passenger Transit",
    "42": "Trucking & Warehousing", "44": "Water Transportation",
    "45": "Air Transportation", "46": "Pipelines", "47": "Transportation Services",
    "48": "Communications", "49": "Electric, Gas & Sanitary Services",
    "50": "Wholesale Durable Goods", "51": "Wholesale Nondurable Goods",
    "52": "Building Materials & Garden", "53": "General Merchandise Stores",
    "54": "Food Stores", "55": "Auto Dealers & Service Stations",
    "56": "Apparel & Accessory Stores", "57": "Furniture & Home Furnishings",
    "58": "Eating & Drinking Places", "59": "Miscellaneous Retail",
    "60": "Depository Institutions", "61": "Nondepository Credit",
    "62": "Security & Commodity Brokers", "63": "Insurance Carriers",
    "64": "Insurance Agents & Brokers", "65": "Real Estate",
    "67": "Holding & Investment Offices", "70": "Hotels & Lodging",
    "72": "Personal Services", "73": "Business Services",
    "75": "Auto Repair & Services", "78": "Motion Pictures",
    "79": "Amusement & Recreation", "80": "Health Services",
    "82": "Educational Services", "83": "Social Services",
    "87": "Engineering & Management Services", "99": "Nonclassifiable",
}

_MAP: dict | None = None


def _load() -> dict:
    global _MAP
    if _MAP is None:
        _MAP = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    return _MAP


def division(ticker: str) -> str:
    """Top level — the closest SIC analogue of a 'sector'."""
    e = _load().get(ticker.upper())
    if not e or not e.get("sic"):
        return "UNK"
    mg = int(str(e["sic"]).zfill(4)[:2])
    for (lo, hi), _code, label in DIVISIONS:
        if lo <= mg <= hi:
            return label
    return "UNK"


def sector(ticker: str) -> str:
    """Alias so callers can swap gics.sector -> sic.sector without churn."""
    return division(ticker)


def subsector(ticker: str) -> str:
    """Major group (2-digit) — the drill-down level below division."""
    e = _load().get(ticker.upper())
    if not e or not e.get("sic"):
        return "UNK"
    mg = str(e["sic"]).zfill(4)[:2]
    return MAJOR.get(mg, f"SIC {mg}")


def industry(ticker: str) -> str:
    """4-digit industry — the leaf, as the SEC describes it."""
    e = _load().get(ticker.upper())
    if not e:
        return "UNK"
    return e.get("desc") or (f"SIC {e['sic']}" if e.get("sic") else "UNK")


def code(ticker: str) -> str | None:
    e = _load().get(ticker.upper())
    return str(e["sic"]).zfill(4) if e and e.get("sic") else None


# ---------------------------------------------------------------------------
def fetch(universe: list[str] | None = None) -> None:
    import httpx
    h = {"User-Agent": UA, "Accept-Encoding": "gzip, deflate"}
    with httpx.Client(headers=h, timeout=30, follow_redirects=True) as c:
        print("fetching ticker -> CIK map ...")
        tj = c.get("https://www.sec.gov/files/company_tickers.json").json()
        t2c: dict[str, int] = {}
        for row in tj.values():
            t2c.setdefault(str(row["ticker"]).upper(), int(row["cik_str"]))
        print(f"  {len(t2c)} tickers have a CIK")

        out = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        want = [t for t in (universe or list(t2c)) if t.upper() not in out]
        hit = [t for t in want if t.upper() in t2c]
        print(f"need {len(want)}, of which {len(hit)} are SEC filers "
              f"({len(want) - len(hit)} not registered with the SEC)")
        for i, tk in enumerate(hit, 1):
            cik = t2c[tk.upper()]
            try:
                r = c.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
                if r.status_code != 200:
                    continue
                j = r.json()
                out[tk.upper()] = {"cik": cik, "sic": j.get("sic") or None,
                                   "desc": j.get("sicDescription") or None,
                                   "name": j.get("name")}
            except Exception as exc:
                print(f"  {tk}: {exc}")
            time.sleep(0.11)          # SEC asks for <=10 req/s
            if i % 50 == 0:
                print(f"  {i}/{len(hit)}")
                CACHE.write_text(json.dumps(out, indent=0))
        for tk in want:
            out.setdefault(tk.upper(), {"cik": None, "sic": None, "desc": None,
                                        "name": None})
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(out, indent=0))
        got = sum(1 for v in out.values() if v.get("sic"))
        print(f"wrote {CACHE}: {len(out)} tickers, {got} with a SIC code")


def universe_from_graph() -> list[str]:
    G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
    tks = set()
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        if j.get("resolution_status") == "resolved_security" and j.get("resolved_ticker"):
            tks.add(j["resolved_ticker"].upper())
    return sorted(tks)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    a = ap.parse_args()
    if a.fetch:
        fetch(universe_from_graph())
    elif a.coverage:
        import collections
        u = universe_from_graph()
        m = _load()
        got = [t for t in u if m.get(t, {}).get("sic")]
        print(f"universe {len(u)}  with SIC {len(got)} ({len(got)/len(u):.0%})")
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from gics import sector as gs
            gcov = sum(1 for t in u if gs(t) != "UNK")
            print(f"  hand-curated gics.py covered {gcov} ({gcov/len(u):.0%})")
        except Exception:
            pass
        d = collections.Counter(division(t) for t in got)
        print("\ndivision:")
        for k, n in d.most_common():
            print(f"  {k:38} {n:4}")
        s = collections.Counter(subsector(t) for t in got)
        print(f"\n{len(s)} distinct major groups (subsectors); top 12:")
        for k, n in s.most_common(12):
            print(f"  {k:38} {n:4}")
    else:
        ap.print_help()
