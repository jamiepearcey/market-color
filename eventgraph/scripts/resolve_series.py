# /// script
# requires-python = ">=3.10"
# ///
"""
Series-slug normalisation pass (lake/0010_series_alias.sql): map the extractor's
free-form event.series_id slugs onto canonical calendar ids so v_event_surprise
actually joins. Downstream mapping only -- fact rows are never touched, so it
works on already-extracted AND in-flight graphs.

Two passes over DISTINCT (series_id, event_type) in <graph-dir>/lake/event.jsonl:
  1. rules      word-boundary keyword regexes -> macro canon (US-CPI, US-NFP, ...).
                Word-boundary matters ('rate' must not match 'corporate' --
                same trap as resolve_tickers).
  2. earnings   event_type in (earnings, guidance, dividend, buyback) + a
                resolved symbol for the issuer (entity_symbol.jsonl from the
                ticker-resolution passes, --symbols) -> '{TICKER}-8-K'
                (US listings only: symbols with '.' are skipped, matching the
                signal doctrine).

Appends eg.series_alias rows (alias, series_id, method, ingested_at) to
<graph-dir>/lake/series_alias.jsonl; reruns append with a fresh ingested_at and
the view's latest-wins dedup picks them up. --dry to preview, --unmapped to
dump the residual tail for manual mapping.

Usage:
  uv run scripts/resolve_series.py --graph-dir /tmp/eg_6k [--symbols .../entity_symbol.jsonl] [--dry]
"""
import argparse, json, re, datetime as dt
from pathlib import Path

# rule name -> (regex, canonical series). First hit wins; order = specificity.
RULES = [
    ("cpi",          r"\bcpi\b|consumer.?price",                    "US-CPI"),
    ("ppi",          r"\bppi\b|producer.?price",                    "US-PPI"),
    ("nfp",          r"non.?farm|payrolls?\b|jobs.?report|employment.?(situation|report)", "US-NFP"),
    ("unemployment", r"unemployment.?rate|jobless.?rate",           "US-UNEMPLOYMENT"),
    ("gdp",          r"\bgdp\b|gross.?domestic",                    "US-GDP"),
    ("retail",       r"retail.?sales",                              "US-RETAIL-SALES"),
    ("trade_bal",    r"trade.?(balance|deficit|gap)",               "US-TRADE-BAL"),
    ("indpro",       r"industrial.?production",                     "US-INDPRO"),
    ("housing",      r"housing.?starts",                            "US-HOUSING-STARTS"),
    ("umich",        r"(michigan|consumer).?(sentiment|confidence)","US-UMICH-SENT"),
    ("fomc",         r"\bfomc\b|fed(eral)?.?(reserve.?|funds.?)?rate.?(decision|cut|hike)|\bfed\b.?meeting", "US-FOMC"),
]
EARNINGS_TYPES = {"earnings", "guidance", "dividend", "buyback"}


def main(a):
    lake = Path(a.graph_dir) / "lake"
    seen = {}   # (slug) -> (event_type, issuer)
    for l in open(lake / "event.jsonl"):
        e = json.loads(l)
        sid = e.get("series_id")
        if sid and sid not in seen:
            seen[sid] = (e.get("event_type") or "", e.get("issuer_entity") or "")
    symbols = {}
    if a.symbols:
        for l in open(a.symbols):  # entity_symbol.jsonl: {"entity_id":..,"symbol":..,"kind":..}
            s = json.loads(l)
            if s.get("kind", "security").startswith("security") and "." not in (s.get("symbol") or "."):
                symbols[s["entity_id"]] = s["symbol"]

    rules = [(n, re.compile(rx, re.I), canon) for n, rx, canon in RULES]
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out, unmapped = [], []
    for slug, (etype, issuer) in sorted(seen.items()):
        text = slug.replace("_", " ")
        hit = next(((n, c) for n, rx, c in rules if rx.search(text)), None)
        if hit:
            out.append({"alias": slug, "series_id": hit[1], "method": f"rule:{hit[0]}", "ingested_at": now})
        elif etype in EARNINGS_TYPES and issuer in symbols:
            out.append({"alias": slug, "series_id": f"{symbols[issuer]}-8-K",
                        "method": "earnings-symbol", "ingested_at": now})
        else:
            unmapped.append(slug)

    n_rule = sum(1 for r in out if r["method"].startswith("rule"))
    print(f"{len(seen)} distinct slugs -> {n_rule} rule-mapped, {len(out)-n_rule} earnings-mapped, {len(unmapped)} unmapped")
    if a.unmapped:
        for s in unmapped: print(f"  ? {s}")
    if a.dry:
        for r in out[:30]: print(f"  {r['alias']} -> {r['series_id']} ({r['method']})")
        return
    with open(lake / "series_alias.jsonl", "a") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    print(f"  +{len(out)} -> {lake / 'series_alias.jsonl'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--symbols", help="entity_symbol.jsonl from resolve_tickers/resolve_yahoo")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--unmapped", action="store_true")
    main(ap.parse_args())
