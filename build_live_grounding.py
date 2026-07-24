# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0"]
# ///
"""Entity-GROUND the live corpus by dictionary linking (company-name -> ticker) over the FULL 17k articles.
This is the grounding layer plain embeddings lack — precise, no hallucination, and it kills same-name
collisions (Nuvve vs Nuvoco). Output: ticker -> [doc_id] over the whole corpus."""
import json
import re
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "data" / "news_corpus"
ENTSYM = HERE / "data" / "eg_runs" / "eg_live2" / "entity_symbol.jsonl"
OUT = HERE / "data" / "tmp" / "live_ticker_docs.json"

# Company names that are also common English / business words → unresolvable by dictionary, skip the name.
STOP = {"man", "spar", "pnr", "key", "all", "one", "on", "it", "for", "are", "cava", "so", "now",
        "strategy", "intelligence", "intel", "note", "core", "block", "square", "match", "open", "arm",
        "sky", "wolf", "unity", "race", "perf", "principal", "global", "market", "group", "world", "new",
        "first", "capital", "energy", "gold", "silver", "oil", "national", "general", "american", "united",
        "science", "technology", "systems", "digital", "power", "trust", "grid", "signal", "target",
        "peloton", "shift", "sound", "affirm", "roku", "chip", "phone", "under", "space", "star", "sun",
        "money", "bank", "life", "health", "sofi", "twist", "root", "rocket", "coty", "vale", "cia", "ec",
        "bullish", "people", "reliability", "perfect", "coffee", "citizens", "national", "rs", "wit",
        "reliance", "calm", "cal", "hut", "opera", "amc", "for", "run", "join", "lulu", "wolfspeed"}


def _wb(term: str) -> str:
    """word-boundary regex fragment (avoids 'intel' matching 'intelligence')."""
    return r"(^|\W)" + term.replace(" ", r"\s+") + r"($|\W)"


def patterns(sym: str, entity_id: str) -> list[str]:
    """Regex alternatives (word-boundary) that specifically identify this firm."""
    name = entity_id.split("__")[0].replace("_", " ").strip()
    pats = []
    toks = name.split()
    # multi-word names are unambiguous; single-word only if long and not a common word
    if len(toks) >= 2 or (len(name) >= 5 and name not in STOP):
        pats.append(_wb(name))
    low = sym.lower()
    if len(low) >= 4 and low not in STOP:
        pats.append(_wb(low))
    return pats


def main():
    # ticker -> patterns, restricted to securities
    tick = {}
    for line in open(ENTSYM):
        j = json.loads(line)
        if j.get("kind") == "security" and j.get("symbol"):
            p = patterns(j["symbol"], j.get("entity_id", ""))
            if p:
                tick[j["symbol"]] = p
    print(f"[ground] {len(tick)} tickers with usable name patterns")

    con = duckdb.connect()
    glob = str(CORPUS / "dt=*" / "part-*.parquet")
    con.execute(f"""CREATE TABLE c AS
        SELECT doc_id, lower(coalesce(title,'')||' '||coalesce(body_text,'')) t
        FROM read_parquet('{glob}') WHERE extraction_ok = TRUE""")
    n = con.execute("SELECT count(*) FROM c").fetchone()[0]
    print(f"[ground] scanning {n} full-body articles")

    ticker_docs = {}
    for sym, pats in tick.items():
        cond = " OR ".join(f"regexp_matches(t, '{p.replace(chr(39), chr(39)*2)}')" for p in pats)
        rows = con.execute(f"SELECT doc_id FROM c WHERE {cond}").fetchall()
        if rows:
            ticker_docs[sym] = [r[0] for r in rows]

    OUT.write_text(json.dumps(ticker_docs))
    tagged = len({d for ds in ticker_docs.values() for d in ds})
    print(f"[ground] {len(ticker_docs)} tickers matched; {tagged} distinct docs tagged; wrote {OUT}")
    for s in ("IBM", "PYPL", "PNR", "SGRP", "NVVE", "AARD", "MAN", "ITVPF"):
        print(f"    {s}: {len(ticker_docs.get(s, []))} docs")


if __name__ == "__main__":
    main()
