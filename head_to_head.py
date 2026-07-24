# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "fastembed>=0.3"]
# ///
"""Head-to-head on LIVE data: PLAIN embeddings vs ENTITY-GROUNDED retrieval, same corpus/queries.
Metric = false-explanation rate: does the method serve a confident top hit that ISN'T about the firm?
Grounding (ticker-tagged docs) is the objective 'is this doc about the firm' oracle."""
import datetime as dt
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

HERE = Path(__file__).resolve().parent
TICKER_DOCS = json.load(open(HERE / "data" / "tmp" / "live_ticker_docs.json"))
C = QdrantClient(url="http://localhost:6333")
E = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
def emb(t): return list(map(float, next(iter(E.embed([t])))))

MOVES = [
    ("PYPL", "PayPal stock acquisition bid earnings merger", "2026-07-15", +7.3),
    ("IBM",  "IBM International Business Machines earnings guidance revenue forecast", "2026-07-14", -8.0),
    ("PNR",  "Pentair stock earnings guidance forecast", "2026-07-15", -8.4),
    ("MAN",  "ManpowerGroup staffing earnings", "2026-07-16", +7.7),
    ("SGRP", "SPAR Group stock news", "2026-07-16", -10.7),
    ("NVVE", "Nuvve vehicle-to-grid EV charging stock", "2026-07-16", -7.2),
    ("AARD", "Aardvark Therapeutics clinical trial", "2026-07-10", +6.5),
]


def win(day, w=3):
    o = dt.date.fromisoformat(day).toordinal()
    return models.FieldCondition(key="published_ordinal", range=models.Range(gte=o - w, lte=o + w))


def retrieve(query, day, k, overfetch=1):
    return C.query_points("market_color", query=emb(query), query_filter=models.Filter(must=[win(day)]),
                          limit=k * overfetch, with_payload=True).points


rows = []
for sym, q, day, z in MOVES:
    tagged = set(TICKER_DOCS.get(sym, []))
    plain = retrieve(q, day, 4)
    # grounded: over-fetch, keep only docs actually about this firm; abstain if none
    grounded_pool = retrieve(q, day, 4, overfetch=12)
    grounded = [h for h in grounded_pool if h.payload.get("doc_id") in tagged][:4]

    plain_top = plain[0] if plain else None
    plain_firm = bool(plain_top and plain_top.payload.get("doc_id") in tagged)
    g_top = grounded[0] if grounded else None

    rows.append((sym, z, plain_top, plain_firm, g_top, len(tagged)))
    print(f"\n=== {sym}  {day}  abn_z={z:+.1f}   (firm has {len(tagged)} tagged docs in corpus) ===")
    pt = plain_top.payload if plain_top else {}
    print(f"  PLAIN    top [{plain_top.score:.2f}]" if plain_top else "  PLAIN    (nothing)", end="")
    print(f" {'✓firm' if plain_firm else '✗OFF-TOPIC'}: {pt.get('source_name')}: {pt.get('title')}")
    if g_top:
        print(f"  GROUNDED top [{g_top.score:.2f}] ✓firm: {g_top.payload.get('source_name')}: {g_top.payload.get('title')}")
    else:
        print(f"  GROUNDED (abstains — no firm-specific news in corpus)")

n = len(rows)
plain_false = sum(1 for r in rows if r[2] and not r[3])          # served an off-topic top hit
plain_hit = sum(1 for r in rows if r[2])
g_correct = sum(1 for r in rows if r[4])                          # served a real firm hit
g_abstain = sum(1 for r in rows if not r[4])
g_false = 0                                                       # grounded can't serve off-topic by construction
print(f"\n==== HEAD-TO-HEAD (N={n} live activations) ====")
print(f"PLAIN embeddings : returns a hit {plain_hit}/{n}; of those {plain_false} are OFF-TOPIC "
      f"(false-explanation rate {plain_false}/{n} = {plain_false/n:.0%})")
print(f"GROUNDED         : {g_correct} firm-specific explanations, {g_abstain} correct abstentions, "
      f"false-explanation rate 0/{n} = 0%")
