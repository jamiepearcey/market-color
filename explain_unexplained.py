# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "fastembed>=0.3"]
# ///
"""Point semantic retrieval at the FULL corpus (market_color) for the moves the extraction subset
flagged 'unexplained'. Tests corpus-gap vs ingestion-gap: does the news exist in the 82% the graph skipped?"""
import datetime as dt
from qdrant_client import QdrantClient
from qdrant_client.http import models
from fastembed import TextEmbedding

C = QdrantClient(url="http://localhost:6333")
E = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
def emb(t): return list(map(float, next(iter(E.embed([t])))))

# unexplained activations: (ticker, semantic query, activation day)
MOVES = [
    ("PYPL", "PayPal stock acquisition bid earnings merger", "2026-07-15", +7.3),
    ("IBM",  "IBM International Business Machines stock earnings guidance revenue forecast", "2026-07-14", -8.0),
    ("PNR",  "Pentair stock earnings guidance", "2026-07-15", -8.4),
    ("SGRP", "SPAR Group stock news", "2026-07-16", -10.7),
    ("NVVE", "Nuvve stock news offering", "2026-07-16", -7.2),
    ("AARD", "Aardvark Therapeutics stock trial news", "2026-07-10", +6.5),
]

for sym, q, day, z in MOVES:
    lo = dt.date.fromisoformat(day).toordinal() - 3
    hi = dt.date.fromisoformat(day).toordinal() + 3
    hits = C.query_points("market_color", query=emb(q),
        query_filter=models.Filter(must=[models.FieldCondition(
            key="published_ordinal", range=models.Range(gte=lo, lte=hi))]),
        limit=4, with_payload=True).points
    print(f"\n=== {sym}  {day}  abn_z={z:+.1f}  ({len(hits)} in-window hits) ===")
    for h in hits:
        p = h.payload
        print(f"  {h.score:.2f} [{p.get('published_date')}] {p.get('source_name')}: {p.get('title')}")
