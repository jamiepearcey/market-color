#!/usr/bin/env python3
"""Bridge our graph facts into the UI's `market_facts` Qdrant collection.

The UI/MCP system (mcp/qdrant_facts_server.py) reads a `market_facts` collection
whose payload has entities but no causal graph. Our facts (facts_work/facts.parquet,
produced by graph_experiment.py from the Codex extraction) carry the full graph:
subject/predicate/object/cause/entities/confidence + provenance.

This loads our facts into `market_facts` with a SUPERSET payload — the UI's
existing fields PLUS the graph fields — using the same embedding model + TurboQuant
config the MCP server expects. Lights up the UI with grounded, graph-bearing facts.

Run (needs Qdrant at :6333):
  uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed \
    --with python-dateutil python bridge_facts_to_market_facts.py --recreate
"""
from __future__ import annotations

import argparse
import uuid
from pathlib import Path

import index_corpus as ic  # embed, _client, _date_ordinal, _epoch, _chunked

HERE = Path(__file__).resolve().parent
DEFAULT_FACTS = HERE / "facts_work" / "facts.parquet"
COLLECTION = "market_facts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--facts", type=Path, default=DEFAULT_FACTS)
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--recreate", action="store_true")
    args = ap.parse_args()

    import pyarrow.parquet as pq
    rows = pq.read_table(args.facts).to_pylist()
    if not rows:
        raise SystemExit("no facts in " + str(args.facts))

    client = ic._client(args.qdrant_url, None)
    from qdrant_client.http import models

    dims = None
    total = 0
    for batch in ic._chunked(rows, 64):
        vecs = ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL,
                        [str(r.get("claim") or "") for r in batch], "", None)
        if dims is None:
            dims = len(vecs[0])
            if args.recreate and client.collection_exists(COLLECTION):
                client.delete_collection(COLLECTION)
            if not client.collection_exists(COLLECTION):
                client.create_collection(
                    collection_name=COLLECTION, on_disk_payload=True,
                    vectors_config=models.VectorParams(
                        size=dims, distance=models.Distance.COSINE,
                        quantization_config=models.TurboQuantization(
                            turbo=models.TurboQuantQuantizationConfig(
                                bits=models.TurboQuantBitSize("bits2"), always_ram=True))))
                for fld, sc in (("entities", models.PayloadSchemaType.KEYWORD),
                                ("desk", models.PayloadSchemaType.KEYWORD),
                                ("predicate", models.PayloadSchemaType.KEYWORD),
                                ("published_ordinal", models.PayloadSchemaType.INTEGER),
                                ("published_epoch", models.PayloadSchemaType.INTEGER)):
                    try:
                        client.create_payload_index(COLLECTION, field_name=fld, field_schema=sc)
                    except Exception:
                        pass
        pts = []
        for r, v in zip(batch, vecs, strict=True):
            pd = str(r.get("published_date") or "")
            ents = list(r.get("entities") or [])
            desk = r.get("desk") or "other"
            desks = list(r.get("desks") or []) or [desk]
            pts.append(models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(r.get("fact_id")))),
                vector=v,
                payload={
                    # --- UI/market_facts schema ---
                    "fact_id": r.get("fact_id"), "doc_id": r.get("doc_id"),
                    "claim": r.get("claim"), "entities": ents,
                    "desk": desk, "desks": desks,
                    "direction": r.get("direction"), "metric": r.get("magnitude"),
                    "source_name": r.get("source_name"), "source_domain": None,
                    "source_access": None, "url": r.get("url"), "title": r.get("doc_title"),
                    "published_date": pd,
                    "published_ordinal": ic._date_ordinal(pd) if pd else 0,
                    "published_epoch": ic._epoch(r.get("published_utc")),
                    "published_utc": r.get("published_utc"), "lang": None,
                    # --- graph fields (the "inc the graph" part) ---
                    "subject": r.get("subject"), "predicate": r.get("predicate"),
                    "object": r.get("object"), "cause": r.get("cause"),
                    "cause_entities": list(r.get("cause_entities") or []),
                    "time": r.get("time"), "magnitude": r.get("magnitude"),
                    "confidence": float(r.get("confidence") or 0.0),
                }))
        client.upsert(collection_name=COLLECTION, points=pts, wait=True)
        total += len(pts)
    print(f"bridged {total} facts -> Qdrant '{COLLECTION}' (with graph fields cause/subject/predicate/object)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
