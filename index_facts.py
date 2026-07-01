#!/usr/bin/env python3
"""market-color fact index — embed decomposed facts into Qdrant for MCP retrieval.

Companion to `decompose_facts.py`. Reads the facts parquet it produced
(`data/facts/dt=*/part-*.parquet`) and upserts one Qdrant point per fact into
the `market_facts` collection, with the same embedding model and TurboQuant
config as `index_corpus.py` (so the two collections are interchangeable
infrastructure). The MCP server (`mcp/qdrant_facts_server.py`) queries this
collection.

What's embedded is the fact's *claim* (with light desk/entity context), not the
whole article — retrieval returns atomic facts, each carrying provenance back to
its source article in the payload.

Run (self-bootstraps via uv; needs a Qdrant at :6333):
  uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed \
    python index_facts.py --recreate

  uv run --with 'qdrant-client>=1.15' --with fastembed \
    python index_facts.py search --query "OPEC supply cut" --desk energy
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Single source of truth for embeddings + collection shape + provenance fields.
from index_corpus import (
    DEFAULT_FASTEMBED_MODEL,
    DEFAULT_HNSW_EF,
    DEFAULT_OVERSAMPLING,
    DEFAULT_TURBO_BITS,
    _chunked,
    _client,
    _clean,
    _date_ordinal,
    _epoch,
    _point_id,
    embed,
    ensure_collection,
    ensure_runtime,
)

HERE = Path(__file__).resolve().parent
DEFAULT_FACTS = HERE / "data" / "facts"
FACTS_COLLECTION = "market_facts"
DEFAULT_BATCH = 128


def _fact_embedding_text(row: dict[str, Any]) -> str:
    """What we embed for a fact: the claim, with desk + entities as light context
    so semantically-near facts on the same desk cluster together."""
    parts = [f"Desk: {row.get('desk')}"]
    ents = row.get("entities") or []
    if ents:
        parts.append(f"Entities: {', '.join(str(e) for e in ents)}")
    parts.append(f"Fact: {_clean(row.get('claim'))}")
    metric = _clean(row.get("metric"))
    if metric:
        parts.append(f"Figure: {metric}")
    return "\n".join(parts)


def _fact_payload(row: dict[str, Any]) -> dict[str, Any]:
    pub_date = str(row.get("published_date"))
    return {
        "fact_id": row["fact_id"],
        "doc_id": row["doc_id"],
        "claim": _clean(row.get("claim")),
        "entities": list(row.get("entities") or []),
        "desk": row.get("desk"),
        # keep a list mirror so the desk filter can use MatchAny like the corpus index
        "desks": [row.get("desk")] if row.get("desk") else [],
        "direction": row.get("direction"),
        "metric": _clean(row.get("metric")),
        "source_name": row.get("source_name"),
        "source_domain": row.get("source_domain"),
        "source_access": row.get("source_access"),
        "url": row.get("url"),
        "title": row.get("title"),
        "published_date": pub_date,
        "published_ordinal": _date_ordinal(pub_date),
        "published_epoch": _epoch(row.get("published_utc")),
        "published_utc": row.get("published_utc"),
        "lang": row.get("lang"),
    }


def _load_facts(facts_dir: Path, start_date: str | None, end_date: str | None,
                desk: str | None, limit: int | None) -> list[dict[str, Any]]:
    import duckdb

    glob = str(facts_dir / "dt=*" / "part-*.parquet")
    where = ["1=1"]
    params: list[Any] = []
    if start_date:
        where.append("published_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("published_date <= ?")
        params.append(end_date)
    if desk:
        where.append("desk = ?")
        params.append(desk)
    sql = f"""
        SELECT fact_id, doc_id, claim, entities, desk, direction, metric,
               source_name, source_domain, source_access, url, title,
               CAST(published_date AS VARCHAR) AS published_date, published_utc, lang
        FROM read_parquet('{glob}')
        WHERE {" AND ".join(where)}
        ORDER BY published_date ASC, fact_id ASC
    """
    if limit:
        sql += f" LIMIT {int(limit)}"
    con = duckdb.connect()
    try:
        cur = con.execute(sql, params)
        cols = [c[0] for c in cur.description]
        return [dict(zip(cols, r, strict=True)) for r in cur.fetchall()]
    finally:
        con.close()


def _ensure_facts_collection(client: Any, name: str, dims: int, turbo_bits: str,
                             recreate: bool, vector_on_disk: bool) -> None:
    """Same TurboQuant config as the corpus collection, plus a `desk` keyword index."""
    from qdrant_client.http import models

    ensure_collection(client, name, dims, turbo_bits, recreate, vector_on_disk)
    for field in ("desk", "direction"):
        try:
            client.create_payload_index(name, field_name=field,
                                        field_schema=models.PayloadSchemaType.KEYWORD)
        except Exception:
            pass


def run_index(args: argparse.Namespace) -> int:
    rows = _load_facts(Path(args.facts_dir), args.start_date, args.end_date, args.desk, args.row_limit)
    if not rows:
        raise RuntimeError(
            "no facts found — run decompose_facts.py first to populate data/facts/"
        )
    client = _client(args.qdrant_url, args.qdrant_api_key)
    from qdrant_client.http import models

    indexed = 0
    dims: int | None = None
    for batch in _chunked(rows, args.batch_size):
        vectors = embed(args.embedding_provider, args.embedding_model,
                        [_fact_embedding_text(r) for r in batch], args.ollama_url, None)
        if dims is None:
            dims = len(vectors[0])
            _ensure_facts_collection(client, args.collection, dims, args.turbo_bits,
                                     args.recreate, args.vector_on_disk)
        points = [
            models.PointStruct(id=_point_id(r["fact_id"]), vector=v, payload=_fact_payload(r))
            for r, v in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        indexed += len(points)
        print(f"  indexed {indexed}/{len(rows)} facts", file=sys.stderr)

    print(json.dumps({
        "collection": args.collection, "indexed_facts": indexed, "dimensions": dims,
        "embedding_model": args.embedding_model, "facts_dir": str(args.facts_dir),
    }, indent=2))
    return 0


def run_search(args: argparse.Namespace) -> int:
    """Smoke-test retrieval without the MCP server."""
    client = _client(args.qdrant_url, args.qdrant_api_key)
    from qdrant_client.http import models

    qvec = embed(args.embedding_provider, args.embedding_model, [args.query], args.ollama_url, None)[0]
    must: list[Any] = []
    if args.desk:
        must.append(models.FieldCondition(key="desk", match=models.MatchValue(value=args.desk)))
    if args.since:
        must.append(models.FieldCondition(
            key="published_ordinal", range=models.Range(gte=_date_ordinal(args.since))))
    resp = client.query_points(
        collection_name=args.collection,
        query=qvec,
        query_filter=models.Filter(must=must) if must else None,
        limit=args.limit,
        with_payload=True,
        search_params=models.SearchParams(
            hnsw_ef=DEFAULT_HNSW_EF,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=DEFAULT_OVERSAMPLING),
        ),
    )
    hits = [{
        "score": round(p.score, 4),
        "claim": (p.payload or {}).get("claim"),
        "desk": (p.payload or {}).get("desk"),
        "direction": (p.payload or {}).get("direction"),
        "metric": (p.payload or {}).get("metric"),
        "source_name": (p.payload or {}).get("source_name"),
        "published_date": (p.payload or {}).get("published_date"),
        "url": (p.payload or {}).get("url"),
    } for p in resp.points if p.score >= args.min_score]
    print(json.dumps({"query": args.query, "desk": args.desk, "hits": hits}, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command")

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--collection", default=FACTS_COLLECTION)
        sp.add_argument("--qdrant-url", default="http://localhost:6333")
        sp.add_argument("--qdrant-api-key")
        sp.add_argument("--embedding-provider",
                        choices=["fastembed", "sentence-transformers", "ollama"],
                        default="fastembed")
        sp.add_argument("--embedding-model", default=DEFAULT_FASTEMBED_MODEL)
        sp.add_argument("--ollama-url", default="http://localhost:11434")

    ip = sub.add_parser("index", help="Embed facts and upsert into Qdrant (default).")
    common(ip)
    ip.add_argument("--facts-dir", default=str(DEFAULT_FACTS))
    ip.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ip.add_argument("--start-date")
    ip.add_argument("--end-date")
    ip.add_argument("--desk")
    ip.add_argument("--row-limit", type=int)
    ip.add_argument("--turbo-bits", choices=["bits1", "bits1_5", "bits2", "bits4"],
                    default=DEFAULT_TURBO_BITS)
    ip.add_argument("--recreate", action="store_true")
    ip.add_argument("--vector-on-disk", action="store_true")
    ip.set_defaults(func=run_index)

    sp = sub.add_parser("search", help="Smoke-test fact retrieval.")
    common(sp)
    sp.add_argument("--query", required=True)
    sp.add_argument("--desk")
    sp.add_argument("--since", help="ISO date; only facts published on/after this")
    sp.add_argument("--limit", type=int, default=8)
    sp.add_argument("--min-score", type=float, default=0.0)
    sp.set_defaults(func=run_search)
    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if not getattr(args, "func", None):
        # default subcommand is `index`
        args = parser.parse_args(["index", *sys.argv[1:]])
    ensure_runtime(args.embedding_provider)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
