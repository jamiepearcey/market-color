#!/usr/bin/env python3
"""market-color facts MCP server — Qdrant retrieval over decomposed facts.

A small Model Context Protocol (stdio) server that exposes ONE tool,
`search_market_facts`, backed by the `market_facts` Qdrant collection that
`index_facts.py` populates. The chat backend (ui/server/chat-core.mjs) spawns
this server as an MCP client and hands the tool to Claude, so the LLM answers
market questions by retrieving structured, source-attributed facts rather than
from its own (stale, ungrounded) memory.

The query is embedded with the SAME fastembed model as the index (imported from
index_corpus.py), so query and fact vectors live in the same space.

Run standalone for a quick check (stdio; needs a Qdrant at :6333):
  uv run --with mcp --with 'qdrant-client>=1.15' --with fastembed \
    python mcp/qdrant_facts_server.py

Env:
  QDRANT_URL                (default http://localhost:6333)
  QDRANT_API_KEY            (optional)
  MARKET_FACTS_COLLECTION   (default market_facts)
  MARKET_EMBED_MODEL        (default sentence-transformers/all-MiniLM-L6-v2)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# Reuse the indexer's embedding + date helpers so query vectors match the index.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from index_corpus import DEFAULT_FASTEMBED_MODEL, _date_ordinal, embed  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY") or None
COLLECTION = os.environ.get("MARKET_FACTS_COLLECTION", "market_facts")
EMBED_MODEL = os.environ.get("MARKET_EMBED_MODEL", DEFAULT_FASTEMBED_MODEL)
HNSW_EF = 128
OVERSAMPLING = 2.0

mcp = FastMCP("market-color-facts")

_client: Any = None


def _qdrant() -> Any:
    global _client
    if _client is None:
        from qdrant_client import QdrantClient

        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _client


@mcp.tool()
def search_market_facts(
    query: str,
    desk: str = "",
    since: str = "",
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Search the market-color corpus for structured facts relevant to a query.

    Returns atomic, source-attributed facts decomposed from recent financial
    news. Use this for ANY question about markets, companies, commodities,
    central banks, or macro/geopolitical events — ground every claim you make
    in the facts it returns, and cite the source for each.

    Args:
        query: Natural-language description of what you're looking for
            (e.g. "OPEC supply cuts and crude oil prices").
        desk: Optional desk filter — one of rates, fx, energy, metals, crypto,
            equities, geopolitics, macro, asia, other. Empty = all desks.
        since: Optional ISO date (YYYY-MM-DD); only facts published on/after it.
        limit: Max number of facts to return (default 8).

    Returns:
        A list of facts, each: {fact_id, claim, desk, direction, metric,
        entities, source_name, published_date, url, score}. Empty list if
        nothing relevant is indexed.
    """
    if not query or not query.strip():
        return []
    from qdrant_client.http import models

    qvec = embed("fastembed", EMBED_MODEL, [query], "", None)[0]

    must: list[Any] = []
    if desk.strip():
        must.append(models.FieldCondition(key="desk", match=models.MatchValue(value=desk.strip())))
    if since.strip():
        try:
            must.append(models.FieldCondition(
                key="published_ordinal", range=models.Range(gte=_date_ordinal(since.strip()))))
        except ValueError:
            pass

    resp = _qdrant().query_points(
        collection_name=COLLECTION,
        query=qvec,
        query_filter=models.Filter(must=must) if must else None,
        limit=max(1, min(int(limit), 25)),
        with_payload=True,
        search_params=models.SearchParams(
            hnsw_ef=HNSW_EF,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=OVERSAMPLING),
        ),
    )
    out: list[dict[str, Any]] = []
    for p in resp.points:
        pl = p.payload or {}
        out.append({
            "fact_id": pl.get("fact_id"),
            "claim": pl.get("claim"),
            "desk": pl.get("desk"),
            "direction": pl.get("direction"),
            "metric": pl.get("metric"),
            "entities": pl.get("entities") or [],
            "source_name": pl.get("source_name"),
            "published_date": pl.get("published_date"),
            "url": pl.get("url"),
            "title": pl.get("title"),
            "score": round(float(p.score), 4),
        })
    return out


if __name__ == "__main__":
    mcp.run()
