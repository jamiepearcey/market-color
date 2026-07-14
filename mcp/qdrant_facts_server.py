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
import re
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


_GRAPH: dict[str, Any] | None = None


def _get_graph() -> dict[str, Any]:
    """Build + cache the entity co-occurrence graph from the fact collection.

    Returns {adj: entity->{entity: co-occurrence weight}, fact_entities: fact_id->[ents],
    fact_payload: fact_id->payload}. Cached for the process lifetime (rebuild the
    process after re-indexing facts).
    """
    global _GRAPH
    if _GRAPH is not None:
        return _GRAPH
    from collections import defaultdict

    adj: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    fact_entities: dict[str, list[str]] = {}
    fact_payload: dict[str, dict[str, Any]] = {}
    ent_facts: dict[str, list[str]] = defaultdict(list)
    client = _qdrant()
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION, limit=512, offset=offset,
            with_payload=True, with_vectors=False)
        for p in points:
            pl = p.payload or {}
            fid = pl.get("fact_id")
            if not fid:
                continue
            ents = [e for e in (pl.get("entities") or []) if e]
            fact_entities[fid] = ents
            fact_payload[fid] = pl
            for a in ents:
                ent_facts[a].append(fid)
                for b in ents:
                    if a != b:
                        adj[a][b] += 1.0
        if not offset:
            break
    _GRAPH = {"adj": {k: dict(v) for k, v in adj.items()},
              "ent_facts": {k: v for k, v in ent_facts.items()},
              "fact_entities": fact_entities, "fact_payload": fact_payload}
    return _GRAPH


_LEX: dict[str, Any] | None = None


def _get_lex() -> dict[str, Any]:
    """BM25-lite index over claim+cause text. The lexical route measurably
    complements dense retrieval (held-out chain-coverage +5pts on its own):
    it catches exact names/figures that embeddings rank poorly."""
    global _LEX
    if _LEX is not None:
        return _LEX
    import math
    from collections import Counter

    g = _get_graph()
    tf: dict[str, Counter] = {}
    df: Counter = Counter()
    lengths: dict[str, int] = {}
    for fid, pl in g["fact_payload"].items():
        toks = re.findall(r"[a-z0-9][a-z0-9.%$/-]{1,}",
                          f"{pl.get('claim') or ''} {pl.get('cause') or ''}".lower())
        c = Counter(toks)
        tf[fid] = c
        lengths[fid] = len(toks)
        df.update(c.keys())
    n = max(len(tf), 1)
    idf = {t: math.log(1 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    _LEX = {"tf": tf, "idf": idf,
            "avg_len": sum(lengths.values()) / n, "lengths": lengths}
    return _LEX


def _lexical_top(query: str, k: int) -> list[tuple[float, str]]:
    lex = _get_lex()
    q_toks = set(re.findall(r"[a-z0-9][a-z0-9.%$/-]{1,}", query.lower()))
    scored = []
    for fid, tf in lex["tf"].items():
        s = 0.0
        for t in q_toks:
            cnt = tf.get(t)
            if cnt:
                dl = lex["lengths"][fid]
                s += lex["idf"].get(t, 0) * cnt * 2.2 / (
                    cnt + 1.2 * (0.25 + 0.75 * dl / lex["avg_len"]))
        if s > 0:
            scored.append((s, fid))
    scored.sort(key=lambda t: -t[0])
    return scored[:k]


def _personalized_pagerank(adj: dict[str, dict[str, float]],
                           personalization: dict[str, float],
                           alpha: float = 0.85, iters: int = 25) -> dict[str, float]:
    """Sparse personalized PageRank (pure-Python power iteration).

    Spreads relevance from the seed entities (`personalization`) across the
    co-occurrence graph, so facts connected to the seeds through intermediate
    entities (2-3 hops) still score — not just 1-hop neighbours.
    """
    nodes = set(adj) | set(personalization)
    if not nodes:
        return {}
    tot = sum(personalization.values()) or 1.0
    p = {n: personalization.get(n, 0.0) / tot for n in nodes}
    deg = {n: sum(adj.get(n, {}).values()) for n in nodes}
    r = dict(p)
    for _ in range(iters):
        nr = {n: (1.0 - alpha) * p[n] for n in nodes}
        dangling = 0.0
        for n in nodes:
            rn = r.get(n, 0.0)
            if rn <= 0.0:
                continue
            d = deg[n]
            if d <= 0.0:
                dangling += alpha * rn
                continue
            share = alpha * rn / d
            for nbr, w in adj[n].items():
                nr[nbr] = nr.get(nbr, 0.0) + share * w
        if dangling:
            for n in nodes:
                nr[n] += dangling * p[n]
        r = nr
    return r


def _fact_dict(pl: dict[str, Any], relation: str, score: float | None = None,
               shared: int | None = None) -> dict[str, Any]:
    d = {
        "fact_id": pl.get("fact_id"),
        "claim": pl.get("claim"),
        "desk": pl.get("desk"),
        "direction": pl.get("direction"),
        "metric": pl.get("metric"),
        "entities": pl.get("entities") or [],
        "subject": pl.get("subject"),
        "predicate": pl.get("predicate"),
        "cause": pl.get("cause"),
        "source_name": pl.get("source_name"),
        "published_date": pl.get("published_date"),
        "url": pl.get("url"),
        "title": pl.get("title"),
        "relation": relation,
    }
    if score is not None:
        d["score"] = round(float(score), 4)
    if shared is not None:
        d["shared_entities"] = shared
    return d


def _resolve_entity(entity: str, g: dict[str, Any]) -> str | None:
    """Match a free-text entity to a canonical node. Exact, else shortest node
    containing all query tokens (light fuzzy match — canonicalization is a TODO)."""
    if not entity:
        return None
    k = " ".join(entity.lower().split()).strip(".,'\"`")
    if k in g["ent_facts"]:
        return k
    toks = k.split()
    cands = [e for e in g["ent_facts"] if all(t in e for t in toks)]
    return min(cands, key=len) if cands else None


@mcp.tool()
def search_market_facts(
    query: str,
    desk: str = "",
    since: str = "",
    limit: int = 40,
    expand: bool = True,
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
        limit: Max number of facts to return (default 40 — measured: recall on
            hard causal questions rises ~15pts going from ~10 to ~50 results;
            facts are small, cite selectively).
        expand: If true (default), also return graph-connected facts — a 1-hop
            expansion over facts that share an entity with the seeds. Lets you
            trace transmission (e.g. a supply shock -> demand response in another
            market) rather than only the top-similar facts.

    Returns:
        A list of facts, each: {fact_id, claim, desk, direction, metric,
        entities, subject, predicate, cause, source_name, published_date, url,
        title, score, relation}. `relation` is "seed" (matched the query),
        "cause-driver" (the upstream fact behind a seed's stated cause —
        retrieved by embedding the cause text itself; use these to explain WHY),
        or "graph-neighbor" (reached via a shared entity; also carries
        `shared_entities`). `cause` is the stated causal driver — the graph edge.
        Empty list if nothing relevant is indexed.
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
        limit=max(1, min(int(limit), 60)),
        with_payload=True,
        search_params=models.SearchParams(
            hnsw_ef=HNSW_EF,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=OVERSAMPLING),
        ),
    )
    def _fd(pl: dict[str, Any], score: float, relation: str,
            shared: int | None = None) -> dict[str, Any]:
        d = {
            "fact_id": pl.get("fact_id"),
            "claim": pl.get("claim"),
            "desk": pl.get("desk"),
            "direction": pl.get("direction"),
            "metric": pl.get("metric"),
            "entities": pl.get("entities") or [],
            "subject": pl.get("subject"),      # graph node
            "predicate": pl.get("predicate"),  # typed relation
            "cause": pl.get("cause"),          # causal driver = graph edge
            "source_name": pl.get("source_name"),
            "published_date": pl.get("published_date"),
            "url": pl.get("url"),
            "title": pl.get("title"),
            "score": round(float(score), 4),
            "relation": relation,
        }
        if shared is not None:
            d["shared_entities"] = shared
        return d

    seeds = list(resp.points)
    seed_ids = {p.id for p in seeds}
    seed_ents: set[str] = set()
    for p in seeds:
        seed_ents.update((p.payload or {}).get("entities") or [])
    out = [_fd(p.payload or {}, p.score, "seed") for p in seeds]

    # LEXICAL ROUTE (validated: +5pts held-out chain coverage on its own):
    # append top BM25 hits missing from the dense seeds, respecting filters.
    seed_fids = {(p.payload or {}).get("fact_id") for p in seeds}
    lex_added = 0
    for s, fid in _lexical_top(query, 40):
        if lex_added >= max(3, int(limit) // 5):
            break
        if fid in seed_fids:
            continue
        pl = _get_graph()["fact_payload"].get(fid) or {}
        if desk.strip() and pl.get("desk") != desk.strip():
            continue
        if since.strip():
            try:
                if (pl.get("published_ordinal") or 0) < _date_ordinal(since.strip()):
                    continue
            except ValueError:
                pass
        seed_fids.add(fid)
        out.append(_fd(pl, s, "seed-lex"))
        lex_added += 1

    # CAUSE-DRIVER ATTACHMENT (trigger-gated): for top seeds whose stated cause
    # points OUTSIDE the fact itself (an unresolved upstream hop), embed the raw
    # cause text as a fresh query and attach the best cross-doc fact for it.
    # This is the measured fix for chain-end questions where the root event is
    # semantically invisible to the original query at any limit.
    if expand:
        triggered = []
        for p in seeds[:4]:
            pl = p.payload or {}
            cause = pl.get("cause")
            c_ents = set(pl.get("cause_entities") or []) - set(pl.get("entities") or [])
            if cause and str(cause).lower() not in ("none", "null") and c_ents:
                triggered.append((p, str(cause)))
        if triggered:
            cvecs = embed("fastembed", EMBED_MODEL, [c for _, c in triggered], "", None)
            for (p, _), cv in zip(triggered, cvecs):
                pl = p.payload or {}
                cresp = _qdrant().query_points(
                    collection_name=COLLECTION, query=cv, limit=4, with_payload=True,
                    search_params=models.SearchParams(
                        hnsw_ef=HNSW_EF,
                        quantization=models.QuantizationSearchParams(
                            rescore=True, oversampling=OVERSAMPLING)))
                for cp in cresp.points:
                    cpl = cp.payload or {}
                    if (cp.id in seed_ids
                            or cpl.get("doc_id") == pl.get("doc_id")
                            or cpl.get("fact_id") == pl.get("fact_id")):
                        continue
                    seed_ids.add(cp.id)
                    out.append(_fd(cpl, cp.score, "cause-driver"))
                    break

    # GRAPH EXPANSION via personalized PageRank: spread relevance from the seed
    # entities (weighted by seed score) over the co-occurrence graph, then rank
    # non-seed facts by graph proximity. Surfaces multi-hop-connected facts, not
    # just 1-hop. Cross-desk on purpose, to trace transmission.
    if expand and seed_ents:
        cap = max(1, min(int(limit), 25))
        graph = _get_graph()
        if graph["adj"]:
            pers: dict[str, float] = {}
            for p in seeds:
                pl = p.payload or {}
                s = max(float(p.score), 0.0)
                for e in (pl.get("entities") or []):
                    pers[e] = pers.get(e, 0.0) + s
            ent_score = _personalized_pagerank(graph["adj"], pers)
            seed_fids = {(p.payload or {}).get("fact_id") for p in seeds}
            cand: list[tuple[float, int, str]] = []
            for fid, ents in graph["fact_entities"].items():
                if fid in seed_fids:
                    continue
                prox = sum(ent_score.get(e, 0.0) for e in ents)
                if prox > 0.0:
                    shared = len(set(ents) & seed_ents)
                    cand.append((prox, shared, fid))
            cand.sort(key=lambda t: (-t[0], -t[1]))
            for prox, shared, fid in cand[:cap]:
                out.append(_fd(graph["fact_payload"][fid], prox, "graph-neighbor", shared))
    return out


@mcp.tool()
def facts_for_entity(entity: str, limit: int = 10, since: str = "") -> list[dict[str, Any]]:
    """All facts that mention a specific entity (a graph node's incident facts).

    Use to inspect what the corpus actually says about one company / commodity /
    country / central bank before you reason about it. Newest first.
    """
    g = _get_graph()
    key = _resolve_entity(entity, g)
    if not key:
        return []
    facts = [g["fact_payload"][f] for f in g["ent_facts"].get(key, []) if f in g["fact_payload"]]
    if since.strip():
        try:
            floor = _date_ordinal(since.strip())
            facts = [f for f in facts if int(f.get("published_ordinal") or 0) >= floor]
        except ValueError:
            pass
    facts.sort(key=lambda f: (int(f.get("published_ordinal") or 0), float(f.get("confidence") or 0)),
               reverse=True)
    return [_fact_dict(f, "entity") for f in facts[:max(1, min(int(limit), 25))]]


@mcp.tool()
def neighbors(entity: str, top: int = 15) -> list[dict[str, Any]]:
    """The entities most co-occurring with a given entity (its graph neighbourhood),
    with co-occurrence weights. Use to see what an entity is connected to and decide
    which link to follow next when tracing transmission."""
    g = _get_graph()
    key = _resolve_entity(entity, g)
    if not key:
        return []
    nbrs = sorted(g["adj"].get(key, {}).items(), key=lambda kv: -kv[1])
    return [{"entity": e, "weight": w} for e, w in nbrs[:max(1, min(int(top), 40))]]


@mcp.tool()
def trace_causes(entity: str, limit: int = 8) -> dict[str, Any]:
    """Causal view for an entity: what DRIVES it (facts about it that carry a stated
    cause) and what IT DRIVES (facts elsewhere whose cause text names it). Use to build
    a transmission chain (driver -> asset -> second-order effect)."""
    g = _get_graph()
    key = _resolve_entity(entity, g)
    if not key:
        return {"entity": entity, "drivers": [], "effects": []}
    own = set(g["ent_facts"].get(key, []))
    drivers = [_fact_dict(g["fact_payload"][f], "driver") for f in own
               if g["fact_payload"][f].get("cause")]
    effects = []
    for fid, pl in g["fact_payload"].items():
        if fid in own:
            continue
        cause = (pl.get("cause") or "").lower()
        if cause and key in cause:
            effects.append(_fact_dict(pl, "effect"))
    cap = max(1, min(int(limit), 15))
    return {"entity": key, "drivers": drivers[:cap], "effects": effects[:cap]}


@mcp.tool()
def entity_path(entity_a: str, entity_b: str, max_hops: int = 3) -> dict[str, Any]:
    """Shortest path between two entities over the co-occurrence graph (<= 3 hops),
    with the fact connecting each step. Use to trace how a shock in one place reaches
    another market/asset."""
    from collections import deque

    g = _get_graph()
    adj = g["adj"]
    a = _resolve_entity(entity_a, g)
    b = _resolve_entity(entity_b, g)
    if not a or not b:
        return {"found": False, "reason": "entity not found", "path": []}
    max_hops = max(1, min(int(max_hops), 3))
    FRONTIER = 30
    prev: dict[str, str | None] = {a: None}
    depth = {a: 0}
    q: deque[str] = deque([a])
    found = False
    while q:
        n = q.popleft()
        if n == b:
            found = True
            break
        if depth[n] >= max_hops:
            continue
        for nb, _w in sorted(adj.get(n, {}).items(), key=lambda kv: -kv[1])[:FRONTIER]:
            if nb not in prev:
                prev[nb] = n
                depth[nb] = depth[n] + 1
                q.append(nb)
    if not found:
        return {"found": False, "path": [], "entity_a": a, "entity_b": b}
    path: list[str] = []
    c: str | None = b
    while c is not None:
        path.append(c)
        c = prev[c]
    path.reverse()
    links = []
    for x, y in zip(path, path[1:]):
        shared_f = set(g["ent_facts"].get(x, [])) & set(g["ent_facts"].get(y, []))
        ex = next(iter(shared_f), None)
        links.append({"from": x, "to": y,
                      "via_fact": _fact_dict(g["fact_payload"][ex], "link") if ex else None})
    return {"found": True, "path": path, "links": links}


if __name__ == "__main__":
    mcp.run()
