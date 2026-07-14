#!/usr/bin/env python3
"""market-color GraphRAG experiment — facts -> fact-embeddings + entity graph -> seed/expand.

Pipeline:
  build   merge codex fact batches -> facts.parquet, normalize entities, embed each
          fact CLAIM into a Qdrant collection (TurboQuant) with entities indexed as a
          keyword payload (the graph adjacency).
  search  seed by vector search over facts, then EXPAND one hop along shared entities
          (Qdrant entity filter = adjacency) and surface causal chains. The seed->expand
          retrieval experiment.

Reuses embed/_client/ensure_collection from index_corpus.py. Needs Qdrant at :6333.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import index_corpus as ic  # embed, _client, _epoch, _date_ordinal, _chunked

HERE = Path(__file__).resolve().parent
DEFAULT_FACTS_GLOB = str(HERE / "facts_work" / "batches" / "*_facts_*.jsonl")
DEFAULT_FACTS_PARQUET = HERE / "facts_work" / "facts.parquet"
FACTS_COLLECTION = "market_color_facts"

# Light entity canonicalization (string-normalize + a few obvious aliases).
ALIASES = {
    "us": "united states", "u.s.": "united states", "usa": "united states",
    "the fed": "federal reserve", "fed": "federal reserve", "fomc": "federal reserve",
    "ecb": "european central bank", "boj": "bank of japan", "boe": "bank of england",
    "pboc": "people's bank of china", "uk": "united kingdom", "u.k.": "united kingdom",
    "opec+": "opec", "wti": "wti crude", "brent": "brent crude",
}


def norm_entity(e: str) -> str:
    k = " ".join(str(e).lower().split()).strip(".,'\"`")
    return ALIASES.get(k, k)


def fact_id(doc_id: str, idx: int, claim: str) -> str:
    return hashlib.sha256(f"{doc_id}|{idx}|{claim}".encode("utf-8", "ignore")).hexdigest()[:20]


# --------------------------------------------------------------------------- #
def merge_facts(facts_glob: str, corpus: Path) -> list[dict[str, Any]]:
    import duckdb

    files = sorted(glob.glob(facts_glob))
    if not files:
        raise RuntimeError(f"no fact files at {facts_glob}")
    # doc metadata for provenance
    con = duckdb.connect()
    meta = {r[0]: r for r in con.execute(
        f"select doc_id, source_name, CAST(published_date AS VARCHAR), published_utc, url, title "
        f"from read_parquet('{corpus}/dt=*/part-*.parquet')").fetchall()}
    out: list[dict[str, Any]] = []
    for fp in files:
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            doc_id = rec.get("doc_id")
            m = meta.get(doc_id, (doc_id, None, None, None, None, rec.get("title")))
            facts_list = rec.get("facts") or []
            if not isinstance(facts_list, list):
                continue
            for i, f in enumerate(facts_list):
                if not isinstance(f, dict):
                    continue
                claim = str(f.get("claim") or "").strip()
                if not claim:
                    continue
                raw_ents = f.get("entities")
                if not isinstance(raw_ents, list):
                    raw_ents = []
                ents = [norm_entity(x) for x in raw_ents if x and isinstance(x, str)]
                # fold subject/object ONLY if they look like entities (not clauses);
                # never fold `cause` (it is free-text and pollutes the node set).
                for extra in (f.get("subject"), f.get("object")):
                    if extra and len(str(extra).split()) <= 4:
                        ents.append(norm_entity(extra))
                # keep entities that look like nodes: <= 5 words, drop sentence-fragments.
                ents = sorted({e for e in ents if e and 1 < len(e) <= 48 and len(e.split()) <= 5})
                def _s(key: str) -> str | None:
                    v = f.get(key)
                    return None if v is None or v == "" else str(v)

                try:
                    conf = float(f.get("confidence") or 0.0)
                except (TypeError, ValueError):
                    conf = 0.0
                out.append({
                    "fact_id": fact_id(doc_id, i, claim),
                    "doc_id": doc_id, "source_name": m[1], "published_date": m[2],
                    "published_utc": m[3], "url": m[4], "doc_title": m[5],
                    "claim": claim, "subject": _s("subject"), "predicate": _s("predicate"),
                    "object": _s("object"), "direction": _s("direction"),
                    "magnitude": _s("magnitude"), "time": _s("time"),
                    "cause": _s("cause"), "entities": ents,
                    "confidence": conf,
                })
    # dedup (same doc may appear in more than one batch file)
    seen: set[str] = set()
    deduped = []
    for f in out:
        if f["fact_id"] in seen:
            continue
        seen.add(f["fact_id"])
        deduped.append(f)
    return deduped


# --------------------------------------------------------------------------- #
# Canonicalization: merge near-duplicate entity nodes, resolve causes to nodes
# --------------------------------------------------------------------------- #
def _token_jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / max(1, len(ta | tb))


def canonicalize_entities(facts: list[dict[str, Any]], provider: str, model: str,
                          sim_floor: float = 0.92) -> dict[str, Any]:
    """Merge near-duplicate entities (embedding cosine >= sim_floor AND a lexical
    guard: token overlap or substring) via union-find; canonical form = the most
    frequent surface form. Returns {"map": {entity->canonical}, "vecs": {canonical->vec}}.
    """
    import numpy as np
    from collections import Counter

    count: Counter = Counter()
    for f in facts:
        count.update(f["entities"])
    ents = sorted(count)
    if not ents:
        return {"map": {}, "vecs": {}}
    vecs = []
    for chunk in ic._chunked(ents, 256):
        vecs.extend(ic.embed(provider, model, chunk, ic.DEFAULT_OLLAMA_URL, None))
    mat = np.asarray(vecs, dtype="float32")
    mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9

    parent = list(range(len(ents)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # chunked all-pairs cosine to bound memory on large vocabularies
    step = 1024
    for s in range(0, len(ents), step):
        sims = mat[s:s + step] @ mat.T
        rows, cols = np.nonzero(sims >= sim_floor)
        for r, c in zip(rows.tolist(), cols.tolist()):
            i, j = s + r, c
            if i >= j:
                continue
            a, b = ents[i], ents[j]
            if _token_jaccard(a, b) >= 0.5 or a in b or b in a:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(len(ents)):
        groups.setdefault(find(i), []).append(i)
    emap: dict[str, str] = {}
    cvecs: dict[str, Any] = {}
    merged = 0
    for members in groups.values():
        canon_i = max(members, key=lambda i: (count[ents[i]], -len(ents[i])))
        canon = ents[canon_i]
        cvecs[canon] = mat[canon_i]
        for i in members:
            emap[ents[i]] = canon
        merged += len(members) - 1
    print(f"[canon] entities {len(ents)} -> {len(groups)} nodes ({merged} near-dups merged)",
          file=sys.stderr)
    return {"map": emap, "vecs": cvecs}


def resolve_causes(facts: list[dict[str, Any]], canon: dict[str, Any],
                   provider: str, model: str, sim_floor: float = 0.60) -> None:
    """Resolve each fact's free-text `cause` to canonical entity nodes:
    (1) word-boundary containment of an entity name in the cause text (longest
    matches win), else (2) embedding top-1 >= sim_floor. Sets f["cause_entities"].
    """
    import re as _re
    import numpy as np

    cvecs = canon["vecs"]
    names = sorted(cvecs, key=len, reverse=True)
    mat = np.asarray([cvecs[n] for n in names], dtype="float32") if names else None

    pending: list[tuple[dict[str, Any], str]] = []
    for f in facts:
        cause = f.get("cause")
        f["cause_entities"] = []
        if not cause:
            continue
        low = norm_entity(str(cause))
        hits = []
        for n in names:
            if len(n) < 3:
                continue
            if _re.search(rf"(?<![a-z0-9]){_re.escape(n)}(?![a-z0-9])", low):
                if not any(n in h for h in hits):  # skip subsets of an already-matched name
                    hits.append(n)
            if len(hits) >= 3:
                break
        if hits:
            f["cause_entities"] = hits
        else:
            pending.append((f, str(cause)))
    if pending and mat is not None:
        vecs = []
        for chunk in ic._chunked([c for _, c in pending], 256):
            vecs.extend(ic.embed(provider, model, chunk, ic.DEFAULT_OLLAMA_URL, None))
        q = np.asarray(vecs, dtype="float32")
        q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
        sims = q @ mat.T
        for (f, _), row in zip(pending, sims, strict=True):
            best = int(np.argmax(row))
            if float(row[best]) >= sim_floor:
                f["cause_entities"] = [names[best]]
    total = sum(1 for f in facts if f.get("cause"))
    resolved = sum(1 for f in facts if f["cause_entities"])
    print(f"[canon] causes resolved to entity nodes: {resolved}/{total}", file=sys.stderr)


def cmd_build(args: argparse.Namespace) -> int:
    import numpy as np

    facts = merge_facts(args.facts_glob, Path(args.corpus))
    if not facts:
        raise RuntimeError("no facts merged")

    # 1) embed all claims (kept in memory; the fact set is small)
    claim_vecs: list[list[float]] = []
    for batch in ic._chunked(facts, 64):
        claim_vecs.extend(ic.embed(args.embedding_provider, args.embedding_model,
                                   [f["claim"] for f in batch], ic.DEFAULT_OLLAMA_URL, None))

    # 2) per-fact desk via embedding zero-shot vs DESK_ANCHORS (replaces the old
    #    hardcoded desk: facts inherit the desk their CLAIM is about)
    anchor_keys = list(ic.DESK_ANCHORS)
    avecs = ic.embed(args.embedding_provider, args.embedding_model,
                     [ic.DESK_ANCHORS[k] for k in anchor_keys], ic.DEFAULT_OLLAMA_URL, None)
    am = np.asarray(avecs, dtype="float32")
    am /= np.linalg.norm(am, axis=1, keepdims=True) + 1e-9
    for f, v in zip(facts, claim_vecs, strict=True):
        desks = ic.assign_desks(v, am, anchor_keys)
        f["desk"], f["desks"] = desks[0], desks

    # 3) entity canonicalization + cause -> entity-node resolution
    canon = canonicalize_entities(facts, args.embedding_provider, args.embedding_model)
    emap = canon["map"]
    for f in facts:
        f["entities"] = sorted({emap.get(e, e) for e in f["entities"]})
    resolve_causes(facts, canon, args.embedding_provider, args.embedding_model)

    # persist facts.parquet
    import pyarrow as pa, pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(facts), args.out_parquet)

    # entity stats
    from collections import Counter
    ent_count: Counter = Counter()
    desk_count: Counter = Counter()
    for f in facts:
        ent_count.update(f["entities"])
        desk_count[f["desk"]] += 1
    print(f"[build] facts={len(facts)}  unique_entities={len(ent_count)}  "
          f"causal_facts={sum(1 for f in facts if f.get('cause'))}", file=sys.stderr)
    print("[build] desks: " + ", ".join(f"{d}={c}" for d, c in desk_count.most_common()),
          file=sys.stderr)
    print("[build] top entities: " + ", ".join(f"{e}({c})" for e, c in ent_count.most_common(12)),
          file=sys.stderr)

    # 4) upsert -> Qdrant
    client = ic._client(args.qdrant_url, None)
    from qdrant_client.http import models
    dims = len(claim_vecs[0])
    if args.recreate and client.collection_exists(FACTS_COLLECTION):
        client.delete_collection(FACTS_COLLECTION)
    if not client.collection_exists(FACTS_COLLECTION):
        client.create_collection(
            collection_name=FACTS_COLLECTION, on_disk_payload=True,
            vectors_config=models.VectorParams(
                size=dims, distance=models.Distance.COSINE,
                quantization_config=models.TurboQuantization(
                    turbo=models.TurboQuantQuantizationConfig(
                        bits=models.TurboQuantBitSize("bits2"), always_ram=True))))
        for fld, sc in (("entities", models.PayloadSchemaType.KEYWORD),
                        ("cause_entities", models.PayloadSchemaType.KEYWORD),
                        ("desk", models.PayloadSchemaType.KEYWORD),
                        ("desks", models.PayloadSchemaType.KEYWORD),
                        ("predicate", models.PayloadSchemaType.KEYWORD),
                        ("published_ordinal", models.PayloadSchemaType.INTEGER),
                        ("published_epoch", models.PayloadSchemaType.INTEGER)):
            try:
                client.create_payload_index(FACTS_COLLECTION, field_name=fld, field_schema=sc)
            except Exception:
                pass
    for i in range(0, len(facts), 64):
        pts = []
        for f, v in zip(facts[i:i + 64], claim_vecs[i:i + 64], strict=True):
            pd = f.get("published_date")
            pl = dict(f)
            pl["published_ordinal"] = ic._date_ordinal(pd) if pd else 0
            pl["published_epoch"] = ic._epoch(f.get("published_utc"))
            pts.append(models.PointStruct(id=_uuid(f["fact_id"]), vector=v, payload=pl))
        client.upsert(collection_name=FACTS_COLLECTION, points=pts, wait=True)
    print(json.dumps({"facts_indexed": len(facts), "collection": FACTS_COLLECTION,
                      "facts_parquet": str(args.out_parquet)}, indent=2))
    return 0


def _uuid(s: str) -> str:
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_URL, s))


# --------------------------------------------------------------------------- #
def cmd_search(args: argparse.Namespace) -> int:
    client = ic._client(args.qdrant_url, None)
    from qdrant_client.http import models
    qv = ic.embed(args.embedding_provider, args.embedding_model, [args.query],
                  ic.DEFAULT_OLLAMA_URL, None)[0]
    # 1) seed by vector search over facts
    seeds = client.query_points(
        collection_name=FACTS_COLLECTION, query=qv, limit=args.seed_k, with_payload=True,
        search_params=models.SearchParams(
            hnsw_ef=128, quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)),
    ).points
    seed_ids = {p.id for p in seeds}
    seed_ents: set[str] = set()
    for p in seeds:
        seed_ents.update((p.payload or {}).get("entities") or [])

    print(f"\n=== SEED facts (vector search: '{args.query}') ===")
    for p in seeds:
        pl = p.payload or {}
        _print_fact(p.score, pl)

    if not args.hops:
        return 0

    # 2) EXPAND one hop: facts that share an entity with the seed set (entity index = adjacency)
    neigh = client.query_points(
        collection_name=FACTS_COLLECTION, query=qv, limit=args.expand_k * 3, with_payload=True,
        query_filter=models.Filter(must=[models.FieldCondition(
            key="entities", match=models.MatchAny(any=sorted(seed_ents)))]),
        search_params=models.SearchParams(hnsw_ef=128,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)),
    ).points
    # rank neighbours by # entities shared with seed set, then score
    ranked = []
    for p in neigh:
        if p.id in seed_ids:
            continue
        ents = set((p.payload or {}).get("entities") or [])
        shared = len(ents & seed_ents)
        if shared:
            ranked.append((shared, p.score, p))
    ranked.sort(key=lambda t: (-t[0], -t[1]))

    print(f"\n=== EXPANDED neighbourhood (1 hop via shared entities; {len(ranked)} connected) ===")
    for shared, score, p in ranked[:args.expand_k]:
        pl = p.payload or {}
        _print_fact(score, pl, prefix=f"[+{shared} shared]")

    # 3) causal chains among seed+neighbour facts
    chains = []
    for p in seeds + [t[2] for t in ranked[:args.expand_k]]:
        pl = p.payload or {}
        if pl.get("cause"):
            chains.append((pl.get("cause"), pl.get("subject"), pl.get("direction"), pl.get("claim")))
    if chains:
        print(f"\n=== CAUSAL CHAINS (cause -> subject) ===")
        for cause, subj, direc, claim in chains[:12]:
            print(f"   {str(cause)[:48]}  ->  {str(subj)[:40]} ({direc})")
    return 0


def _print_fact(score: float, pl: dict, prefix: str = "") -> None:
    mag = f" [{pl.get('magnitude')}]" if pl.get("magnitude") else ""
    cause = f"  ⟵ {pl.get('cause')}" if pl.get("cause") else ""
    print(f"  {score:.3f} {prefix} ({pl.get('predicate')}/{pl.get('direction')}) "
          f"{(pl.get('claim') or '')[:96]}{mag}")
    print(f"        {pl.get('source_name')} {pl.get('published_date')} | "
          f"entities={(pl.get('entities') or [])[:6]}{cause}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("build", "search"):
        sp = sub.add_parser(name)
        sp.add_argument("--qdrant-url", default="http://localhost:6333")
        sp.add_argument("--embedding-provider", default=ic.DEFAULT_PROVIDER)
        sp.add_argument("--embedding-model", default=ic.DEFAULT_FASTEMBED_MODEL)
    b = sub.choices["build"]
    b.add_argument("--facts-glob", default=DEFAULT_FACTS_GLOB)
    b.add_argument("--corpus", default=str(HERE / "data" / "news_corpus"))
    b.add_argument("--out-parquet", type=Path, default=DEFAULT_FACTS_PARQUET)
    b.add_argument("--recreate", action="store_true")
    b.set_defaults(func=cmd_build)
    s = sub.choices["search"]
    s.add_argument("--query", required=True)
    s.add_argument("--seed-k", type=int, default=6)
    s.add_argument("--expand-k", type=int, default=10)
    s.add_argument("--hops", type=int, default=1)
    s.set_defaults(func=cmd_search)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
