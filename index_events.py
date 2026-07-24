# /// script
# requires-python = ">=3.10"
# dependencies = ["qdrant-client>=1.15", "fastembed>=0.3"]
# ///
"""Build the ENTITY-ENRICHED event docset in Qdrant (the hybrid-retrieval collection).

Unlike `market_color` (one point per whole-article body, doc-level payload only), this
collection is CHUNK-GRANULAR and carries the entity/sector/mechanism payload the
decomposition router needs to SCOPE a search:

  idiosyncratic-dominant -> filter `tickers` = the firm
  sector-dominant        -> filter `tickers` in the beta-correlated peer subset
  macro-dominant         -> filter `is_macro` = true (+ point-in-time `published_ordinal`)

Every point BACK-LINKS to its source: `chunk_id` -> `chunk.jsonl` text, `doc_id` ->
`document.jsonl` metadata. The enrichment comes from the causal graph: a chunk is
indexed iff >=1 causal edge was extracted from it, and all edges on that chunk are
aggregated into its payload (entities, mechanisms, directions, macro flag, a sample quote).

Vector space matches `market_color` (fastembed all-MiniLM-L6-v2, 384-dim cosine) so the
two collections are directly comparable.

Run (serverless smoke test, no Qdrant server needed):
  uv run index_events.py --url local:/tmp/qdrant_events --limit-chunks 300 --recreate
Run (against the compose Qdrant):
  docker compose up -d qdrant
  uv run index_events.py --url http://localhost:6333 --recreate
"""
import argparse
import datetime as dt
import json
import uuid
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_GRAPH = HERE / "data" / "eg_runs" / "eg100k_graph"
DEFAULT_COLLECTION = "market_color_events"
DEFAULT_URL = "http://localhost:6333"
FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DIMS = 384
MAX_EMBED_CHARS = 8000
BATCH = 128

# Cause kinds that make a chunk a MACRO catalyst (kept in sync with residual_subsector.py).
MACRO_KINDS = {
    "equity_index", "sovereign", "central_bank", "commodity", "currency",
    "rate_or_bond", "economic_indicator", "sector", "market", "exchange",
}


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(chunk_id)))


def _ordinal(date_str: str) -> int | None:
    try:
        return dt.date.fromisoformat(date_str[:10]).toordinal()
    except (ValueError, TypeError):
        return None


def load_reference(graph: Path) -> tuple[dict, dict, dict]:
    """entity_id -> ticker (securities only); ticker -> sector; doc_id -> doc metadata."""
    sym: dict[str, str] = {}
    for line in open(graph / "entity_symbol.jsonl"):
        j = json.loads(line)
        if j.get("kind") == "security" and j.get("symbol"):
            sym[j["entity_id"]] = j["symbol"]

    sec_of: dict[str, str] = {}
    attr = json.load(open(graph / "attribution.json"))
    for f in attr["firms"]:
        sec_of[f["t"]] = f["sec"]

    docmeta: dict[str, dict] = {}
    for line in open(graph / "lake" / "document.jsonl"):
        j = json.loads(line)
        pub = (j.get("published_at") or "")
        docmeta[j["doc_id"]] = {
            "published_date": pub[:10],
            "published_ordinal": _ordinal(pub),
            "published_epoch": j.get("epoch"),
            "month": pub[:7],
            "headline": j.get("headline"),
            "source": j.get("source"),
            "url": j.get("url"),
        }
    return sym, sec_of, docmeta


def aggregate_edges(graph: Path, sym: dict, sec_of: dict) -> dict[str, dict]:
    """chunk_id -> aggregated enrichment across every causal edge extracted from that chunk."""
    agg: dict[str, dict] = {}
    for line in open(graph / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(line)
        cid = j.get("chunk_id")
        if not cid:
            continue
        a = agg.get(cid)
        if a is None:
            a = agg[cid] = {
                "doc_id": j.get("doc_id"),
                "effect_entities": set(), "cause_entities": set(),
                "tickers": set(), "sectors": set(),
                "mechanisms": set(), "effect_dirs": set(),
                "cause_kinds": set(), "effect_kinds": set(),
                "is_macro": False, "n_edges": 0, "quote": None,
            }
        a["n_edges"] += 1
        for role, kindfld in (("cause_entity", "cause_kind"), ("effect_entity", "effect_kind")):
            ent = j.get(role)
            if not ent:
                continue
            (a["cause_entities"] if role == "cause_entity" else a["effect_entities"]).add(ent)
            tick = sym.get(ent)
            if tick:
                a["tickers"].add(tick)
                sec = sec_of.get(tick)
                if sec:
                    a["sectors"].add(sec)
        for fld, key in (("mechanism", "mechanisms"), ("effect_dir", "effect_dirs"),
                         ("cause_kind", "cause_kinds"), ("effect_kind", "effect_kinds")):
            v = j.get(fld)
            if v:
                a[key].add(v)
        if j.get("cause_kind") in MACRO_KINDS:
            a["is_macro"] = True
        if not a["quote"] and j.get("quote"):
            a["quote"] = j["quote"][:280]
    return agg


def build_payload(cid: str, a: dict, docmeta: dict) -> dict:
    dm = docmeta.get(a["doc_id"], {})
    seq = None
    if ":" in cid:
        tail = cid.rsplit(":", 1)[-1]
        seq = int(tail) if tail.isdigit() else None
    return {
        # back-link to source ------------------------------------------------
        "chunk_id": cid,
        "doc_id": a["doc_id"],
        "seq": seq,
        # point-in-time ------------------------------------------------------
        "published_date": dm.get("published_date"),
        "published_ordinal": dm.get("published_ordinal"),
        "published_epoch": dm.get("published_epoch"),
        "month": dm.get("month"),
        # entity scope (the missing filter axis) -----------------------------
        "tickers": sorted(a["tickers"]),
        "sectors": sorted(a["sectors"]),
        "effect_entities": sorted(a["effect_entities"]),
        "cause_entities": sorted(a["cause_entities"]),
        # channel routing ----------------------------------------------------
        "is_macro": a["is_macro"],
        "mechanisms": sorted(a["mechanisms"]),
        "effect_dirs": sorted(a["effect_dirs"]),
        "cause_kinds": sorted(a["cause_kinds"]),
        "effect_kinds": sorted(a["effect_kinds"]),
        # provenance / display ----------------------------------------------
        "quote": a["quote"],
        "headline": dm.get("headline"),
        "source": dm.get("source"),
        "url": dm.get("url"),
        "n_edges": a["n_edges"],
    }


def iter_event_chunks(graph: Path, wanted: set[str], limit: int | None):
    """Stream chunk.jsonl, yielding (chunk_id, text) for chunks that carry a causal edge."""
    n = 0
    for line in open(graph / "lake" / "chunk.jsonl"):
        j = json.loads(line)
        cid = j.get("chunk_id")
        if cid not in wanted:
            continue
        text = (j.get("text") or "")[:MAX_EMBED_CHARS]
        if not text.strip():
            continue
        yield cid, text
        n += 1
        if limit and n >= limit:
            return


def _client(url: str, api_key: str | None):
    from qdrant_client import QdrantClient
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith("local:"):
        return QdrantClient(path=url.removeprefix("local:"))
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client, name: str, recreate: bool) -> None:
    from qdrant_client.http import models
    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if not client.collection_exists(name):
        client.create_collection(
            collection_name=name,
            on_disk_payload=True,
            vectors_config=models.VectorParams(size=DIMS, distance=models.Distance.COSINE),
        )
    # Native filter indexes for the routing axes.
    for field in ("tickers", "sectors", "effect_entities", "cause_entities", "mechanisms", "month"):
        client.create_payload_index(name, field, field_schema=models.PayloadSchemaType.KEYWORD)
    client.create_payload_index(name, "is_macro", field_schema=models.PayloadSchemaType.BOOL)
    client.create_payload_index(name, "published_ordinal", field_schema=models.PayloadSchemaType.INTEGER)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    ap.add_argument("--url", default=DEFAULT_URL, help="Qdrant URL, ':memory:', or 'local:<path>'")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--collection", default=DEFAULT_COLLECTION)
    ap.add_argument("--recreate", action="store_true")
    ap.add_argument("--limit-chunks", type=int, default=None, help="cap indexed chunks (smoke test)")
    args = ap.parse_args()

    from fastembed import TextEmbedding
    from qdrant_client.http import models

    print(f"[events] loading reference maps from {args.graph}")
    sym, sec_of, docmeta = load_reference(args.graph)
    print(f"[events] {len(sym)} security symbols, {len(sec_of)} sectors, {len(docmeta)} docs")

    print("[events] aggregating causal edges per chunk ...")
    agg = aggregate_edges(args.graph, sym, sec_of)
    print(f"[events] {len(agg)} event-bearing chunks")

    client = _client(args.url, args.api_key)
    ensure_collection(client, args.collection, args.recreate)
    embedder = TextEmbedding(model_name=FASTEMBED_MODEL)

    wanted = set(agg)
    indexed = 0
    batch_cids: list[str] = []
    batch_text: list[str] = []

    def flush() -> None:
        nonlocal indexed
        if not batch_cids:
            return
        vectors = [list(map(float, v)) for v in embedder.embed(batch_text)]
        points = [
            models.PointStruct(id=_point_id(cid), vector=vec, payload=build_payload(cid, agg[cid], docmeta))
            for cid, vec in zip(batch_cids, vectors)
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        indexed += len(points)
        print(f"[events] upserted {indexed} points")
        batch_cids.clear()
        batch_text.clear()

    for cid, text in iter_event_chunks(args.graph, wanted, args.limit_chunks):
        batch_cids.append(cid)
        batch_text.append(text)
        if len(batch_cids) >= BATCH:
            flush()
    flush()

    info = client.get_collection(args.collection)
    print(f"[events] done: collection '{args.collection}' now holds {info.points_count} points")


if __name__ == "__main__":
    main()
