#!/usr/bin/env python3
"""market-color semantic index — fast Qdrant TurboQuant retrieval over the corpus.

Ports the proven retrieval recipe from the old news-narrative engine
(`qdrant_turboquant_rag.py`) onto the v4 full-text corpus schema:

- TurboQuant quantization (bits2): FLOAT32 vectors on disk + a 2-bit quantized
  copy always in RAM for fast ANN.
- Query-time oversampling + rescore against full-precision vectors (fast AND
  accurate), tunable hnsw_ef.
- Embeds the real article BODY (title + body_text), not metadata.
- Desk-aware payload + filter (`desks` MatchAny) so retrieval is scoped to the
  trading desk you care about -- the market-color differentiator.
- Point-in-time fields: `published_ordinal` (day range) and `published_epoch`
  (as-of-second filtering) for temporally consistent, leak-free retrieval.

Reads the corpus parquet directly with DuckDB; only `extraction_ok` rows
(real bodies) are indexed.

Run (local, zero-server embeddings via fastembed; needs a Qdrant at :6333):
  uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed \
    python index_corpus.py index --recreate --start-date 2026-06-28

  uv run --with 'qdrant-client>=1.15' --with fastembed \
    python index_corpus.py search --desk energy \
      --query "OPEC+ supply cut tightening crude" --limit 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import urllib.error
import urllib.request
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "data" / "news_corpus"
DEFAULT_COLLECTION = "market_color"
DEFAULT_BATCH = 64
DEFAULT_LIMIT = 10
DEFAULT_OVERSAMPLING = 2.0
DEFAULT_HNSW_EF = 128
DEFAULT_TURBO_BITS = "bits2"
DEFAULT_MAX_EMBED_CHARS = 8000
DEFAULT_PROVIDER = "fastembed"
DEFAULT_FASTEMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "embeddinggemma"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OPENAI_MODEL = "text-embedding-3-large"

_FASTEMBED_CACHE: dict[str, Any] = {}
_ST_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# Runtime bootstrap (self-install via uv, mirrors the old engine)
# --------------------------------------------------------------------------- #
def _reexec_with(packages: list[str]) -> None:
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError("missing runtime packages and `uv` not found to bootstrap them")
    cmd = [uv, "run"]
    for p in packages:
        cmd += ["--with", p]
    cmd += [str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execvp(uv, cmd)


def ensure_runtime(provider: str) -> None:
    needs: list[str] = []
    try:
        import qdrant_client  # noqa: F401
    except ModuleNotFoundError:
        needs.append("qdrant-client>=1.15")
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        needs.append("duckdb>=1.0")
    if provider == "fastembed":
        try:
            import fastembed  # noqa: F401
        except ModuleNotFoundError:
            needs.append("fastembed>=0.3")
    elif provider == "sentence-transformers":
        try:
            import sentence_transformers  # noqa: F401
        except ModuleNotFoundError:
            needs.append("sentence-transformers>=3.0")
    elif provider == "openai":
        try:
            import openai  # noqa: F401
        except ModuleNotFoundError:
            needs.append("openai>=1.35")
    if needs:
        # keep already-present deps in the re-exec set too
        base = ["qdrant-client>=1.15", "duckdb>=1.0"]
        _reexec_with(sorted(set(base + needs)))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _default_model(provider: str) -> str:
    return {
        "fastembed": DEFAULT_FASTEMBED_MODEL,
        "sentence-transformers": DEFAULT_ST_MODEL,
        "ollama": DEFAULT_OLLAMA_MODEL,
        "openai": DEFAULT_OPENAI_MODEL,
    }[provider]


def _date_ordinal(value: str) -> int:
    return date.fromisoformat(value[:10]).toordinal()


def _epoch(value: str | None) -> int:
    if not value:
        return 0
    try:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return 0


def _point_id(doc_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, str(doc_id)))


def _clean(value: Any) -> str:
    if not value:
        return ""
    return " ".join(str(value).split())


def _embedding_text(row: dict[str, Any], max_chars: int) -> str:
    parts: list[str] = []
    title = _clean(row.get("title"))
    if title:
        parts.append(f"Title: {title}")
    desks = row.get("desks") or []
    if desks:
        parts.append(f"Desks: {', '.join(desks)}")
    src = _clean(row.get("source_name"))
    if src:
        parts.append(f"Source: {src}")
    body = _clean(row.get("body_text"))
    if body:
        parts.append(f"Body: {body}")
    text = "\n".join(parts).strip() or _clean(row.get("url"))
    return text[:max_chars]


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    pub_date = str(row["published_date"])
    body = _clean(row.get("body_text"))
    return {
        "doc_id": row["doc_id"],
        "published_date": pub_date,
        "published_ordinal": _date_ordinal(pub_date),
        "published_epoch": _epoch(row.get("published_utc")),
        "published_utc": row.get("published_utc"),
        "source_name": row.get("source_name"),
        "source_domain": row.get("source_domain"),
        "source_access": row.get("source_access"),
        "discovery": row.get("discovery"),
        "desks": list(row.get("_desks") or row.get("desks") or []),
        "source_desks": list(row.get("desks") or []),
        "title": row.get("title"),
        "url": row.get("url"),
        "word_count": int(row.get("word_count") or 0),
        "lang": row.get("lang"),
        "snippet": body[:600],
    }


def _chunked(values: list[Any], size: int) -> Iterable[list[Any]]:
    for i in range(0, len(values), size):
        yield values[i : i + size]


# --------------------------------------------------------------------------- #
# Quality gate — drop section/listing/nav/boilerplate pages (not real articles)
# --------------------------------------------------------------------------- #
_SECTION_TITLE_RE = re.compile(r"\bpage\s+\d+\b|\|\s*page\s*\d+", re.I)
_GENERIC_TITLE_RE = re.compile(
    r"(blockchain news|independent statistics|breaking news|latest news|"
    r"news\s*-\s*page|home\s*page|all the latest)", re.I)
_SECTION_LABELS = {
    "economy", "finance", "markets", "market", "business", "news", "data",
    "analysis", "opinion", "world", "home", "sport", "sports", "technology",
    "tech", "politics", "money", "investing", "companies", "video", "podcasts",
}
_SECTION_TAIL = {
    "report", "data", "outlook", "projections", "overview", "statistics",
    "monthly", "finance", "markets", "market", "analysis", "index", "prices",
    "news", "home", "economy", "summary", "dashboard", "calendar",
    "sales", "sources", "profile", "charts", "tools", "guide",
}
_MIN_ARTICLE_WORDS = 120


def _prose_sentences(body: str) -> int:
    if not body:
        return 0
    return sum(1 for s in re.split(r"[.!?]+", body) if len(s.split()) >= 8)


def is_real_article(row: dict[str, Any]) -> bool:
    """Heuristic: is this a real prose article vs a section/listing/nav page?"""
    title = (row.get("title") or "").strip()
    body = row.get("body_text") or ""
    wc = int(row.get("word_count") or 0)
    if wc < _MIN_ARTICLE_WORDS:
        return False
    # Source-specific landing/section pages: EIA articles live under
    # /todayinenergy/; OilPrice articles end in .html (sections don't).
    url = (row.get("url") or "").lower()
    if "eia.gov" in url and "/todayinenergy/" not in url:
        return False
    if "oilprice.com" in url and not url.split("?")[0].rstrip("/").endswith(".html"):
        return False
    if _SECTION_TITLE_RE.search(title) or _GENERIC_TITLE_RE.search(title):
        return False
    low = title.lower().strip(" |-—·")
    if low in _SECTION_LABELS:
        return False
    if len(title.split()) <= 2 and (title.isupper() or low in _SECTION_LABELS):
        return False
    # Short section/report/landing titles (e.g. EIA "Natural Gas Data",
    # "Refinery Capacity Report", "Analysis & Projections") -- strip any
    # " - Source" / " | Source" suffix, then check the core title's tail word.
    core = re.split(r"\s+[|\-–—]\s+", title)[0].strip()
    core_words = core.split()
    if core_words and len(core_words) <= 5 and core_words[-1].lower().strip(":") in _SECTION_TAIL:
        return False
    # The strongest signal: nav/listing pages have little real prose.
    if _prose_sentences(body) < 3:
        return False
    return True


# --------------------------------------------------------------------------- #
# Per-article desk classification — embedding zero-shot vs desk "anchors"
# --------------------------------------------------------------------------- #
DESK_ANCHORS = {
    "rates": "central bank interest rate policy, Federal Reserve and ECB rate decisions, "
             "Treasury and government bond yields, the yield curve, rate hikes and monetary tightening",
    "fx": "foreign exchange currency markets, the US dollar index DXY, euro, yen, sterling and "
          "currency pairs, FX trading and central bank intervention",
    "energy": "crude oil Brent and WTI prices, OPEC supply and production, natural gas and LNG, "
              "refineries and oil inventories, energy markets",
    "metals": "gold and silver precious metals, copper and industrial base metals, mining and "
              "smelting, metals prices and demand",
    "crypto": "Bitcoin and Ethereum cryptocurrency, digital assets, stablecoins, crypto ETFs and exchanges",
    "equities": "stock markets, S&P 500 and Nasdaq indices, company earnings and single stocks, "
                "equity rally and selloff",
    "geopolitics": "war conflict and military strikes, sanctions, elections, coups and political "
                   "instability, geopolitical risk",
    "macro": "macroeconomic data, inflation GDP and employment, recession risk, the global economy "
             "and trade, central bank policy",
    "asia": "China and Japan economies and markets, the yuan and yen, Chinese property and exports, "
            "Asian equities and Asia session",
}


def assign_desks(vec: list[float], anchor_mat: Any, anchor_keys: list[str],
                 floor: float = 0.28, top_k: int = 3, margin: float = 0.08) -> list[str]:
    import numpy as np

    v = np.asarray(vec, dtype="float32")
    v = v / (np.linalg.norm(v) + 1e-9)
    sims = anchor_mat @ v  # anchor_mat rows are pre-normalized
    order = np.argsort(-sims)
    top = float(sims[order[0]])
    desks = [anchor_keys[i] for i in order[:top_k]
             if float(sims[i]) >= max(floor, top - margin)]
    return desks or [anchor_keys[int(order[0])]]


# --------------------------------------------------------------------------- #
# Corpus loading (DuckDB over parquet)
# --------------------------------------------------------------------------- #
def load_rows(
    corpus: Path,
    start_date: str | None,
    end_date: str | None,
    desk: str | None,
    source: str | None,
    only_full_body: bool,
    limit: int | None,
) -> list[dict[str, Any]]:
    import duckdb

    glob = str(corpus / "dt=*" / "part-*.parquet")
    where = ["1=1"]
    params: list[Any] = []
    if only_full_body:
        where.append("extraction_ok = TRUE")
    if start_date:
        where.append("published_date >= ?")
        params.append(start_date)
    if end_date:
        where.append("published_date <= ?")
        params.append(end_date)
    if desk:
        where.append("list_contains(desks, ?)")
        params.append(desk)
    if source:
        where.append("source_name = ?")
        params.append(source)
    sql = f"""
        SELECT doc_id, CAST(published_date AS VARCHAR) AS published_date,
               published_utc, source_name, source_domain, source_access,
               discovery, desks, title, body_text, url, word_count, lang
        FROM read_parquet('{glob}')
        WHERE {" AND ".join(where)}
        ORDER BY published_date ASC, doc_id ASC
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


# --------------------------------------------------------------------------- #
# Embeddings
# --------------------------------------------------------------------------- #
def embed(provider: str, model: str, texts: list[str], ollama_url: str,
          openai_api_key: str | None) -> list[list[float]]:
    if provider == "fastembed":
        from fastembed import TextEmbedding

        inst = _FASTEMBED_CACHE.get(model)
        if inst is None:
            inst = TextEmbedding(model_name=model)
            _FASTEMBED_CACHE[model] = inst
        return [list(map(float, v)) for v in inst.embed(texts)]
    if provider == "sentence-transformers":
        from sentence_transformers import SentenceTransformer

        inst = _ST_CACHE.get(model)
        if inst is None:
            inst = SentenceTransformer(model)
            _ST_CACHE[model] = inst
        return [list(map(float, v)) for v in inst.encode(texts, normalize_embeddings=True).tolist()]
    if provider == "openai":
        from openai import OpenAI

        resp = OpenAI(api_key=openai_api_key).embeddings.create(model=model, input=texts)
        return [list(d.embedding) for d in resp.data]
    if provider == "ollama":
        req = urllib.request.Request(
            f"{ollama_url.rstrip('/')}/api/embed",
            data=json.dumps({"model": model, "input": texts}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            payload = json.loads(r.read().decode())
        return [list(map(float, v)) for v in payload["embeddings"]]
    raise ValueError(f"unsupported provider: {provider}")


# --------------------------------------------------------------------------- #
# Qdrant
# --------------------------------------------------------------------------- #
def _client(url: str, api_key: str | None) -> Any:
    from qdrant_client import QdrantClient

    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith("local:"):
        return QdrantClient(path=url.removeprefix("local:"))
    return QdrantClient(url=url, api_key=api_key)


def ensure_collection(client: Any, name: str, dims: int, turbo_bits: str,
                      recreate: bool, vector_on_disk: bool) -> None:
    from qdrant_client.http import models

    if recreate and client.collection_exists(name):
        client.delete_collection(name)
    if client.collection_exists(name):
        return
    client.create_collection(
        collection_name=name,
        on_disk_payload=True,
        vectors_config=models.VectorParams(
            size=dims,
            distance=models.Distance.COSINE,
            on_disk=vector_on_disk,
            datatype=models.Datatype.FLOAT32,
            hnsw_config=models.HnswConfigDiff(on_disk=vector_on_disk, inline_storage=True),
            quantization_config=models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=models.TurboQuantBitSize(turbo_bits),
                    always_ram=True,
                )
            ),
        ),
    )
    # payload indexes for fast filtering
    for field, schema in (
        ("published_ordinal", models.PayloadSchemaType.INTEGER),
        ("published_epoch", models.PayloadSchemaType.INTEGER),
        ("desks", models.PayloadSchemaType.KEYWORD),
        ("source_name", models.PayloadSchemaType.KEYWORD),
        ("source_access", models.PayloadSchemaType.KEYWORD),
    ):
        try:
            client.create_payload_index(name, field_name=field, field_schema=schema)
        except Exception:
            pass


def _filter(start_date: str | None, end_date: str | None, desks: list[str],
            sources: list[str], access: list[str], as_of_epoch: int | None) -> Any:
    from qdrant_client.http import models

    must: list[Any] = []
    if start_date or end_date:
        must.append(models.FieldCondition(
            key="published_ordinal",
            range=models.Range(
                gte=_date_ordinal(start_date) if start_date else None,
                lte=_date_ordinal(end_date) if end_date else None,
            ),
        ))
    if as_of_epoch is not None:
        must.append(models.FieldCondition(
            key="published_epoch", range=models.Range(lte=as_of_epoch)))
    if desks:
        must.append(models.FieldCondition(key="desks", match=models.MatchAny(any=desks)))
    if sources:
        must.append(models.FieldCondition(key="source_name", match=models.MatchAny(any=sources)))
    if access:
        must.append(models.FieldCondition(key="source_access", match=models.MatchAny(any=access)))
    return models.Filter(must=must) if must else None


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def run_index(args: argparse.Namespace) -> int:
    rows = load_rows(Path(args.corpus), args.start_date, args.end_date,
                     args.desk, args.source, not args.include_headline_only, args.row_limit)
    if not rows:
        raise RuntimeError("no corpus rows matched (is the crawl done / filters too tight?)")

    # #1 quality gate — drop section/listing/nav/boilerplate pages.
    if not args.no_quality_gate:
        before = len(rows)
        rows = [r for r in rows if is_real_article(r)]
        print(f"[quality-gate] kept {len(rows)}/{before} real articles "
              f"(dropped {before - len(rows)} section/listing/stub pages)", file=sys.stderr)
        if not rows:
            raise RuntimeError("quality gate removed everything")

    client = _client(args.qdrant_url, args.qdrant_api_key)
    from qdrant_client.http import models

    # #2 desk anchors — embed once for per-article zero-shot desk assignment.
    anchor_keys: list[str] = []
    anchor_mat = None
    if not args.no_content_desks:
        import numpy as np
        anchor_keys = list(DESK_ANCHORS)
        avecs = embed(args.embedding_provider, args.embedding_model,
                      [DESK_ANCHORS[k] for k in anchor_keys], args.ollama_url, args.openai_api_key)
        am = np.asarray(avecs, dtype="float32")
        anchor_mat = am / (np.linalg.norm(am, axis=1, keepdims=True) + 1e-9)

    indexed = 0
    dims: int | None = None
    for batch in _chunked(rows, args.batch_size):
        vectors = embed(args.embedding_provider, args.embedding_model,
                        [_embedding_text(r, args.max_embed_chars) for r in batch],
                        args.ollama_url, args.openai_api_key)
        if anchor_mat is not None:
            for r, v in zip(batch, vectors, strict=True):
                r["_desks"] = assign_desks(v, anchor_mat, anchor_keys)
        if dims is None:
            dims = len(vectors[0])
            ensure_collection(client, args.collection, dims, args.turbo_bits,
                              args.recreate, args.vector_on_disk)
        points = [
            models.PointStruct(id=_point_id(r["doc_id"]), vector=v, payload=_payload(r))
            for r, v in zip(batch, vectors, strict=True)
        ]
        client.upsert(collection_name=args.collection, points=points, wait=True)
        indexed += len(points)
        print(f"  indexed {indexed}/{len(rows)}", file=sys.stderr)

    print(json.dumps({
        "collection": args.collection, "indexed_points": indexed, "dimensions": dims,
        "embedding_provider": args.embedding_provider, "embedding_model": args.embedding_model,
        "turbo_bits": args.turbo_bits, "corpus": str(args.corpus),
        "start_date": args.start_date, "end_date": args.end_date, "desk": args.desk,
    }, indent=2))
    return 0


def run_search(args: argparse.Namespace) -> int:
    client = _client(args.qdrant_url, args.qdrant_api_key)
    qvec = embed(args.embedding_provider, args.embedding_model, [args.query],
                 args.ollama_url, args.openai_api_key)[0]
    from qdrant_client.http import models

    as_of = _epoch(args.as_of) if args.as_of else None
    resp = client.query_points(
        collection_name=args.collection,
        query=qvec,
        query_filter=_filter(args.start_date, args.end_date, args.desk,
                             args.source, args.access, as_of),
        limit=args.limit,
        with_payload=True,
        with_vectors=False,
        search_params=models.SearchParams(
            hnsw_ef=args.hnsw_ef,
            quantization=models.QuantizationSearchParams(
                rescore=True, oversampling=args.oversampling),
        ),
    )
    hits = [{
        "score": p.score,
        "published_date": (p.payload or {}).get("published_date"),
        "source_name": (p.payload or {}).get("source_name"),
        "desks": (p.payload or {}).get("desks"),
        "title": (p.payload or {}).get("title"),
        "url": (p.payload or {}).get("url"),
        "snippet": (p.payload or {}).get("snippet"),
    } for p in resp.points if p.score >= args.min_score]
    out = {"query": args.query, "desk_filter": args.desk, "as_of": args.as_of,
           "min_score": args.min_score}
    if not hits:
        out["result"] = "no_strong_match"  # #3 abstention
    out["hits"] = hits
    print(json.dumps(out, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def common_embed(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--collection", default=DEFAULT_COLLECTION)
        sp.add_argument("--qdrant-url", default="http://localhost:6333")
        sp.add_argument("--qdrant-api-key")
        sp.add_argument("--openai-api-key")
        sp.add_argument("--embedding-provider",
                        choices=["fastembed", "sentence-transformers", "ollama", "openai"],
                        default=DEFAULT_PROVIDER)
        sp.add_argument("--embedding-model")
        sp.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)

    ip = sub.add_parser("index", help="Embed corpus bodies and upsert into Qdrant.")
    common_embed(ip)
    ip.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    ip.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    ip.add_argument("--start-date")
    ip.add_argument("--end-date")
    ip.add_argument("--desk", help="only index this desk")
    ip.add_argument("--source", help="only index this source name")
    ip.add_argument("--row-limit", type=int)
    ip.add_argument("--max-embed-chars", type=int, default=DEFAULT_MAX_EMBED_CHARS)
    ip.add_argument("--turbo-bits", choices=["bits1", "bits1_5", "bits2", "bits4"],
                    default=DEFAULT_TURBO_BITS)
    ip.add_argument("--include-headline-only", action="store_true",
                    help="also index rows without a full body (default: full-body only)")
    ip.add_argument("--no-quality-gate", action="store_true",
                    help="skip the section/listing/boilerplate filter")
    ip.add_argument("--no-content-desks", action="store_true",
                    help="keep source desks instead of per-article embedding-derived desks")
    ip.add_argument("--recreate", action="store_true")
    ip.add_argument("--vector-on-disk", action="store_true")
    ip.set_defaults(func=run_index)

    sp = sub.add_parser("search", help="Desk-scoped, point-in-time semantic search.")
    common_embed(sp)
    sp.add_argument("--query", required=True)
    sp.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    sp.add_argument("--oversampling", type=float, default=DEFAULT_OVERSAMPLING)
    sp.add_argument("--hnsw-ef", type=int, default=DEFAULT_HNSW_EF)
    sp.add_argument("--start-date")
    sp.add_argument("--end-date")
    sp.add_argument("--as-of", help="ISO time; only return docs published at/before this (no look-ahead)")
    sp.add_argument("--desk", action="append", default=[], help="restrict to desk(s)")
    sp.add_argument("--source", action="append", default=[])
    sp.add_argument("--access", action="append", default=[], help="e.g. open / headline")
    sp.add_argument("--min-score", type=float, default=0.0,
                    help="abstain: drop hits below this score; empty result => no_strong_match")
    sp.set_defaults(func=run_search)
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.embedding_model = args.embedding_model or _default_model(args.embedding_provider)
    ensure_runtime(args.embedding_provider)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
