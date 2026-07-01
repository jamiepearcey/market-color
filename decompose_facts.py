#!/usr/bin/env python3
"""market-color fact decomposition — turn crawled articles into atomic facts.

The chat UI does not retrieve whole articles; it retrieves *structured facts*
decomposed from the corpus. This script is that decomposition step. For each
real article in `data/news_corpus/` it asks Claude to extract a short list of
atomic, self-contained market facts — each one desk-tagged, direction-tagged,
and carrying the entities and any numeric magnitude it mentions — then writes
them to `data/facts/dt=YYYY-MM-DD/part-000.parquet` with full provenance back
to the source article (doc_id, url, source, title, published_at).

`index_facts.py` then embeds each fact's claim and upserts it into the
`market_facts` Qdrant collection, and `mcp/qdrant_facts_server.py` serves those
facts to the LLM via MCP.

Provenance is the whole point: a fact is only useful here if you can click
through to the article it came from. We never invent facts — the model is
instructed to extract only what the article states, and we keep the source
ids on every row.

Run (self-bootstraps deps via uv; needs ANTHROPIC_API_KEY in the env):
  ANTHROPIC_API_KEY=sk-ant-... \
  uv run --with 'anthropic>=0.49' --with 'duckdb>=1.0' --with 'pyarrow>=15' \
    python decompose_facts.py --start-date 2026-06-28

  # cheaper bulk pass over thousands of articles:
  ANTHROPIC_API_KEY=... uv run ... python decompose_facts.py --model claude-haiku-4-5
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import date
from pathlib import Path
from typing import Any

# Reuse the corpus loader + quality gate + desk taxonomy from the indexer so
# "what counts as a real article" and "what desks exist" stay in one place.
from index_corpus import DESK_ANCHORS, _clean, is_real_article, load_rows

HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = HERE / "data" / "news_corpus"
DEFAULT_FACTS = HERE / "data" / "facts"
DEFAULT_MODEL = "claude-opus-4-8"  # the skill default; override with --model for bulk runs
DEFAULT_MAX_FACTS = 8
DEFAULT_BODY_CHARS = 6000

DEFAULT_DESKS = list(DESK_ANCHORS) + ["other"]


def _load_desk_keys(path: str | None) -> list[str]:
    """Desk keys for the extraction enum. From --desks-file (config materialized
    from Postgres) if given, else the built-in taxonomy. 'other' is always allowed."""
    if not path:
        return DEFAULT_DESKS
    data = json.loads(Path(path).read_text())
    if isinstance(data, dict):
        keys = list(data.keys())
    elif isinstance(data, list):
        keys = [d["key"] if isinstance(d, dict) else str(d) for d in data]
    else:
        keys = []
    keys = [k for k in keys if k]
    if "other" not in keys:
        keys.append("other")
    return keys or DEFAULT_DESKS


def build_facts_schema(desks: list[str]) -> dict[str, Any]:
    """Structured-output schema — Claude is constrained to return exactly this,
    so the parse never has to be defensive about shape. Every field is required
    (structured outputs has no "optional"); empty string / [] means "not stated".
    The desk enum is driven by the configured desk taxonomy."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "One atomic, self-contained market fact stated by the "
                            "article, in a single sentence. No pronouns or 'the company' — name "
                            "the subject so the fact stands alone out of context.",
                        },
                        "entities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Companies, tickers, commodities, institutions, people, "
                            "or countries the fact is about.",
                        },
                        "desk": {
                            "type": "string",
                            "enum": desks,
                            "description": "Which trading desk this fact is relevant to.",
                        },
                        "direction": {
                            "type": "string",
                            "enum": ["bullish", "bearish", "neutral"],
                            "description": "Market color: directional implication of the fact for "
                            "the named subject/desk, or 'neutral' if purely informational.",
                        },
                        "metric": {
                            "type": "string",
                            "description": "The key number or magnitude in the fact (e.g. '2.2m "
                            "bpd', '+3.4%', '$78.50', '25bps'), or '' if the fact has none.",
                        },
                    },
                    "required": ["claim", "entities", "desk", "direction", "metric"],
                },
            }
        },
        "required": ["facts"],
    }

SYSTEM = (
    "You are a markets desk analyst. You decompose a news article into a small set of "
    "atomic, self-contained facts a trader could act on or file. Extract ONLY what the "
    "article actually states — never infer, never add outside knowledge, never speculate. "
    "Each fact must be a single declarative sentence that stands on its own without the "
    "article for context. Prefer facts with a concrete subject, a verb, and where present a "
    "number. Skip opinion, boilerplate, and navigation text. If the article contains no "
    "substantive market facts, return an empty list."
)


def _bootstrap() -> None:
    """Self-install the runtime deps via uv, mirroring index_corpus.py."""
    needs: list[str] = []
    try:
        import anthropic  # noqa: F401
    except ModuleNotFoundError:
        needs.append("anthropic>=0.49")
    try:
        import duckdb  # noqa: F401
    except ModuleNotFoundError:
        needs.append("duckdb>=1.0")
    try:
        import pyarrow  # noqa: F401
    except ModuleNotFoundError:
        needs.append("pyarrow>=15")
    if not needs:
        return
    uv = shutil.which("uv")
    if uv is None:
        raise RuntimeError(
            "missing runtime packages and `uv` not found to bootstrap them; "
            f"pip install {' '.join(needs)}"
        )
    cmd = [uv, "run"]
    for p in sorted({"anthropic>=0.49", "duckdb>=1.0", "pyarrow>=15", *needs}):
        cmd += ["--with", p]
    cmd += [str(Path(__file__).resolve()), *sys.argv[1:]]
    os.execvp(uv, cmd)


def _already_done(facts_dir: Path) -> set[str]:
    """doc_ids already decomposed, so re-runs are incremental (resumable)."""
    glob = facts_dir / "dt=*" / "part-*.parquet"
    matches = list(facts_dir.glob("dt=*/part-*.parquet")) if facts_dir.exists() else []
    if not matches:
        return set()
    import duckdb

    con = duckdb.connect()
    try:
        rows = con.execute(
            f"SELECT DISTINCT doc_id FROM read_parquet('{glob}')"
        ).fetchall()
        return {r[0] for r in rows}
    except Exception:
        return set()
    finally:
        con.close()


def _extract(client: Any, model: str, row: dict[str, Any], body_chars: int,
             schema: dict[str, Any]) -> list[dict[str, Any]]:
    title = _clean(row.get("title"))
    source = _clean(row.get("source_name"))
    body = _clean(row.get("body_text"))[:body_chars]
    user = (
        f"Source: {source}\nTitle: {title}\n\nArticle:\n{body}\n\n"
        f"Extract up to {DEFAULT_MAX_FACTS} atomic market facts."
    )
    resp = client.messages.create(
        model=model,
        max_tokens=4096,
        system=SYSTEM,
        messages=[{"role": "user", "content": user}],
        output_config={"format": {"type": "json_schema", "schema": schema}},
    )
    # Structured outputs may still refuse; guard before reading content.
    if getattr(resp, "stop_reason", None) == "refusal":
        return []
    text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
    if not text.strip():
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return []
    facts = parsed.get("facts") if isinstance(parsed, dict) else None
    return facts if isinstance(facts, list) else []


def _fact_rows(row: dict[str, Any], facts: list[dict[str, Any]],
               valid_desks: list[str]) -> list[dict[str, Any]]:
    pub = str(row.get("published_date"))
    out: list[dict[str, Any]] = []
    for i, f in enumerate(facts):
        claim = _clean(f.get("claim"))
        if not claim:
            continue
        desk = f.get("desk") if f.get("desk") in valid_desks else "other"
        direction = f.get("direction") if f.get("direction") in ("bullish", "bearish", "neutral") else "neutral"
        out.append(
            {
                "fact_id": f"{row['doc_id']}:{i}",
                "doc_id": row["doc_id"],
                "claim": claim,
                "entities": [str(e) for e in (f.get("entities") or []) if str(e).strip()],
                "desk": desk,
                "direction": direction,
                "metric": _clean(f.get("metric")),
                "source_name": row.get("source_name"),
                "source_domain": row.get("source_domain"),
                "source_access": row.get("source_access"),
                "url": row.get("url"),
                "title": row.get("title"),
                "published_date": pub,
                "published_utc": row.get("published_utc"),
                "lang": row.get("lang"),
            }
        )
    return out


def _write_partition(facts_dir: Path, pub_date: str, rows: list[dict[str, Any]]) -> None:
    """Append facts to dt=<date>/part-000.parquet, merged with any existing rows
    (so an incremental run that adds facts for a date does not clobber prior ones)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    part_dir = facts_dir / f"dt={pub_date}"
    part_dir.mkdir(parents=True, exist_ok=True)
    target = part_dir / "part-000.parquet"

    existing: list[dict[str, Any]] = []
    if target.exists():
        existing = pq.read_table(target).to_pylist()
    seen = {r["fact_id"] for r in existing}
    merged = existing + [r for r in rows if r["fact_id"] not in seen]
    pq.write_table(pa.Table.from_pylist(merged), target)


def run(args: argparse.Namespace) -> int:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    client = anthropic.Anthropic(api_key=api_key)

    facts_dir = Path(args.facts_dir)
    valid_desks = _load_desk_keys(args.desks_file)
    schema = build_facts_schema(valid_desks)

    if args.input_json:
        # Ad-hoc ingest: rows come from a JSON list (manually uploaded docs),
        # not the crawled corpus. Each row mirrors the corpus schema fields.
        raw = json.loads(Path(args.input_json).read_text())
        rows = raw["rows"] if isinstance(raw, dict) else raw
    else:
        rows = load_rows(
            Path(args.corpus),
            args.start_date,
            args.end_date,
            args.desk,
            args.source,
            only_full_body=True,
            limit=args.row_limit,
        )
        rows = [r for r in rows if is_real_article(r)]
    done = set() if (args.overwrite or args.input_json) else _already_done(facts_dir)
    todo = [r for r in rows if r["doc_id"] not in done]
    print(
        f"[decompose] {len(todo)} articles to process "
        f"({len(rows) - len(todo)} already decomposed, {len(rows)} real articles total)",
        file=sys.stderr,
    )

    by_date: dict[str, list[dict[str, Any]]] = {}
    total_facts = 0
    for n, row in enumerate(todo, 1):
        try:
            facts = _extract(client, args.model, row, args.body_chars, schema)
        except Exception as exc:  # one bad article must not sink the batch
            print(f"  ! {row['doc_id']}: {exc}", file=sys.stderr)
            continue
        frows = _fact_rows(row, facts, valid_desks)
        if frows:
            by_date.setdefault(str(row["published_date"]), []).extend(frows)
            total_facts += len(frows)
        if n % args.flush_every == 0:
            for d, rs in by_date.items():
                _write_partition(facts_dir, d, rs)
            by_date.clear()
            print(f"  {n}/{len(todo)} articles, {total_facts} facts so far", file=sys.stderr)

    for d, rs in by_date.items():
        _write_partition(facts_dir, d, rs)

    print(
        json.dumps(
            {
                "articles_processed": len(todo),
                "facts_written": total_facts,
                "facts_dir": str(facts_dir),
                "model": args.model,
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    p.add_argument("--facts-dir", default=str(DEFAULT_FACTS))
    p.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id for extraction")
    p.add_argument("--start-date")
    p.add_argument("--end-date")
    p.add_argument("--desk", help="only decompose this desk")
    p.add_argument("--source", help="only decompose this source name")
    p.add_argument("--row-limit", type=int)
    p.add_argument("--body-chars", type=int, default=DEFAULT_BODY_CHARS)
    p.add_argument("--flush-every", type=int, default=25, help="write parquet every N articles")
    p.add_argument("--overwrite", action="store_true", help="re-decompose already-done doc_ids")
    p.add_argument("--desks-file", help="JSON of desk keys/anchors (overrides the built-in taxonomy)")
    p.add_argument("--input-json", help="JSON list of article rows to decompose (ad-hoc ingest, "
                                        "bypasses the crawled corpus)")
    return p


def main() -> int:
    _bootstrap()
    args = build_parser().parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
