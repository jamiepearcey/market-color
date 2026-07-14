#!/usr/bin/env python3
"""Export quality-gated corpus docs into Codex fact-extraction batches.

Writes facts_work/batches/<prefix>_batch_NNN.jsonl (one doc per line:
{doc_id, source_name, published_date, url, title, body_text}), skipping any
doc_id that already appears in an existing *_facts_*.jsonl or in
facts_work/facts.parquet — so re-runs only queue NEW documents (the
incremental path used by run_pipeline.sh).

Run:
  uv run --with 'duckdb>=1.0' python make_fact_batches.py --prefix all
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCHES = HERE / "facts_work" / "batches"

from index_corpus import is_real_article, load_rows  # noqa: E402


def already_extracted() -> set[str]:
    done: set[str] = set()
    for fp in glob.glob(str(BATCHES / "*_facts_*.jsonl")):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    parquet = HERE / "facts_work" / "facts.parquet"
    if parquet.exists():
        import duckdb
        for (doc_id,) in duckdb.connect().execute(
                f"select distinct doc_id from read_parquet('{parquet}')").fetchall():
            done.add(doc_id)
    # also skip docs already QUEUED in pending batch inputs (extraction may be running)
    for fp in glob.glob(str(BATCHES / "*_batch_*.jsonl")):
        for line in open(fp):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["doc_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", default=str(HERE / "data" / "news_corpus"))
    ap.add_argument("--prefix", default="all", help="batch filename prefix")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--start-date")
    ap.add_argument("--end-date")
    ap.add_argument("--max-body-chars", type=int, default=12000)
    ap.add_argument("--sources", help="comma-separated source_name filter "
                    "(e.g. the EM feed pack) — keeps drains scoped and cheap")
    args = ap.parse_args()

    rows = load_rows(Path(args.corpus), args.start_date, args.end_date,
                     desk=None, source=None, only_full_body=True, limit=None)
    before = len(rows)
    rows = [r for r in rows if is_real_article(r)]
    if args.sources:
        want = {s.strip().lower() for s in args.sources.split(",")}
        rows = [r for r in rows
                if str(r.get("source_name") or "").lower() in want]
    done = already_extracted()
    rows = [r for r in rows if r["doc_id"] not in done]
    print(f"[batches] corpus={before} gated={len(rows) + sum(1 for _ in ())} "
          f"already-extracted/queued={len(done)} to-queue={len(rows)}", file=sys.stderr)
    if not rows:
        print(json.dumps({"batches_written": 0, "docs_queued": 0}))
        return 0

    BATCHES.mkdir(parents=True, exist_ok=True)
    # continue numbering after any existing batches with this prefix
    existing = sorted(glob.glob(str(BATCHES / f"{args.prefix}_batch_*.jsonl")))
    start = 0
    if existing:
        start = int(existing[-1].rsplit("_", 1)[1].split(".")[0]) + 1

    n_batches = 0
    for i in range(0, len(rows), args.batch_size):
        chunk = rows[i:i + args.batch_size]
        idx = start + n_batches
        out = BATCHES / f"{args.prefix}_batch_{idx:03d}.jsonl"
        with open(out, "w") as fh:
            for r in chunk:
                fh.write(json.dumps({
                    "doc_id": r["doc_id"], "source_name": r.get("source_name"),
                    "published_date": str(r.get("published_date")),
                    "url": r.get("url"), "title": r.get("title"),
                    "body_text": (r.get("body_text") or "")[:args.max_body_chars],
                }, ensure_ascii=False) + "\n")
        n_batches += 1
    print(json.dumps({"batches_written": n_batches, "docs_queued": len(rows),
                      "prefix": args.prefix, "first_index": start}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
