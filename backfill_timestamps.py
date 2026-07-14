#!/usr/bin/env python3
"""Backfill time-of-day publication timestamps for date-floored corpus docs.

The original crawl used trafilatura's date-only `date` field, flooring
published_utc to midnight. This re-fetches ONLY those docs' URLs and parses
the real timestamp from HTML metadata (JSON-LD / meta tags — no LLM).
Safe update rule: only apply a fetched timestamp whose DATE matches the
stored published_date (no partition moves, no re-dating).

Resume-safe: fetched results cached in facts_work/ts_backfill.jsonl.

  uv run --with httpx --with pyarrow --with python-dateutil \
      python backfill_timestamps.py fetch    # crawl metadata (resumable)
  uv run --with pyarrow python backfill_timestamps.py apply  # rewrite parquet
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from crawl_corpus import extract_pub_ts  # noqa: E402

CORPUS = HERE / "data" / "news_corpus"
CACHE = HERE / "facts_work" / "ts_backfill.jsonl"
WORKERS = 8


def _read_part(part):
    """Raw single-file read. NEVER pq.read_table(path) on a hive-partitioned
    file: it infers and EMBEDS the dt partition column, corrupting the file
    for future hive reads."""
    import pyarrow.parquet as pq
    return pq.ParquetFile(part).read().to_pylist()


def _floored_docs():
    docs = []
    for part in sorted(CORPUS.glob("dt=*/part-*.parquet")):
        for r in _read_part(part):
            if "T00:00:00" in str(r.get("published_utc") or ""):
                docs.append({"doc_id": r["doc_id"], "url": r["url"],
                             "published_date": r["published_date"]})
    return docs


def cmd_fetch(args):
    import httpx
    docs = _floored_docs()
    done = set()
    if CACHE.exists():
        for l in open(CACHE):
            if l.strip():
                done.add(json.loads(l)["doc_id"])
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"floored docs: {len(docs)}, cached: {len(done)}, to fetch: {len(todo)}",
          file=sys.stderr)
    lock = threading.Lock()
    stats = {"ok": 0, "no_ts": 0, "date_mismatch": 0, "http_fail": 0}

    def job(d):
        rec = {"doc_id": d["doc_id"], "ts": None, "status": "http_fail"}
        try:
            with httpx.Client(timeout=15, follow_redirects=True, headers={
                    "User-Agent": "Mozilla/5.0 (market-color ts-backfill)"}) as c:
                r = c.get(d["url"])
            if r.status_code < 400 and r.text:
                dt = extract_pub_ts(r.text)
                if dt is None or (dt.hour, dt.minute, dt.second) == (0, 0, 0):
                    rec["status"] = "no_ts"
                elif dt.date().isoformat() != d["published_date"]:
                    rec["status"] = "date_mismatch"
                    rec["ts"] = dt.isoformat()
                else:
                    rec["status"] = "ok"
                    rec["ts"] = dt.isoformat()
        except Exception:
            pass
        with lock:
            stats[rec["status"]] += 1
            with open(CACHE, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            n = sum(stats.values())
            if n % 200 == 0:
                print(f"  [{n}/{len(todo)}] {stats}", file=sys.stderr)
        return rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        list(ex.map(job, todo))
    print(json.dumps(stats))


def cmd_apply(args):
    import pyarrow as pa
    import pyarrow.parquet as pq
    ts = {}
    for l in open(CACHE):
        if l.strip():
            r = json.loads(l)
            if r["status"] == "ok" and r["ts"]:
                ts[r["doc_id"]] = r["ts"]
    updated = 0
    for part in sorted(CORPUS.glob("dt=*/part-*.parquet")):
        rows = _read_part(part)
        changed = False
        for r in rows:
            new = ts.get(r["doc_id"])
            if new and "T00:00:00" in str(r.get("published_utc") or ""):
                r["published_utc"] = new
                r["published_is_estimated"] = False
                changed = True
                updated += 1
        if changed:
            pq.write_table(pa.Table.from_pylist(rows), part)
    print(f"applied {updated} timestamps across corpus "
          f"({len(ts)} fetched ok)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["fetch", "apply"])
    args = p.parse_args()
    {"fetch": cmd_fetch, "apply": cmd_apply}[args.stage](args)


if __name__ == "__main__":
    main()
