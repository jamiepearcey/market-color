# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
eventgraph :: retrieve finished Groq BATCHES by id and land them in the
extraction checkpoint -- for when the batches.json resume-state is gone (e.g. a
run submitted in a prior session) so extract_batch.py can't adopt them.

Downloads each batch's output_file_id (works for COMPLETED and EXPIRED batches
alike -- an expired batch still serves its completed requests), parses each
line custom_id(=doc_id) -> salvaged DocExtraction JSON, and APPENDS the exact
checkpoint line extract_batch writes ({doc_id, ex, headline, published_at,
source}), joining doc metadata back from the feed. Dedup by doc_id against the
existing checkpoint (same `done`-set semantics as extract_batch), so it's safe
to re-run and to point at overlapping batches.

Then: `eventgraph ingest --mock --feed <feed> --out <out>` normalizes + gates +
loads the new docs (no Rust rebuild).

Usage:
  source data/tmp/groq.env
  uv run eventgraph/scripts/retrieve_batches.py --feed /tmp/eg100k_feed.jsonl \
      --out /tmp/eg100k_graph --batch-ids batch_a,batch_b,batch_c
  # or discover every terminal batch on the account:
  uv run eventgraph/scripts/retrieve_batches.py --feed ... --out ... --discover
"""
import os, sys, json, argparse
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_batch import BASE, hdr, salvage, _retry, download_batch

TERMINAL = {"completed", "expired", "cancelled", "failed"}


def list_batches(client):
    """Page through every batch on the account (newest first)."""
    out, after = [], None
    while True:
        params = {"limit": 100}
        if after:
            params["after"] = after
        d = _retry(lambda: client.get(f"{BASE}/batches", headers=hdr(), params=params, timeout=60).json(),
                   what="list")
        data = d.get("data", [])
        out.extend(data)
        if not d.get("has_more") or not data:
            break
        after = data[-1]["id"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-ids", help="comma-separated batch ids")
    ap.add_argument("--discover", action="store_true", help="pull every terminal batch on the account")
    args = ap.parse_args()
    if not args.batch_ids and not args.discover:
        ap.error("give --batch-ids or --discover")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "extractions.jsonl"
    manifest = out / "processed.jsonl"

    # doc metadata from the feed (for the checkpoint line)
    meta = {}
    for l in open(args.feed):
        l = l.strip()
        if l:
            d = json.loads(l)
            meta[d["doc_id"]] = d
    print(f"feed={len(meta)} docs", flush=True)

    # already-checkpointed doc_ids (any status) -> skip, matching extract_batch resume
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            try: done.add(json.loads(l)["doc_id"])
            except Exception: pass
    print(f"checkpoint already has {len(done)} docs", flush=True)

    client = httpx.Client()
    if args.discover:
        batches = [b for b in list_batches(client) if b.get("status") in TERMINAL]
        print(f"discovered {len(batches)} terminal batches", flush=True)
    else:
        ids = [b.strip() for b in args.batch_ids.split(",") if b.strip()]
        batches = [_retry(lambda i=i: client.get(f"{BASE}/batches/{i}", headers=hdr(), timeout=60).json(),
                          what="get") for i in ids]

    ckf = open(ckpt, "a"); mff = open(manifest, "a")
    n_new = n_ok = n_null = n_skip = n_nometa = 0
    for b in batches:
        bid, status, ofid = b.get("id"), b.get("status"), b.get("output_file_id")
        if status not in TERMINAL:
            print(f"  {bid}: {status} (not terminal) -- skipped", flush=True); continue
        res = download_batch(client, ofid)
        b_new = b_ok = 0
        for doc_id, ex in res.items():
            if doc_id in done:
                n_skip += 1; continue
            d = meta.get(doc_id)
            if d is None:  # extraction from a batch for a DIFFERENT feed -- can't be
                n_nometa += 1; continue   # grounded against this corpus, so skip it
            ok = isinstance(ex, dict)
            ckf.write(json.dumps({"doc_id": doc_id, "ex": ex,
                                  "headline": (d or {}).get("headline"),
                                  "published_at": (d or {}).get("published_at"),
                                  "source": (d or {}).get("source")}) + "\n")
            mff.write(json.dumps({"doc_id": doc_id, "published_at": (d or {}).get("published_at"), "ok": ok}) + "\n")
            done.add(doc_id); n_new += 1; b_new += 1
            n_ok += ok; n_null += not ok; b_ok += ok
        ckf.flush(); mff.flush()
        print(f"  {bid}: {status} -> {len(res)} rows, +{b_new} new ({b_ok} ok)", flush=True)

    print(f"DONE +{n_new} new docs ({n_ok} parsed, {n_null} null), {n_skip} already-had"
          f"{f', {n_nometa} not-in-feed' if n_nometa else ''} -> {ckpt}", flush=True)


if __name__ == "__main__":
    main()
