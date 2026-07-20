# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
eventgraph :: Groq BATCH API extraction (eg-rich-v2), the cheap-at-scale path.

Flow per (sub)batch: build request JSONL -> upload as a file (purpose=batch) ->
create a batch (endpoint /v1/chat/completions) -> poll -> download output -> parse
each line's JSON into the DocExtraction checkpoint format {"doc_id","ex":...}.

Batch API = 50% cheaper than on-demand and does NOT touch rate limits; window is
24h-7d. Resumable: skips doc_ids already in <out>/extractions.jsonl AND records a
processed-manifest so a 100k run knows exactly what remains.

Usage:
  source data/tmp/groq.env
  uv run eventgraph/scripts/extract_batch.py --feed /tmp/eg100_feed.jsonl \
      --out /tmp/eg100 --model llama-3.1-8b-instant --max-tokens 3500
"""
import os, sys, io, json, time, argparse
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_prompt_v2 import build_prompt, chunk_text, SYSTEM, PROMPT_CHUNKS, PROMPT_VERSION, SCHEMA_VERSION

BASE = "https://api.groq.com/openai/v1"

def hdr():
    key = os.environ.get("GROQ_API_KEY")
    if not key:
        sys.exit("GROQ_API_KEY not set (source data/tmp/groq.env)")
    return {"Authorization": f"Bearer {key}"}

def load_feed(path):
    docs = []
    for line in open(path):
        line = line.strip()
        if line:
            docs.append(json.loads(line))
    return docs

def build_request(doc, model, max_tokens):
    ch = chunk_text(doc.get("headline", ""), doc.get("article", ""))[:PROMPT_CHUNKS]
    return {
        "custom_id": doc["doc_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(doc.get("headline", ""), ch)},
            ],
        },
    }

def salvage(txt):
    import re
    t = re.sub(r"^```(json)?", "", (txt or "").strip())
    t = re.sub(r"```$", "", t).strip()
    i = t.find("{")
    if i > 0:
        t = t[i:]
    for cut in (t, re.sub(r",\s*([}\]])", r"\1", t)):
        try:
            return json.loads(cut)
        except Exception:
            pass
    return None

def _retry(fn, tries=6, base=3.0, what="request"):
    """Retry a network call through transient failures (TLS resets, timeouts)."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 -- transient network/TLS
            last = e
            print(f"    [{what}] attempt {i+1}/{tries} failed: {type(e).__name__} {str(e)[:80]}; retrying", flush=True)
            time.sleep(base * (i + 1))
    raise last

def submit_subbatch(client, reqs, model, max_tokens, window, label):
    """Upload + create one batch; return its batch id (does NOT wait). Each
    HTTP step retries independently through transient TLS/network errors."""
    payload = "\n".join(json.dumps(build_request(d, model, max_tokens)) for d in reqs).encode()
    def upload():
        up = client.post(f"{BASE}/files", headers=hdr(),
                         files={"file": (f"{label}.jsonl", io.BytesIO(payload), "application/json")},
                         data={"purpose": "batch"}, timeout=300)
        up.raise_for_status()
        return up.json()["id"]
    file_id = _retry(upload, what=f"{label} upload")
    def create():
        cr = client.post(f"{BASE}/batches", headers=hdr(), json={
            "input_file_id": file_id, "endpoint": "/v1/chat/completions",
            "completion_window": window}, timeout=60)
        cr.raise_for_status()
        return cr.json()["id"]
    return _retry(create, what=f"{label} create")

def batch_status(client, bid):
    return _retry(lambda: client.get(f"{BASE}/batches/{bid}", headers=hdr(), timeout=60).json(), what="status")

def download_batch(client, out_file_id):
    if not out_file_id:
        return {}
    content = _retry(lambda: client.get(f"{BASE}/files/{out_file_id}/content", headers=hdr(), timeout=600).text, what="download")
    results = {}
    for line in content.splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        cid = row.get("custom_id")
        try:
            results[cid] = salvage(row["response"]["body"]["choices"][0]["message"]["content"])
        except Exception:
            results[cid] = None
    return results

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--max-tokens", type=int, default=3500)
    ap.add_argument("--window", default="24h")
    ap.add_argument("--poll", type=int, default=15)
    ap.add_argument("--max-per-batch", type=int, default=20000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "extractions.jsonl"
    manifest = out / "processed.jsonl"
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            try: done.add(json.loads(l)["doc_id"])
            except Exception: pass

    docs = [d for d in load_feed(args.feed) if d["doc_id"] not in done]
    print(f"feed={len(done)+len(docs)} already={len(done)} to_process={len(docs)} "
          f"model={args.model} prompt={PROMPT_VERSION} schema={SCHEMA_VERSION}", flush=True)
    if not docs:
        print("nothing to do."); return

    client = httpx.Client()
    # deterministic sub-batches
    subs = [docs[i:i + args.max_per_batch] for i in range(0, len(docs), args.max_per_batch)]

    # resumable state: sub-batch idx -> {batch_id, downloaded}. Lets us ADOPT an
    # already-submitted batch (e.g. one launched by a prior run) instead of
    # resubmitting, and survive restarts.
    state_path = out / "batches.json"
    state = {}
    if state_path.exists():
        try: state = json.loads(state_path.read_text())
        except Exception: state = {}
    def save_state():
        state_path.write_text(json.dumps(state, indent=2))

    # PHASE 1 — submit every not-yet-submitted sub-batch UP FRONT (all run
    # concurrently on Groq's side; the Batch API doesn't touch rate limits).
    for idx, grp in enumerate(subs):
        k = str(idx)
        if state.get(k, {}).get("batch_id"):
            print(f"  sb{idx}: adopting existing batch {state[k]['batch_id']}", flush=True)
            continue
        bid = submit_subbatch(client, grp, args.model, args.max_tokens, args.window, f"sb{idx}")
        state[k] = {"batch_id": bid, "downloaded": False}
        save_state()
        print(f"  sb{idx}: submitted {bid} ({len(grp)} reqs)", flush=True)

    # PHASE 2 — poll all sub-batches together; download each as it finishes.
    ckf = open(ckpt, "a"); mff = open(manifest, "a")
    t0 = time.time(); n_ok = n_bad = 0
    remaining = {idx for idx in range(len(subs)) if not state.get(str(idx), {}).get("downloaded")}
    while remaining:
        for idx in sorted(remaining):
            k = str(idx); bid = state[k]["batch_id"]
            st = batch_status(client, bid)
            status = st.get("status"); rc = st.get("request_counts", {})
            if status in ("completed", "failed", "expired", "cancelled"):
                res = download_batch(client, st.get("output_file_id"))
                for d in subs[idx]:
                    ex = res.get(d["doc_id"])
                    ok = isinstance(ex, dict); n_ok += ok; n_bad += not ok
                    ckf.write(json.dumps({"doc_id": d["doc_id"], "ex": ex, "headline": d.get("headline"),
                                          "published_at": d.get("published_at"), "source": d.get("source")}) + "\n")
                    mff.write(json.dumps({"doc_id": d["doc_id"], "published_at": d.get("published_at"), "ok": ok}) + "\n")
                ckf.flush(); mff.flush()
                state[k]["downloaded"] = True; save_state()
                remaining.discard(idx)
                print(f"  sb{idx}: {status} -> downloaded {len(res)} ({len(remaining)} sub-batches left)", flush=True)
            else:
                print(f"  sb{idx}: {status} {rc.get('completed',0)}/{rc.get('total',0)} failed={rc.get('failed',0)}", flush=True)
        if remaining:
            time.sleep(args.poll)
    dt = time.time() - t0
    print(f"DONE {n_ok+n_bad} docs ({n_ok} parsed, {n_bad} null) in {dt:.0f}s -> {ckpt}", flush=True)

if __name__ == "__main__":
    main()
