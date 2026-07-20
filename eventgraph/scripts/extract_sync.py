# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
eventgraph :: SYNCHRONOUS Groq extraction (eg-rich-v2) -- instant results for
small runs / iteration. Async httpx with a pooled client + bounded concurrency +
Retry-After backoff. Same checkpoint format as extract_batch.py so the assessor and
the Rust ingest are identical. Use extract_batch.py (Batch API) for the big cheap runs.

Usage:
  source data/tmp/groq.env
  uv run extract_sync.py --feed /tmp/eg100_feed.jsonl --out /tmp/eg100 \
      --model llama-3.1-8b-instant --concurrency 12 --max-tokens 3500
"""
import os, sys, json, time, argparse, asyncio, re
from pathlib import Path
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_prompt_v2 import build_prompt, chunk_text, SYSTEM, PROMPT_CHUNKS, PROMPT_VERSION, SCHEMA_VERSION

URL = "https://api.groq.com/openai/v1/chat/completions"

def salvage(txt):
    t = re.sub(r"^```(json)?", "", (txt or "").strip())
    t = re.sub(r"```$", "", t).strip()
    i = t.find("{")
    if i > 0: t = t[i:]
    for cut in (t, re.sub(r",\s*([}\]])", r"\1", t)):
        try: return json.loads(cut)
        except Exception: pass
    return None

async def one(client, sem, doc, model, max_tokens, key):
    ch = chunk_text(doc.get("headline", ""), doc.get("article", ""))[:PROMPT_CHUNKS]
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": build_prompt(doc.get("headline", ""), ch)}]}
    async with sem:
        for attempt in range(5):
            try:
                r = await client.post(URL, headers={"Authorization": f"Bearer {key}"}, json=body, timeout=180)
                if r.status_code == 429:
                    await asyncio.sleep(float(r.headers.get("retry-after", 2)) + attempt)
                    continue
                r.raise_for_status()
                return doc, salvage(r.json()["choices"][0]["message"]["content"])
            except Exception:
                await asyncio.sleep(1.5 * (attempt + 1))
        return doc, None

async def main_async(args):
    key = os.environ.get("GROQ_API_KEY")
    if not key: sys.exit("GROQ_API_KEY not set (source data/tmp/groq.env)")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "extractions.jsonl"; manifest = out / "processed.jsonl"
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            try: done.add(json.loads(l)["doc_id"])
            except Exception: pass
    docs = [json.loads(l) for l in open(args.feed)]
    docs = [d for d in docs if d["doc_id"] not in done]
    print(f"to_process={len(docs)} (skipped {len(done)}) model={args.model} "
          f"prompt={PROMPT_VERSION} schema={SCHEMA_VERSION}", flush=True)
    if not docs: return
    sem = asyncio.Semaphore(args.concurrency)
    ckf = open(ckpt, "a"); mff = open(manifest, "a")
    t0 = time.time(); n_ok = n_bad = 0; n = 0
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=args.concurrency + 4)) as client:
        tasks = [one(client, sem, d, args.model, args.max_tokens, key) for d in docs]
        for fut in asyncio.as_completed(tasks):
            doc, ex = await fut
            ok = isinstance(ex, dict)
            n_ok += ok; n_bad += not ok; n += 1
            ckf.write(json.dumps({"doc_id": doc["doc_id"], "ex": ex, "headline": doc.get("headline"),
                                  "published_at": doc.get("published_at"), "source": doc.get("source")}) + "\n")
            mff.write(json.dumps({"doc_id": doc["doc_id"], "published_at": doc.get("published_at"), "ok": ok}) + "\n")
            if n % 25 == 0:
                ckf.flush(); mff.flush()
                print(f"  {n}/{len(docs)}  ok={n_ok} null={n_bad}  {time.time()-t0:.0f}s", flush=True)
    ckf.flush(); mff.flush()
    dt = time.time() - t0
    print(f"DONE {n} docs ({n_ok} parsed, {n_bad} null) in {dt:.1f}s -> {ckpt}", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=3500)
    asyncio.run(main_async(ap.parse_args()))
