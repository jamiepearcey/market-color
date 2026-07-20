# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Fast narrative extraction (the 'yesterday' way): async httpx with a POOLED client
(keep-alive, no per-call TLS handshake) + Retry-After-aware backoff, writing the
eventgraph rich schema straight into the Rust checkpoint format.

Output: <out>/extractions.jsonl  lines of {"doc_id","ex":<DocExtraction>} -- the
exact format `eventgraph ingest --mock` resumes from, so Rust does normalize +
quality gates + DuckLake with NO re-extraction and NO Rust rebuild.

Resumable: skips doc_ids already in the checkpoint; appends as it goes.

Usage:
  source data/tmp/groq.env
  uv run eventgraph/scripts/extract.py --feed /tmp/eg_500_feed.jsonl --out /tmp/eg_500 \
      --model llama-3.1-8b-instant --concurrency 8
"""
import os, re, json, argparse, asyncio, time
from pathlib import Path
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

ENTITY_TYPES = "company|bank|central_bank|sovereign|regulator|commodity|currency|equity_index|rate_or_bond|sector|person|exchange|economic_indicator|other"

WORDS, MAXC = 120, 6
def chunk_text(title, body):
    ws = (body or "").split()
    out = []
    for i in range(0, min(len(ws), WORDS * MAXC), WORDS):
        out.append((f"{(title or '').strip()} — " + " ".join(ws[i:i + WORDS]))[:2000])
    return out or [((title or "").strip() or "untitled")[:2000]]

SYSTEM = "You are a financial news information extractor. Output ONLY the JSON object."

def prompt(headline, chunks):
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    return f"""HEADLINE: {headline}

ARTICLE (numbered chunks):
{numbered}

Extract ONE JSON object (only what the article asserts; do not speculate). Every causal edge,
sensitivity, sentiment, event and proposition MUST cite the chunk it comes from (evidence_chunk =
the [i]) and quote the exact supporting words verbatim.

{{
  "doc_type": "news|analysis|opinion|data_release|corporate_statement|non_financial",
  "sentiment_overall": <-1..1 or null>,
  "entities": [{{"name":"canonical (no tickers/Inc.)","type":"{ENTITY_TYPES}","identifier":"ticker/ISO or null","sector":null,"country":null,"role":"subject|driver|counterparty|mentioned"}}],
  "events": [{{"event_type":"rate_decision|cpi|employment|gdp|pmi|earnings|guidance|mergers_acquisitions|debt_auction|rating_review|election|opec_meeting|eia_inventory|other","series_hint":null,"issuer":null,"region":null,"scheduled":true,"event_time":"ISO or null","expected":null,"actual":null,"prior":null,"unit":null,"evidence_chunk":0,"quote":"verbatim"}}],
  "causal_edges": [{{"cause":"entities.name","effect":"a DIFFERENT entities.name","mechanism":"monetary_policy|rate_decision|supply_shock|demand_change|earnings|guidance|mergers_acquisitions|default|rating_action|regulation|geopolitics|fund_flows|data_surprise|contagion|other","effect_direction":"up|down|widen|tighten|volatile|unchanged","magnitude_value":null,"magnitude_unit":null,"modality":"happened|ongoing|forecast|hypothetical|denied","attribution_source":"reporter|named_analyst|company|official|market_consensus|market_implied","lag":"immediate|days|longer","confidence":"strong|tentative","evidence_chunk":0,"quote":"verbatim contiguous span"}}],
  "sensitivities": [{{"asset":"entities.name","factor":"oil|rates|usd|credit spread|<name>","factor_type":"rates|credit_spread|oil|commodity|usd|fx|equity_beta|inflation|specific|other","sign":1,"magnitude_qual":"high|medium|low","magnitude_value":null,"magnitude_unit":null,"basis":"fundamental|stated_beta|historical","evidence_chunk":0,"quote":"verbatim"}}],
  "sentiments": [{{"target":"entities.name","polarity":-1.0,"intensity":"strong|mild","type":"directional|risk|credit|surprise","source":"reporter|named_analyst|official|market_implied","horizon":"immediate|short|long","evidence_chunk":0,"quote":"verbatim"}}],
  "propositions": [{{"text":"resolvable forward statement","subject":null,"resolution_date":null,"resolution_criteria":"objective settle condition","probability":null,"source":"reporter|named_analyst|market_implied","source_instrument":null,"contract_hint":null,"evidence_chunk":0,"quote":"verbatim"}}],
  "figures": [{{"entity":null,"kind":"price|level|pct|bps|volume|estimate","value":0.0,"unit":null,"evidence_chunk":0,"quote":"verbatim"}}]
}}

Rules: entities 3-8; cause/effect/asset/target/subject MUST be an entities.name; a causal_edge's cause and effect MUST be DIFFERENT entities; use vocab values exactly; empty arrays where nothing applies; quote copied verbatim from the cited chunk."""

# ---- lenient clean: keep only well-formed array elements so the Rust strict
#      checkpoint loader parses every line. ------------------------------------
def num(x):
    if isinstance(x, bool): return None
    if isinstance(x, (int, float)): return x
    try: return float(x)
    except (TypeError, ValueError): return None

def clean_ex(raw):
    if not isinstance(raw, dict): raw = {}
    def arr(key): return raw.get(key) if isinstance(raw.get(key), list) else []
    def keep(el, req):  # req = required keys
        return isinstance(el, dict) and all(el.get(k) not in (None, "") for k in req)
    ents = [e for e in arr("entities") if keep(e, ["name"])]
    events = [e for e in arr("events") if keep(e, ["event_type"])]
    edges = [e for e in arr("causal_edges") if keep(e, ["cause", "effect"])]
    sens = [s for s in arr("sensitivities") if keep(s, ["asset", "factor"])]
    sents = [s for s in arr("sentiments") if keep(s, ["target"]) and num(s.get("polarity")) is not None]
    props = [p for p in arr("propositions") if keep(p, ["text"])]
    figs = [f for f in arr("figures") if keep(f, ["kind"]) and num(f.get("value")) is not None]
    for s in sents: s["polarity"] = num(s["polarity"])
    for f in figs: f["value"] = num(f["value"])
    for s in sens:
        if s.get("sign") is not None:
            v = num(s["sign"]); s["sign"] = int(v) if v is not None else None
    return {
        "doc_type": raw.get("doc_type") if isinstance(raw.get("doc_type"), str) else None,
        "sentiment_overall": num(raw.get("sentiment_overall")),
        "entities": ents, "events": events, "causal_edges": edges,
        "sensitivities": sens, "sentiments": sents, "propositions": props, "figures": figs,
    }

def salvage(txt):
    try: return json.loads(txt)
    except Exception: pass
    a, b = txt.find("{"), txt.rfind("}")
    if a >= 0 and b > a:
        try: return json.loads(txt[a:b + 1])
        except Exception: return None
    return None

class TokenBucket:
    """Pace requests to stay under Groq's tokens-per-minute limit (the real cap)."""
    def __init__(self, per_min):
        self.cap = per_min; self.tokens = float(per_min); self.rate = per_min / 60.0
        self.last = time.monotonic(); self.lock = asyncio.Lock()
    async def acquire(self, n):
        n = min(n, self.cap)
        async with self.lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate)
                self.last = now
                if self.tokens >= n:
                    self.tokens -= n; return
                await asyncio.sleep((n - self.tokens) / self.rate)

# Groq debits the RESERVED max_tokens against the 250k TPM, so keep it tight (the
# rich JSON for a typical article fits; truncation is recovered by the balancer).
MAX_TOKENS = 1200
PROMPT_CHUNKS = 4          # feed the first 4 chunks (~480 words) -- lead carries the causal content
async def extract_one(client, sem, bucket, model, key, doc, max_tokens=MAX_TOKENS):
    chunks = chunk_text(doc["headline"], doc.get("article", ""))[:PROMPT_CHUNKS]
    user = prompt(doc["headline"], chunks)
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": user}]}
    est = (len(SYSTEM) + len(user)) // 4 + max_tokens   # TPM debit = input + RESERVED max_tokens
    await bucket.acquire(est)                            # pace to match real accounting
    async with sem:
        for attempt in range(6):
            try:
                r = await client.post(GROQ_URL, json=body,
                                      headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 429:            # respect Retry-After (the real fix)
                    ra = r.headers.get("retry-after")
                    await asyncio.sleep(min(float(ra), 20) if ra else 2 * (attempt + 1))
                    continue
                if r.status_code >= 500:
                    await asyncio.sleep(1.5 * (attempt + 1)); continue
                if r.status_code != 200:
                    return doc["doc_id"], None      # 400 etc: skip
                txt = r.json()["choices"][0]["message"]["content"]
                return doc["doc_id"], clean_ex(salvage(txt))
            except Exception:
                await asyncio.sleep(1.5 * (attempt + 1))
        return doc["doc_id"], None

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="1200 fits 8b lean output; reasoning models (gpt-oss-120b) need ~3000+")
    a = ap.parse_args()
    key = os.environ.get("GROQ_API_KEY")
    if not key: raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ckpt = out / "extractions.jsonl"

    docs = [json.loads(l) for l in open(a.feed) if l.strip()]
    if a.limit: docs = docs[:a.limit]
    done = set()
    if ckpt.exists():
        for l in open(ckpt):
            try: done.add(json.loads(l)["doc_id"])
            except Exception: pass
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"{len(docs)} docs, {len(done)} already in checkpoint, {len(todo)} to extract "
          f"[{a.model}] conc={a.concurrency} (pooled + Retry-After)")

    limits = httpx.Limits(max_connections=a.concurrency * 2, max_keepalive_connections=a.concurrency * 2)
    sem = asyncio.Semaphore(a.concurrency)
    bucket = TokenBucket(240_000)          # stay just under the 250k TPM ceiling
    t0, ok, fail, wrote = time.time(), 0, 0, 0
    async with httpx.AsyncClient(timeout=180, limits=limits) as client:
        with open(ckpt, "a") as f:
            tasks = [extract_one(client, sem, bucket, a.model, key, d, a.max_tokens) for d in todo]
            for i, fut in enumerate(asyncio.as_completed(tasks), 1):
                did, ex = await fut
                if ex is not None:
                    f.write(json.dumps({"doc_id": did, "ex": ex}) + "\n"); f.flush(); ok += 1
                else:
                    fail += 1
                if i % 50 == 0 or i == len(todo):
                    el = time.time() - t0
                    print(f"  {i}/{len(todo)}  {el:5.0f}s  {i/max(el,1):.1f}/s  ok={ok} fail={fail}", flush=True)
    print(f"done: {ok} extracted, {fail} failed -> {ckpt}")

if __name__ == "__main__":
    asyncio.run(main())
