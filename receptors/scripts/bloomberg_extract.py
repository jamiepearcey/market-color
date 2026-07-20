# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "pyarrow", "numpy"]
# ///
"""
Extract a chunk-anchored causal graph from Bloomberg articles via Groq.

Provenance is first-class: every causal edge is tied back to the ORIGINAL CHUNK
(the project's 120-word `doc_id:ci` span) via a VERBATIM quote that we then
mechanically re-locate against the source bytes -- the model's pointer is
verified, never trusted (same principle as verify_citations.py: "citations
cannot be asserted into existence").

How the LLM is made to attribute correctly:
  1. discrete anchors  -> the article is shown as numbered chunks [0] [1] ...
  2. self-verifying ptr -> each edge carries {evidence_chunk, verbatim quote}
  3. relocate + gate    -> repair() finds the quote in the source, snaps the
                           chunk_id to the chunk that actually contains it, and
                           DROPS edges whose quote can't be located.

Outputs (receptors/data_bloomberg/):
  chunks.jsonl        {chunk_id, doc_id, epoch}      (drop-in for the receptor)
  chunk_texts.jsonl   {"t": "<chunk text>"}          (aligned to chunks.jsonl)
  docs.jsonl          receptor-compatible per-doc rows (flat cause_entities)
  facts.parquet       per-EDGE rows WITH provenance {cause, subject, direction,
                      event_type, asset_class, evidence_chunk_id, quote, verified}
  data/tmp/bbg_raw_<slice>_v2_<model>.jsonl   raw checkpoint (resume-safe)

Usage:
  source data/tmp/groq.env
  uv run receptors/scripts/bloomberg_extract.py --ym 2011-09 --limit 3000 \
      --model openai/gpt-oss-120b --concurrency 10
"""
import os, re, json, time, argparse, asyncio, hashlib, difflib
from pathlib import Path
import httpx
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
BBG  = Path("/Users/jamiepearcey/Downloads/bloomberg_financial_data.parquet.gzip")
OUTD = ROOT / "receptors" / "data_bloomberg"
TMP  = ROOT / "data" / "tmp"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# controlled vocab (kept in sync with bloomberg_bench.py)
ENTITY_TYPES = {"company","country","central_bank","government_agency","commodity",
                "currency","equity_index","rate_or_bond","sector","person",
                "organization","market","other"}
EVENT_TYPES  = {"price_move","monetary_policy","fiscal_policy","earnings","guidance",
                "mergers_acquisitions","supply","demand","regulation","legal",
                "geopolitics","credit_rating","default","labor","macro_data","other"}
ASSET_CLASS  = {"equities","rates","fx","commodities","credit","macro","other"}
DIRECTION    = {"up","down","mixed","none"}
DIR_SIGN     = {"up":1.0,"down":-1.0,"mixed":0.0,"none":0.0}

ALIASES = {"us":"united states","u.s.":"united states","usa":"united states",
           "fed":"federal reserve","the fed":"federal reserve","uk":"united kingdom",
           "ecb":"european central bank","boj":"bank of japan"}
SUFFIX = re.compile(r"\b(inc|corp|corporation|co|ltd|plc|llc|llp|sa|ag|nv|group|holdings?|bhd)\.?$", re.I)

def norm_name(s):
    if isinstance(s, dict): s = s.get("name") or ""
    s = re.sub(r"\s+", " ", str(s or "")).strip().strip(".,")
    s = SUFFIX.sub("", s).strip()
    return ALIASES.get(s.lower(), s.lower())

# ---- canonical chunker (identical to export_rag.py) -------------------------
WORDS, MAXC = 120, 6
def chunk_text(title, body):
    body = body or ""; ws = body.split(); out = []
    for i in range(0, min(len(ws), WORDS*MAXC), WORDS):
        out.append(((title or "") + " — " + " ".join(ws[i:i+WORDS])).strip()[:2000])
    if not out: out = [((title or "").strip() or "untitled")[:2000]]
    return out

_DASH = re.compile(r"[‐‑‒–—―−]")
def _canon(s):  # for quote<->chunk matching: case/space/dash/smart-quote insensitive
    s = (s or "").lower().replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"')
    s = _DASH.sub("-", s).replace(" "," ").replace(" "," ").replace(" "," ")
    return re.sub(r"\s+", " ", s).strip()

def locate(quote, canon_chunks):
    """Return the chunk index that contains `quote`, else -1 -- what makes
    attribution VERIFIED not asserted. Models sometimes stitch non-contiguous
    fragments with '...'; we split on that and verify the longest fragment, and
    attribute the edge to the chunk that holds it."""
    q = _canon(quote)
    if len(q) < 12: return -1
    frags = [f.strip() for f in re.split(r"\.\.\.|…", q) if len(f.strip()) >= 15] or [q]
    frags.sort(key=len, reverse=True)
    for frag in frags:                             # exact (normalized) substring, longest first
        for j, c in enumerate(canon_chunks):
            if frag in c: return j
    frag = frags[0]                                # fuzzy fallback on the longest fragment
    best, bj = 0.0, -1
    for j, c in enumerate(canon_chunks):
        r = difflib.SequenceMatcher(None, frag, c).find_longest_match(0, len(frag), 0, len(c))
        cov = r.size / max(len(frag), 1)
        if cov > best: best, bj = cov, j
    return bj if best >= 0.80 else -1

SYSTEM = ("You are a financial news information extractor. You read one article, shown as "
    "NUMBERED CHUNKS, and output a single JSON object describing its causal-graph structure. "
    "For every causal link you MUST cite the chunk it comes from and quote the exact supporting "
    "words verbatim. Extract only what the article asserts; do not speculate. Output ONLY the JSON.")

def user_prompt(headline, chunks):
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    return f"""HEADLINE: {headline}

ARTICLE (numbered chunks):
{numbered}

Extract this schema as one JSON object:

{{
  "entities": [{{"name": "<canonical, e.g. 'Apple','Federal Reserve','crude oil'>", "type": "<{'|'.join(sorted(ENTITY_TYPES))}>"}}],
  "subject": "<the single primary entity AFFECTED / the story is about, or ''>",
  "direction": "<{'|'.join(sorted(DIRECTION))}: sign of the main effect on the subject>",
  "event_type": "<{'|'.join(sorted(EVENT_TYPES))}>",
  "asset_class": "<{'|'.join(sorted(ASSET_CLASS))}>",
  "cause_edges": [
    {{"cause": "<an entities.name the article says DROVE/CAUSED the effect on the subject>",
      "evidence_chunk": <integer chunk number [i] the causal claim appears in>,
      "quote": "<8-25 words copied VERBATIM as ONE CONTIGUOUS run from that chunk; do NOT join phrases with '...'>"}}
  ]
}}

Rules:
- entities: specific named actors/instruments/places the article is ABOUT (3-8). Normalize (no tickers, no 'Inc.'/'Corp.').
- cause_edges: one per distinct driver the article attributes causation to ("because of X", "driven by X", "after X"). cause MUST be in entities.name. quote MUST be copied exactly from the cited chunk. Empty list if no cause is stated.
- Use the controlled vocab values exactly."""

def snap(v, vocab, default):
    v = str(v or "").strip().lower()
    return v if v in vocab else default

def repair(p, doc_id, chunks):
    """Coerce raw output into a clean, provenance-verified record. None if unusable."""
    if not isinstance(p, dict): return None
    ents, seen = [], set()
    for e in p.get("entities", []) or []:
        name = norm_name(e.get("name") if isinstance(e, dict) else e)
        if not name or name in seen: continue
        seen.add(name)
        ents.append({"name": name, "type": snap(e.get("type") if isinstance(e, dict) else "", ENTITY_TYPES, "other")})
    if not ents: return None
    names = {e["name"] for e in ents}
    canon = [_canon(c) for c in chunks]
    edges, dropped = [], 0
    for ed in p.get("cause_edges", []) or []:
        if not isinstance(ed, dict): continue
        cause = norm_name(ed.get("cause"))
        if cause not in names: dropped += 1; continue     # cause not an entity
        j = locate(ed.get("quote",""), canon)             # RELOCATE against source
        if j < 0: dropped += 1; continue                  # quote unverifiable -> drop
        edges.append({"cause": cause, "evidence_chunk_id": f"{doc_id}:{j}",
                      "quote": str(ed.get("quote","")).strip()})
    # dedup edges by (cause, chunk)
    ded, keys = [], set()
    for e in edges:
        k = (e["cause"], e["evidence_chunk_id"])
        if k not in keys: keys.add(k); ded.append(e)
    return {"entities": ents, "cause_edges": ded, "edges_dropped": dropped,
            "subject": norm_name(p.get("subject")) or "",
            "direction": snap(p.get("direction"), DIRECTION, "none"),
            "event_type": snap(p.get("event_type"), EVENT_TYPES, "other"),
            "asset_class": snap(p.get("asset_class"), ASSET_CLASS, "other")}

def is_market(rec):
    return (rec["asset_class"] != "other" or rec["event_type"] != "other" or len(rec["cause_edges"]) > 0)

async def call(client, model, art, sem, key):
    chunks = chunk_text(art["headline"], art["article"])
    body = {"model": model, "temperature": 0, "max_tokens": 1400,
            "response_format": {"type": "json_object"},
            "messages": [{"role":"system","content":SYSTEM},
                         {"role":"user","content":user_prompt(art["headline"], chunks)}]}
    if model.startswith("openai/gpt-oss"): body["reasoning_effort"] = "low"
    async with sem:
        t0, last = time.time(), "none"
        for attempt in range(6):
            try:
                r = await client.post(GROQ_URL, json=body, timeout=180,
                                      headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 429: last="429"; await asyncio.sleep(3*(attempt+1)); continue
                if r.status_code == 400 and "response_format" in body:
                    body.pop("response_format"); last="no-json"; continue
                r.raise_for_status()
                j = r.json(); txt = j["choices"][0]["message"]["content"]; u = j.get("usage",{})
                try: parsed = json.loads(txt)
                except Exception:
                    m = re.search(r"\{.*\}", txt or "", re.S)
                    parsed = json.loads(m.group(0)) if m else None
                return {"doc_id":art["doc_id"],"parsed":parsed,"pt":u.get("prompt_tokens",0),"ct":u.get("completion_tokens",0)}
            except Exception as e:
                last=f"{type(e).__name__}:{e}"; await asyncio.sleep(1.5*(attempt+1))
        return {"doc_id":art["doc_id"],"parsed":None,"err":last,"pt":0,"ct":0}

def load_slice(ym, limit, min_chars, max_chars):
    t = pq.ParquetFile(BBG).read()
    y, m = map(int, ym.split("-"))
    t = t.filter(pc.and_(pc.equal(pc.year(t.column("Date")), y), pc.equal(pc.month(t.column("Date")), m)))
    al = pc.utf8_length(t.column("Article"))
    t = t.filter(pc.and_(pc.greater_equal(al, min_chars), pc.less_equal(al, max_chars)))
    t = t.sort_by([("Date","ascending")])
    n = t.num_rows
    idx = range(n) if (limit<=0 or limit>=n) else [int(i*n/limit) for i in range(limit)]
    H=t.column("Headline").to_pylist(); A=t.column("Article").to_pylist()
    D=t.column("Date").to_pylist(); L=t.column("Link").to_pylist()
    out=[]
    for i in idx:
        did="bbg_"+hashlib.sha1((L[i] or H[i]).encode()).hexdigest()[:16]
        out.append({"doc_id":did,"date":D[i].isoformat(),"epoch":int(D[i].timestamp()),
                    "headline":H[i],"article":(A[i] or "")[:max_chars]})
    return out

async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ym",default="2011-09"); ap.add_argument("--limit",type=int,default=3000)
    ap.add_argument("--model",default="openai/gpt-oss-120b")
    ap.add_argument("--concurrency",type=int,default=10)
    ap.add_argument("--min-chars",type=int,default=800); ap.add_argument("--max-chars",type=int,default=6000)
    a=ap.parse_args()
    key=os.environ.get("GROQ_API_KEY")
    if not key: raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")
    OUTD.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
    ckpt = TMP / f"bbg_raw_{a.ym}_v2_{a.model.split('/')[-1]}.jsonl"

    arts = load_slice(a.ym, a.limit, a.min_chars, a.max_chars)
    by_id = {x["doc_id"]: x for x in arts}
    done = {}
    if ckpt.exists():
        for l in open(ckpt):
            try: r=json.loads(l); done[r["doc_id"]]=r
            except Exception: pass
    todo=[x for x in arts if x["doc_id"] not in done]
    print(f"slice {a.ym}: {len(arts)} articles, {len(done)} cached, {len(todo)} to extract  [{a.model}] (chunk-anchored v2)")

    sem=asyncio.Semaphore(a.concurrency); t0=time.time(); pt=ct=0
    async with httpx.AsyncClient() as client:
        with open(ckpt,"a") as ck:
            for i in range(0,len(todo),a.concurrency*4):
                chunk=todo[i:i+a.concurrency*4]
                res=await asyncio.gather(*[call(client,a.model,x,sem,key) for x in chunk])
                for r in res:
                    ck.write(json.dumps(r)+"\n"); pt+=r["pt"]; ct+=r["ct"]; done[r["doc_id"]]=r
                ck.flush()
                el=time.time()-t0; n=min(i+len(chunk),len(todo))
                print(f"  {n}/{len(todo)}  {el:5.0f}s  {n/max(el,1):.1f}/s  tok in={pt} out={ct}", end="\r")
    print()

    # ---- repair + assemble artifacts ----
    docs=[]; edge_rows=[]; chunk_meta=[]; chunk_txts=[]
    parsed_ok=market=total_edges=total_dropped=0
    for x in arts:
        r=done.get(x["doc_id"])
        if not r: continue
        chunks = chunk_text(x["headline"], x["article"])
        rec=repair(r.get("parsed"), x["doc_id"], chunks)
        if rec is None: continue
        parsed_ok+=1; total_dropped+=rec["edges_dropped"]
        # emit chunks for every parsed doc (so the receptor/citation layer can resolve any chunk_id)
        for ci,txt in enumerate(chunks):
            chunk_meta.append({"chunk_id":f'{x["doc_id"]}:{ci}',"doc_id":x["doc_id"],"epoch":x["epoch"]})
            chunk_txts.append(txt)
        causes=[e["cause"] for e in rec["cause_edges"]]
        for e in rec["cause_edges"]:
            total_edges+=1
            edge_rows.append({"doc_id":x["doc_id"],"published_epoch":x["epoch"],"date":x["date"][:10],
                              "cause":e["cause"],"subject":rec["subject"],"direction":rec["direction"],
                              "event_type":rec["event_type"],"asset_class":rec["asset_class"],
                              "evidence_chunk_id":e["evidence_chunk_id"],"quote":e["quote"]})
        if not is_market(rec): continue
        market+=1
        docs.append({"doc_id":x["doc_id"],"published_epoch":x["epoch"],"date":x["date"][:10],
                     "entities":[e["name"] for e in rec["entities"]],
                     "cause_entities":sorted(set(causes)),
                     "cause_edges":rec["cause_edges"],           # provenance kept on the doc row too
                     "predicate":rec["event_type"],"direction":DIR_SIGN[rec["direction"]],
                     "confidence":1.0,"source_name":"bloomberg","desk":rec["asset_class"],
                     "desks":[rec["asset_class"]] if rec["asset_class"]!="other" else []})
    docs.sort(key=lambda d:(d["published_epoch"],d["doc_id"]))
    with open(OUTD/"docs.jsonl","w") as f:
        for d in docs: f.write(json.dumps(d)+"\n")
    with open(OUTD/"chunks.jsonl","w") as f:
        for c in chunk_meta: f.write(json.dumps(c)+"\n")
    with open(OUTD/"chunk_texts.jsonl","w") as f:
        for t in chunk_txts: f.write(json.dumps({"t":t})+"\n")
    pq.write_table(pa.Table.from_pylist(edge_rows) if edge_rows else pa.table({"doc_id":[]}), OUTD/"facts.parquet")

    n_cause_docs=sum(1 for d in docs if d["cause_entities"])
    print(f"\n=== {a.ym} [{a.model}] chunk-anchored ===")
    print(f"parsed_ok={parsed_ok}/{len(arts)}  market={market}  docs-with-causes={n_cause_docs}")
    print(f"verified causal edges={total_edges}  dropped(unverifiable/not-entity)={total_dropped}"
          f"  keep-rate={total_edges/max(total_edges+total_dropped,1):.2f}")
    print(f"chunks={len(chunk_meta)}  tokens in={pt} out={ct}")
    print(f"wrote docs.jsonl({len(docs)}) chunks.jsonl({len(chunk_meta)}) chunk_texts.jsonl facts.parquet({len(edge_rows)} edges)")

if __name__=="__main__":
    asyncio.run(main())
