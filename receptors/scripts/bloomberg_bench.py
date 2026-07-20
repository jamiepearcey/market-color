# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Economics + quality benchmark for Bloomberg fact/graph extraction on Groq.

Goal: find the cheapest model that extracts our lean causal-graph schema well
enough to build a receptor training set, and project the $ to scale.

We extract ONE compact JSON object per article (doc-level, NOT per-fact
decomposition — that's the key economy: the receptor export aggregates facts
back to one row per doc anyway, so we skip the fact-explosion and its tokens).

Schema (deliberately lean — every field earns its tokens):
  entities:       typed graph NODES + the binding set the receptor needs
  cause_entities: the DIRECTIONAL LABELS — the one thing operator W trains on
  subject:        primary affected entity (edge head)
  direction:      signed effect  -> signed edges / future per-direction operator
  event_type:     relation type  -> future per-relation operator bank
  asset_class:    coarse desk taxonomy
We deliberately DROP: free-text claim, S-P-O triples, magnitude, confidence,
per-fact rows. They multiply output tokens and the receptor never reads them.

Usage:
  source data/tmp/groq.env
  uv run receptors/scripts/bloomberg_bench.py            # all models on sample
  uv run receptors/scripts/bloomberg_bench.py --n 12     # fewer articles
"""
import os, json, time, argparse, asyncio, statistics as st
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "tmp" / "bench_sample.jsonl"
OUT = ROOT / "data" / "tmp" / "bench_results.json"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# ---- controlled vocabularies (the taxonomy / graph model) -------------------
ENTITY_TYPES = ["company","country","central_bank","government_agency","commodity",
                "currency","equity_index","rate_or_bond","sector","person",
                "organization","market","other"]
EVENT_TYPES  = ["price_move","monetary_policy","fiscal_policy","earnings","guidance",
                "mergers_acquisitions","supply","demand","regulation","legal",
                "geopolitics","credit_rating","default","labor","macro_data","other"]
ASSET_CLASS  = ["equities","rates","fx","commodities","credit","macro","other"]
DIRECTION    = ["up","down","mixed","none"]

SYSTEM = (
    "You are a financial news information extractor. You read one news article "
    "and output a single compact JSON object describing its causal-graph structure. "
    "Extract only what the article actually asserts; do not speculate. "
    "Output ONLY the JSON object, no prose."
)

def user_prompt(headline: str, article: str) -> str:
    return f"""Extract this schema as one JSON object:

{{
  "entities": [{{"name": "<canonical name, e.g. 'Apple', 'Federal Reserve', 'crude oil'>", "type": "<one of: {'|'.join(ENTITY_TYPES)}>"}}],
  "cause_entities": ["<names, subset of entities.name, that the article says are the DRIVERS/CAUSES of what happened>"],
  "subject": "<the single primary entity that is AFFECTED / the story is about, or ''>",
  "direction": "<{'|'.join(DIRECTION)}: sign of the main effect on the subject>",
  "event_type": "<one of: {'|'.join(EVENT_TYPES)}>",
  "asset_class": "<one of: {'|'.join(ASSET_CLASS)}>"
}}

Rules:
- entities: the specific named actors/instruments/places the article is ABOUT (3-8 typical). Normalize names (no tickers, no 'Inc.'/'Corp.').
- cause_entities: only entities the article attributes causation to ("because of X", "driven by X", "after X did Y"). Must be a subset of entities.name. Empty list if the article states no cause.
- Every cause_entity MUST also appear in entities.
- Use the controlled vocab values exactly.

HEADLINE: {headline}

ARTICLE:
{article}"""

# ---- candidate models + approx Groq list pricing ($/1M tok in,out) ----------
# Prices approximate (public Groq pricing, may drift); token counts below are
# exact from the API so you can reprice. Reasoning models (gpt-oss) may emit
# hidden reasoning tokens billed as completion — that shows up in the numbers.
MODELS = {
    "llama-3.1-8b-instant":                       (0.05, 0.08),
    "meta-llama/llama-4-scout-17b-16e-instruct":  (0.11, 0.34),
    "openai/gpt-oss-20b":                         (0.10, 0.50),
    "qwen/qwen3.6-27b":                           (0.29, 0.59),
    "llama-3.3-70b-versatile":                    (0.59, 0.79),
    "openai/gpt-oss-120b":                        (0.15, 0.75),
}
REFERENCE = "openai/gpt-oss-120b"   # strongest -> pseudo-gold for agreement
FULL_CORPUS = 446_762

def norm(s):
    if isinstance(s, dict): s = s.get("name") or s.get("entity") or ""
    return str(s or "").strip().lower()

async def call(client, model, art, sem):
    body = {
        "model": model,
        "messages": [{"role":"system","content":SYSTEM},
                     {"role":"user","content":user_prompt(art["headline"], art["article"])}],
        "temperature": 0,
        "response_format": {"type":"json_object"},
        "max_tokens": 1200,
    }
    if model.startswith("openai/gpt-oss"):
        body["reasoning_effort"] = "low"   # keep reasoning-token cost honest/low
    async with sem:
        t0 = time.time()
        last = "no attempt"
        for attempt in range(6):
            try:
                r = await client.post(GROQ_URL, json=body, timeout=180)
                if r.status_code == 429:                     # rate limit -> back off
                    last = "429"; await asyncio.sleep(3*(attempt+1)); continue
                if r.status_code == 400 and "response_format" in body:
                    body.pop("response_format")              # model rejects json mode
                    last = "no-json-mode"; continue
                r.raise_for_status()
                j = r.json()
                txt = j["choices"][0]["message"]["content"]
                usage = j.get("usage",{})
                try: parsed = json.loads(txt)
                except Exception:                            # salvage a fenced/embedded object
                    import re
                    mt = re.search(r"\{.*\}", txt or "", re.S)
                    parsed = None
                    if mt:
                        try: parsed = json.loads(mt.group(0))
                        except Exception: parsed = None
                return {"doc_id":art["doc_id"],"latency":time.time()-t0,"parsed":parsed,"raw":txt,
                        "pt":usage.get("prompt_tokens",0),"ct":usage.get("completion_tokens",0)}
            except Exception as e:
                last = f"{type(e).__name__}: {e}"
                await asyncio.sleep(1.5*(attempt+1))
        return {"doc_id":art["doc_id"],"latency":time.time()-t0,"parsed":None,
                "raw":f"ERR after retries: {last}","pt":0,"ct":0}

async def run_model(client, model, arts):
    sem = asyncio.Semaphore(4)
    res = await asyncio.gather(*[call(client, model, a, sem) for a in arts])
    return {r["doc_id"]: r for r in res}

def schema_ok(p):
    if not isinstance(p, dict): return False
    ents = p.get("entities")
    if not isinstance(ents, list) or not ents: return False
    names = set()
    for e in ents:
        if not isinstance(e, dict) or not e.get("name"): return False
        if e.get("type") not in ENTITY_TYPES: return False
        names.add(norm(e["name"]))
    ce = p.get("cause_entities", [])
    if not isinstance(ce, list): return False
    if any(norm(c) not in names for c in ce): return False
    if p.get("direction") not in DIRECTION: return False
    if p.get("event_type") not in EVENT_TYPES: return False
    if p.get("asset_class") not in ASSET_CLASS: return False
    return True

def ents_of(p): return {norm(e.get("name")) for e in p.get("entities",[]) if isinstance(e,dict) and e.get("name")} if isinstance(p,dict) else set()
def causes_of(p): return {norm(c) for c in p.get("cause_entities",[]) if c} if isinstance(p,dict) else set()

def f1(a, b):
    if not a and not b: return 1.0
    if not a or not b: return 0.0
    tp = len(a & b);
    prec = tp/len(b); rec = tp/len(a)
    return 0.0 if prec+rec==0 else 2*prec*rec/(prec+rec)

def score(model_out, ref_out, arts):
    rows=[]
    for a in arts:
        d=a["doc_id"]; m=model_out[d]; p=m["parsed"]; rp=ref_out[d]["parsed"]
        rows.append({
            "valid": p is not None,
            "schema": schema_ok(p),
            "ent_f1": f1(ents_of(rp), ents_of(p)),
            "cause_f1": f1(causes_of(rp), causes_of(p)),
            "dir_match": isinstance(p,dict) and isinstance(rp,dict) and p.get("direction")==rp.get("direction"),
            "evt_match": isinstance(p,dict) and isinstance(rp,dict) and p.get("event_type")==rp.get("event_type"),
            "n_ent": len(ents_of(p)), "n_cause": len(causes_of(p)),
            "pt": m["pt"], "ct": m["ct"], "lat": m["latency"],
        })
    return rows

async def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=24)
    ap.add_argument("--models",default=""); a=ap.parse_args()
    if not os.environ.get("GROQ_API_KEY"): raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")
    arts=[json.loads(l) for l in open(SAMPLE)][:a.n]
    models=[m.strip() for m in a.models.split(",") if m.strip()] or list(MODELS)
    print(f"benchmark: {len(arts)} articles x {len(models)} models\n")
    headers={"Authorization":f"Bearer {os.environ['GROQ_API_KEY']}"}
    outs={}
    async with httpx.AsyncClient(headers=headers) as client:
        for m in models:
            t0=time.time(); outs[m]=await run_model(client,m,arts)
            print(f"  ran {m:44s} in {time.time()-t0:5.1f}s")
    ref=outs[REFERENCE]
    print(f"\n{'model':44s} {'valid':>6} {'schm':>6} {'entF1':>6} {'causF1':>7} {'dir':>5} {'evt':>5} {'nEnt':>5} {'nCau':>5} {'tok(in/out)':>13} {'lat_s':>6} {'$/3k':>7} {'$/446k':>8}")
    summary={}
    for m in models:
        rows=score(outs[m],ref,arts)
        mean=lambda k: st.mean(r[k] for r in rows)
        rate=lambda k: sum(1 for r in rows if r[k])/len(rows)
        pin,pout=MODELS.get(m,(0,0))
        tin,tout=mean("pt"),mean("ct")
        cost1=(tin*pin+tout*pout)/1e6
        summary[m]={"valid":rate("valid"),"schema":rate("schema"),"ent_f1":mean("ent_f1"),
                    "cause_f1":mean("cause_f1"),"dir":mean("dir_match"),"evt":mean("evt_match"),
                    "n_ent":mean("n_ent"),"n_cause":mean("n_cause"),"tin":tin,"tout":tout,
                    "lat":mean("lat"),"cost_3k":cost1*3000,"cost_full":cost1*FULL_CORPUS}
        s=summary[m]
        tag=" (REF)" if m==REFERENCE else ""
        print(f"{m:44s} {rate('valid'):6.2f} {rate('schema'):6.2f} {mean('ent_f1'):6.2f} {mean('cause_f1'):7.2f} "
              f"{mean('dir_match'):5.2f} {mean('evt_match'):5.2f} {mean('n_ent'):5.1f} {mean('n_cause'):5.1f} "
              f"{tin:5.0f}/{tout:<5.0f} {mean('lat'):6.2f} {cost1*3000:6.2f} {cost1*FULL_CORPUS:8.2f}{tag}")
    json.dump({"summary":summary,"raw":{m:outs[m] for m in models}}, open(OUT,"w"), indent=1, default=str)
    print(f"\nwrote {OUT}")
    print("\nNote: entF1/causF1/dir/evt are AGREEMENT vs the reference model "
          f"({REFERENCE}), not human gold — read them as 'captures what the strong model captures'.")

if __name__=="__main__":
    asyncio.run(main())
