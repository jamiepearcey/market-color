# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""Catalyst normalization with a RUNNING canonical dictionary (each batch reuses keys already
assigned) + distinctive-token lexical merge for fragments. Collapses raw 'cause' entities into
canonical CATALYSTS (both Almunia IDs -> one; Fed variants -> one) with a proper type.
Output catalyst_map.json {cause_entity_id: {key,name,type}}."""
import os, json, time, collections, difflib, re, sys
from pathlib import Path
import httpx
G=Path("../data/eg_runs/eg100k_graph")
GROQ="https://api.groq.com/openai/v1/chat/completions"; KEY=os.environ["GROQ_API_KEY"]; MODEL="openai/gpt-oss-120b"
BATCH=45; TOPN=800; STOP={"the","of","and","for","a","an","to","in","on","at","by","inc","corp","co","ltd","plc","sa","ag","llc","s","u"}
DIRV={"up","down","widen","tighten"}
sym=set(json.loads(l)["entity_id"] for l in open(G/"entity_symbol.jsonl") if json.loads(l)["kind"]=="security")
freq=collections.Counter(); ctx={}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); c=j.get("cause_entity"); e=j.get("effect_entity"); d=j.get("effect_dir")
    if not c or d not in DIRV or e not in sym: continue
    freq[c]+=1; ctx.setdefault(c, j.get("quote") or "")
cands=[c for c,_ in freq.most_common(TOPN) if freq[c]>=2]
print(f"{len(freq)} cause entities driving priceable names; canonicalizing top {len(cands)} (running dict)",flush=True)
SYSTEM=("You normalize raw news 'cause' entities. For each, give a clean canonical NAME and TYPE, and a "
 "snake_case KEY. ONLY merge (same key) obvious title/spelling/fragment variants of the SAME specific entity "
 "(e.g. 'jpmorgan' & 'jpmorgan_chase' -> 'jpmorgan_chase'; 'president_barack_obama' & 'barack_obama' -> "
 "'barack_obama'). Do NOT merely group RELATED-but-distinct entities: different central banks, different "
 "companies, different countries each keep their OWN key (reserve_bank_of_india != federal_reserve; japan != china).\n"
 "- name: concise Title-Case label. - type: regulator|regulation|legal|monetary|fiscal|labor|political|company|person|disaster|macro|other.\n"
 'Return ONLY JSON {"results":[{"i":<int>,"key":<snake>,"name":<str>,"type":<str>}]}')
def groq(batch,existing):
    lines=[f'{i+1}. {c.split("__")[0].replace("_"," ")} [{c.split("__")[-1]}] — ctx: {ctx.get(c,"")[:50]}' for i,c in enumerate(batch)]
    body={"model":MODEL,"temperature":0,"max_tokens":3000,"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":SYSTEM},
                      {"role":"user","content":"Canonicalize these:\n"+"\n".join(lines)}]}
    for att in range(5):
        try:
            r=httpx.post(GROQ,json=body,headers={"Authorization":f"Bearer {KEY}"},timeout=120)
            if r.status_code==429: time.sleep(3*(att+1)); continue
            if r.status_code!=200: time.sleep(2*(att+1)); continue
            t=r.json()["choices"][0]["message"]["content"]; a,b=t.find("{"),t.rfind("}")
            return json.loads(t[a:b+1]).get("results",[])
        except Exception: time.sleep(2*(att+1))
    return []
raw={}; keyfreq=collections.Counter(); keyname={}
CACHE=G/"catalyst_raw.json"
if CACHE.exists():
    raw=json.load(open(CACHE)); print(f"loaded cached LLM stage: {len(raw)} entities",flush=True)
for i in ([] if raw else range(0,len(cands),BATCH)):
    b=cands[i:i+BATCH]
    res=groq(b,None); by={r.get("i"):r for r in res if isinstance(r,dict)}
    for k,c in enumerate(b):
        r=by.get(k+1)
        if not r or not r.get("key"): continue
        key=re.sub(r"[^a-z0-9_]","",r["key"].strip().lower().replace(" ","_"))[:40] or "x"
        raw[c]={"key":key,"name":(r.get("name") or c.split("__")[0].replace("_"," ").title()).strip(),"type":(r.get("type") or "other").strip().lower()}
        keyfreq[key]+=freq[c]; keyname.setdefault(key,raw[c]["name"])
    print(f"  batch {i//BATCH+1}/{(len(cands)+BATCH-1)//BATCH}: {len(raw)} mapped, {len(keyfreq)} keys",flush=True)
if not CACHE.exists(): CACHE.write_text(json.dumps(raw))
# union-find merge: (a) identical key, (b) shared distinctive token in RAW id, (c) fuzzy key/name
ents=list(raw)
toks={c:set(t for t in c.split("__")[0].split("_") if len(t)>=2 and t not in STOP) for c in ents}
df=collections.Counter()                       # token document-frequency across entities
for c in ents:
    for t in toks[c]: df[t]+=1
ts={e:frozenset(toks[e]) for e in ents}
# representative entity per DISTINCT token-set (highest freq) — merges 'the_federal_reserve' == 'federal_reserve'
setrep={}
for e in ents:
    if ts[e] and (ts[e] not in setrep or freq[e]>freq[setrep[ts[e]]]): setrep[ts[e]]=e
sets=list(setrep)
GENERIC={"bank","central","credit","capital","group","markets","market","investors","investor","government",
 "commission","committee","department","authority","ministry","union","prices","price","futures","index","crisis",
 "holdings","financial","finance","national","international","global","president","chief","executive","company",
 "corporation","states","state","court","agency","board","service","services","system","reserve","fund","funds",
 "regulators","regulator","lenders","lender","banks","officials","office","council","association","exchange",
 "european","europe","america","american","united","china","chinese","japan","japanese","britain","british",
 "eurozone","asia","asian","western","germany","german","french","france","spain","spanish","italy","italian"}
# a set is a valid canonical PARENT only if specific: a single RARE token (df<=2), OR >=2 tokens with at least
# one NON-generic (proper-noun-ish) token. Blocks generic bridges ({central,bank},{capital}) but keeps real forms.
def valid_parent(t):
    if len(t)==1: return df[next(iter(t))]<=2
    return len(t)>=2 and any(x not in GENERIC for x in t)
# DIRECTIONAL, non-transitive: a token-set maps to its highest-freq STRICT-SUBSET valid parent (its own
# shorter canonical form). A compound entity maps to ONE parent, so two distinct entities are never bridged.
parent={s:s for s in sets}
for s in sets:
    cands=[t for t in sets if t<s and valid_parent(t)]
    if cands: parent[s]=max(cands,key=lambda t:(freq[setrep[t]],len(t)))
def root(s):
    seen=set()
    while parent[s]!=s and s not in seen: seen.add(s); s=parent[s]
    return s
comp=collections.defaultdict(list)
for e in ents: comp[setrep[root(ts[e])] if ts[e] else e].append(e)
out={}
for rep,cs in comp.items():
    # name sanity: LLM name must share a token with the rep's raw string, else fall back to humanized raw
    reptok=set(rep.split("__")[0].split("_")); llmtok=set(raw[rep]["name"].lower().replace("'","").split())
    nm=raw[rep]["name"] if (reptok & llmtok) else rep.split("__")[0].replace("_"," ").title()
    ck=raw[rep]["key"]
    tc=collections.Counter()
    for c in cs: tc[raw[c]["type"]]+=freq[c]
    ty=tc.most_common(1)[0][0]
    for c in cs: out[c]={"key":ck,"name":nm,"type":ty}
(G/"catalyst_map.json").write_text(json.dumps(out))
ncanon=len(set(v["key"] for v in out.values()))
print(f"\n-> catalyst_map.json  {len(out)} raw causes -> {ncanon} canonical catalysts")
merged=[(r,cs) for r,cs in comp.items() if len(cs)>=2]
print(f"   {len(merged)} catalysts absorb >=2 fragments; top merges:")
for r,cs in sorted(merged,key=lambda x:-sum(freq[c] for c in x[1]))[:12]:
    print(f"     {out[cs[0]]['name'][:32]:32} [{out[cs[0]]['type'][:9]:9}] <- {len(cs)}: {', '.join(c.split('__')[0][:22] for c in sorted(cs,key=lambda c:-freq[c])[:3])}")
