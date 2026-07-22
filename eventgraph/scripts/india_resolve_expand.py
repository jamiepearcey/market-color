# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""Expand the India resolution beyond Nifty50: LLM maps unresolved company/bank entities to NSE
tickers; each survives ONLY if the .NS ticker has real 2020-22 price history AND its Yahoo issuer
name matches (anti-hallucination). Merges into nifty50_resolved.json so india_* scripts gain power."""
import os, re, json, time, collections, difflib, sys, datetime as dt
from pathlib import Path
import httpx
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/india2021"); cache=G/"prices"
GROQ="https://api.groq.com/openai/v1/chat/completions"; KEY=os.environ["GROQ_API_KEY"]
MODEL="openai/gpt-oss-120b"; LIMIT=600; BATCH=40
DIRSET={"up","down","widen","tighten"}
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
SYSTEM=("You map company/brand names from INDIAN financial news to their primary NSE (National Stock "
 "Exchange of India) ticker SYMBOL, no suffix. Rules:\n"
 "- Give the NSE symbol for companies LISTED on NSE (Reliance->RELIANCE, Tata Consultancy Services->TCS, "
 "Vodafone Idea->IDEA, Zomato->ZOMATO, Vedanta->VEDL, ONGC->ONGC, Paytm->PAYTM, IRCTC->IRCTC, "
 "Nestle India->NESTLEIND, Bharti Airtel->BHARTIARTL).\n"
 "- listed=false for: NON-NSE-listed firms (US/foreign like Apple/Tesla/Amazon/GameStop even if in Indian "
 "news; unlisted startups like Flipkart; private cos), govt bodies/regulators/central banks, indices, "
 "people, and generic aggregates ('banks','companies','businesses','lenders').\n"
 "- Map a brand/subsidiary to its listed NSE PARENT only when unambiguous. Do NOT guess; unsure -> listed=false.\n"
 "- Include 'issuer': official company name.\n"
 'Return ONLY JSON {"results":[{"i":<int>,"listed":<bool>,"ticker":<str|null>,"issuer":<str|null>,"confidence":"high|med|low"}]}')
resolved=json.load(open(G/"nifty50_resolved.json")); rset=set(resolved)
docm={json.loads(l)["doc_id"]:json.loads(l).get("headline","") for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); ctx={}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); e=j.get("effect_entity",""); 
    if e.split("__")[-1] not in ("company","bank") or j.get("effect_dir") not in DIRSET or e in rset: continue
    freq[e]+=1; ctx.setdefault(e, docm.get(j.get("doc_id"),""))
targets=sorted(freq,key=lambda e:-freq[e])[:LIMIT]
ents=[{"eid":e,"name":e.split("__")[0].replace("_"," "),"type":e.split("__")[-1],"ctx":ctx.get(e,"")} for e in targets]
print(f"{len(freq)} unresolved company/bank entities; proposing NSE tickers for top {len(ents)}",flush=True)
def groq(batch):
    lines=[f'{i+1}. {e["name"]} [{e["type"]}] — ctx: {e["ctx"][:80]}' for i,e in enumerate(batch)]
    body={"model":MODEL,"temperature":0,"max_tokens":2400,"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":SYSTEM},{"role":"user","content":"Resolve each:\n"+"\n".join(lines)}]}
    for att in range(5):
        try:
            r=httpx.post(GROQ,json=body,headers={"Authorization":f"Bearer {KEY}"},timeout=90)
            if r.status_code==429: time.sleep(3*(att+1)); continue
            if r.status_code!=200: time.sleep(2*(att+1)); continue
            t=r.json()["choices"][0]["message"]["content"]; a,b=t.find("{"),t.rfind("}")
            return json.loads(t[a:b+1]).get("results",[])
        except Exception: time.sleep(2*(att+1))
    return []
proposals=[]
for i in range(0,len(ents),BATCH):
    batch=ents[i:i+BATCH]; res=groq(batch); by={r.get("i"):r for r in res if isinstance(r,dict)}
    for k,e in enumerate(batch):
        r=by.get(k+1)
        if r and r.get("listed") and r.get("ticker"): proposals.append((e,r["ticker"].strip().upper(),(r.get("issuer") or "").strip(),r.get("confidence","low")))
    print(f"  batch {i//BATCH+1}/{(len(ents)+BATCH-1)//BATCH}: cumulative proposals {len(proposals)}",flush=True)
sess=httpx.Client(headers={"User-Agent":"Mozilla/5.0"})
def yname(t):
    try:
        r=sess.get("https://query2.finance.yahoo.com/v1/finance/search",params={"q":t,"quotesCount":6,"newsCount":0},timeout=15)
        for x in (r.json().get("quotes",[]) if r.status_code==200 else []):
            if x.get("symbol","").upper()==t.upper(): return x.get("shortname") or x.get("longname") or ""
    except Exception: pass
    time.sleep(0.15); return ""
accepted={}; rej=0
for e,tk,iss,conf in proposals:
    base=tk[:-3] if tk.endswith(".NS") else tk; nse=base+".NS"
    if e["eid"] in resolved or nse in set(resolved.values()): continue
    r=logret(yahoo(nse,cache,p1,p2))
    if len(r)<200: rej+=1; continue                                  # hard gate: real 2021 price history
    yn=yname(nse); sim=max(difflib.SequenceMatcher(None,iss.lower(),yn.lower()).ratio() if yn else 0,
                            difflib.SequenceMatcher(None,e["name"].lower(),yn.lower()).ratio() if yn else 0)
    if not yn or (sim<0.4 and conf!="high"): rej+=1; continue        # name-match gate
    accepted[e["eid"]]=nse
    print(f"  ✓ {e['name'][:30]:30} -> {nse:14} yahoo='{yn[:24]}' sim{sim:.2f} {conf}",flush=True)
print(f"\nproposed {len(proposals)} | ACCEPTED {len(accepted)} new | rejected {rej}")
merged=dict(resolved); merged.update(accepted)
import shutil; shutil.copy(G/"nifty50_resolved.json", G/"nifty50_resolved.bak46.json")
json.dump(merged, open(G/"nifty50_resolved.json","w"), indent=1)
print(f"universe: {len(set(resolved.values()))} -> {len(set(merged.values()))} distinct NSE tickers  (backup nifty50_resolved.bak46.json)")
