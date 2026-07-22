# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""Does the exposure model's accuracy VARY by EVENT TYPE? (tests whether we need type-routing / independent
channels). LLM-classify each event into 6 types; compute exposure-vs-realized IC per event (from the
landscape JSON, no re-run); aggregate by type. High variation -> a universal embedding is wrong; route by type."""
import os, json, collections, math, sys, time
from pathlib import Path
import numpy as np, httpx
G=Path("../data/eg_runs/eg100k_graph")
D=json.loads((G/"exposure_landscape_emb.json").read_text())["events"]
GROQ="https://api.groq.com/openai/v1/chat/completions"; KEY=os.environ["GROQ_API_KEY"]; MODEL="openai/gpt-oss-120b"
TYPES=["company_specific","supply_chain","regulatory","macro_surprise","policy","geopolitical"]
SYS=("Classify each financial-news EVENT into ONE exposure type:\n"
 "company_specific (one firm's earnings/M&A/product/litigation), supply_chain (supplier/customer/input linkage), "
 "regulatory (a regulator/antitrust/rule/probe), macro_surprise (growth/inflation/rates/demand data or broad economy), "
 "policy (government/central-bank/fiscal/trade policy), geopolitical (country/conflict/sanctions/commodity-nation).\n"
 'Return ONLY JSON {"r":[{"i":<int>,"t":"<type>"}]}')
def clf(batch):
    lines=[f'{i+1}. {e["label"]} [{e["type"]}] — {(e["facts"][0]["q"][:80] if e.get("facts") else "")}' for i,e in enumerate(batch)]
    body={"model":MODEL,"temperature":0,"max_tokens":2000,"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":SYS},{"role":"user","content":"Classify:\n"+"\n".join(lines)}]}
    for a in range(4):
        try:
            r=httpx.post(GROQ,json=body,headers={"Authorization":f"Bearer {KEY}"},timeout=90)
            if r.status_code!=200: time.sleep(2*(a+1)); continue
            t=r.json()["choices"][0]["message"]["content"]; x,y=t.find("{"),t.rfind("}")
            return json.loads(t[x:y+1]).get("r",[])
        except Exception: time.sleep(2*(a+1))
    return []
cat={}
for i in range(0,len(D),40):
    b=D[i:i+40]; res=clf(b); by={r.get("i"):r.get("t") for r in res if isinstance(r,dict)}
    for k,e in enumerate(b): cat[i+k]=by.get(k+1,"?")
def ic(e):
    xs=[(r["exp"],r["real"]) for r in e["rows"] if not r["m"] and r["real"] is not None]
    if len(xs)<8: return None
    x=np.array([a for a,_ in xs]); y=np.array([b for _,b in xs]); x=x-x.mean(); y=y-y.mean()
    if x.std()==0 or y.std()==0: return None
    return float((x@y)/(np.sqrt(x@x)*np.sqrt(y@y)))
byT=collections.defaultdict(list); breadth=collections.defaultdict(list)
for i,e in enumerate(D):
    v=ic(e)
    if v is not None: byT[cat[i]].append(v)
    exps=sorted((r["exp"] for r in e["rows"]),reverse=True)
    if len(exps)>=10: breadth[cat[i]].append(exps[9]/max(exps[0],1e-6))  # top10/top1 = how broad
print("=== exposure-model accuracy (exp vs realized IC) BY EVENT TYPE ===")
print(f"  {'type':18} {'mean IC':>8} {'n':>4} {'breadth(top10/1)':>17}")
for t in TYPES+["?"]:
    a=byT.get(t,[])
    if not a: continue
    br=np.mean(breadth.get(t,[0]))
    print(f"  {t:18} {np.mean(a):+8.3f} {len(a):4}  {br:16.2f}")
allic=[v for a in byT.values() for v in a]
print(f"\n  overall mean IC {np.mean(allic):+.3f} (n{len(allic)})")
print("  => if IC varies a lot by type (macro/geopolitical low, company/regulatory high), a single embedding")
print("     is the wrong channel for some types -> route by type (your point). breadth: higher=exposure more diffuse.")
