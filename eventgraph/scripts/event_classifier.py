# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""EVENT CLASSIFIER (top of the hierarchy): tag each event with mechanism-type + domain + which
decomposition layer DOMINATES (mechanism / factor / regime) + whether direction is factor-determined
(recoverable) or idiosyncratic (not). This is the routing key. Validate against per-type exposure IC."""
import os, json, collections, sys, time
from pathlib import Path
import numpy as np, httpx
G=Path("../data/eg_runs/eg100k_graph")
EVs=json.loads((G/"exposure_landscape_emb.json").read_text())["events"]
GROQ="https://api.groq.com/openai/v1/chat/completions"; KEY=os.environ["GROQ_API_KEY"]; MODEL="openai/gpt-oss-120b"
SYS=("You are an equity event classifier. For each financial-news event, output:\n"
 "- mech: the dominant mechanism — one of: regulatory, supply_shock, macro_surprise, policy, "
 "company_specific, geopolitical, credit_rating, legal.\n"
 "- domain: primary affected domain — one of: financials, manufacturing, technology, energy, "
 "consumer, materials, healthcare, broad_market.\n"
 "- layer: which decomposition layer carries the exposure — 'mechanism' (specific named-set/relations), "
 "'factor' (a common risk factor: rates, growth, oil), or 'regime' (broad risk-on/off).\n"
 "- dir: is the SIGN of firms' reactions set by a common FACTOR (so it's recoverable & can differ by "
 "firm's beta) = 'factor_signed', or idiosyncratic/uniform = 'idiosyncratic'.\n"
 'Return ONLY JSON {"r":[{"i":<int>,"mech":..,"domain":..,"layer":..,"dir":..}]}')
def clf(batch):
    lines=[f'{i+1}. {e["label"]} [{e["type"]}] — {(e["facts"][0]["q"][:90] if e.get("facts") else "")}' for i,e in enumerate(batch)]
    body={"model":MODEL,"temperature":0,"max_tokens":2600,"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":SYS},{"role":"user","content":"Classify:\n"+"\n".join(lines)}]}
    for a in range(4):
        try:
            r=httpx.post(GROQ,json=body,headers={"Authorization":f"Bearer {KEY}"},timeout=90)
            if r.status_code!=200: time.sleep(2*(a+1)); continue
            t=r.json()["choices"][0]["message"]["content"]; x,y=t.find("{"),t.rfind("}")
            return json.loads(t[x:y+1]).get("r",[])
        except Exception: time.sleep(2*(a+1))
    return []
C={}
for i in range(0,len(EVs),22):
    b=EVs[i:i+22]; res=clf(b); by={r.get("i"):r for r in res if isinstance(r,dict)}
    for k,e in enumerate(b):
        r=by.get(k+1) or {}
        C[i+k]={"mech":r.get("mech","?"),"domain":r.get("domain","?"),"layer":r.get("layer","?"),"dir":r.get("dir","?")}
# save routing key
route={f"{EVs[i]['month']}|{EVs[i]['label']}":C[i] for i in range(len(EVs))}
(G/"event_routing.json").write_text(json.dumps(route))
def ic(e):
    xs=[(r["exp"],r["real"]) for r in e["rows"] if not r["m"] and r["real"] is not None]
    if len(xs)<8: return None
    x=np.array([a for a,_ in xs]);y=np.array([b for _,b in xs]);x=x-x.mean();y=y-y.mean()
    return None if x.std()==0 or y.std()==0 else float((x@y)/(np.sqrt(x@x)*np.sqrt(y@y)))
print("=== sample classifications ===")
for want in ["Almunia","Project Merlin","Growth Forecast","Earthquake","Dodd","Bmw"]:
    for i,e in enumerate(EVs):
        if want.lower() in e["label"].lower():
            c=C[i]; print(f"  {e['label'][:34]:34} -> {c['mech']} + {c['domain']} | layer={c['layer']} dir={c['dir']}"); break
byM=collections.defaultdict(list); byL=collections.defaultdict(list)
for i,e in enumerate(EVs):
    v=ic(e)
    if v is not None: byM[C[i]["mech"]].append(v); byL[C[i]["layer"]].append(v)
print("\n=== embedding-exposure IC by MECHANISM (where embedding is right vs wrong channel) ===")
for m,a in sorted(byM.items(),key=lambda x:-np.mean(x[1])): print(f"  {m:16} IC {np.mean(a):+.3f}  n{len(a)}")
print("\n=== IC by dominant LAYER (embedding suits 'mechanism' layer; 'factor'/'regime' need other channels) ===")
for m,a in sorted(byL.items(),key=lambda x:-np.mean(x[1])): print(f"  {m:12} IC {np.mean(a):+.3f}  n{len(a)}")
print(f"\n-> event_routing.json written (routing key per event)")
