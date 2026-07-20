# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
import json, collections, re, sys, time
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
from pathlib import Path
G="../data/eg_runs/eg100k_graph"
EM={"china","brazil","india","russia","south korea","korea","indonesia","turkey","mexico","poland",
    "hungary","south africa","thailand","malaysia","chile","colombia","egypt","saudi arabia","taiwan","philippines","peru"}
SUFF={"china":[".SS",".SZ",".HK"],"brazil":[".SA"],"india":[".NS",".BO"],"russia":[".ME"],"korea":[".KS",".KQ"],
      "south korea":[".KS"],"indonesia":[".JK"],"turkey":[".IS"],"mexico":[".MX"],"poland":[".WA"],"hungary":[".BD"],
      "south africa":[".JO"],"thailand":[".BK"],"malaysia":[".KL"],"chile":[".SN"],"colombia":[".CL"],"saudi arabia":[".SR"],
      "taiwan":[".TW"],"egypt":[".CA"],"philippines":[".PS"],"peru":[".LM"]}
# EM company -> (suggested_ticker, country) from extraction
em_ent={}
for l in open(f"{G}/extractions.jsonl"):
    try: ex=json.loads(l).get("ex")
    except: continue
    if not isinstance(ex,dict): continue
    for e in ex.get("entities") or []:
        cty=(e.get("country") or "").lower(); nm=e.get("name")
        if cty in EM and nm and e.get("type") in ("company","bank"):
            slug=re.sub(r'[^a-z0-9]+','_',nm.lower()).strip('_')+"__"+e.get("type")
            if slug not in em_ent: em_ent[slug]=(e.get("suggested_ticker"), cty)
freq=collections.Counter()
for l in open(f"{G}/lake/causal_event_edge.jsonl"):
    e=json.loads(l).get("effect_entity")
    if e: freq[e]+=1
em=[(freq[eid],eid,tk,cty) for eid,(tk,cty) in em_ent.items() if freq[eid]>=3]
em.sort(reverse=True)
print(f"EM-country company effect-entities (>=3 edges): {len(em)}")
cache=Path(f"{G}/prices"); cache.mkdir(exist_ok=True); p1=1230768000; p2=1420070400
priceable=[]
for f,eid,tk,cty in em:
    variants=[]
    if tk:
        tks=str(tk).replace(".","")
        if "." in str(tk): variants=[tk]
        else: variants=[tks+s for s in SUFF.get(cty,[".SS"])]+[tks]  # US ADR fallback
    for v in variants:
        try: r=logret(yahoo(v,cache,p1,p2))
        except: continue
        if len(r)>250: priceable.append((f,eid.split("__")[0],cty,v,len(r))); break
    time.sleep(0.03)
print(f"PRICEABLE EM companies (>250d): {len(priceable)}")
cc=collections.Counter(x[2] for x in priceable)
print("by country:", dict(cc.most_common()))
for f,n,c,v,nd in sorted(priceable,reverse=True)[:30]: print(f"  {f:4} {n[:28]:28} {c[:10]:10} {v:12} {nd}d")
json.dump([{"eid":n,"country":c,"symbol":v,"freq":f} for f,n,c,v,nd in priceable], open("/tmp/em_universe.json","w"))
