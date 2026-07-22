# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""ROUTED DIRECTION test. Claim (F27): direction is recoverable for factor_signed events (sign set by a
common factor) but not idiosyncratic ones — averaging both gave F24's chance. Test: does the event's
shock direction D_E propagate to NON-mentioned firms (market-relative, drift-removed), split by the
routing tag? factor_signed should beat chance; idiosyncratic should sit at chance."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import SKIP, yahoo, logret
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2009,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2013,6,1,tzinfo=dt.UTC).timestamp())
EV=json.loads((G/"exposure_landscape_emb.json").read_text())["events"]
ROUTE=json.loads((G/"event_routing.json").read_text())
tick=sorted({r["t"] for e in EV for r in e["rows"]})
R={s:logret(yahoo(s,cache,p1,p2)) for s in tick}; R={s:v for s,v in R.items() if len(v)>200}
alld=sorted(set().union(*[set(R[s]) for s in R]))
# market-relative, drift-removed returns: subtract cross-sectional mean each day, then subtract each firm's own mean
day_mean={d:np.mean([R[s][d] for s in R if d in R[s]]) for d in alld}
Rel={s:{d:R[s][d]-day_mean[d] for d in R[s]} for s in R}
mu={s:np.mean(list(Rel[s].values())) for s in Rel}
Rel={s:{d:v-mu[s] for d,v in Rel[s].items()} for s in Rel}
mo=[f"{y}-{m:02d}" for y in (2010,2011,2012) for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def wret(s,win): 
    v=[Rel[s][d] for d in win if s in Rel and d in Rel[s]]
    return float(np.sum(v)) if len(v)>=8 else None
buck=collections.defaultdict(lambda:[0,0]); mag=collections.defaultdict(list)
for e in EV:
    key=f"{e['month']}|{e['label']}"; tag=ROUTE.get(key,{}).get("dir","?"); lay=ROUTE.get(key,{}).get("layer","?")
    i=idx.get(e["month"]); 
    if i is None: continue
    win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    named=[s for s in e["named"] if s in Rel]
    if len(named)<2: continue
    D=[wret(s,win) for s in named]; D=[x for x in D if x is not None]
    if not D: continue
    D_E=np.sign(np.mean(D)) or 1.0                        # event shock direction (market-relative)
    for r in e["rows"]:
        s=r["t"]
        if r["m"] or s not in Rel: continue
        w=wret(s,win)
        if w is None: continue
        agree = np.sign(w)==D_E
        buck[tag][0]+=int(agree); buck[tag][1]+=1
        mag[tag].append(D_E*w)                            # signed magnitude alignment
print("=== does event direction D_E propagate to NON-mentioned firms? (market-relative) ===")
print(f"  {'routing tag':16} {'hit-rate':>9} {'n':>6} {'mean signed align':>18}")
for tag in ["factor_signed","idiosyncratic","?"]:
    h,n=buck[tag]
    if n<20: continue
    print(f"  {tag:16} {h/n:9.1%} {n:6}  {np.mean(mag[tag]):+18.4f}")
print("\n  (F24 pooled direction was 51.4%. If factor_signed > idiosyncratic, direction routes as F27 predicts.)")
# also split by dominant layer
lb=collections.defaultdict(lambda:[0,0])
for e in EV:
    key=f"{e['month']}|{e['label']}"; lay=ROUTE.get(key,{}).get("layer","?")
    i=idx.get(e["month"]); 
    if i is None: continue
    win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]; named=[s for s in e["named"] if s in Rel]
    if len(named)<2: continue
    D=[wret(s,win) for s in named]; D=[x for x in D if x is not None]
    if not D: continue
    D_E=np.sign(np.mean(D)) or 1.0
    for r in e["rows"]:
        if r["m"] or r["t"] not in Rel: continue
        w=wret(r["t"],win)
        if w is None: continue
        lb[lay][0]+=int(np.sign(w)==D_E); lb[lay][1]+=1
print("=== by dominant LAYER ===")
for lay in ["factor","regime","mechanism","?"]:
    h,n=lb[lay]
    if n>=20: print(f"  {lay:12} hit {h/n:.1%}  n{n}")
