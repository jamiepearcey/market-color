# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""EM vs DM persistence test. Industry thesis: EM info environments are less efficient,
so a news-causal graph should be MORE valuable there. Mirror the core persistence test
(elevated abnormal-corr pairs: do news-linked persist > unlinked?) on the 59 priceable
EM companies, neutralized against EM-appropriate factors (EEM/ACWI/UUP/GLD residuals),
and compare to the established DM(US) baseline (unlinked 20% / linked 48%)."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
EMU={j["symbol"]:j for j in json.load(open("/tmp/em_universe.json"))}
eid2sym={j["eid"]+"__company":j["symbol"] for j in EMU.values()}
eid2sym.update({j["eid"]+"__bank":j["symbol"] for j in EMU.values()})
PROX=["EEM","ACWI","UUP","GLD"]
p1=1230768000;p2=1420070400
def fetch(names): return {s:logret(yahoo(s,cache,p1,p2)) for s in names}
prox={p:v for p,v in fetch(PROX).items() if len(v)>250}
px={s:v for s,v in fetch(list(EMU)).items() if len(v)>250}
# neutralize
pdays=sorted(set.intersection(*[set(prox[p]) for p in prox]))
ar={}
for s,r in px.items():
    days=[d for d in sorted(r) if d in set(pdays)]
    if len(days)<250: continue
    y=np.array([r[d] for d in days]); X=np.column_stack([np.ones(len(days))]+[[prox[p][d] for d in days] for p in prox])
    b,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@b; ar[s]=dict(zip(days,res))
print(f"{len(ar)} EM names with neutralized series")
# shared-cause links among EM names, by month
docdate={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
cause_m=collections.defaultdict(lambda:collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docdate.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity")
    if not m or m[:4] not in {"2010","2011","2012"} or not c: continue
    s=eid2sym.get(e)
    if s and s in ar: cause_m[m][c].add(s)
def links_in(months):
    out=set()
    for m in months:
        for c,ss in cause_m.get(m,{}).items():
            ss=sorted(ss)
            for i in range(len(ss)):
                for k in range(i+1,len(ss)): out.add((ss[i],ss[k]))
    return out
def wcorr(a,b,days):
    common=[d for d in days if d in a and d in b]
    if len(common)<12: return None
    x=np.array([a[d] for d in common]);y=np.array([b[d] for d in common])
    if x.std()==0 or y.std()==0: return None
    return float(np.corrcoef(x,y)[0,1])
months=[f"{y}-{m:02d}" for y in (2010,2011,2012) for m in range(1,13)]; idx={m:i for i,m in enumerate(months)}
alldays=sorted(set().union(*[set(v) for v in ar.values()]))
d2m=lambda d:d[:7]
names=sorted(ar)
rows=[]
for m in months:
    i=idx[m]
    if i<3 or i+2>=len(months): continue
    wm=months[i-3:i+1]; nm=months[i+1:i+3]
    td=[d for d in alldays if d2m(d) in wm]; fd=[d for d in alldays if d2m(d) in nm]
    lk=links_in(wm)
    for x in range(len(names)):
        for k in range(x+1,len(names)):
            p=(names[x],names[k])
            tc=wcorr(ar[p[0]],ar[p[1]],td); fc=wcorr(ar[p[0]],ar[p[1]],fd)
            if tc is None or fc is None: continue
            rows.append((tc,fc,1 if p in lk else 0))
R=np.array(rows)
print(f"EM pair-months: {len(R)} | linked {int(R[:,2].sum())}")
hi=R[R[:,0]>0.3]
for lbl,sub in [("unlinked",hi[hi[:,2]==0]),("news-LINKED",hi[hi[:,2]==1])]:
    if len(sub): print(f"  EM {lbl:12} persists {np.mean(sub[:,1]>=sub[:,0]):.0%}  next {sub[:,1].mean():+.3f} (from {sub[:,0].mean():+.3f})  n={len(sub)}")
print("  DM(US) baseline: unlinked 20% / linked 48%")
