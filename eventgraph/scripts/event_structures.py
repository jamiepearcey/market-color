# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""Precompute EVENT-LINKAGE structures for visualization. Each = a (month, specific news driver)
that binds >=2 priceable names; we measure the cluster's SECTOR-ORTHOGONAL residual co-movement
(after macro+sector removed) over the event window. Output events.json for the timeline app."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=150
SN={"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLV":"Health Care","XLI":"Industrials","XLY":"Cons Disc","XLP":"Cons Staples","XLU":"Utilities","XLB":"Materials"}
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}; name={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
        sym[j["entity_id"]]=j["symbol"]; name.setdefault(j["symbol"], j["entity_id"].split("__")[0].replace("_"," ").title())
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); cause_m=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))); dlabel={}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cause_m[m][c][sym[e]]+=DIRV[d]; dlabel[c]=(c.split("__")[0].replace("_"," ").title(),c.split("__")[-1])
years={"2010","2011","2012"}
def ret(t): return logret(yahoo(t,cache,p1,p2))
spdr={e:ret(e) for e in SPDR}
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=N_UNIV: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in years)>400: uni.append(s)
R={s:ret(s) for s in uni}
alld=sorted({d for s in uni for d in R[s] if d[:4] in years and d in Fmap and all(d in spdr[e] for e in SPDR)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
def design(days):
    X=np.column_stack([[Fmap[d][i] for d in days] for i in mcol]+[[spdr[e][d] for d in days] for e in SPDR])
    X=np.nan_to_num(X-np.nanmean(X,0)); return np.column_stack([np.ones(len(days)),X])
R2={}
for s in uni:
    days=[d for d in alld if d in R[s]]
    if len(days)<200: continue
    y=np.array([R[s][d] for d in days]); X=design(days); b,*_=np.linalg.lstsq(X,y,rcond=None); R2[s]=dict(zip(days,y-X@b))
def sec_of(s):
    r=R[s]; best=("?",-1)
    for e in SPDR:
        cm=[d for d in alld if d in r and d in spdr[e]]
        if len(cm)<40: continue
        x=np.array([r[d] for d in cm]);y=np.array([spdr[e][d] for d in cm])
        if x.std() and y.std():
            c=abs(np.corrcoef(x,y)[0,1])
            if c>best[1]: best=(SN[e],c)
    return best[0]
SEC={s:sec_of(s) for s in R2}
mo=[f"{y}-{m:02d}" for y in (2010,2011,2012) for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def rescorr(a,b,days):
    cm=[d for d in days if d in R2[a] and d in R2[b]]
    if len(cm)<20: return None
    x=np.array([R2[a][d] for d in cm]);y=np.array([R2[b][d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
events=[]
for m in mo:
    i=idx[m]
    win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]  # event window m-1..m+1
    for c,sgn in cause_m[m].items():
        if dlabel[c][1] in MACRO_T: continue
        ns=sorted(s for s in sgn if s in R2 and sgn[s]!=0)
        if len(ns)<2: continue
        pairs=[]; vals=[]
        for x in range(len(ns)):
            for y in range(x+1,len(ns)):
                rc=rescorr(ns[x],ns[y],win)
                if rc is not None: pairs.append([x,y,round(rc,3)]); vals.append(rc)
        if not vals: continue
        events.append({"month":m,"label":dlabel[c][0],"type":dlabel[c][1],"size":len(ns),
                       "res":round(float(np.mean(vals)),3),"resmax":round(float(np.max(vals)),3),
                       "names":[{"t":s,"n":name.get(s,s),"sec":SEC[s],"dir":int(np.sign(sgn[s]))} for s in ns],
                       "pairs":pairs})
events.sort(key=lambda e:(e["month"],-e["res"]))
Path("../data/eg_runs/eg100k_graph/events.json").write_text(json.dumps({"events":events,"sectors":list(SN.values())}))
print(f"-> events.json  {len(events)} event-cluster structures over {len(mo)} months")
strong=[e for e in events if e["res"]>=0.15 and e["size"]>=2]
print(f"   {len(strong)} with residual co-movement >=0.15")
print("   top by residual strength:")
for e in sorted(events,key=lambda e:-e["res"])[:10]:
    print(f"     {e['month']}  {e['label'][:34]:34} [{e['type'][:9]:9}] res {e['res']:+.2f} n{e['size']}  {'/'.join(sorted(set(x['sec'][:4] for x in e['names'])))}")
