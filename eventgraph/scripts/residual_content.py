# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""WHAT INFORMATION does the residual contain? After macro+sector removed, the news-linked residual
co-movement (+0.062, t6.2) is characterized: (A) is it concentrated in CROSS-SECTOR pairs (genuinely
non-sector links)? (B) which specific NEWS DRIVERS carry it, and what kind are they? (C) relation types."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=90
SN={"XLF":"Fin","XLK":"Tech","XLE":"Enrgy","XLV":"Hlth","XLI":"Indl","XLY":"Disc","XLP":"Stpl","XLU":"Util","XLB":"Matl"}
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); cause_m=collections.defaultdict(lambda: collections.defaultdict(set)); dlabel={}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cause_m[m][c].add(sym[e]); dlabel[c]=(c.split("__")[0].replace("_"," ").title(),c.split("__")[-1])
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
names=sorted(R2)
# sector via residual... no, via raw SPDR corr
def sec_of(s):
    r=R[s]; best=("?",-1)
    for e in SPDR:
        cm=[d for d in alld if d in r and d in spdr[e]]
        x=np.array([r[d] for d in cm]);y=np.array([spdr[e][d] for d in cm])
        if x.std() and y.std():
            c=abs(np.corrcoef(x,y)[0,1])
            if c>best[1]: best=(SN[e],c)
    return best[0]
SEC={s:sec_of(s) for s in names}
def corr(a,b):
    cm=[d for d in alld if d in a and d in b]
    if len(cm)<40: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
# linked pairs (specific drivers) with the driver that links them
pair_drv=collections.defaultdict(set)
for m in cause_m:
    for c,ss in cause_m[m].items():
        if dlabel[c][1] in MACRO_T: continue
        ss=sorted(x for x in ss if x in R2)
        for i in range(len(ss)):
            for j in range(i+1,len(ss)): pair_drv[tuple(sorted((ss[i],ss[j])))].add(c)
linked=set(pair_drv)
# A) cross-sector vs same-sector residual structure
buckets={("linked","same"):[],("linked","cross"):[],("unlinked","same"):[],("unlinked","cross"):[]}
rescorr={}
for i in range(len(names)):
    for j in range(i+1,len(names)):
        A,B=names[i],names[j]; c=corr(R2[A],R2[B])
        if c is None: continue
        rescorr[(A,B)]=c
        lk="linked" if (A,B) in linked else "unlinked"; xs="same" if SEC[A]==SEC[B] else "cross"
        buckets[(lk,xs)].append(c)
print("=== A) residual co-movement: linked vs unlinked, SAME-sector vs CROSS-sector ===")
for xs in ("same","cross"):
    l=buckets[("linked",xs)]; u=buckets[("unlinked",xs)]
    print(f"  {xs:5}-sector: linked {np.mean(l):+.3f} (n{len(l)})  unlinked {np.mean(u):+.3f} (n{len(u)})  excess {np.mean(l)-np.mean(u):+.3f}")
print("  => where does the news residual structure live — same or cross sector?")
# B) which drivers carry the residual co-movement
drv_res=collections.defaultdict(list)
for (A,B),cs in pair_drv.items():
    if (A,B) in rescorr:
        for c in cs: drv_res[c].append(((A,B),rescorr[(A,B)]))
rows=[(np.mean([v for _,v in ps]),len(ps),c) for c,ps in drv_res.items() if len(ps)>=3]
print("\n=== B) top news drivers by RESIDUAL co-movement (sector-orthogonal channels they carry) ===")
for mres,n,c in sorted(rows,reverse=True)[:14]:
    ex=max(drv_res[c],key=lambda x:x[1])[0]; xs="cross" if SEC[ex[0]]!=SEC[ex[1]] else "same"
    print(f"  {dlabel[c][0][:30]:30} [{dlabel[c][1][:11]:11}] res-corr {mres:+.2f} n{n:2}  e.g. {ex[0]}-{ex[1]} ({SEC[ex[0]]}/{SEC[ex[1]]},{xs})")
