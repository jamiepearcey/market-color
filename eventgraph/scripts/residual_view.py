# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""OBSERVE THE RESIDUAL: what's left after macro + sector are removed?
A) GS-MS residual co-movement over time — episodic (missing factor) or flat noise?
B) Universe residual correlation structure — mean off-diagonal + absorption ratio (top eigenvalue
   share). If ~0 / flat, factors captured the systematic part; if >0 / concentrated, a factor is missing.
C) Is the leftover structure NEWS-shaped? mean residual corr for news-linked vs unlinked pairs."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=90
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
freq=collections.Counter(); cause_m=collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cause_m[m][c].add(sym[e])
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
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]  # drop all-NaN macro (crypto)
def design(days):
    X=np.column_stack([[Fmap[d][i] for d in days] for i in mcol]+[[spdr[e][d] for d in days] for e in SPDR])
    X=np.nan_to_num(X-np.nanmean(X,0)); return np.column_stack([np.ones(len(days)),X])
# residualize each name on macro+sector over its own days
R2={}
for s in uni:
    days=[d for d in alld if d in R[s]]
    if len(days)<200: continue
    y=np.array([R[s][d] for d in days]); X=design(days); b,*_=np.linalg.lstsq(X,y,rcond=None)
    R2[s]=dict(zip(days,y-X@b))
def corr(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<40: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
# A) GS-MS residual over time
print("=== A) GS-MS residual co-movement over time (after macro+sector removed) ===")
if "GS" in R2 and "MS" in R2:
    a,b=R2["GS"],R2["MS"]; days=sorted(set(a)&set(b))
    win=[]; 
    for i in range(63,len(days),10):
        c=corr(a,b,days[i-63:i]); 
        if c is not None: win.append((days[i],c))
    cs=[c for _,c in win]
    print(f"  full-period residual corr(GS,MS) = {corr(a,b,days):+.3f}")
    print(f"  rolling 63d residual corr: mean {np.mean(cs):+.3f}  sd {np.std(cs):.3f}  min {min(cs):+.3f}  max {max(cs):+.3f}")
    hi=sorted(win,key=lambda x:-x[1])[:3]; lo=sorted(win,key=lambda x:x[1])[:2]
    print(f"  peaks: "+", ".join(f"{d[:7]} {c:+.2f}" for d,c in hi)+" | troughs: "+", ".join(f"{d[:7]} {c:+.2f}" for d,c in lo))
# B) universe residual correlation structure
print("\n=== B) universe residual structure — missing factor or idiosyncratic noise? ===")
names=sorted(R2); rc=[]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        c=corr(R2[names[i]],R2[names[j]],alld)
        if c is not None: rc.append(c)
rc=np.array(rc)
# absorption ratio of residuals (top eigenvalue share) vs raw
def absorption(dct):
    common=alld; Mx=np.array([[dct[s].get(d,np.nan) for d in common] for s in names])
    ok=~np.isnan(Mx).any(0); Mx=Mx[:,ok]
    Mx=Mx-Mx.mean(1,keepdims=True); C=np.corrcoef(Mx); C=np.nan_to_num(C)
    ev=np.linalg.eigvalsh(C); ev=ev[::-1]; return ev[0]/ev.sum(), ev[:3]/ev.sum()
rawR={s:{d:R[s][d] for d in R[s] if d in alld} for s in names}
ar_raw,_=absorption(rawR); ar_res,top3=absorption(R2)
print(f"  mean pairwise residual corr = {rc.mean():+.3f}  (sd {rc.std():.2f}; raw ~0.4-0.5 typical)")
print(f"  absorption ratio (top eigenvalue share): RAW {ar_raw:.2f} -> RESIDUAL {ar_res:.2f}  (top3 resid {', '.join(f'{x:.2f}' for x in top3)})")
print(f"  => {'FLAT/idiosyncratic: factors captured the systematic part' if ar_res<0.15 and abs(rc.mean())<0.08 else 'STRUCTURE REMAINS: a common factor is still in the residual'}")
# C) is the leftover structure news-shaped?
print("\n=== C) is the residual co-movement NEWS-shaped? (linked vs unlinked residual corr) ===")
linked=set()
for m in cause_m:
    for c,ss in cause_m[m].items():
        ss=sorted(x for x in ss if x in R2)
        if c.split("__")[-1] in {"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}: continue
        for x in range(len(ss)):
            for y in range(x+1,len(ss)): linked.add(tuple(sorted((ss[x],ss[y]))))
lk=[]; un=[]
for i in range(len(names)):
    for j in range(i+1,len(names)):
        p=tuple(sorted((names[i],names[j]))); c=corr(R2[names[i]],R2[names[j]],alld)
        if c is None: continue
        (lk if p in linked else un).append(c)
print(f"  linked pairs (specific news driver): mean residual corr {np.mean(lk):+.3f}  n={len(lk)}")
print(f"  unlinked pairs:                      mean residual corr {np.mean(un):+.3f}  n={len(un)}")
d=np.mean(lk)-np.mean(un); se=(np.var(lk)/len(lk)+np.var(un)/len(un))**0.5
print(f"  difference {d:+.3f}  t {d/se:+.1f}  => {'news structure SURVIVES into the residual (beyond macro+sector)' if d/se>2 else 'no significant news structure left in the residual'}")
