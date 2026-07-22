# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""ORDER-INDEPENDENT (Shapley) factor decomposition of a pair's correlation.
Blocks: MACRO (9), SECTOR (9 GICS SPDRs), NEWS (the pair's named driver baskets). For each of the
7 subsets, v(S) = cov of A,B projected onto that subset's factor span (order-free). Shapley value of
a block = its average marginal v() contribution over all orderings. Sums (by efficiency) to the total
explained covariance; residual = idiosyncratic. Removes the 'in this order' asterisk from F17."""
import json, collections, csv, sys, itertools, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
PAIR=(sys.argv[1] if len(sys.argv)>1 else "GS", sys.argv[2] if len(sys.argv)>2 else "MS"); A,B=PAIR
KDRV=6; SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
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
drv_net=collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    drv_net[c][sym[e]]+=DIRV[d]
shared=[c for c in drv_net if A in drv_net[c] and B in drv_net[c] and drv_net[c][A]!=0 and drv_net[c][B]!=0]
shared=[c for c in shared if c.split("__")[-1] not in MACRO_T]
shared=sorted(shared,key=lambda c:-len(drv_net[c]))[:KDRV]
def ret(t): return logret(yahoo(t,cache,p1,p2))
rA,rB=ret(A),ret(B); spdr={e:ret(e) for e in SPDR}; years={"2010","2011","2012"}
def basket_series(c):
    names=[s for s in drv_net[c] if s not in (A,B) and drv_net[c][s]!=0]; rr={s:ret(s) for s in names}
    sgn={s:np.sign(drv_net[c][s]) for s in names}; days=set().union(*[set(rr[s]) for s in names]) if names else set(); out={}
    for d in days:
        v=[sgn[s]*rr[s][d] for s in names if d in rr[s]]
        if len(v)>=2: out[d]=float(np.mean(v))
    return out
baskets=[basket_series(c) for c in shared]
def okday(d): return d[:4] in years and d in rA and d in rB and d in Fmap and all(d in spdr[e] for e in SPDR)
days=sorted(d for d in rA if okday(d)); N=len(days)
def mat(series_list):
    X=np.array([[s.get(d,np.nan) if isinstance(s,dict) else s[d] for d in days] for s in series_list],float).T
    keep=[j for j in range(X.shape[1]) if np.isfinite(X[:,j]).sum()>=N*0.5]; X=X[:,keep]
    for j in range(X.shape[1]):
        mm=~np.isfinite(X[:,j]); 
        if mm.any(): X[mm,j]=np.nanmean(X[~mm,j])
    return X-X.mean(0)
Mblk=mat([{d:Fmap[d][i] for d in days} for i in range(len(FN))])
Sblk=mat([spdr[e] for e in SPDR])
Nblk=mat(baskets) if baskets else np.zeros((N,0))
BLK={"MACRO":Mblk,"SECTOR":Sblk,"NEWS":Nblk}
a=np.array([rA[d] for d in days]); b=np.array([rB[d] for d in days]); a=a-a.mean(); b=b-b.mean()
sA=a.std(); sB=b.std(); totcov=(a@b)/N
def v(subset):
    mats=[BLK[k] for k in subset if BLK[k].shape[1]>0]
    if not mats: return 0.0
    X=np.hstack(mats); 
    pa=X@np.linalg.lstsq(X,a,rcond=None)[0]; pb=X@np.linalg.lstsq(X,b,rcond=None)[0]
    return (pa@pb)/N
blocks=["MACRO","SECTOR","NEWS"]; n=len(blocks)
from math import factorial
vcache={frozenset(S):v(S) for r in range(n+1) for S in itertools.combinations(blocks,r)}
shap={}
for k in blocks:
    rest=[x for x in blocks if x!=k]; tot=0.0
    for r in range(len(rest)+1):
        for S in itertools.combinations(rest,r):
            w=factorial(len(S))*factorial(n-len(S)-1)/factorial(n)
            tot+=w*(vcache[frozenset(S+(k,))]-vcache[frozenset(S)])
    shap[k]=tot
resid=totcov-sum(shap.values())
print(f"=== SHAPLEY (order-independent) decomposition of corr({A},{B}) = {totcov/(sA*sB):+.3f}  ({N} days, {len(shared)} news drivers) ===")
print(f"  {'block':28} {'corr share':>11} {'% of total':>11}")
for k in blocks: print(f"  {k:28} {shap[k]/(sA*sB):+11.3f} {100*shap[k]/totcov:+10.0f}%")
print(f"  {'IDIOSYNCRATIC':28} {resid/(sA*sB):+11.3f} {100*resid/totcov:+10.0f}%")
print(f"\n  for reference — NEWS block alone v(NEWS) (generous, news-first) = {vcache[frozenset(['NEWS'])]/(sA*sB):+.3f} "
      f"({100*vcache[frozenset(['NEWS'])]/totcov:+.0f}%);  news-LAST increment = {(vcache[frozenset(blocks)]-vcache[frozenset(['MACRO','SECTOR'])])/(sA*sB):+.3f}")
print(f"  Shapley news share sits between those two bounds by construction.")
