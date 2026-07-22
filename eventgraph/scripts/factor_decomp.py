# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""FACTOR DECOMPOSITION of a pair's co-movement: what makes up the correlation?
Split cov(A,B) EXACTLY into orthogonalized blocks in economic order:
  macro (9 factors) -> sector (9 GICS SPDRs) -> each named news driver -> idiosyncratic.
Orthogonalization (Gram-Schmidt) makes the blocks additive: cov(A,B) = sum_k betaA_k betaB_k var(f_k)
+ resid, so every % is a real, non-overlapping share of the correlation. Order matters (reported)."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
PAIR=(sys.argv[1] if len(sys.argv)>1 else "GS", sys.argv[2] if len(sys.argv)>2 else "MS")
KDRV=5
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
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
inv={v:k for k,v in sym.items()}
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
drv_net=collections.defaultdict(lambda: collections.defaultdict(int)); drv_pair=collections.Counter()
A,B=PAIR; eA,eB=inv.get(A),inv.get(B)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    drv_net[c][sym[e]]+=DIRV[d]
# shared SPECIFIC drivers of the pair (drv_net keyed by SYMBOL), ranked by connection breadth
shared=[c for c in drv_net if A in drv_net[c] and B in drv_net[c] and drv_net[c][A]!=0 and drv_net[c][B]!=0]
shared=[c for c in shared if c.split("__")[-1] not in MACRO_T]
shared=sorted(shared,key=lambda c:-len(drv_net[c]))[:KDRV]
print(f"pair {A}-{B}: {len(shared)} specific shared news drivers used")
# returns
def ret(t): return logret(yahoo(t,cache,p1,p2))
rA,rB=ret(A),ret(B); spdr={e:ret(e) for e in SPDR}
years={"2010","2011","2012"}
# driver basket raw returns (signed leave-two-out over the driver's OTHER connected priceable names)
def basket_series(c):
    names=[s for s in drv_net[c] if s in sym.values() and s not in (A,B) and drv_net[c][s]!=0]
    rr={s:ret(s) for s in names}; sgn={s:np.sign(drv_net[c][s]) for s in names}
    days=set().union(*[set(rr[s]) for s in names]) if names else set()
    out={}
    for d in days:
        v=[sgn[s]*rr[s][d] for s in names if d in rr[s]]
        if len(v)>=2: out[d]=float(np.mean(v))
    return out
baskets=[(c,basket_series(c)) for c in shared]
# common days (2010-12): A,B, Fmap row present, all SPDR present (macro NaNs handled per-column)
def okday(d):
    return d[:4] in years and d in rA and d in rB and d in Fmap and all(d in spdr[e] for e in SPDR)
days=sorted(d for d in rA if okday(d))
# build factor matrix (columns) with block labels
cols=[]; labels=[]; groups=[]
for i,f in enumerate(FN): cols.append([Fmap[d][i] for d in days]); labels.append(f"macro:{f}"); groups.append("macro")
for e in SPDR: cols.append([spdr[e][d] for d in days]); labels.append(f"sector:{e}"); groups.append("sector")
for c,bs in baskets:
    cols.append([bs.get(d,np.nan) for d in days]); labels.append(c.split("__")[0].replace("_"," ").title()); groups.append("driver")
X=np.array(cols,float).T  # days x factors
# drop all-NaN / near-empty columns (e.g. crypto factor is all-NaN in 2010-12), mean-fill the rest
keep=[j for j in range(X.shape[1]) if np.isfinite(X[:,j]).sum()>=len(days)*0.5]
X=X[:,keep]; labels=[labels[j] for j in keep]; groups=[groups[j] for j in keep]
for j in range(X.shape[1]):
    m=~np.isfinite(X[:,j])
    if m.any(): X[m,j]=np.nanmean(X[~m,j])
a=np.array([rA[d] for d in days]); b=np.array([rB[d] for d in days])
a=a-a.mean(); b=b-b.mean(); X=X-X.mean(0)
# Gram-Schmidt orthogonalize columns in order
Q=np.zeros_like(X)
for j in range(X.shape[1]):
    v=X[:,j].copy()
    for k in range(j):
        if Q[:,k]@Q[:,k]>1e-12: v=v-(Q[:,k]@v)/(Q[:,k]@Q[:,k])*Q[:,k]
    Q[:,j]=v
N=len(days); sA=a.std(); sB=b.std()
contrib=[]
for j in range(X.shape[1]):
    q=Q[:,j]; qq=q@q
    if qq<1e-12: contrib.append(0.0); continue
    bA=(q@a)/qq; bB=(q@b)/qq
    contrib.append(bA*bB*qq/N)   # covariance contribution
eA_=a-sum(((Q[:,j]@a)/(Q[:,j]@Q[:,j]) if Q[:,j]@Q[:,j]>1e-12 else 0)*Q[:,j] for j in range(X.shape[1]))
eB_=b-sum(((Q[:,j]@b)/(Q[:,j]@Q[:,j]) if Q[:,j]@Q[:,j]>1e-12 else 0)*Q[:,j] for j in range(X.shape[1]))
resid_cov=(eA_@eB_)/N
totcov=(a@b)/N; totcorr=totcov/(sA*sB)
def pct(cv): return 100*cv/totcov
gc=collections.OrderedDict()
for lab,gr,cv in zip(labels,groups,contrib):
    key = "MACRO (9 factors)" if gr=="macro" else ("SECTOR (9 GICS SPDRs)" if gr=="sector" else lab)
    gc[key]=gc.get(key,0)+cv
print(f"\n=== FACTOR DECOMPOSITION of corr({A},{B}) = {totcorr:+.3f}  (2010-12 raw returns, {N} days) ===")
print(f"  {'block / driver':32} {'corr share':>11} {'% of total':>11}")
for k,cv in gc.items():
    print(f"  {k:32} {cv/(sA*sB):+11.3f} {pct(cv):+10.0f}%")
print(f"  {'IDIOSYNCRATIC (residual)':32} {resid_cov/(sA*sB):+11.3f} {pct(resid_cov):+10.0f}%")
chk=sum(gc.values())+resid_cov
print(f"  {'—— total (check)':32} {chk/(sA*sB):+11.3f} {pct(chk):+10.0f}%")
print(f"\n  (orthogonalized in order macro->sector->drivers->idio; order-dependent by construction)")
