# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""ECONOMIC SIGNIFICANCE: does acting on the persistence signal improve a real risk decision
out-of-sample? The value isn't the whole covariance matrix (0.4% coverage -> +0.4%); it's the
DECISION on the elevated-correlation pairs where a manager is tempted to assume diversification.

Two tests on DM (US 2010-12, chronological train/test):
 A. DIVERSIFICATION-TRAP: a naive manager forecasts next-period corr by mean-reversion
    (shrink trailing). Linked pairs DON'T revert -> naive UNDERESTIMATES their risk. Measure
    the risk-underestimate (realized corr - forecast corr) for linked vs unlinked, baseline
    vs news-aware forecast, on elevated pairs. Underestimate = the dangerous direction.
 B. PAIR RISK-FORECAST error: RMSE of the 2-asset variance forecast, baseline vs news-aware.
"""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret, wls, FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
DIR={"up","down","widen","tighten"}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
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
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or j.get("effect_dir") not in DIR: continue
    freq[sym[e]]+=1
    if c: cause_m[m][c].add(sym[e])
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=160: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
names=sorted(AR); U=set(names)
def links_in(ms):
    out=set()
    for m in ms:
        for c,ss in cause_m.get(m,{}).items():
            ss=sorted(s for s in ss if s in U)
            for i in range(len(ss)):
                for k in range(i+1,len(ss)): out.add((ss[i],ss[k]))
    return out
allm=months_between(2010,2012); idx={m:i for i,m in enumerate(allm)}
rows=[]; mon=[]
for m in allm:
    i=idx[m]
    if i<3 or i+2>=len(allm): continue
    tr=corr_matrix(AR,names,allm[i-3:i+1]); nx=corr_matrix(AR,names,allm[i+1:i+3]); lk=links_in(allm[i-3:i+1])
    for p,tc in tr.items():
        if p in nx: rows.append((tc,nx[p],1 if p in lk else 0)); mon.append(i)
P=np.array(rows); mon=np.array(mon)
# chronological split
cut=sorted(set(mon))[int(len(set(mon))*0.6)]
tr_m=mon<cut; te_m=mon>=cut
# fit forecasts on TRAIN: baseline next~1+trailing ; news-aware next~1+trailing+linked+linked*trailing
def fit(P,news):
    X=[np.ones(len(P)),P[:,0]]+([P[:,2],P[:,2]*P[:,0]] if news else [])
    X=np.column_stack(X); b,*_=np.linalg.lstsq(X,P[:,1],rcond=None); return b
def pred(P,b,news):
    X=[np.ones(len(P)),P[:,0]]+([P[:,2],P[:,2]*P[:,0]] if news else [])
    return np.column_stack(X)@b
bb=fit(P[tr_m],False); nb=fit(P[tr_m],True)
te=P[te_m]; hi=te[te[:,0]>0.3]  # elevated pairs = the decision-relevant set
fb=pred(hi,bb,False); fn=pred(hi,nb,True); real=hi[:,1]; lk=hi[:,2]==1
print(f"OOS elevated pairs (trailing>0.3): {len(hi)} ({int(lk.sum())} linked)\n")
print("=== A. DIVERSIFICATION-TRAP: risk-underestimate (realized corr - forecast) on LINKED pairs ===")
print(f"  baseline (mean-reversion) forecast: linked underestimate = {np.mean(real[lk]-fb[lk]):+.3f}  (positive = risk UNDER-estimated)")
print(f"  news-aware forecast:                linked underestimate = {np.mean(real[lk]-fn[lk]):+.3f}")
print(f"  => news-aware cuts the linked risk-underestimate by {(np.mean(real[lk]-fb[lk])-np.mean(real[lk]-fn[lk]))/max(abs(np.mean(real[lk]-fb[lk])),1e-9)*100:.0f}%")
print(f"\n=== B. correlation-forecast RMSE (all elevated / linked-only) ===")
print(f"  baseline  RMSE all {np.sqrt(np.mean((real-fb)**2)):.3f} | linked-only {np.sqrt(np.mean((real[lk]-fb[lk])**2)):.3f}")
print(f"  news-aware RMSE all {np.sqrt(np.mean((real-fn)**2)):.3f} | linked-only {np.sqrt(np.mean((real[lk]-fn[lk])**2)):.3f}")
print(f"  MSE reduction on LINKED pairs: {(np.mean((real[lk]-fb[lk])**2)-np.mean((real[lk]-fn[lk])**2))/np.mean((real[lk]-fb[lk])**2)*100:+.0f}%")
# translate to 2-asset portfolio vol error (equal-weight, standardized => var ∝ (1+corr)/2)
print(f"\n=== C. 2-asset portfolio VOL forecast error on linked pairs (equal-wt, abnormal) ===")
volerr_b=np.mean(np.abs(np.sqrt((1+real[lk])/2)-np.sqrt((1+fb[lk])/2)))
volerr_n=np.mean(np.abs(np.sqrt((1+real[lk])/2)-np.sqrt((1+fn[lk])/2)))
print(f"  baseline vol-error {volerr_b:.4f} | news-aware {volerr_n:.4f} | reduction {(volerr_b-volerr_n)/volerr_b*100:+.0f}%")
