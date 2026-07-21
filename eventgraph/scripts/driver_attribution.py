# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""COUNTERFACTUAL DRIVER ATTRIBUTION — does the news-named shared driver explain a pair's
correlation MORE than a random driver? For each news-linked elevated-corr pair (A,B) with
shared cause D: build D as a synthetic factor (equal-wt abnormal return of the OTHER names
D drives), residualize A,B on it, measure the correlation DROP = D's attribution. Compare
to removing a RANDOM driver (placebo). If named-D drop >> random drop, the graph gives real,
quantified, counterfactual attribution. Falsifiable."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret, wls, FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
DIRV={"up","down","widen","tighten"}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"; rng=np.random.RandomState(0)
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
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or j.get("effect_dir") not in DIRV: continue
    freq[sym[e]]+=1
    if c: cause_m[m][c].add(sym[e])
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=160: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR)
def dfactor(driver_syms, days):
    """equal-wt abnormal return series of the driver's names, on given days."""
    out={}
    for d in days:
        r=[AR[s][d] for s in driver_syms if s in AR and d in AR[s]]
        if len(r)>=2: out[d]=np.mean(r)
    return out
def resid_on(a, f, days):
    common=[d for d in days if d in a and d in f]
    if len(common)<20: return None
    y=np.array([a[d] for d in common]); X=np.column_stack([np.ones(len(common)),[f[d] for d in common]])
    b,*_=np.linalg.lstsq(X,y,rcond=None); return dict(zip(common,y-X@b))
def wc(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<20: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
allm=months_between(2010,2012); idx={m:i for i,m in enumerate(allm)}
named_drop=[]; rand_drop=[]; examples=[]
for m in allm:
    i=idx[m]
    if i<3: continue
    wm=allm[i-3:i+1]; days=sorted(d for d in set().union(*[set(AR[s]) for s in AR]) if d[:7] in wm)
    cm=cause_m; allcauses=[c for c in cm[m] if len(cm[m][c] & U)>=3]
    for c in cm[m]:
        effs=sorted(cm[m][c] & U)
        for x in range(len(effs)):
            for y in range(x+1,len(effs)):
                A,B=effs[x],effs[y]
                c0=wc(AR[A],AR[B],days)
                if c0 is None or c0<0.3: continue
                # NAMED driver: other names c drives (excl A,B)
                others=[s for s in effs if s not in (A,B)]
                if len(others)<2: continue
                fD=dfactor(others,days)
                rA=resid_on(AR[A],fD,days); rB=resid_on(AR[B],fD,days)
                if rA and rB:
                    c1=wc(rA,rB,days)
                    if c1 is not None:
                        named_drop.append(c0-c1)
                        if len(examples)<6 and (c0-c1)>0.15: examples.append((A,B,c.split("__")[0],round(c0,2),round(c1,2)))
                # RANDOM driver placebo: a random OTHER cause's names
                rc=allcauses[rng.randint(len(allcauses))] if allcauses else None
                if rc and rc!=c:
                    ro=[s for s in (cm[m][rc]&U) if s not in (A,B)]
                    if len(ro)>=2:
                        fR=dfactor(ro,days); rrA=resid_on(AR[A],fR,days); rrB=resid_on(AR[B],fR,days)
                        if rrA and rrB:
                            cr=wc(rrA,rrB,days)
                            if cr is not None: rand_drop.append(c0-cr)
nd=np.array(named_drop); rd=np.array(rand_drop)
print(f"=== counterfactual driver attribution ({len(nd)} named removals, {len(rd)} random placebos) ===")
print(f"  NAMED shared driver removed:  mean corr drop {nd.mean():+.3f}  (median {np.median(nd):+.3f})")
print(f"  RANDOM driver removed (placebo): mean corr drop {rd.mean():+.3f}")
se=np.sqrt(nd.var(ddof=1)/len(nd)+rd.var(ddof=1)/len(rd))
print(f"  EXCESS attribution of the named driver = {nd.mean()-rd.mean():+.3f}  t={ (nd.mean()-rd.mean())/se:+.1f}")
print(f"  => named driver explains {nd.mean()/max(0.3,1e-9)*100:.0f}% of the (>=0.3) correlation on average\n")
print("  examples (pair | named driver | corr before -> after removing driver):")
for A,B,d,c0,c1 in examples: print(f"    {A}~{B:6} | {d[:30]:30} | {c0:+.2f} -> {c1:+.2f}")
