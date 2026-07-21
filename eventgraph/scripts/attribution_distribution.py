# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""How accurate is the driver-attribution DISTRIBUTION (not just the mean)? Measures:
(1) full percentile distribution of named-driver vs random-driver correlation drop,
(2) PAIRED per-instance discrimination (named vs its own random placebo) + bootstrap CI,
(3) OUT-OF-SAMPLE stability: does the attribution measured in-window hold in the NEXT window."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret, wls, FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between
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
U=set(AR); ALLD=sorted(set().union(*[set(AR[s]) for s in AR]))
def dfac(names,days):
    out={}
    for d in days:
        r=[AR[s][d] for s in names if s in AR and d in AR[s]]
        if len(r)>=2: out[d]=np.mean(r)
    return out
def resid(a,f,days):
    cm=[d for d in days if d in a and d in f]
    if len(cm)<20: return None
    y=np.array([a[d] for d in cm]);X=np.column_stack([np.ones(len(cm)),[f[d] for d in cm]])
    b,*_=np.linalg.lstsq(X,y,rcond=None); return dict(zip(cm,y-X@b))
def wc(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<20: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
allm=months_between(2010,2012); idx={m:i for i,m in enumerate(allm)}
named=[]; rand=[]; paired=[]; oos_in=[]; oos_out=[]
for m in allm:
    i=idx[m]
    if i<3 or i+2>=len(allm): continue
    wd=[d for d in ALLD if d[:7] in allm[i-3:i+1]]; fd=[d for d in ALLD if d[:7] in allm[i+1:i+3]]
    cs=[c for c in cause_m[m] if len(cause_m[m][c]&U)>=3]
    for c in cause_m[m]:
        effs=sorted(cause_m[m][c]&U)
        for x in range(len(effs)):
            for y in range(x+1,len(effs)):
                A,B=effs[x],effs[y]; c0=wc(AR[A],AR[B],wd)
                if c0 is None or c0<0.3: continue
                oth=[s for s in effs if s not in (A,B)]
                if len(oth)<2: continue
                fD=dfac(oth,wd); rA=resid(AR[A],fD,wd); rB=resid(AR[B],fD,wd)
                if not(rA and rB): continue
                c1=wc(rA,rB,wd)
                if c1 is None: continue
                nd=c0-c1; named.append(nd)
                # matched random placebo
                rc=cs[rng.randint(len(cs))] if cs else None; rdv=None
                if rc and rc!=c:
                    ro=[s for s in (cause_m[m][rc]&U) if s not in (A,B)]
                    if len(ro)>=2:
                        fR=dfac(ro,wd); rrA=resid(AR[A],fR,wd); rrB=resid(AR[B],fR,wd)
                        if rrA and rrB:
                            cr=wc(rrA,rrB,wd)
                            if cr is not None: rdv=c0-cr; rand.append(rdv)
                if rdv is not None: paired.append(nd-rdv)
                # OOS: same driver's attribution in the FORWARD window
                c0f=wc(AR[A],AR[B],fd)
                if c0f is not None and len(fd)>20:
                    fDf=dfac(oth,fd); rAf=resid(AR[A],fDf,fd); rBf=resid(AR[B],fDf,fd)
                    if rAf and rBf:
                        c1f=wc(rAf,rBf,fd)
                        if c1f is not None: oos_in.append(nd); oos_out.append(c0f-c1f)
named=np.array(named); rand=np.array(rand); paired=np.array(paired); oi=np.array(oos_in); oo=np.array(oos_out)
pc=lambda a,q: np.percentile(a,q)
print(f"=== (1) DISTRIBUTION of correlation-drop attribution (n_named={len(named)}) ===")
print(f"  {'pctile':>8} {'p10':>7} {'p25':>7} {'p50':>7} {'p75':>7} {'p90':>7}")
print(f"  named    {pc(named,10):+7.2f} {pc(named,25):+7.2f} {pc(named,50):+7.2f} {pc(named,75):+7.2f} {pc(named,90):+7.2f}")
print(f"  random   {pc(rand,10):+7.2f} {pc(rand,25):+7.2f} {pc(rand,50):+7.2f} {pc(rand,75):+7.2f} {pc(rand,90):+7.2f}")
print(f"  named-drop NEGATIVE (driver 'explains' nothing/anti): {np.mean(named<0):.0%}")
print(f"\n=== (2) PER-PAIR DISCRIMINATION (paired named vs its own random placebo, n={len(paired)}) ===")
print(f"  P(named driver drop > random driver drop) = {np.mean(paired>0):.0%}  (50% = no discrimination)")
bs=[np.mean(paired[rng.randint(0,len(paired),len(paired))]) for _ in range(2000)]
print(f"  mean paired excess {paired.mean():+.3f}  bootstrap 95% CI [{pc(bs,2.5):+.3f}, {pc(bs,97.5):+.3f}]")
# AUC: how separable is a named attribution from the random null
allv=np.concatenate([named,rand]); lab=np.concatenate([np.ones(len(named)),np.zeros(len(rand))])
order=np.argsort(allv); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(len(allv))
auc=(ranks[lab==1].sum()-len(named)*(len(named)-1)/2)/(len(named)*len(rand))
print(f"  AUC (named vs random attribution separability) = {auc:.2f}  (0.5=none, 1.0=perfect)")
print(f"\n=== (3) OUT-OF-SAMPLE STABILITY (n={len(oi)}) — does in-window attribution hold next window? ===")
if len(oi)>30:
    r=np.corrcoef(oi,oo)[0,1]
    print(f"  corr(in-window attribution, next-window attribution) = {r:+.2f}")
    print(f"  in-window mean {oi.mean():+.3f} -> next-window mean {oo.mean():+.3f} ({oo.mean()/oi.mean()*100:.0f}% persists)")
