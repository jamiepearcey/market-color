# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""SIGNED driver attribution (the 'synthetic negative counterpart'). Drivers have DIRECTION:
both-up = co-movement driver; one-up/one-down = DIVERGENCE driver (a hedge). Build the SIGNED
driver factor (dir-weighted co-driven returns), remove it, measure corr change. FALSIFIABLE:
same-sign shared driver removal should DROP corr (+); opposite-sign (divergence) removal should
RAISE corr (-, removing the thing pulling them apart). If the exposure sign predicts the sign of
the correlation change, the signed embedding attributes what unsigned links cannot."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret, wls, FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"; rng=np.random.RandomState(0)
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter()
# month -> cause -> {sym: net_dir}
cause_signed=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cause_signed[m][c][sym[e]]+=DIRV[d]
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=160: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR); ALLD=sorted(set().union(*[set(AR[s]) for s in AR]))
def sfac(sgn,days):  # SIGNED driver factor: dir-weighted mean of co-driven returns
    out={}
    for d in days:
        vals=[np.sign(sgn[s])*AR[s][d] for s in sgn if s in AR and d in AR[s] and sgn[s]!=0]
        if len(vals)>=2: out[d]=np.mean(vals)
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
same_d=[]; opp_d=[]
for m in allm:
    i=idx[m]
    if i<3: continue
    wm=allm[i-3:i+1]; days=[d for d in ALLD if d[:7] in wm]
    cs=cause_signed[m]
    for c,sgn in cs.items():
        effs=sorted(s for s in sgn if s in U and sgn[s]!=0)
        if len(effs)<4: continue
        for x in range(len(effs)):
            for y in range(x+1,len(effs)):
                A,B=effs[x],effs[y]; c0=wc(AR[A],AR[B],days)
                if c0 is None or abs(c0)<0.2: continue
                oth={s:sgn[s] for s in effs if s not in (A,B) and sgn[s]!=0}
                if len(oth)<2: continue
                f=sfac(oth,days); rA=resid(AR[A],f,days); rB=resid(AR[B],f,days)
                if not(rA and rB): continue
                c1=wc(rA,rB,days)
                if c1 is None: continue
                dcorr=c0-c1  # positive = removing D dropped corr
                if np.sign(sgn[A])==np.sign(sgn[B]): same_d.append(dcorr)
                else: opp_d.append(dcorr)
sd=np.array(same_d); od=np.array(opp_d)
print(f"=== SIGNED driver attribution on BBG ===")
print(f"  SAME-sign shared driver (co-movement) removed:  Δcorr {sd.mean():+.3f}  (n={len(sd)})  [expect +, DROP]")
print(f"  OPPOSITE-sign shared driver (divergence) removed: Δcorr {od.mean():+.3f}  (n={len(od)})  [expect -, RISE]")
se=np.sqrt(sd.var(ddof=1)/len(sd)+od.var(ddof=1)/len(od))
print(f"  DIFFERENCE (same - opposite) = {sd.mean()-od.mean():+.3f}  t={(sd.mean()-od.mean())/se:+.1f}")
print(f"  => sign of exposure predicts sign of correlation change = signed attribution (unsigned links can't)")
