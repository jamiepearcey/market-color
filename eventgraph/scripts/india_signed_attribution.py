# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""EM GENERALIZATION of F13/F15: signed driver attribution on India (Nifty), with India factors.
Same test as the US: for a correlated pair joined by a shared news driver, remove the driver's
SIGNED (direction-weighted) factor and measure the correlation drop. Same-sign (co-movement)
drivers should drop it more than opposite-sign (divergence). Plus the OOS version (fit betas W1,
apply to forward W2). If it holds in a structurally different EM, the finding broadens."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/india2021"); cache=G/"prices"
resolved=json.load(open(G/"nifty50_resolved.json"))
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
FAC={"mkt":"^NSEI","bank":"^NSEBANK","it":"^CNXIT","inr":"INR=X","brent":"BZ=F"}
facr={k:logret(yahoo(t,cache,p1,p2)) for k,t in FAC.items()}
px={s:logret(yahoo(s,cache,p1,p2)) for s in sorted(set(resolved.values()))}; px={s:v for s,v in px.items() if len(v)>200}
facdays=sorted(set.intersection(*[set(facr[k]) for k in facr]))   # master calendar (factor days)
AR={}                                                            # residualize each stock on ITS OWN days
for s,r in px.items():
    days=sorted(set(r)&set(facdays))
    if len(days)<200: continue
    y=np.array([r[d] for d in days]); X=np.column_stack([np.ones(len(days))]+[[facr[k][d] for d in days] for k in facr])
    b,*_=np.linalg.lstsq(X,y,rcond=None); AR[s]=dict(zip(days,y-X@b))
U=set(AR); common=facdays
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
cs=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or not m.startswith("2021") or e not in resolved or resolved[e] not in U or not c or d not in DIRV: continue
    cs[m][c][resolved[e]]+=DIRV[d]
mo=[f"2021-{m:02d}" for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def wc(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<12: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
def sfac(sgn,days):
    out={}
    for d in days:
        v=[np.sign(sgn[s])*AR[s][d] for s in sgn if s in AR and d in AR[s] and sgn[s]!=0]
        if len(v)>=2: out[d]=np.mean(v)
    return out
def beta(a,f,days):
    cm=[d for d in days if d in a and d in f]
    if len(cm)<12: return None
    x=np.array([f[d] for d in cm]);y=np.array([a[d] for d in cm]);x=x-x.mean()
    return float((x@(y-y.mean()))/(x@x)) if x@x>0 else 0.0
def cresid(a,b,f,days,ba,bb):
    cm=[d for d in days if d in a and d in b and d in f]
    if len(cm)<12: return None,None
    A=np.array([a[d] for d in cm]);B=np.array([b[d] for d in cm]);F=np.array([f[d] for d in cm])
    c0=np.corrcoef(A,B)[0,1] if A.std() and B.std() else None
    rA=A-ba*F; rB=B-bb*F
    c1=np.corrcoef(rA,rB)[0,1] if rA.std() and rB.std() else None
    return c0,c1
same=[]; opp=[]; nIS=[]; nOOS=[]
for m in mo:
    i=idx[m]
    if i<3: continue
    w1=mo[i-3:i+1]; days=[d for d in common if d[:7] in w1]
    fwd=mo[i+1:i+4] if i+3<len(mo) else None; fdays=[d for d in common if fwd and d[:7] in fwd] if fwd else []
    for c,sgn in cs[m].items():
        effs=sorted(s for s in sgn if s in U and sgn[s]!=0)
        if len(effs)<4: continue
        for x in range(len(effs)):
            for y in range(x+1,len(effs)):
                A,B=effs[x],effs[y]; c0=wc(AR[A],AR[B],days)
                if c0 is None or c0<0.2: continue
                oth={s:sgn[s] for s in effs if s not in (A,B) and sgn[s]!=0}
                if len(oth)<2: continue
                f=sfac(oth,days); bA=beta(AR[A],f,days); bB=beta(AR[B],f,days)
                if bA is None or bB is None: continue
                _,c1=cresid(AR[A],AR[B],f,days,bA,bB)
                if c1 is None: continue
                drop=c0-c1
                (same if np.sign(sgn[A])==np.sign(sgn[B]) else opp).append(drop)
                if np.sign(sgn[A])==np.sign(sgn[B]) and fdays:
                    f2=sfac(oth,fdays)
                    c0f,c1is=cresid(AR[A],AR[B],f2,fdays,beta(AR[A],f2,fdays) or 0,beta(AR[B],f2,fdays) or 0)
                    _,c1oos=cresid(AR[A],AR[B],f2,fdays,bA,bB)
                    if c0f is not None and c1is is not None: nIS.append(c0f-c1is)
                    if c0f is not None and c1oos is not None: nOOS.append(c0f-c1oos)
def st(a):
    a=np.array(a)
    return f"mean {a.mean():+.3f}  median {np.median(a):+.3f}  t {a.mean()/(a.std(ddof=1)/len(a)**0.5):+.1f}  n {len(a)}" if len(a)>1 else f"n {len(a)} (insufficient)"
print("=== INDIA signed driver attribution (F13 generalization, Nifty + India factors, 2021) ===")
print(f"  same-sign (co-movement) removal : {st(same)}")
print(f"  opposite-sign (divergence)      : {st(opp)}")
if len(same)>1 and len(opp)>1:
    s=np.array(same);o=np.array(opp);d=s.mean()-o.mean()
    se=(s.var(ddof=1)/len(s)+o.var(ddof=1)/len(o))**0.5
    print(f"  DIFFERENCE same-opp = {d:+.3f}  t {d/se:+.1f}   (US: +0.121, t7.5)")
print(f"\n=== INDIA out-of-sample (F15 generalization; fit betas W1 -> forward W2) ===")
print(f"  same-sign IN-SAMPLE (W2 betas)  : {st(nIS)}")
print(f"  same-sign OUT-OF-SAMPLE (W1 beta): {st(nOOS)}")
if len(nIS)>1 and len(nOOS)>1: print(f"  OOS retains {np.mean(nOOS)/np.mean(nIS)*100:.0f}% of in-sample  (US: 99%)")
