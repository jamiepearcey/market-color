# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""Driver attribution on INDIA (cross-market test of the BBG result). India specified-factor
neutralized (Nifty/BankNifty/IT/INR/Brent); then named-shared-driver removal vs random placebo,
same falsifiable test as BBG (named should explain more than random)."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/india2021"); cache=G/"prices"; rng=np.random.RandomState(0)
resolved=json.load(open(G/"nifty50_resolved.json")); names=sorted(set(resolved.values()))
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
FAC={"mkt":"^NSEI","inr":"INR=X","brent":"BZ=F"}  # MACRO only (keep sector for attribution to find, mirrors BBG 9-macro)
facr={k:logret(yahoo(t,cache,p1,p2)) for k,t in FAC.items()}
px={s:logret(yahoo(s,cache,p1,p2)) for s in names}; px={s:v for s,v in px.items() if len(v)>200}
common=sorted(set.intersection(*[set(px[s]) for s in px],*[set(facr[k]) for k in FAC]))
AR={}
for s,r in px.items():
    y=np.array([r[d] for d in common]); X=np.column_stack([np.ones(len(common))]+[[facr[k][d] for d in common] for k in FAC])
    b,*_=np.linalg.lstsq(X,y,rcond=None); AR[s]=dict(zip(common,y-X@b))
U=set(AR); print(f"{len(AR)} India names, specified-factor neutralized")
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
DIR={"up","down","widen","tighten"}
cause_m=collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity")
    if m and c and e in resolved and resolved[e] in U and j.get("effect_dir") in DIR: cause_m[m][c].add(resolved[e])
def dfac(nm,days):
    out={}
    for d in days:
        r=[AR[s][d] for s in nm if s in AR and d in AR[s]]
        if len(r)>=2: out[d]=np.mean(r)
    return out
def resid(a,f,days):
    cm=[d for d in days if d in a and d in f]
    if len(cm)<15: return None
    y=np.array([a[d] for d in cm]);X=np.column_stack([np.ones(len(cm)),[f[d] for d in cm]])
    b,*_=np.linalg.lstsq(X,y,rcond=None); return dict(zip(cm,y-X@b))
def wc(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<15: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
mo=[f"2021-{m:02d}" for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
named=[]; rand=[]
for m in mo:
    i=idx[m]
    if i<3: continue
    wd=[d for d in common if d[:7] in mo[i-3:i+1]]
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
                if rA and rB:
                    c1=wc(rA,rB,wd)
                    if c1 is not None: named.append(c0-c1)
                rc=cs[rng.randint(len(cs))] if cs else None
                if rc and rc!=c:
                    ro=[s for s in (cause_m[m][rc]&U) if s not in (A,B)]
                    if len(ro)>=2:
                        fR=dfac(ro,wd); rrA=resid(AR[A],fR,wd); rrB=resid(AR[B],fR,wd)
                        if rrA and rrB:
                            cr=wc(rrA,rrB,wd)
                            if cr is not None: rand.append(c0-cr)
nd=np.array(named); rd=np.array(rand)
print(f"\n=== INDIA driver attribution (n_named={len(nd)}, n_random={len(rd)}) ===")
if len(nd):
    print(f"  NAMED driver corr drop:  {nd.mean():+.3f} (median {np.median(nd):+.3f})")
    print(f"  RANDOM driver corr drop: {rd.mean():+.3f}")
    se=np.sqrt(nd.var(ddof=1)/len(nd)+rd.var(ddof=1)/max(len(rd),2))
    print(f"  EXCESS = {nd.mean()-rd.mean():+.3f}  t={(nd.mean()-rd.mean())/se:+.1f}")
    print(f"  named-drop NEGATIVE: {np.mean(nd<0):.0%}")
print(f"\n  BBG: named +0.266 / random +0.121 / excess +0.145 (t14), OOS-stable 0.64")
