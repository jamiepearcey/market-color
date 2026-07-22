# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""RETEST the descriptive-not-predictive verdict (F14) on India. Does a driver's news activation
LEAD its names' forward co-movement, controlling for mean-reversion + market-wide co-movement?
If news leads in India too -> predictive (would flip the verdict). If not -> descriptive confirmed
cross-market. Mirrors lead_lag.py with India factors / 140-name universe / 2021."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
H=15; STEP=5; MINCO=10
G=Path("../data/eg_runs/india2021"); cache=G/"prices"
resolved=json.load(open(G/"nifty50_resolved.json"))
DIRV={"up","down","widen","tighten"}
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
FAC={"mkt":"^NSEI","bank":"^NSEBANK","it":"^CNXIT","inr":"INR=X","brent":"BZ=F"}
facr={k:logret(yahoo(t,cache,p1,p2)) for k,t in FAC.items()}
facdays=sorted(set.intersection(*[set(facr[k]) for k in facr]))
px={s:logret(yahoo(s,cache,p1,p2)) for s in sorted(set(resolved.values()))}
AR={}
for s,r in px.items():
    days=sorted(set(r)&set(facdays))
    if len(days)<200: continue
    y=np.array([r[d] for d in days]); X=np.column_stack([np.ones(len(days))]+[[facr[k][d] for d in days] for k in facr])
    b,*_=np.linalg.lstsq(X,y,rcond=None); AR[s]=dict(zip(days,y-X@b))
U=set(AR)
DAYS=[d for d in facdays if any(d in AR[s] for s in AR)]; DAYS=sorted(DAYS); DI={d:i for i,d in enumerate(DAYS)}
M={s:np.array([AR[s].get(d,np.nan) for d in DAYS]) for s in U}
docd={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:10] for l in open(G/"lake/document.jsonl")}
drv_names=collections.defaultdict(set); drv_day=collections.defaultdict(lambda: collections.Counter())
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); dd=docd.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not dd or not dd.startswith("2021") or e not in resolved or resolved[e] not in U or not c or d not in DIRV: continue
    drv_names[c].add(resolved[e]); drv_day[c][dd]+=1
def pc(a,b):
    ok=~(np.isnan(a)|np.isnan(b))
    if ok.sum()<MINCO: return None
    x=a[ok]-a[ok].mean();y=b[ok]-b[ok].mean()
    return float((x*y).mean()/(x.std()*y.std())) if x.std() and y.std() else None
def setcorr(names,sl):
    v=[pc(M[names[i]][sl],M[names[j]][sl]) for i in range(len(names)) for j in range(i+1,len(names))]
    v=[x for x in v if x is not None]; return float(np.mean(v)) if v else None
# weeks within 2021 (need H before and H after)
w2021=[i for i,d in enumerate(DAYS) if d.startswith("2021")]
weeks=[i for i in range(min(w2021)+H, max(w2021)-H, STEP)]
uni=sorted(U); rng=np.random.RandomState(1)
allp=[(uni[i],uni[j]) for i in range(len(uni)) for j in range(i+1,len(uni))]
mp=[allp[k] for k in rng.choice(len(allp),size=min(400,len(allp)),replace=False)]
def mkt(sl):
    v=[pc(M[a][sl],M[b][sl]) for a,b in mp]; v=[x for x in v if x is not None]; return float(np.mean(v)) if v else None
mkt_p={t:mkt(slice(t-H,t)) for t in weeks}; mkt_f={t:mkt(slice(t,t+H)) for t in weeks}
usable=[c for c in drv_names if len([s for s in drv_names[c] if s in U])>=4]
rows=[]
for c in usable:
    names=[s for s in drv_names[c] if s in U]
    for t in weeks:
        cp=setcorr(names,slice(t-H,t)); cf=setcorr(names,slice(t,t+H))
        if cp is None or cf is None or mkt_p[t] is None or mkt_f[t] is None: continue
        news=sum(drv_day[c][DAYS[k]] for k in range(t-H,t))
        rows.append((news,cp,cf-cp,mkt_f[t]-mkt_p[t],t))
R=np.array(rows,float)
if len(R)<30:
    print(f"insufficient ({len(R)} driver-weeks)"); sys.exit()
news=R[:,0]; news=(news-news.mean())/news.std()
X=np.column_stack([np.ones(len(R)),news,R[:,1],R[:,3]]); y=R[:,2]
b,*_=np.linalg.lstsq(X,y,rcond=None); u=y-X@b
wk=R[:,4].astype(int); gid={w:i for i,w in enumerate(sorted(set(wk)))}; g=np.array([gid[w] for w in wk])
S=X*u[:,None]; Gm=np.zeros((len(gid),X.shape[1])); np.add.at(Gm,g,S)
bread=np.linalg.inv(X.T@X); V=bread@(Gm.T@Gm)@bread; se=np.sqrt(np.diag(V))
labs=["intercept","NEWS (per SD)","C_past","dC_market"]
print(f"=== INDIA lead-lag retest (F14): does news LEAD forward co-movement? (n={len(R)} driver-weeks, {len(gid)} week-clusters, H={H}d) ===")
for l,bb,ss in zip(labs,b,se): print(f"  {l:16} coef {bb:+.4f}   week-clustered t {bb/ss:+.1f}")
print(f"\n  NEWS coef {b[1]:+.4f} (t {b[1]/se[1]:+.1f}) — US was -0.0044 (t-3.9), i.e. news does NOT lead.")
print(f"  {'=> descriptive CONFIRMED in India (news not a positive leading signal)' if b[1]/se[1]<2 else '=> WARNING: news leads in India — investigate (would flip verdict)'}")
