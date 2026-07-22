# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""THE DECISIVE TEST: does a driver's NEWS activation LEAD its names' correlation rising?
For each (driver D, week t): does a surge in D's news over the trailing window predict D's
connected names co-moving MORE over the FORWARD window, controlling for (a) past co-movement
(mean reversion) and (b) market-wide co-movement change (crisis confound)? News uses NO forward
info. If the news coefficient is +ve and significant, news leads the correlation regime — the
one thing a price-defined beta regression cannot give. If not, it's an interpretable factor model."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP
from news_contagion import ar_series
H=15; STEP=5; N=120; MINCO=10
DIRV={"up","down","widen","tighten"}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docdate={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:10] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); drv_names=collections.defaultdict(set); drv_day=collections.defaultdict(lambda: collections.Counter())
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); dd=docdate.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not dd or dd[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; drv_names[c].add(sym[e]); drv_day[c][dd]+=1
print(f"building AR for top {N} ...", flush=True)
AR={}
for s,_ in freq.most_common():
    if len(AR)>=N: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR); DAYS=sorted(set().union(*[set(AR[s]) for s in AR])); DI={d:i for i,d in enumerate(DAYS)}
M={s:np.array([AR[s].get(d,np.nan) for d in DAYS]) for s in U}
def pc(a,b):
    ok=~(np.isnan(a)|np.isnan(b))
    if ok.sum()<MINCO: return None
    x=a[ok]-a[ok].mean();y=b[ok]-b[ok].mean()
    return float((x*y).mean()/(x.std()*y.std())) if x.std() and y.std() else None
def setcorr(names,sl):
    vals=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            c=pc(M[names[i]][sl],M[names[j]][sl])
            if c is not None: vals.append(c)
    return float(np.mean(vals)) if vals else None
weeks=[i for i in range(2*H,len(DAYS)-H,STEP)]
# market co-movement per week (sampled fixed pairs) for the confound control
uni=sorted(U); rng=np.random.RandomState(1)
mp=[(uni[i],uni[j]) for i in range(len(uni)) for j in range(i+1,len(uni))]
mp=[mp[k] for k in rng.choice(len(mp),size=min(400,len(mp)),replace=False)]
def mkt(sl):
    vals=[pc(M[a][sl],M[b][sl]) for a,b in mp]; vals=[v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None
mkt_past={t:mkt(slice(t-H,t)) for t in weeks}; mkt_fwd={t:mkt(slice(t,t+H)) for t in weeks}
# per-driver panel
rows=[]  # (news_z-ready news, C_past, dC, dC_mkt, weekidx)
usable=[c for c in drv_names if len([s for s in drv_names[c] if s in U])>=4]
for c in usable:
    names=[s for s in drv_names[c] if s in U]
    for t in weeks:
        cp=setcorr(names,slice(t-H,t)); cf=setcorr(names,slice(t,t+H))
        if cp is None or cf is None or mkt_past[t] is None or mkt_fwd[t] is None: continue
        news=sum(drv_day[c][DAYS[k]] for k in range(t-H,t))
        rows.append((news,cp,cf-cp,mkt_fwd[t]-mkt_past[t],t))
R=np.array(rows,float)
news=R[:,0]; news=(news-news.mean())/news.std()          # per-SD news activation
X=np.column_stack([np.ones(len(R)),news,R[:,1],R[:,3]])   # 1, news, C_past, dC_mkt
y=R[:,2]                                                   # dC (forward change in co-movement)
b,*_=np.linalg.lstsq(X,y,rcond=None); u=y-X@b
# week-clustered SEs
wk=R[:,4].astype(int); gid={w:i for i,w in enumerate(sorted(set(wk)))}; g=np.array([gid[w] for w in wk])
S=X*u[:,None]; G=np.zeros((len(gid),X.shape[1])); np.add.at(G,g,S)
bread=np.linalg.inv(X.T@X); V=bread@(G.T@G)@bread; se=np.sqrt(np.diag(V))
labs=["intercept","NEWS (per SD)","C_past","dC_market"]
print(f"\n=== does news activation LEAD forward co-movement change?  (n={len(R)} driver-weeks, {len(gid)} week-clusters, H={H}d) ===")
print(f"  DV = forward Δ co-movement of the driver's names")
for l,bb,ss in zip(labs,b,se): print(f"  {l:16} coef {bb:+.4f}   week-clustered t {bb/ss:+.1f}")
print(f"\n  => NEWS coef {b[1]:+.4f} (t {b[1]/se[1]:+.1f}) per 1-SD news surge, over the next {H} trading days,")
print(f"     controlling for mean-reversion (C_past) and market-wide co-movement change.")
# intuitive lead-lag: peak cross-correlation lag (news_t vs comovement_{t+k}), per driver, in weeks
best=[]
for c in usable:
    names=[s for s in drv_names[c] if s in U]
    nv=np.array([sum(drv_day[c][DAYS[k]] for k in range(t-H,t)) for t in weeks],float)
    cvv=np.array([setcorr(names,slice(t-H,t)) or np.nan for t in weeks])
    if np.nanstd(cvv)==0 or nv.std()==0: continue
    ok=~np.isnan(cvv)
    if ok.sum()<20: continue
    lags=range(-4,5); cc=[]
    for k in lags:
        a=nv; bb=np.roll(cvv,-k)
        m=ok&~np.isnan(np.roll(cvv,-k))
        if m.sum()<15: cc.append(np.nan); continue
        cc.append(np.corrcoef(a[m],np.roll(cvv,-k)[m])[0,1])
    if np.all(np.isnan(cc)): continue
    best.append(list(lags)[int(np.nanargmax(cc))])
best=np.array(best)
print(f"\n=== intuitive lead-lag: peak news→co-movement correlation lag (weeks), {len(best)} drivers ===")
print(f"  median peak lag {np.median(best):+.1f} wk | mean {best.mean():+.2f} wk | % leading(>0) {np.mean(best>0):.0%} | % same-week(0) {np.mean(best==0):.0%} | % lagging(<0) {np.mean(best<0):.0%}")
print("  (+lag = news leads correlation; 0 = contemporaneous; - = news trails)")
