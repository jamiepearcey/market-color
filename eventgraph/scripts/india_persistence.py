# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""India (Nifty50) persistence test — the F12 EM thesis on genuinely single-name-dense
native press. Neutralize .NS returns vs the Nifty cross-sectional market factor; monthly
abnormal-corr; shared-cause linked vs unlinked persistence; compare to DM (20%/48%)."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/india2021"); cache=G/"prices"; cache.mkdir(exist_ok=True)
resolved=json.load(open(G/"nifty50_resolved.json"))  # eid -> NSE.NS
names=sorted(set(resolved.values()))
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
px={s:logret(yahoo(s,cache,p1,p2)) for s in names}
px={s:v for s,v in px.items() if len(v)>200}
print(f"{len(px)} Nifty50 names priced")
# market factor = equal-weight cross-sectional mean per day; abnormal = residual on market
alldays=sorted(set().union(*[set(v) for v in px.values()]))
mkt={}
for d in alldays:
    r=[px[s][d] for s in px if d in px[s]]
    if len(r)>=10: mkt[d]=np.mean(r)
ar={}
for s,r in px.items():
    days=[d for d in sorted(r) if d in mkt]
    if len(days)<200: continue
    y=np.array([r[d] for d in days]); X=np.column_stack([np.ones(len(days)),[mkt[d] for d in days]])
    b,*_=np.linalg.lstsq(X,y,rcond=None); res=y-X@b; ar[s]=dict(zip(days,res))
print(f"{len(ar)} names with abnormal (market-neutralized) series")
# shared-cause links by month
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
cause_m=collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity")
    if m and c and e in resolved and resolved[e] in ar: cause_m[m][c].add(resolved[e])
def links_in(ms):
    out=set()
    for m in ms:
        for c,ss in cause_m.get(m,{}).items():
            ss=sorted(ss)
            for i in range(len(ss)):
                for k in range(i+1,len(ss)): out.add((ss[i],ss[k]))
    return out
def wcorr(a,b,days):
    common=[d for d in days if d in a and d in b]
    if len(common)<12: return None
    x=np.array([a[d] for d in common]);y=np.array([b[d] for d in common])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
months=[f"2021-{m:02d}" for m in range(1,13)]; idx={m:i for i,m in enumerate(months)}
d2m=lambda d:d[:7]
rows=[]; L=sorted(ar)
for m in months:
    i=idx[m]
    if i<3 or i+2>=len(months): continue
    td=[d for d in alldays if d2m(d) in months[i-3:i+1]]; fd=[d for d in alldays if d2m(d) in months[i+1:i+3]]
    lk=links_in(months[i-3:i+1])
    for x in range(len(L)):
        for k in range(x+1,len(L)):
            p=(L[x],L[k]); tc=wcorr(ar[p[0]],ar[p[1]],td); fc=wcorr(ar[p[0]],ar[p[1]],fd)
            if tc is None or fc is None: continue
            rows.append((tc,fc,1 if p in lk else 0))
R=np.array(rows)
print(f"India pair-months: {len(R)} | linked {int(R[:,2].sum())}")
for thr in (0.3,0.2):
    hi=R[R[:,0]>thr]
    print(f"  --- elevated trailing > {thr} ---")
    for lbl,sub in [("unlinked",hi[hi[:,2]==0]),("news-LINKED",hi[hi[:,2]==1])]:
        if len(sub): print(f"    India {lbl:12} persists {np.mean(sub[:,1]>=sub[:,0]):.0%}  next {sub[:,1].mean():+.3f} (from {sub[:,0].mean():+.3f})  n={len(sub)}")
print("  DM baseline: unlinked 20% / linked 48% | Bloomberg-EM: 21% / 23% (n=43)")
