# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""India persistence with SPECIFIED (named, economically-motivated) India factors — removes
the PCA-K arbitrariness. Nested specs (market -> +sectors -> +macro) shown for transparency:
if the news-linked persistence advantage is robust across specifications, it's real."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/india2021"); cache=G/"prices"
resolved=json.load(open(G/"nifty50_resolved.json")); names=sorted(set(resolved.values()))
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
FAC={"mkt":"^NSEI","bank":"^NSEBANK","it":"^CNXIT","inr":"INR=X","brent":"BZ=F"}
facr={k:logret(yahoo(t,cache,p1,p2)) for k,t in FAC.items()}
px={s:logret(yahoo(s,cache,p1,p2)) for s in names}; px={s:v for s,v in px.items() if len(v)>200}
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
cause_m=collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity")
    if m and c and e in resolved and resolved[e] in px: cause_m[m][c].add(resolved[e])

def neutralize(fset):
    fk=[k for k in fset]
    common=sorted(set.intersection(*[set(px[s]) for s in px], *[set(facr[k]) for k in fk]))
    ar={}
    for s,r in px.items():
        y=np.array([r[d] for d in common]); X=np.column_stack([np.ones(len(common))]+[[facr[k][d] for d in common] for k in fk])
        b,*_=np.linalg.lstsq(X,y,rcond=None); ar[s]=dict(zip(common,y-X@b))
    return ar, common

def links_in(ms):
    out=set()
    for m in ms:
        for c,ss in cause_m.get(m,{}).items():
            ss=sorted(ss)
            for i in range(len(ss)):
                for k in range(i+1,len(ss)): out.add((ss[i],ss[k]))
    return out
def wc(a,b,dd):
    cm=[d for d in dd if d in a and d in b]
    if len(cm)<12: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None

def run(ar,common,label):
    mo=[f"2021-{m:02d}" for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}; L=sorted(ar); rows=[]
    for m in mo:
        i=idx[m]
        if i<3 or i+2>=len(mo): continue
        td=[d for d in common if d[:7] in mo[i-3:i+1]]; fd=[d for d in common if d[:7] in mo[i+1:i+3]]
        lk=links_in(mo[i-3:i+1])
        for x in range(len(L)):
            for k in range(x+1,len(L)):
                p=(L[x],L[k]); tc=wc(ar[p[0]],ar[p[1]],td); fc=wc(ar[p[0]],ar[p[1]],fd)
                if tc is not None and fc is not None: rows.append((tc,fc,1 if p in lk else 0))
    R=np.array(rows); hi=R[R[:,0]>0.3]; hi2=R[R[:,0]>0.2]
    def pr(sub): return (np.mean(sub[:,1]>=sub[:,0]),len(sub))
    u,nu=pr(hi[hi[:,2]==0]); l,nl=pr(hi[hi[:,2]==1]); u2,nu2=pr(hi2[hi2[:,2]==0]); l2,nl2=pr(hi2[hi2[:,2]==1])
    print(f"  {label:34} >0.3: unlink {u:.0%}(n{nu}) link {l:.0%}(n{nl}) [{l-u:+.0%}]  |  >0.2: unlink {u2:.0%} link {l2:.0%}(n{nl2}) [{l2-u2:+.0%}]")

print("SPECIFIED-FACTOR India persistence (nested; robust across specs = real):")
for label,fset in [("market only [mkt]",["mkt"]),
                   ("+ sectors [mkt,bank,it]",["mkt","bank","it"]),
                   ("+ macro [mkt,bank,it,inr,brent]",["mkt","bank","it","inr","brent"])]:
    ar,common=neutralize(fset); run(ar,common,label)
print("\n  DM +28pp | Bloomberg-EM +2pp | PCA-3 +26pp/+28pp | crude market-mean null/+7pp")
