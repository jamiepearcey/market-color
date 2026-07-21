# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""India persistence with a PROPER India factor model (statistical PCA multi-factor +
specified-factor variant), vs the crude market-mean. Tests whether weak India signal was
a neutralization artifact."""
import json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret
G=Path("../data/eg_runs/india2021"); cache=G/"prices"
resolved=json.load(open(G/"nifty50_resolved.json")); names=sorted(set(resolved.values()))
p1=int(dt.datetime(2020,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2022,12,31,tzinfo=dt.UTC).timestamp())
px={s:logret(yahoo(s,cache,p1,p2)) for s in names}; px={s:v for s,v in px.items() if len(v)>200}
days=sorted(set.intersection(*[set(v) for v in px.values()]))
S=sorted(px); M=np.array([[px[s][d] for d in days] for s in S])  # names x days
print(f"{len(S)} names x {len(days)} common days")

def neutralize_pca(M, k):
    Md=M-M.mean(axis=1,keepdims=True)
    # PCs across TIME: SVD of names x days -> right singular vecs are day-factors
    U,sv,Vt=np.linalg.svd(Md,full_matrices=False)
    F=Vt[:k]  # k x days factor series
    res=np.zeros_like(M)
    X=np.column_stack([np.ones(len(days))]+[F[j] for j in range(k)])
    for i in range(M.shape[0]):
        b,*_=np.linalg.lstsq(X,M[i],rcond=None); res[i]=M[i]-X@b
    varexp=(sv[:k]**2).sum()/(sv**2).sum()
    return res, varexp

def run(res, label):
    ar={S[i]:dict(zip(days,res[i])) for i in range(len(S))}
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
    def wc(a,b,dd):
        cm=[d for d in dd if d in a and d in b]
        if len(cm)<12: return None
        x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
        return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
    mo=[f"2021-{m:02d}" for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
    rows=[]; L=sorted(ar)
    for m in mo:
        i=idx[m]
        if i<3 or i+2>=len(mo): continue
        td=[d for d in days if d[:7] in mo[i-3:i+1]]; fd=[d for d in days if d[:7] in mo[i+1:i+3]]
        lk=links_in(mo[i-3:i+1])
        for x in range(len(L)):
            for k in range(x+1,len(L)):
                p=(L[x],L[k]); tc=wc(ar[p[0]],ar[p[1]],td); fc=wc(ar[p[0]],ar[p[1]],fd)
                if tc is not None and fc is not None: rows.append((tc,fc,1 if p in lk else 0))
    R=np.array(rows); print(f"\n=== {label} ({len(R)} pair-months, {int(R[:,2].sum())} linked) ===")
    for thr in (0.3,0.2):
        hi=R[R[:,0]>thr]
        for lbl,sub in [("unlinked",hi[hi[:,2]==0]),("LINKED",hi[hi[:,2]==1])]:
            if len(sub): print(f"  >{thr} {lbl:9} {np.mean(sub[:,1]>=sub[:,0]):.0%} persist  (trail {sub[:,0].mean():.2f})  n={len(sub)}")

for k in (3,5):
    res,ve=neutralize_pca(M,k)
    run(res, f"PCA-{k}factor (var explained {ve:.0%})")
print("\n  DM +28pp | Bloomberg-EM +2pp | India market-mean: >0.3 null, >0.2 +7pp")
