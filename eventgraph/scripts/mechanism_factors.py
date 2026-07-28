# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx","sentence-transformers","torch","scikit-learn"]
# ///
"""STEP 4 first cut: latent mechanism factors over event embeddings + return-anchored company exposures.
(1) cluster event embeddings -> K interpretable mechanism factors (show top events each). (2) test three
exposures vs realized co-movement: raw embedding cosine (baseline +0.057), factor-mediated (semantic),
and RETURN-ANCHORED (company loads on each mechanism via its return beta to the factor's mimicking portfolio)."""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=140; K=24
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
BROADCANON={"macro","monetary","fiscal"}; DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
CANON=json.load(open(G/"catalyst_map.json"))
dmeta={json.loads(l)["doc_id"]:((json.loads(l).get("published_at") or "")[:7], json.loads(l).get("headline") or "") for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); comp_q=collections.defaultdict(list); ev_q=collections.defaultdict(list); ev_names=collections.defaultdict(lambda: collections.defaultdict(int)); ev_lab={}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); mh=dmeta.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not mh or mh[0][:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    m=mh[0]; freq[sym[e]]+=1; q=(j.get("quote") or "").strip()
    if q and len(comp_q[sym[e]])<50: comp_q[sym[e]].append((q+". "+mh[1])[:200])
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c.split("__")[0].replace("_"," ").title()
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    if q and len(ev_q[(m,key)])<40: ev_q[(m,key)].append((lab+": "+q+" ("+(j.get("mechanism") or "")+")")[:200])
    ev_names[(m,key)][sym[e]]+=DIRV[d]; ev_lab[(m,key)]=lab
years={"2010","2011","2012"}
def ret(t): return logret(yahoo(t,cache,p1,p2))
spdr={e:ret(e) for e in SPDR}
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=N_UNIV: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in years)>400: uni.append(s)
R={s:ret(s) for s in uni}
alld=sorted({d for s in uni for d in R[s] if d[:4] in years and d in Fmap and all(d in spdr[e] for e in SPDR)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
def design(days):
    X=np.column_stack([[Fmap[d][i] for d in days] for i in mcol]+[[spdr[e][d] for d in days] for e in SPDR])
    X=np.nan_to_num(X-np.nanmean(X,0)); return np.column_stack([np.ones(len(days)),X])
R2={}
for s in uni:
    days=[d for d in alld if d in R[s]]
    if len(days)<200: continue
    y=np.array([R[s][d] for d in days]); X=design(days); b,*_=np.linalg.lstsq(X,y,rcond=None); R2[s]=dict(zip(days,y-X@b))
uni=[s for s in uni if s in R2 and comp_q[s]]
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
dev="mps" if torch.backends.mps.is_available() else "cpu"
model=SentenceTransformer("all-MiniLM-L6-v2",device=dev)
evs=[(m,k) for (m,k),nm in ev_names.items() if len([x for x in nm if x in R2])>=3 and ev_q[(m,k)]]
allq=[]; im={}
for s in uni: im[("C",s)]=(len(allq),len(allq)+len(comp_q[s])); allq+=comp_q[s]
for k in evs: im[("E",k)]=(len(allq),len(allq)+len(ev_q[k])); allq+=ev_q[k]
print(f"encoding {len(allq)} quotes ...",flush=True)
emb=model.encode(allq,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
# EMB_CENTER=1 -> mean-centre the embedding space (F43). Transformer spaces are
# anisotropic: uncentred cosine is dominated by proximity to the corpus centroid,
# which tracks coverage volume and therefore firm size. Two results died here.
if __import__("os").environ.get("EMB_CENTER"):
    import numpy as _np
    emb = emb - emb.mean(0)
    _n = _np.linalg.norm(emb, axis=1, keepdims=True); emb = emb / _np.where(_n > 0, _n, 1)
    print("[EMB_CENTER] embedding space mean-centred", flush=True)
def vec(kk): a,b=im[kk]; v=emb[a:b].mean(0); n=np.linalg.norm(v); return v/n if n else v
cv={s:vec(("C",s)) for s in uni}; ev={k:vec(("E",k)) for k in evs}
# (1) K mechanism factors via KMeans over EVENT embeddings
EM=np.array([ev[k] for k in evs]); km=KMeans(K,n_init=5,random_state=0).fit(EM); cen=km.cluster_centers_
cen=cen/np.linalg.norm(cen,axis=1,keepdims=True)
print(f"\n=== {K} latent MECHANISM factors (top events per cluster) — are they economic mechanisms? ===")
for f in range(K):
    mem=[(float(ev[k]@cen[f]),k) for k in evs]; mem.sort(reverse=True)
    labs=[]
    for _,k in mem[:5]:
        L=ev_lab[k]
        if L not in labs: labs.append(L)
    print(f"  f{f:2}: {', '.join(labs[:4])}")
# factor loadings
Lev={k:(ev[k]@cen.T) for k in evs}     # event -> K
Lco={s:(cv[s]@cen.T) for s in uni}     # company -> K
# return-anchored: factor mimicking portfolio = mean residual return of top-loading companies; company beta to it
def series(s): return R2[s]
comdays=alld
fac_ret={}
for f in range(K):
    w=np.array([Lco[s][f] for s in uni]); top=[uni[i] for i in np.argsort(-w)[:12]]
    fr={}
    for d in comdays:
        vals=[R2[s][d] for s in top if d in R2[s]]
        if len(vals)>=4: fr[d]=float(np.mean(vals))
    fac_ret[f]=fr
def beta(s,f):
    fr=fac_ret[f]; cm=[d for d in fr if d in R2[s]]
    if len(cm)<60: return 0.0
    x=np.array([fr[d] for d in cm]); y=np.array([R2[s][d] for d in cm]); x=x-x.mean()
    return float((x@(y-y.mean()))/(x@x)) if x@x>0 else 0.0
Bco={s:np.array([beta(s,f) for f in range(K)]) for s in uni}
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def corr(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<15: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
E_cos=[]; E_fac=[]; E_ret=[]; Yr=[]
for (m,key) in evs:
    nm=ev_names[(m,key)]; M=[s for s in nm if s in R2]; e=ev[(m,key)]; lf=Lev[(m,key)]
    i=idx[m]; win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    basket={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in win if sum(1 for s in M if d in R2[s])>=2}
    for s in uni:
        if s in M: continue
        rl=corr(R2[s],basket,win)
        if rl is None: continue
        E_cos.append(float(e@cv[s])); E_fac.append(float(lf@Lco[s])); E_ret.append(float(lf@Bco[s])); Yr.append(rl)
def pc(x):
    x=np.array(x); y=np.array(Yr); x=x-x.mean(); y=y-y.mean()
    r=float((x@y)/(np.sqrt(x@x)*np.sqrt(y@y))); return r, r*math.sqrt((len(x)-2)/(1-r*r))
print(f"\n=== exposure vs realized co-movement (NON-mentioned, n={len(Yr)}) ===")
for lbl,x in [("raw embedding cosine (baseline)",E_cos),("factor-mediated (semantic)",E_fac),("RETURN-ANCHORED mechanism factors",E_ret)]:
    r,t=pc(x); print(f"  {lbl:36} corr {r:+.3f}  t {t:+.0f}")
