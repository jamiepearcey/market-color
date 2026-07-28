# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx","sentence-transformers","torch","scikit-learn"]
# ///
"""STEP 4 + DIRECTION: signed mechanism model. Event has a SHOCK SIGN D_E (net effect_dir of its named
firms); company has SIGNED return betas to each mechanism (leakage-clean: firm excluded from its own
mimicking portfolio). Predicted signed response = D_E * sum_f Lev[E,f]*beta[s,f]. Tests:
 (A) co-movement magnitude (confirm +0.076 holds leakage-clean),
 (B) DIRECTION: does predicted signed response predict the SIGN of a non-mentioned firm's abnormal return?"""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=140; K=24; TOPF=12
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
freq=collections.Counter(); comp_q=collections.defaultdict(list); ev_q=collections.defaultdict(list); ev_names=collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); mh=dmeta.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not mh or mh[0][:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    m=mh[0]; freq[sym[e]]+=1; q=(j.get("quote") or "").strip()
    if q and len(comp_q[sym[e]])<50: comp_q[sym[e]].append((q+". "+mh[1])[:200])
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c.split("__")[0].replace("_"," ").title()
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    if q and len(ev_q[(m,key)])<40: ev_q[(m,key)].append((lab+": "+q+" ("+(j.get("mechanism") or "")+")")[:200])
    ev_names[(m,key)][sym[e]]+=DIRV[d]
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
if __import__("os").environ.get("EMB_CENTER"):
    emb = emb - emb.mean(0)
    _n = np.linalg.norm(emb, axis=1, keepdims=True); emb = emb / np.where(_n > 0, _n, 1)
    print("[EMB_CENTER] embedding space mean-centred", flush=True)
def vec(kk): a,b=im[kk]; v=emb[a:b].mean(0); n=np.linalg.norm(v); return v/n if n else v
cv={s:vec(("C",s)) for s in uni}; ev={k:vec(("E",k)) for k in evs}
EM=np.array([ev[k] for k in evs]); cen=KMeans(K,n_init=5,random_state=0).fit(EM).cluster_centers_
cen=cen/np.linalg.norm(cen,axis=1,keepdims=True)
Lev={k:(ev[k]@cen.T) for k in evs}; Lco={s:(cv[s]@cen.T) for s in uni}
# leakage-clean mimicking portfolios: per factor, top-loading companies; beta excludes the scored firm
top={f:[uni[i] for i in np.argsort(-np.array([Lco[s][f] for s in uni]))[:TOPF]] for f in range(K)}
def facret(f,excl=None):
    mem=[s for s in top[f] if s!=excl]; fr={}
    for d in alld:
        vals=[R2[s][d] for s in mem if d in R2[s]]
        if len(vals)>=4: fr[d]=float(np.mean(vals))
    return fr
base_fr={f:facret(f) for f in range(K)}
def beta(s,f):
    fr=facret(f,excl=s) if s in top[f] else base_fr[f]
    cm=[d for d in fr if d in R2[s]]
    if len(cm)<60: return 0.0
    x=np.array([fr[d] for d in cm]); y=np.array([R2[s][d] for d in cm]); x=x-x.mean()
    return float((x@(y-y.mean()))/(x@x)) if x@x>0 else 0.0
Bco={s:np.array([beta(s,f) for f in range(K)]) for s in uni}
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def corr(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<15: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
comov_pred=[]; comov_real=[]; dir_pred=[]; dir_real=[]
for (m,key) in evs:
    nm=ev_names[(m,key)]; M=[s for s in nm if s in R2]
    D_E=np.sign(sum(nm[s] for s in nm)) or 1.0                      # event shock sign
    lf=Lev[(m,key)]; i=idx[m]; win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    basket={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in win if sum(1 for s in M if d in R2[s])>=2}
    for s in uni:
        if s in M: continue
        exp=float(lf@Bco[s])
        rl=corr(R2[s],basket,win)
        if rl is not None: comov_pred.append(abs(exp)); comov_real.append(rl)      # A: magnitude
        rr=[R2[s][d] for d in win if d in R2[s]]
        if len(rr)>=15:
            dir_pred.append(D_E*exp); dir_real.append(float(np.mean(rr)))           # B: signed return
def pc(x,y):
    x=np.array(x)-np.mean(x); y=np.array(y)-np.mean(y)
    r=float((x@y)/(np.sqrt(x@x)*np.sqrt(y@y))); return r, r*math.sqrt((len(x)-2)/(1-r*r))
r1,t1=pc(comov_pred,comov_real)
print(f"\n=== A) co-movement magnitude (leakage-clean) ===")
print(f"  |signed exposure| vs realized co-move : corr {r1:+.3f} t {t1:+.0f}   (n{len(comov_pred)})  [was +0.076 w/ self-incl]")
dp=np.array(dir_pred); dr=np.array(dir_real); r2,t2=pc(dp,dr)
q=np.quantile(np.abs(dp),0.5); conf=np.abs(dp)>=q
hit=np.mean(np.sign(dp[conf])==np.sign(dr[conf]))
print(f"\n=== B) DIRECTION: predicted signed response vs realized SIGNED abnormal return ===")
print(f"  corr(D_E*exposure, realized signed return) = {r2:+.3f} t {t2:+.0f}  (n{len(dp)})")
print(f"  directional hit-rate (top-half confidence): {hit:.1%}  (50% = chance) — can we call UP vs DOWN?")
