# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx","sentence-transformers","torch"]
# ///
"""Event-exposure with REAL semantic embeddings (all-MiniLM-L6-v2, Metal/MPS). Company vector = mean
embedding of its news quotes; event vector = mean embedding of its quotes+catalyst+mechanism. Exposure
= cosine. Re-runs the F22 validation (does embedding exposure predict realized co-movement for NON-
mentioned firms?) and the landscape, and compares to the tf-idf baseline (corr +0.047)."""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
SN={"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLV":"Health Care","XLI":"Industrials","XLY":"Cons Disc","XLP":"Cons Staples","XLU":"Utilities","XLB":"Materials"}
N_UNIV=140; MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
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
    j=json.loads(l); did=j.get("doc_id"); mh=dmeta.get(did); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not mh or mh[0][:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    m=mh[0]; freq[sym[e]]+=1
    q=(j.get("quote") or "").strip()
    if q and len(comp_q[sym[e]])<50: comp_q[sym[e]].append((q+". "+mh[1])[:200])
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c.split("__")[0].replace("_"," ").title()
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    if q and len(ev_q[(m,key)])<40: ev_q[(m,key)].append((lab+": "+q+" ("+(j.get("mechanism") or "")+")")[:200])
    ev_names[(m,key)][sym[e]]+=DIRV[d]; ev_lab[(m,key)]=(lab,typ)
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
def secof(s):
    r=R[s]; best=("?",-1)
    for e in SPDR:
        cm=[d for d in alld if d in r and d in spdr[e]]
        if len(cm)<40: continue
        x=np.array([r[d] for d in cm]);y=np.array([spdr[e][d] for d in cm])
        if x.std() and y.std():
            cc=abs(np.corrcoef(x,y)[0,1])
            if cc>best[1]: best=(SN[e],cc)
    return best[0]
SEC={s:secof(s) for s in uni}
print("loading embedding model (all-MiniLM-L6-v2) ...",flush=True)
import torch
from sentence_transformers import SentenceTransformer
dev="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
print(f"device = {dev}",flush=True)
model=SentenceTransformer("all-MiniLM-L6-v2",device=dev)
# collect all quote strings, encode once, then mean-pool per company / event
allq=[]; idxmap={}
for s in uni:
    idxmap[("C",s)]=[len(allq)+i for i in range(len(comp_q[s]))]; allq+=comp_q[s]
evs=[(m,k) for (m,k),nm in ev_names.items() if len([x for x in nm if x in R2])>=3 and ev_q[(m,k)]]
for (m,k) in evs:
    idxmap[("E",(m,k))]=[len(allq)+i for i in range(len(ev_q[(m,k)]))]; allq+=ev_q[(m,k)]
print(f"encoding {len(allq)} quote strings ...",flush=True)
emb=model.encode(allq,batch_size=256,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
# EMB_CENTER=1 -> mean-centre the embedding space (F43). Transformer spaces are
# anisotropic: uncentred cosine is dominated by proximity to the corpus centroid,
# which tracks coverage volume and therefore firm size. Two results died here.
if __import__("os").environ.get("EMB_CENTER"):
    import numpy as _np
    emb = emb - emb.mean(0)
    _n = _np.linalg.norm(emb, axis=1, keepdims=True); emb = emb / _np.where(_n > 0, _n, 1)
    print("[EMB_CENTER] embedding space mean-centred", flush=True)
def vecof(kk):
    idl=idxmap.get(kk,[])
    if not idl: return None
    v=emb[idl].mean(0); n=np.linalg.norm(v); return v/n if n else v
cvec={s:vecof(("C",s)) for s in uni}
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def corr(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<15: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
Xe=[]; Yr=[]; landscapes=[]
for (m,key) in evs:
    nm=ev_names[(m,key)]; M=[s for s in nm if s in R2]
    ev=vecof(("E",(m,key)))
    if ev is None: continue
    i=idx[m]; win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    basket={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in win if sum(1 for s in M if d in R2[s])>=2}
    rows=[]
    for s in uni:
        ex=float(cvec[s]@ev) if cvec[s] is not None else 0.0; rl=corr(R2[s],basket,win)
        rows.append({"t":s,"sec":SEC[s],"exp":round(ex,3),"real":None if rl is None else round(rl,3),"m":int(s in M)})
        if s not in M and rl is not None: Xe.append(ex); Yr.append(rl)
    rows.sort(key=lambda r:-r["exp"]); lab,typ=ev_lab[(m,key)]
    landscapes.append({"month":m,"label":lab,"type":typ,"n_named":len(M),"named":M,"rows":rows[:34]})
Xe=np.array(Xe); Yr=np.array(Yr); xe=Xe-Xe.mean(); yr=Yr-Yr.mean()
r=float((xe@yr)/(np.sqrt(xe@xe)*np.sqrt(yr@yr))); t=r*math.sqrt((len(Xe)-2)/(1-r*r))
print(f"\n=== EMBEDDING exposure vs realized co-movement (NON-mentioned firms) ===")
print(f"  corr(embedding-exposure, realized) = {r:+.3f}  t {t:+.0f}  (n={len(Xe)})   [tf-idf baseline was +0.047]")
Path("../data/eg_runs/eg100k_graph/exposure_landscape_emb.json").write_text(json.dumps({"events":landscapes,"sectors":list(SN.values())}))
alm=next((e for e in landscapes if "Almunia" in e["label"]),landscapes[0])
print(f"\n  LIBOR/Almunia — named {alm['n_named']}: {','.join(alm['named'][:8])}")
print("  top surfaced NON-mentioned (embedding):")
for rr in [x for x in alm["rows"] if not x["m"]][:8]: print(f"     {rr['t']:6} ({rr['sec'][:4]}) exp {rr['exp']:.2f} realized {rr['real']}")
