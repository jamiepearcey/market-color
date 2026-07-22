# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx","sentence-transformers","torch"]
# ///
"""THE INFORMATION-GAP / alpha test. Not direction — TIMING. For each event, find NON-mentioned firms with
HIGH embedding exposure that have NOT reacted yet (early-window co-move ~0). Do they DRIFT toward the event
in the LATE window (delayed reaction = tradeable), or stay flat (efficient / no edge)? Compare to low-exposure
unreacted firms. If high-exposure-unreacted drift > 0 and > low-exposure, the exposure model leads the price."""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=140
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
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    if q and len(ev_q[(m,key)])<40: ev_q[(m,key)].append((lab+": "+q)[:200])
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
model=SentenceTransformer("all-MiniLM-L6-v2",device="mps" if torch.backends.mps.is_available() else "cpu")
evs=[(m,k) for (m,k),nm in ev_names.items() if len([x for x in nm if x in R2])>=3 and ev_q[(m,k)]]
allq=[]; im={}
for s in uni: im[("C",s)]=(len(allq),len(allq)+len(comp_q[s])); allq+=comp_q[s]
for k in evs: im[("E",k)]=(len(allq),len(allq)+len(ev_q[k])); allq+=ev_q[k]
emb=model.encode(allq,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
def vec(kk): a,b=im[kk]; v=emb[a:b].mean(0); n=np.linalg.norm(v); return v/n if n else v
cv={s:vec(("C",s)) for s in uni}; ev={k:vec(("E",k)) for k in evs}
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def co(a,basket,days):
    cm=[d for d in days if d in a and d in basket]
    if len(cm)<12: return None
    x=np.array([a[d] for d in cm]);y=np.array([basket[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
rows=[]
for (m,key) in evs:
    i=idx[m]
    if i+3>=len(mo): continue
    M=[s for s in ev_names[(m,key)] if s in R2]
    if len(M)<3: continue
    early=[d for d in alld if d[:7] in mo[i:i+2]]; late=[d for d in alld if d[:7] in mo[i+2:i+4]]
    bE={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in early if sum(1 for s in M if d in R2[s])>=2}
    bL={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in late if sum(1 for s in M if d in R2[s])>=2}
    for s in uni:
        if s in M: continue
        E=float(ev[(m,key)]@cv[s]); ce=co(R2[s],bE,early); cl=co(R2[s],bL,late)
        if ce is not None and cl is not None: rows.append((E,ce,cl))
A=np.array(rows); E=A[:,0]; ce=A[:,1]; cl=A[:,2]
hiE=E>=np.quantile(E,0.75); unreacted=np.abs(ce)<=np.quantile(np.abs(ce),0.33)
def st(mask,lbl):
    v=cl[mask]; m=v.mean(); t=m/(v.std(ddof=1)/len(v)**0.5) if len(v)>1 else 0
    print(f"  {lbl:46} late co-move {m:+.3f}  t{t:+.1f}  n{int(mask.sum())}")
print(f"\n=== INFORMATION GAP: do high-exposure UNREACTED firms drift toward the event LATE? (n={len(A)}) ===")
st(hiE&unreacted, "HIGH exposure + NOT reacted early")
st((~hiE)&unreacted, "low exposure + not reacted early  (control)")
st(hiE&(~unreacted), "HIGH exposure + already reacted (for reference)")
# regression among the unreacted subset: does exposure predict LATE drift?
u=unreacted; x=E[u]-E[u].mean(); y=cl[u]-cl[u].mean(); r=float((x@y)/(np.sqrt(x@x)*np.sqrt(y@y))); t=r*math.sqrt((u.sum()-2)/(1-r*r))
print(f"\n  among UNREACTED firms: corr(exposure, LATE drift) = {r:+.3f}  t{t:+.1f}")
print("  => if HIGH-unreacted late>0 AND beats control AND corr>0 sig -> exposure LEADS price (alpha). else efficient.")
