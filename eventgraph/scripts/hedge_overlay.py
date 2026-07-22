# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx","sentence-transformers","torch"]
# ///
"""FABLE #1 + R3: the decisive rigor recheck. For each event, does TEXT (embedding exposure) predict a
non-mentioned firm's co-movement with the event basket INCREMENTAL to the missing baseline = PRE-EVENT
trailing correlation with the basket? Report CONTEMPORANEOUS (the F22/F24 claim) AND FORWARD (R3: predictive
risk). Event-level inference: per-event IC, mean + bootstrap CI + t ACROSS events (not pooled). Leakage-clean:
trailing/baseline use only pre-event months; target windows are strictly separate."""
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
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
head={json.loads(l)["doc_id"]:(json.loads(l).get("headline") or "") for l in open(G/"lake/document.jsonl")}
# T2: broaden "mentioned" = any firm appearing as causal effect OR sentiment target OR relation endpoint in the event's docs
freq=collections.Counter(); comp_qm=collections.defaultdict(list); ev_q=collections.defaultdict(list); ev_named=collections.defaultdict(lambda: collections.defaultdict(int)); ev_docs=collections.defaultdict(set)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); did=j.get("doc_id"); m=docm.get(did); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; q=(j.get("quote") or "").strip()
    if q and len(comp_qm[sym[e]])<80: comp_qm[sym[e]].append((m,(q+". "+head.get(did,""))[:200]))
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    if q and len(ev_q[(m,key)])<40: ev_q[(m,key)].append((lab+": "+q)[:200])
    ev_named[(m,key)][sym[e]]+=DIRV[d]; ev_docs[(m,key)].add(did)
# broader mention set from sentiment + relation targets sharing the event's docs
doc_ment=collections.defaultdict(set)
for l in open(G/"lake/sentiment_annotation.jsonl"):
    j=json.loads(l); t=j.get("target_entity")
    if t in sym: doc_ment[j.get("doc_id")].add(sym[t])
if (G/"lake/relation_edge.jsonl").exists():
    for l in open(G/"lake/relation_edge.jsonl"):
        j=json.loads(l)
        for k in ("source_entity","target_entity"):
            if j.get(k) in sym: doc_ment[j.get("doc_id")].add(sym[j[k]])
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
uni=[s for s in uni if s in R2 and comp_qm.get(s)]
import torch
from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2",device="mps" if torch.backends.mps.is_available() else "cpu")
evs=[(m,k) for (m,k),nm in ev_named.items() if len([x for x in nm if x in R2])>=3 and ev_q[(m,k)]]
allq=[]; qmeta=[]; im={}
for s in uni:
    for (qm,qt) in comp_qm[s]: qmeta.append((s,qm)); allq.append(qt)
for k in evs: im[("E",k)]=(len(allq),len(allq)+len(ev_q[k])); allq+=ev_q[k]
emb=model.encode(allq,batch_size=256,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
qbyfirm=collections.defaultdict(list)
for gi,(s,qm) in enumerate(qmeta): qbyfirm[s].append((gi,qm))
def ev_vec(k): a,b=im[("E",k)]; v=emb[a:b].mean(0); n=np.linalg.norm(v); return v/n if n else v
ev={k:ev_vec(k) for k in evs}
def preprofile(s,evmonth):
    idxs=[gi for gi,qm in qbyfirm.get(s,()) if qm< evmonth]
    if len(idxs)<2: return None
    v=emb[idxs].mean(0); n=np.linalg.norm(v); return v/n if n else None
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def cw(a,basket,days):
    cm=[d for d in days if d in a and d in basket]
    if len(cm)<12: return None
    x=np.array([a[d] for d in cm]);y=np.array([basket[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
def _f(x): return np.array([np.nan if v is None else v for v in x],float)
def ic(x,y):
    x=_f(x);y=_f(y); m=np.isfinite(x)&np.isfinite(y); x,y=x[m],y[m]
    if len(x)<8 or x.std()==0 or y.std()==0: return None
    return float(np.corrcoef(x,y)[0,1])
def partial(a,b,ctrl):
    a,b,ctrl=_f(a),_f(b),_f(ctrl); m=np.isfinite(a)&np.isfinite(b)&np.isfinite(ctrl); a,b,ctrl=a[m],b[m],ctrl[m]
    if len(a)<10: return None
    def res(v):
        X=np.column_stack([np.ones(len(ctrl)),ctrl]); bb,*_=np.linalg.lstsq(X,v,rcond=None); return v-X@bb
    ra,rb=res(a),res(b)
    return None if ra.std()==0 or rb.std()==0 else float(np.corrcoef(ra,rb)[0,1])
per=collections.defaultdict(list); ROWS=[]   # ROWS: (text_exposure, trailing, forward) per firm-event
for (m,key) in evs:
    i=idx[m]
    if i<3 or i+3>=len(mo): continue
    ment=set(ev_named[(m,key)]) | {t for did in ev_docs[(m,key)] for t in doc_ment.get(did,())}
    M=[s for s in ev_named[(m,key)] if s in R2]
    if len(M)<3: continue
    preW=[d for d in alld if d[:7] in mo[i-3:i]]; conW=[d for d in alld if d[:7] in mo[i:i+2]]; fwdW=[d for d in alld if d[:7] in mo[i+2:i+4]]
    def basket(days): return {d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in days if sum(1 for s in M if d in R2[s])>=2}
    bp,bc,bf=basket(preW),basket(conW),basket(fwdW)
    E=[]; TR=[]; CO=[]; FW=[]
    for s in uni:
        if s in ment: continue                       # T2: broadened non-mentioned
        pp=preprofile(s,m)
        if pp is None: continue
        E.append(float(ev[(m,key)]@pp)); TR.append(cw(R2[s],bp,preW)); CO.append(cw(R2[s],bc,conW)); FW.append(cw(R2[s],bf,fwdW))
    Ea,TRa,FWa=_f(E),_f(TR),_f(FW); mm=np.isfinite(Ea)&np.isfinite(TRa)&np.isfinite(FWa)
    if mm.sum()<10: continue
    Ea,TRa,FWa=Ea[mm],TRa[mm],FWa[mm]
    per["ic_E_dcorr"].append(ic(Ea, FWa-TRa))                       # does text-exposure predict corr CHANGE?
    per["emb|trail->forward"].append(partial(Ea,FWa,TRa))
    for e_,t_,f_ in zip(Ea,TRa,FWa): ROWS.append((e_,t_,f_))
def report(name):
    a=np.array([v for v in per[name] if v is not None])
    if len(a)<5: print(f"  {name:22} n<5"); return
    m=a.mean(); t=m/(a.std(ddof=1)/len(a)**0.5)
    rng=np.random.RandomState(0); bs=[rng.choice(a,len(a)).mean() for _ in range(2000)]; lo,hi=np.percentile(bs,[2.5,97.5])
    print(f"  {name:26} mean {m:+.3f}  t {t:+.1f}  95%CI [{lo:+.3f},{hi:+.3f}]  ({len(a)} ev)")
A=np.array(ROWS); E,TR,FW=A[:,0],A[:,1],A[:,2]; DC=FW-TR
print(f"\n=== HEDGE-FAILURE OVERLAY: does pre-event text-exposure forecast correlation RISING? (n={len(A)} firm-events, leakage-free) ===")
report("ic_E_dcorr"); report("emb|trail->forward")
qlo,qhi=np.quantile(E,[1/3,2/3])
hi=E>=qhi; lo=E<=qlo
print(f"\n  text-exposure tercile:  Δcorr (forward-trailing)   forward corr   %rising")
for lbl,mask in [("HIGH exposure",hi),("LOW exposure",lo)]:
    print(f"    {lbl:14} {DC[mask].mean():+.3f}                  {FW[mask].mean():+.3f}          {np.mean(DC[mask]>0):.0%}")
print(f"    spread (HIGH-LOW Δcorr) = {DC[hi].mean()-DC[lo].mean():+.3f}")
# diversification-trap: among ELEVATED trailing pairs (>0.3), do high-exposure ones persist while low revert?
el=TR>0.3
if el.sum()>50:
    print(f"\n  === diversification trap: ELEVATED pairs (trailing>0.3, n={el.sum()}) — do they mean-revert or persist? ===")
    for lbl,mask in [("HIGH exposure",hi&el),("LOW exposure",lo&el)]:
        if mask.sum()>20: print(f"    {lbl:14} trailing {TR[mask].mean():+.2f} -> forward {FW[mask].mean():+.2f}  (Δ {DC[mask].mean():+.3f}, {'persists/rises' if DC[mask].mean()>-0.02 else 'reverts'})")
    print("    => if high-exposure elevated pairs persist while low-exposure revert, text flags which hedges will fail.")
