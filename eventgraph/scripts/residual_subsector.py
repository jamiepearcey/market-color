# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""DECISIVE: is the news residual genuine event info, or just FINER sub-industry sector?
Add sub-industry ETFs to the broad-9-sector block, re-residualize, re-measure the news-linked
residual excess. If it collapses toward 0 -> the news graph is rediscovering sub-industries.
If it survives -> genuinely event-driven, beyond any sector taxonomy."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
BROAD=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
SUBIND=["KBE","KRE","KIE","IAI","XHB","OIH","SMH","IYT","IBB","ITB","XRT","IYR"]  # bank/reg-bank/insurance/broker/homebuild/oilsvc/semi/transport/biotech/homeconstr/retail/reit
N_UNIV=120
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); cause_m=collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1
    if c.split("__")[-1] not in MACRO_T: cause_m[m][c].add(sym[e])
years={"2010","2011","2012"}
def ret(t): return logret(yahoo(t,cache,p1,p2))
etf={e:ret(e) for e in BROAD+SUBIND}
etf={e:v for e,v in etf.items() if len([d for d in v if d[:4] in years])>300}  # keep ETFs with data
subok=[e for e in SUBIND if e in etf]
print(f"sub-industry ETFs available 2010-12: {subok}")
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=N_UNIV: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in years)>400: uni.append(s)
R={s:ret(s) for s in uni}
alld=sorted({d for s in uni for d in R[s] if d[:4] in years and d in Fmap and all(d in etf[e] for e in BROAD)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
def design(days,secset):
    cols=[[Fmap[d][i] for d in days] for i in mcol]+[[etf[e][d] if d in etf[e] else np.nan for d in days] for e in secset]
    X=np.nan_to_num(np.array(cols,float).T-np.nanmean(np.array(cols,float).T,0)); return np.column_stack([np.ones(len(days)),X])
def residualize(secset):
    R2={}
    for s in uni:
        days=[d for d in alld if d in R[s]]
        if len(days)<200: continue
        y=np.array([R[s][d] for d in days]); X=design(days,secset); b,*_=np.linalg.lstsq(X,y,rcond=None); R2[s]=dict(zip(days,y-X@b))
    return R2
def linked_excess(R2):
    names=sorted(R2); linked=set()
    for m in cause_m:
        for c,ss in cause_m[m].items():
            ss=sorted(x for x in ss if x in R2)
            for i in range(len(ss)):
                for j in range(i+1,len(ss)): linked.add(tuple(sorted((ss[i],ss[j]))))
    def corr(a,b):
        cm=[d for d in alld if d in a and d in b]
        if len(cm)<40: return None
        x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
    lk=[];un=[]
    for i in range(len(names)):
        for j in range(i+1,len(names)):
            c=corr(R2[names[i]],R2[names[j]])
            if c is None: continue
            (lk if tuple(sorted((names[i],names[j]))) in linked else un).append(c)
    d=np.mean(lk)-np.mean(un); se=(np.var(lk)/len(lk)+np.var(un)/len(un))**0.5
    return np.mean(lk),np.mean(un),d,d/se,len(lk)
print("\n=== news-linked residual EXCESS: broad sector vs + sub-industry ===")
for lbl,secset in [("broad 9 GICS only",BROAD),("+ sub-industry ETFs",BROAD+subok)]:
    R2=residualize(secset); l,u,d,t,n=linked_excess(R2)
    print(f"  {lbl:24}: linked {l:+.3f} unlinked {u:+.3f}  EXCESS {d:+.3f}  t{t:+.1f}  (n{n})")
print("\n  if excess collapses -> residual was sub-industry; if it survives -> genuinely event-driven beyond sector.")
