# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""Does catalyst normalization give BETTER results? Compare news-linked residual co-movement under
RAW cause grouping vs CANONICAL catalyst grouping. Key cell: 'canon-only' pairs (linked only after
merging fragments) — if their residual co-movement is high (like linked), the merges captured REAL
links; if ~0 (like unlinked), they added noise."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=110
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
BROADCANON={"macro","monetary","fiscal"}
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
CANON=json.load(open(G/"catalyst_map.json"))
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); raw_m=collections.defaultdict(lambda: collections.defaultdict(set)); can_m=collections.defaultdict(lambda: collections.defaultdict(set))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1
    if c.split("__")[-1] not in MACRO_T: raw_m[m][c].add(sym[e])   # raw specific cause
    ci=CANON.get(c)
    if ci and ci["type"] not in BROADCANON: can_m[m][ci["key"]].add(sym[e])  # canonical specific catalyst
    elif not ci and c.split("__")[-1] not in MACRO_T: can_m[m][c].add(sym[e])
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
names=sorted(R2)
def links(mm):
    out=set()
    for m in mm:
        for c,ss in mm[m].items():
            ss=sorted(x for x in ss if x in R2)
            for i in range(len(ss)):
                for j in range(i+1,len(ss)): out.add((ss[i],ss[j]))
    return out
Lraw=links(raw_m); Lcan=links(can_m)
def corr(a,b):
    cm=[d for d in alld if d in a and d in b]
    if len(cm)<40: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
buck=collections.defaultdict(list)
for i in range(len(names)):
    for j in range(i+1,len(names)):
        p=(names[i],names[j]); c=corr(R2[p[0]],R2[p[1]])
        if c is None: continue
        r=p in Lraw; k=p in Lcan
        buck["raw" if r else "unlinked_raw"].append(c)
        buck["canon" if k else "unlinked_canon"].append(c)
        if k and not r: buck["canon_only"].append(c)      # NEW links from merging fragments
        if r: buck["raw_only_check"].append(c)
def st(a): a=np.array(a); return f"mean {a.mean():+.3f}  n {len(a)}"
print(f"=== news-linked residual co-movement: RAW cause vs CANONICAL catalyst grouping ===")
print(f"  RAW-linked pairs      : {st(buck['raw'])}")
print(f"  CANONICAL-linked pairs: {st(buck['canon'])}")
print(f"  unlinked (canon)      : {st(buck['unlinked_canon'])}")
print(f"\n  *** canon-ONLY pairs (linked only AFTER merging fragments) : {st(buck['canon_only'])} ***")
print(f"  => if canon-only mean is HIGH (~linked), the merges captured REAL links; if ~unlinked, they added noise.")
lc=np.array(buck['canon']); uc=np.array(buck['unlinked_canon']); lr=np.array(buck['raw']); ur=np.array(buck['unlinked_raw'])
er=lr.mean()-ur.mean(); ec=lc.mean()-uc.mean()
print(f"\n  RAW excess (linked-unlinked)      = {er:+.3f}  ({len(lr)} linked)")
print(f"  CANONICAL excess (linked-unlinked)= {ec:+.3f}  ({len(lc)} linked)")
print(f"  => {'BETTER: canonical captures more links at similar/higher strength' if ec>=er-0.005 and len(lc)>len(lr) else 'mixed'}")
