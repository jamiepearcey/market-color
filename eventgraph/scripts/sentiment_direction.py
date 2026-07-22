# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""Does SENTIMENT recover DIRECTION (where the mechanism model failed at 51.4%)? For each sentiment
(signed polarity toward a priceable firm), measure the firm's signed abnormal return around it:
CONTEMPORANEOUS [D..D+5] and LEAD [D+6..D+15]. IC + directional hit-rate. Compare to effect_dir (F6 +0.10).
If contemporaneous works but lead is null, sentiment is descriptive too — consistent with the whole arc."""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP
from news_contagion import ar_series
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
# collect sentiments on priceable targets, 2010-2012
S=collections.Counter(); rows=[]
for l in open(G/"lake/sentiment_annotation.jsonl"):
    j=json.loads(l); t=j.get("target_entity"); pol=j.get("polarity"); d=(j.get("as_of") or "")[:10]
    if t not in sym or pol is None or not d or d[:4] not in {"2010","2011","2012"}: continue
    inten=1.5 if j.get("intensity")=="strong" else 1.0
    rows.append((sym[t], d, float(pol)*inten)); S[sym[t]]+=1
tick=[s for s,_ in S.most_common(150)]
print(f"{len(rows)} sentiments on priceable firms; building AR for top {len(tick)} targets ...",flush=True)
AR={}
for s in tick:
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
DAYS={s:sorted(AR[s]) for s in AR}
def window_ret(s,d,a,b):
    if s not in AR: return None
    ds=DAYS[s]; 
    # first trading day >= d
    import bisect; i=bisect.bisect_left(ds,d)
    if i+b>len(ds) or i>=len(ds): return None
    seg=[AR[s][ds[k]] for k in range(i+a,min(i+b,len(ds)))]
    return float(np.sum(seg)) if seg else None
def run(a,b,label):
    P=[]; Rr=[]
    for s,d,pol in rows:
        r=window_ret(s,d,a,b)
        if r is not None: P.append(pol); Rr.append(r)
    P=np.array(P); Rr=np.array(Rr)
    x=P-P.mean(); y=Rr-Rr.mean(); ic=float((x@y)/(np.sqrt(x@x)*np.sqrt(y@y))); t=ic*math.sqrt((len(P)-2)/(1-ic*ic))
    conf=np.abs(P)>=np.quantile(np.abs(P),0.0)  # all (polarity mostly +-1)
    hit=np.mean(np.sign(P[conf])==np.sign(Rr[conf]))
    print(f"  {label:26} IC {ic:+.3f} t {t:+.0f}  | directional hit-rate {hit:.1%}  (n{len(P)})")
print("\n=== does sentiment polarity predict the SIGN of the firm's abnormal return? ===")
run(0,5,"CONTEMPORANEOUS [D..D+5]")
run(6,16,"LEAD [D+6..D+15]")
run(-5,0,"PRE [D-5..D] (leakage chk)")
print("\n  (mechanism-model direction was 51.4% hit / +0.039; F6 causal-dir IC was +0.10 contemporaneous)")
