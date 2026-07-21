# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""PAIR-LEVEL daily attribution (the atomic, checkable unit).
For specific named pairs, roll a trailing window daily and plot TWO lines:
  raw rolling correlation, and the counterfactual correlation with a named driver's
  channel removed (leave-two-out signed residualization). The gap = attribution.
Emits a self-contained small-multiples HTML. Default: Moody's driver, GS-MS & BAC-C."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP
from news_contagion import ar_series
W=63; MINCO=40; DRIVER="moody_s_investors_service"
PAIRS=[("GS","MS"),("BAC","C")]
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security": sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
net=collections.defaultdict(int)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or not c or d not in DIRV or e not in sym: continue
    if c.split("__")[0]==DRIVER: net[sym[e]]+=DIRV[d]
drv_names=[s for s,v in net.items() if abs(v)>=1]
need=sorted(set(drv_names)|{x for p in PAIRS for x in p})
print(f"driver connects {len(drv_names)} names; building AR for {len(need)} ...", flush=True)
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s in need:
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
DAYS=sorted(set().union(*[set(AR[s]) for s in AR]))
M={s:np.array([AR[s].get(d,np.nan) for d in DAYS]) for s in AR}
signs={s:np.sign(net[s]) for s in drv_names if s in AR}
dn=[s for s in drv_names if s in AR]
def corr(a,b):
    ok=~(np.isnan(a)|np.isnan(b))
    if ok.sum()<MINCO: return None
    x=a[ok]-a[ok].mean(); y=b[ok]-b[ok].mean()
    if x.std()==0 or y.std()==0: return None
    return float((x*y).mean()/(x.std()*y.std()))
def resid(a,f):
    ok=~(np.isnan(a)|np.isnan(f))
    if ok.sum()<MINCO: return a
    y=a[ok]; X=np.column_stack([np.ones(ok.sum()),f[ok]]); b,*_=np.linalg.lstsq(X,y,rcond=None)
    out=np.full_like(a,np.nan); out[ok]=y-X@b; return out
idxs=list(range(W,len(DAYS)))
out={"window":W,"driver":DRIVER.replace("_"," ").title(),"dates":[DAYS[i] for i in idxs],"pairs":[]}
for A,B in PAIRS:
    if A not in M or B not in M: print(f"  skip {A}-{B} (no AR)"); continue
    ser=[]
    for i in idxs:
        sl=slice(i-W,i); a=M[A][sl]; b=M[B][sl]
        c0=corr(a,b)
        others=[s for s in dn if s not in (A,B)]
        basket=np.vstack([signs[s]*M[s][sl] for s in others]) if others else None
        f=np.nanmean(basket,axis=0) if basket is not None else None
        c1=corr(resid(a,f),resid(b,f)) if f is not None else None
        ser.append({"c0":None if c0 is None else round(c0,4),"c1":None if c1 is None else round(c1,4)})
    va=[s["c0"]-s["c1"] for s in ser if s["c0"] is not None and s["c1"] is not None]
    out["pairs"].append({"a":A,"b":B,"label":f"{A}–{B}","series":ser})
    print(f"  {A}-{B}: raw corr mean {np.nanmean([s['c0'] for s in ser if s['c0'] is not None]):+.2f}  attribution mean {np.mean(va):+.3f}  (n {len(va)})", flush=True)
(G/"pair_attribution.json").write_text(json.dumps(out))
print(f"-> pair_attribution.json ({len(idxs)} days, {len(out['pairs'])} pairs)")
