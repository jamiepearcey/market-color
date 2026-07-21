# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""DAILY ROLLING signed attribution monitor (F13 productized).
For a sample of top drivers, roll a trailing W-day window across 2010-12 BBG data one
trading day at a time and record how much each driver explains its connected pairs'
co-movement THAT day. Leave-two-out signed factor (no leakage). Output -> JSON for viz.
This is a daily REPLAY of the historical corpus; it runs identically on a live feed."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP
from news_contagion import ar_series
W=63; MINCO=40; TOPN=160; NDRIVERS=6; STEP=1
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); drv_net=collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; drv_net[c][sym[e]]+=DIRV[d]
print(f"building AR series for top {TOPN} names ...", flush=True)
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=TOPN: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR); DAYS=sorted(set().union(*[set(AR[s]) for s in AR])); DI={d:i for i,d in enumerate(DAYS)}
# arrays for speed: name -> (idx array, val array)
def label(c): return c.split("__")[0].replace("_"," ").title()
# select drivers: net sign per connected priceable name, >=5 connected names, rank by count
cand=[]
for c,nm in drv_net.items():
    names=sorted(s for s,v in nm.items() if s in U and abs(v)>=1)
    if len(names)>=5: cand.append((len(names),c,names))
cand.sort(reverse=True)
sel=cand[:NDRIVERS]
print("selected drivers:")
for n,c,names in sel: print(f"  {label(c):32} {n} names")
# dense matrix of AR aligned to DAYS (nan where missing)
M={s:np.array([AR[s].get(d,np.nan) for d in DAYS]) for s in U}
def corr(a,b):
    ok=~(np.isnan(a)|np.isnan(b))
    if ok.sum()<MINCO: return None
    x=a[ok]-a[ok].mean(); y=b[ok]-b[ok].mean()
    if x.std()==0 or y.std()==0: return None
    return float((x*y).mean()/(x.std()*y.std()))
def resid(a,f):
    ok=~(np.isnan(a)|np.isnan(f))
    if ok.sum()<MINCO: return a
    y=a[ok]; X=np.column_stack([np.ones(ok.sum()),f[ok]])
    b,*_=np.linalg.lstsq(X,y,rcond=None); out=np.full_like(a,np.nan); out[ok]=y-X@b; return out
start=W; idxs=list(range(start,len(DAYS),STEP))
out={"window":W,"dates":[DAYS[i] for i in idxs],"drivers":[]}
for n,c,names in sel:
    names=[s for s in names if s in M]
    signs={s:np.sign(drv_net[c][s]) for s in names}
    series=[]
    for i in idxs:
        sl=slice(i-W,i)
        basket=np.vstack([signs[s]*M[s][sl] for s in names])  # signed
        bsum=np.nansum(basket,axis=0); bcnt=np.sum(~np.isnan(basket),axis=0)
        attrs=[]; raws=[]
        for x in range(len(names)):
            for y in range(x+1,len(names)):
                A,B=names[x],names[y]; a=M[A][sl]; b=M[B][sl]
                c0=corr(a,b)
                if c0 is None or c0<0.2: continue  # sample: meaningfully correlated pairs
                # leave-two-out signed factor
                num=bsum-np.where(np.isnan(a),0,signs[A]*a)-np.where(np.isnan(b),0,signs[B]*b)
                den=bcnt-(~np.isnan(a)).astype(int)-(~np.isnan(b)).astype(int)
                f=np.where(den>=2,num/np.maximum(den,1),np.nan)
                c1=corr(resid(a,f),resid(b,f))
                if c1 is None: continue
                attrs.append(c0-c1); raws.append(c0)
        series.append({"attr":round(float(np.mean(attrs)),4) if attrs else None,
                       "corr":round(float(np.mean(raws)),4) if raws else None,
                       "n":len(attrs)})
    out["drivers"].append({"id":c,"label":label(c),"n_names":n,"series":series})
    va=[s["attr"] for s in series if s["attr"] is not None]
    print(f"  {label(c):32} mean daily attr {np.mean(va):+.3f}  range [{min(va):+.3f},{max(va):+.3f}]  npts {len(va)}", flush=True)
Path("../data/eg_runs/eg100k_graph/attribution_daily.json").write_text(json.dumps(out))
print(f"\n-> {len(idxs)} trading days x {NDRIVERS} drivers written to attribution_daily.json")
