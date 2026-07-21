# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""Precompute the correlation-attribution EXPLORER data (compact JSON for a single-file app).
Per instrument: name, sector (argmax sector-ETF beta), news-driver profile.
Per pair (canonical A|B): weekly rolling 63d correlation.
Per pair x shared-driver: weekly attribution time series (leave-two-out signed residualization).
Universe = top-N news-dense priceable names. Weekly grid over 2010-2012."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
from news_contagion import ar_series
N=64; W=63; STEP=5; MINCO=40
SECTORS=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
SECNAME={"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLV":"Health Care","XLI":"Industrials",
         "XLY":"Cons. Disc.","XLP":"Cons. Staples","XLU":"Utilities","XLB":"Materials"}
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}; name={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
        sym[j["entity_id"]]=j["symbol"]
        name.setdefault(j["symbol"], j["entity_id"].split("__")[0].replace("_"," ").title())
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); drv_net=collections.defaultdict(lambda: collections.defaultdict(int)); drv_cnt=collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; drv_net[c][sym[e]]+=DIRV[d]; drv_cnt[c][sym[e]]+=1
print(f"building AR for top {N} ...", flush=True)
AR={}
for s,_ in freq.most_common():
    if len(AR)>=N: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
uni=sorted(AR); U=set(uni)
print(f"{len(uni)} names; fetching sector SPDRs ...", flush=True)
SEC={e:logret(yahoo(e,cache,p1,p2)) for e in SECTORS}
secdays=sorted(set.intersection(*[set(SEC[e]) for e in SEC]))
# sector = argmax |beta| of AR on each single SPDR (R2-based)
def sector_of(s):
    best=("—",-1)
    cm=[d for d in AR[s] if d in secdays]
    if len(cm)<60: return "—"
    y=np.array([AR[s][d] for d in cm]); y=y-y.mean()
    for e in SECTORS:
        x=np.array([SEC[e][d] for d in cm]); x=x-x.mean()
        if x.std()==0: continue
        r=np.corrcoef(x,y)[0,1]
        if r*r>best[1]: best=(SECNAME[e],r*r)
    return best[0]
DAYS=sorted(set().union(*[set(AR[s]) for s in AR])); DI={d:i for i,d in enumerate(DAYS)}
M={s:np.array([AR[s].get(d,np.nan) for d in DAYS]) for s in uni}
weeks=list(range(W,len(DAYS),STEP)); wdates=[DAYS[i] for i in weeks]
def corrw(a,b):
    ok=~(np.isnan(a)|np.isnan(b))
    if ok.sum()<MINCO: return None
    x=a[ok]-a[ok].mean(); y=b[ok]-b[ok].mean()
    if x.std()==0 or y.std()==0: return None
    return round(float((x*y).mean()/(x.std()*y.std())),3)
def residw(a,f):
    ok=~(np.isnan(a)|np.isnan(f))
    if ok.sum()<MINCO: return None
    y=a[ok]; X=np.column_stack([np.ones(ok.sum()),f[ok]]); b,*_=np.linalg.lstsq(X,y,rcond=None)
    out=np.full_like(a,np.nan); out[ok]=y-X@b; return out
# instruments
insts=[]
inst_drv=collections.defaultdict(dict)  # sym -> {driverId: (label,sign,cnt)}
for c,nm in drv_net.items():
    for s,v in nm.items():
        if s in U and v!=0: inst_drv[s][c]=(c.split("__")[0].replace("_"," ").title(), int(np.sign(v)), drv_cnt[c][s])
for s in uni:
    ds=sorted(inst_drv[s].items(), key=lambda kv:-kv[1][2])[:12]
    insts.append({"sym":s,"name":name.get(s,s),"sector":sector_of(s),"freq":freq[s],
                  "drivers":[{"id":c,"label":v[0],"sign":v[1],"n":v[2]} for c,v in ds]})
print("computing weekly correlations ...", flush=True)
corr={}
for x in range(len(uni)):
    for y in range(x+1,len(uni)):
        A,B=uni[x],uni[y]; ser=[corrw(M[A][i-W:i],M[B][i-W:i]) for i in weeks]
        if max((abs(v) for v in ser if v is not None), default=0)>=0.2: corr[f"{A}|{B}"]=ser
print(f"{len(corr)} correlated pairs; computing driver attribution ...", flush=True)
# pooled signed basket per driver, dense
drv_names={c:[s for s in nm if s in U and nm[s]!=0] for c,nm in drv_net.items()}
attr={}; npair=0
for key,cser in corr.items():
    A,B=key.split("|")
    if max((v for v in cser if v is not None), default=0)<0.3: continue  # only meaningfully +corr pairs
    shared=[c for c in inst_drv[A] if c in inst_drv[B] and len(drv_names[c])>=4]
    if not shared: continue
    pa={}
    for c in shared:
        others=[s for s in drv_names[c] if s not in (A,B)]
        if len(others)<2: continue
        sgn={s:np.sign(drv_net[c][s]) for s in others}
        aser=[]
        for i in weeks:
            sl=slice(i-W,i); a=M[A][sl]; b=M[B][sl]; c0=corrw(a,b)
            if c0 is None: aser.append(None); continue
            basket=np.vstack([sgn[s]*M[s][sl] for s in others]); f=np.nanmean(basket,axis=0)
            rA=residw(a,f); rB=residw(b,f)
            c1=corrw(rA,rB) if rA is not None and rB is not None else None
            aser.append(round(c0-c1,3) if c1 is not None else None)
        va=[v for v in aser if v is not None]
        if va: pa[c]={"label":inst_drv[A][c][0],"sign":int(inst_drv[A][c][1]*inst_drv[B][c][1]),
                      "mean":round(float(np.mean(va)),3),"series":aser}
    if pa: attr[key]=dict(sorted(pa.items(), key=lambda kv:-kv[1]["mean"])[:5]); npair+=1
out={"window":W,"dates":wdates,"instruments":insts,"corr":corr,"attr":attr}
Path("../data/eg_runs/eg100k_graph/explorer.json").write_text(json.dumps(out))
sz=len(json.dumps(out))/1e6
print(f"\n-> explorer.json  {len(insts)} instruments, {len(corr)} corr pairs, {npair} pairs w/ drivers, {len(wdates)} weeks, {sz:.1f}MB")
