# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""Calibration + OOS stability of the SIGNED driver attribution — is it product-ready?
For same-sign (co-movement) driver removals: (1) distribution, (2) OUT-OF-SAMPLE stability
(does the in-window Δcorr hold next window), (3) CALIBRATION — does the embedding-contribution
magnitude predict the realized attribution (so we can quote a per-pair confidence)."""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret, wls, FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"; rng=np.random.RandomState(0)
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter()
cause_signed=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
cause_cnt=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cause_signed[m][c][sym[e]]+=DIRV[d]; cause_cnt[m][c][sym[e]]+=1
seen=collections.defaultdict(set)
for m in cause_cnt:
    for c,ss in cause_cnt[m].items():
        for s in ss: seen[c].add(s)
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=160: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR); ALLD=sorted(set().union(*[set(AR[s]) for s in AR])); N=len(AR)
idf={c:math.log(1+N/len(ss)) for c,ss in seen.items()}
def sfac(sgn,days):
    out={}
    for d in days:
        vals=[np.sign(sgn[s])*AR[s][d] for s in sgn if s in AR and d in AR[s] and sgn[s]!=0]
        if len(vals)>=2: out[d]=np.mean(vals)
    return out
def resid(a,f,days):
    cm=[d for d in days if d in a and d in f]
    if len(cm)<20: return None
    y=np.array([a[d] for d in cm]);X=np.column_stack([np.ones(len(cm)),[f[d] for d in cm]])
    b,*_=np.linalg.lstsq(X,y,rcond=None); return dict(zip(cm,y-X@b))
def wc(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<20: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
allm=months_between(2010,2012); idx={m:i for i,m in enumerate(allm)}
rows=[]  # (contrib_magnitude, in_window_drop, oos_drop)
for m in allm:
    i=idx[m]
    if i<3 or i+2>=len(allm): continue
    wm=allm[i-3:i+1]; days=[d for d in ALLD if d[:7] in wm]; fd=[d for d in ALLD if d[:7] in allm[i+1:i+3]]
    for c,sgn in cause_signed[m].items():
        effs=sorted(s for s in sgn if s in U and sgn[s]!=0)
        if len(effs)<4: continue
        for x in range(len(effs)):
            for y in range(x+1,len(effs)):
                A,B=effs[x],effs[y]
                if np.sign(sgn[A])!=np.sign(sgn[B]): continue  # same-sign only
                c0=wc(AR[A],AR[B],days)
                if c0 is None or c0<0.3: continue
                oth={s:sgn[s] for s in effs if s not in (A,B) and sgn[s]!=0}
                if len(oth)<2: continue
                f=sfac(oth,days); rA=resid(AR[A],f,days); rB=resid(AR[B],f,days)
                if not(rA and rB): continue
                c1=wc(rA,rB,days)
                if c1 is None: continue
                dwin=c0-c1
                # embedding contribution magnitude (signed tf-idf overlap on this driver)
                contrib=abs(cause_cnt[m][c][A]*idf.get(c,0) * cause_cnt[m][c][B]*idf.get(c,0))**0.5
                # OOS: same driver, forward window
                oos=None; c0f=wc(AR[A],AR[B],fd)
                if c0f is not None and len(fd)>20:
                    ff=sfac(oth,fd); rAf=resid(AR[A],ff,fd); rBf=resid(AR[B],ff,fd)
                    if rAf and rBf:
                        c1f=wc(rAf,rBf,fd)
                        if c1f is not None: oos=c0f-c1f
                rows.append((contrib,dwin,oos))
R=[r for r in rows]; dw=np.array([r[1] for r in R])
oo=np.array([(r[1],r[2]) for r in R if r[2] is not None])
pc=lambda a,q:np.percentile(a,q)
print(f"=== SIGNED attribution — distribution ({len(dw)} same-sign co-movement removals) ===")
print(f"  Δcorr pctiles: p10 {pc(dw,10):+.2f} | p25 {pc(dw,25):+.2f} | p50 {pc(dw,50):+.2f} | p75 {pc(dw,75):+.2f} | p90 {pc(dw,90):+.2f} | neg {np.mean(dw<0):.0%}")
print(f"\n=== OUT-OF-SAMPLE STABILITY ({len(oo)}) ===")
if len(oo)>30:
    r=np.corrcoef(oo[:,0],oo[:,1])[0,1]
    print(f"  corr(in-window Δcorr, next-window Δcorr) = {r:+.2f} | magnitude {oo[:,0].mean():+.3f} -> {oo[:,1].mean():+.3f} ({oo[:,1].mean()/oo[:,0].mean()*100:.0f}% persists)")
print(f"\n=== CALIBRATION: does embedding-contribution magnitude predict the realized attribution? ===")
cs=np.array([r[0] for r in R]); 
q=np.quantile(cs,[0,.33,.67,1.0])
print(f"  {'contribution tercile':22} {'mean Δcorr':>11} {'P(Δcorr>0.15)':>14} {'n':>6}")
for lo,hi,lbl in [(q[0],q[1],'low'),(q[1],q[2],'mid'),(q[2],q[3]+1e-9,'high')]:
    mask=(cs>=lo)&(cs<hi)
    if mask.sum()>10: print(f"  {lbl:22} {dw[mask].mean():+11.3f} {np.mean(dw[mask]>0.15):14.0%} {int(mask.sum()):6}")
print("  => monotone rise = the embedding magnitude CALIBRATES per-pair attribution confidence")
