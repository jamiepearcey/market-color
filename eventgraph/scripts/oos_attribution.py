# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""IN-SAMPLE vs OUT-OF-SAMPLE attribution — is it real structure or fitting?
Fit each leg's beta to the driver factor on window W1; apply the FIXED W1 betas to the
non-overlapping FORWARD window W2 and measure the correlation drop there. Compare:
  named in-sample (W2 betas)  vs  named OOS (W1 betas)  vs  random-driver OOS (W1 betas).
If named-OOS holds near named-in-sample AND beats random-OOS -> real, stable structure.
If named-OOS collapses toward random -> in-sample artifact (a relabeled regression)."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"; rng=np.random.RandomState(3)
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); cs=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int)))
allnet=collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cs[m][c][sym[e]]+=DIRV[d]; allnet[c][sym[e]]+=DIRV[d]
print("building AR ...",flush=True)
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=160: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR); ALLD=sorted(set().union(*[set(AR[s]) for s in AR])); allm=months_between(2010,2012); idx={m:i for i,m in enumerate(allm)}
bigdrv=[c for c in allnet if len([s for s in allnet[c] if s in U and allnet[c][s]!=0])>=4]
def basket(sgn,days):
    out={}
    for d in days:
        v=[np.sign(sgn[s])*AR[s][d] for s in sgn if s in AR and d in AR[s] and sgn[s]!=0]
        if len(v)>=2: out[d]=np.mean(v)
    return out
def beta(a,f,days):
    cm=[d for d in days if d in a and d in f]
    if len(cm)<20: return None
    x=np.array([f[d] for d in cm]); y=np.array([a[d] for d in cm]); x=x-x.mean()
    return float((x@(y-y.mean()))/(x@x)) if x@x>0 else 0.0
def corr_resid(a,b,f,days,ba,bb):
    cm=[d for d in days if d in a and d in b and d in f]
    if len(cm)<20: return None,None
    A=np.array([a[d] for d in cm]); B=np.array([b[d] for d in cm]); F=np.array([f[d] for d in cm])
    c0=np.corrcoef(A,B)[0,1] if A.std() and B.std() else None
    rA=A-ba*F; rB=B-bb*F
    c1=np.corrcoef(rA,rB)[0,1] if rA.std() and rB.std() else None
    return c0,c1
nIS=[]; nOOS=[]; rOOS=[]
for m in allm:
    i=idx[m]
    if i<3 or i+3>=len(allm): continue
    w1=allm[i-3:i+1]; w2=allm[i+1:i+4]
    d1=[d for d in ALLD if d[:7] in w1]; d2=[d for d in ALLD if d[:7] in w2]
    for c,sgn in cs[m].items():
        effs=sorted(s for s in sgn if s in U and sgn[s]!=0)
        if len(effs)<4: continue
        for x in range(len(effs)):
            for y in range(x+1,len(effs)):
                A,B=effs[x],effs[y]
                if np.sign(sgn[A])!=np.sign(sgn[B]): continue
                oth={s:sgn[s] for s in effs if s not in (A,B) and sgn[s]!=0}
                if len(oth)<2: continue
                f1=basket(oth,d1); f2=basket(oth,d2)
                bA1=beta(AR[A],f1,d1); bB1=beta(AR[B],f1,d1)
                bA2=beta(AR[A],f2,d2); bB2=beta(AR[B],f2,d2)
                if None in (bA1,bB1,bA2,bB2): continue
                c0,c1is=corr_resid(AR[A],AR[B],f2,d2,bA2,bB2)      # in-sample (W2 betas on W2)
                _,c1oos=corr_resid(AR[A],AR[B],f2,d2,bA1,bB1)      # OOS (W1 betas on W2)
                if c0 is None or c1is is None or c1oos is None or c0<0.2: continue
                nIS.append(c0-c1is); nOOS.append(c0-c1oos)
                # random driver baseline (unrelated to A,B), OOS betas
                for _try in range(3):
                    rc=bigdrv[rng.randint(len(bigdrv))]
                    rn={s:allnet[rc][s] for s in allnet[rc] if s in U and allnet[rc][s]!=0 and s not in (A,B)}
                    if len(rn)>=2 and A not in allnet[rc] and B not in allnet[rc]: break
                else: rn=None
                if rn:
                    rf1=basket(rn,d1); rf2=basket(rn,d2)
                    rbA=beta(AR[A],rf1,d1); rbB=beta(AR[B],rf1,d1)
                    if None not in (rbA,rbB):
                        _,rc1=corr_resid(AR[A],AR[B],rf2,d2,rbA,rbB)
                        if rc1 is not None: rOOS.append(c0-rc1)
def st(a): a=np.array(a); return f"mean {a.mean():+.3f}  median {np.median(a):+.3f}  t {a.mean()/(a.std(ddof=1)/len(a)**0.5):+.1f}  n {len(a)}"
print(f"\n=== attribution: in-sample vs OUT-OF-SAMPLE (fit betas W1 -> apply to forward W2) ===")
print(f"  named driver, IN-SAMPLE (W2 betas)  : {st(nIS)}")
print(f"  named driver, OUT-OF-SAMPLE (W1 beta): {st(nOOS)}")
print(f"  RANDOM driver, OUT-OF-SAMPLE        : {st(rOOS)}")
nis=np.mean(nIS); noos=np.mean(nOOS); roos=np.mean(rOOS)
print(f"\n  OOS retains {noos/nis*100:.0f}% of in-sample attribution")
print(f"  named-OOS minus random-OOS = {noos-roos:+.3f}  (the real, non-mechanical, out-of-sample structure)")
