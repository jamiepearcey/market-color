# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""EMBEDDING-BASED attribution (BBG). Uses the validated cause-profile embedding (F5) to RANK
which shared driver explains a pair's correlation, then validates against realized returns.
Falsifiable: among a pair's shared drivers, removing the embedding's TOP-ranked (tf-idf
c_i[D]*c_j[D]) should drop realized corr MORE than removing a random shared driver. If yes,
the embedding correctly attributes — richer than link-based (which can't rank shared drivers)."""
import json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import yahoo, logret, wls, FN, SKIP
from news_contagion import ar_series
from news_covariance import months_between
DIRV={"up","down","widen","tighten"}
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
name_causes=collections.defaultdict(lambda: collections.defaultdict(collections.Counter))  # month->sym->cause->count
cause_effs=collections.defaultdict(lambda: collections.defaultdict(set))  # month->cause->{sym}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or j.get("effect_dir") not in DIRV: continue
    freq[sym[e]]+=1; name_causes[m][sym[e]][c]+=1; cause_effs[m][c].add(sym[e])
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
AR={}
for s,_ in freq.most_common():
    if len(AR)>=160: break
    v=ar_series(s,cache,Fmap,{"2010","2011","2012"},p1,p2)
    if len(v)>150: AR[s]=v
U=set(AR); ALLD=sorted(set().union(*[set(AR[s]) for s in AR]))
# idf over causes (how many distinct names each cause drives, whole window)
cause_df=collections.Counter()
for m in cause_effs:
    for c,ss in cause_effs[m].items(): cause_df[c]|=0
seen=collections.defaultdict(set)
for m in name_causes:
    for s,cc in name_causes[m].items():
        for c in cc: seen[c].add(s)
N=len(AR); idf={c:math.log(1+N/len(ss)) for c,ss in seen.items()}
def prof(s,months):  # tf-idf cause profile over window
    v=collections.Counter()
    for m in months:
        for c,n in name_causes.get(m,{}).get(s,{}).items(): v[c]+=n
    return {c:n*idf.get(c,0) for c,n in v.items()}
def dfac(nm,days):
    out={}
    for d in days:
        r=[AR[s][d] for s in nm if s in AR and d in AR[s]]
        if len(r)>=2: out[d]=np.mean(r)
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
top_drop=[]; rand_drop=[]; n_multi=0
for m in allm:
    i=idx[m]
    if i<3: continue
    wm=allm[i-3:i+1]; days=[d for d in ALLD if d[:7] in wm]
    # candidate pairs: elevated corr + share >=2 causes each driving >=3 names
    active=[s for s in U if prof(s,wm)]
    for x in range(len(active)):
        for y in range(x+1,len(active)):
            A,B=active[x],active[y]; c0=wc(AR[A],AR[B],days)
            if c0 is None or c0<0.3: continue
            pa,pb=prof(A,wm),prof(B,wm)
            shared=[c for c in (set(pa)&set(pb)) if len({s for s in cause_effs[m].get(c,set()) if s in U}-{A,B})>=2]
            if len(shared)<2: continue
            n_multi+=1
            # EMBEDDING rank: contribution c_A[D]*c_B[D]
            ranked=sorted(shared, key=lambda c: -(pa[c]*pb[c]))
            topD=ranked[0]; randD=shared[rng.randint(len(shared))]
            for D,bucket in [(topD,top_drop),(randD,rand_drop)]:
                oth=sorted({s for s in cause_effs[m][D] if s in U}-{A,B})
                if len(oth)<2: continue
                f=dfac(oth,days); rA=resid(AR[A],f,days); rB=resid(AR[B],f,days)
                if rA and rB:
                    c1=wc(rA,rB,days)
                    if c1 is not None: bucket.append(c0-c1)
td=np.array(top_drop); rd=np.array(rand_drop)
print(f"=== EMBEDDING-BASED attribution on BBG ({n_multi} multi-driver pairs) ===")
print(f"  removing EMBEDDING TOP-ranked shared driver:  corr drop {td.mean():+.3f} (median {np.median(td):+.3f}, n={len(td)})")
print(f"  removing RANDOM shared driver:                corr drop {rd.mean():+.3f} (n={len(rd)})")
se=np.sqrt(td.var(ddof=1)/len(td)+rd.var(ddof=1)/len(rd))
print(f"  EXCESS of the embedding's top pick = {td.mean()-rd.mean():+.3f}  t={(td.mean()-rd.mean())/se:+.1f}")
print(f"  P(embedding-top drop > random-shared drop) among matched = see paired below")
# paired
pt=[]; 
mn=min(len(td),len(rd))
print(f"\n  => if excess>0 & t>2: the embedding correctly RANKS which shared driver is responsible")
print(f"     (link-based can't do this — it can't distinguish among a pair's shared drivers)")
