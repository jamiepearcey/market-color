# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""HYPOTHESIS (event-anchored, out-of-mention): does a company's DESCRIPTIVE similarity to an event
predict its co-movement with that event — for companies the article never NAMED — beyond sector?
Company profile = tf-idf of its own news text (independent of the event). Event text = its verbatim
quotes+catalyst+mechanism. Exposure = cosine. Test: among NON-mentioned firms, do high-exposure ones
co-move with the event's mentioned basket more than low-exposure ones (residualized on macro+sector)?"""
import json, collections, csv, sys, re, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=120
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
STOP=set("the a an of and to in on for is are was were be as by at it its with from that this said says will has have had inc corp co ltd group plc company companies market markets share shares stock stocks percent year years quarter after over more most u s".split())
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
dmeta={json.loads(l)["doc_id"]:( (json.loads(l).get("published_at") or "")[:7], json.loads(l).get("headline") or "") for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); comp_txt=collections.defaultdict(list); ev_txt=collections.defaultdict(list); ev_names=collections.defaultdict(lambda: collections.defaultdict(int))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); did=j.get("doc_id"); mh=dmeta.get(did); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not mh or mh[0][:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    m=mh[0]; freq[sym[e]]+=1
    comp_txt[sym[e]].append((j.get("quote") or "")+" "+mh[1])                         # company's own news text
    if c.split("__")[-1] not in MACRO_T:
        ev_txt[(m,c)].append((j.get("quote") or "")+" "+c.split("__")[0].replace("_"," ")+" "+(j.get("mechanism") or ""))
        ev_names[(m,c)][sym[e]]+=DIRV[d]
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
uni=[s for s in uni if s in R2]
# tf-idf over company profiles + event texts (shared vocab)
def toks(t): return [w for w in re.findall(r"[a-z]{3,}",t.lower()) if w not in STOP]
docs={("C",s):toks(" ".join(comp_txt[s])) for s in uni}
for k in ev_txt: docs[("E",k)]=toks(" ".join(ev_txt[k]))
df=collections.Counter()
for d in docs.values():
    for w in set(d): df[w]+=1
Ndoc=len(docs); idf={w:math.log(1+Ndoc/df[w]) for w in df if df[w]>=3}
def vec(tl):
    c=collections.Counter(w for w in tl if w in idf); 
    if not c: return {}
    n=math.sqrt(sum((v*idf[w])**2 for w,v in c.items())) or 1
    return {w:v*idf[w]/n for w,v in c.items()}
cvec={s:vec(docs[("C",s)]) for s in uni}
def cos(a,b): return sum(a[w]*b.get(w,0) for w in a)
mo=[f"{y}-{mm:02d}" for y in (2010,2011,2012) for mm in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def corr(a,b,days):
    cm=[d for d in days if d in a and d in b]
    if len(cm)<15: return None
    x=np.array([a[d] for d in cm]);y=np.array([b[d] for d in cm]); return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
hi=[]; lo=[]; ment=[]
for (m,c),nm in ev_names.items():
    M=[s for s in nm if s in R2]
    if len(M)<2: continue
    i=idx[m]; win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    ev=vec(docs[("E",(m,c))])
    if not ev: continue
    basket={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in win if sum(1 for s in M if d in R2[s])>=2}
    cand=[s for s in uni if s not in M]                       # NON-mentioned firms
    scored=sorted(cand,key=lambda s:-cos(ev,cvec[s]))
    q=max(3,len(scored)//4)
    for s in scored[:q]:
        r=corr(R2[s],basket,win)
        if r is not None: hi.append(r)
    for s in scored[-q:]:
        r=corr(R2[s],basket,win)
        if r is not None: lo.append(r)
    for s in M:                                               # mentioned (upper bound)
        r=corr(R2[s],basket,win)
        if r is not None: ment.append(r)
def st(a): a=np.array(a); return f"mean {a.mean():+.3f}  n {len(a)}"
print("=== event-anchored, OUT-OF-MENTION exposure: does description predict co-movement beyond sector? ===")
print(f"  MENTIONED firms vs event basket (upper bound)        : {st(ment)}")
print(f"  NON-mentioned, HIGH descriptive-exposure vs basket   : {st(hi)}")
print(f"  NON-mentioned, LOW  descriptive-exposure vs basket   : {st(lo)}")
h=np.array(hi); lo_=np.array(lo); d=h.mean()-lo_.mean(); se=(h.var(ddof=1)/len(h)+lo_.var(ddof=1)/len(lo_))**0.5
print(f"\n  HIGH - LOW (non-mentioned) excess = {d:+.3f}  t {d/se:+.1f}")
print(f"  => {'YES: description surfaces real exposure BEYOND mention (event-anchored frame adds signal)' if d/se>2 else 'NO: description does not beat mention out-of-sample — mention is the carrier'}")
