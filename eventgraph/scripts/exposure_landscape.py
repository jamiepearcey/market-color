# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""EVENT-EXPOSURE LANDSCAPE (F22 productized). For each event, compute EVERY ticker's descriptive
exposure to it (cosine of company news-profile vs event text) across the whole universe — named or
not — plus each ticker's REALIZED co-movement with the event during its window. Validation: does the
description-exposure ranking predict realized co-movement for NON-mentioned firms? Output landscape JSON."""
import json, collections, csv, sys, re, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
SN={"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLV":"Health Care","XLI":"Industrials","XLY":"Cons Disc","XLP":"Cons Staples","XLU":"Utilities","XLB":"Materials"}
N_UNIV=140; MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
BROADCANON={"macro","monetary","fiscal"}
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
STOP=set("the a an of and to in on for is are was were be as by at it its with from that this said says will has have had inc corp co ltd group plc company companies market markets share shares stock stocks percent year years quarter after over more most u s said its".split())
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}; name={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
        sym[j["entity_id"]]=j["symbol"]; name.setdefault(j["symbol"], j["entity_id"].split("__")[0].replace("_"," ").title())
CANON=json.load(open(G/"catalyst_map.json"))
dmeta={json.loads(l)["doc_id"]:((json.loads(l).get("published_at") or "")[:7], json.loads(l).get("headline") or "") for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); comp_txt=collections.defaultdict(list); ev_txt=collections.defaultdict(list); ev_names=collections.defaultdict(lambda: collections.defaultdict(int)); ev_lab={}
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); did=j.get("doc_id"); mh=dmeta.get(did); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not mh or mh[0][:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    m=mh[0]; freq[sym[e]]+=1; comp_txt[sym[e]].append((j.get("quote") or "")+" "+mh[1])
    ci=CANON.get(c); key=ci["key"] if ci else c; typ=ci["type"] if ci else c.split("__")[-1]; lab=ci["name"] if ci else c.split("__")[0].replace("_"," ").title()
    if (ci and typ in BROADCANON) or (not ci and c.split("__")[-1] in MACRO_T): continue
    ev_txt[(m,key)].append((j.get("quote") or "")+" "+lab+" "+(j.get("mechanism") or "")); ev_names[(m,key)][sym[e]]+=DIRV[d]; ev_lab[(m,key)]=(lab,typ)
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
def secof(s):
    r=R[s]; best=("?",-1)
    for e in SPDR:
        cm=[d for d in alld if d in r and d in spdr[e]]
        if len(cm)<40: continue
        x=np.array([r[d] for d in cm]);y=np.array([spdr[e][d] for d in cm])
        if x.std() and y.std():
            c=abs(np.corrcoef(x,y)[0,1])
            if c>best[1]: best=(SN[e],c)
    return best[0]
SEC={s:secof(s) for s in uni}
def toks(t): return [w for w in re.findall(r"[a-z]{3,}",t.lower()) if w not in STOP]
docs={("C",s):toks(" ".join(comp_txt[s])) for s in uni}
for k in ev_txt: docs[("E",k)]=toks(" ".join(ev_txt[k]))
df=collections.Counter()
for dd in docs.values():
    for w in set(dd): df[w]+=1
Ndoc=len(docs); idf={w:math.log(1+Ndoc/df[w]) for w in df if df[w]>=3}
def vec(tl):
    c=collections.Counter(w for w in tl if w in idf)
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
# validation: pooled corr(exposure, realized co-move) for NON-mentioned firms
X_exp=[]; Y_real=[]; landscapes=[]
for (m,key),nm in sorted(ev_names.items(),key=lambda kv:-sum(abs(v) for v in kv[1].values())):
    M=[s for s in nm if s in R2]
    if len(M)<3: continue
    ev=vec(docs[("E",(m,key))])
    if not ev: continue
    i=idx[m]; win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    basket={d:np.mean([R2[s][d] for s in M if d in R2[s]]) for d in win if sum(1 for s in M if d in R2[s])>=2}
    rows=[]
    for s in uni:
        ex=cos(ev,cvec[s]); rl=corr(R2[s],basket,win)
        rows.append({"t":s,"sec":SEC[s],"exp":round(ex,3),"real":None if rl is None else round(rl,3),"m":int(s in M)})
        if s not in M and rl is not None: X_exp.append(ex); Y_real.append(rl)
    rows.sort(key=lambda r:-r["exp"])
    lab,typ=ev_lab[(m,key)]
    landscapes.append({"month":m,"label":lab,"type":typ,"n_named":len(M),
                       "named":[s for s in M],"rows":[r for r in rows[:34]]})
Xe=np.array(X_exp); Yr=np.array(Y_real)
xe=(Xe-Xe.mean()); yr=(Yr-Yr.mean()); r=float((xe@yr)/(np.sqrt(xe@xe)*np.sqrt(yr@yr)))
t=r*math.sqrt((len(Xe)-2)/(1-r*r))
print(f"=== VALIDATION: does descriptive exposure predict realized co-movement for NON-mentioned firms? ===")
print(f"  pooled corr(exposure, realized) = {r:+.3f}  t {t:+.0f}  (n={len(Xe)} non-mentioned firm-events)")
# keep the strongest interpretable landscapes for the viz
landscapes=[e for e in landscapes if e["n_named"]>=3][:60]
Path("../data/eg_runs/eg100k_graph/exposure_landscape.json").write_text(json.dumps({"events":landscapes,"sectors":list(SN.values())}))
print(f"\n-> exposure_landscape.json  {len(landscapes)} events with full-universe exposure maps")
print("  sample — LIBOR/Almunia landscape (top surfaced NON-mentioned firms):")
alm=next((e for e in landscapes if "Almunia" in e["label"]),None) or landscapes[0]
print(f"  [{alm['month']}] {alm['label']} — named {alm['n_named']}: {','.join(alm['named'][:8])}")
sur=[r for r in alm["rows"] if not r["m"]][:8]
for r in sur: print(f"     surfaced {r['t']:6} ({r['sec'][:4]}) exp {r['exp']:.2f} realized {r['real']}")
