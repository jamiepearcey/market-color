# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx","numpy"]
# ///
"""Precompute EVENT-LINKAGE structures WITH the specificity contrast. Per (month, specific driver)
binding >=2 priceable firms we compute: (a) residual co-movement after macro+broad-sector [res],
(b) the EXCESS that survives macro+broad+SUB-INDUSTRY [res_sub] = the irreducible signal, and
(c) the SECTOR COHORT — same-sector universe peers the catalyst did NOT name, with the baseline
co-movement of bound-vs-peer. The UI action (open event) then SHOWS why the selection is specific."""
import json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
BROAD=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
SUBIND=["KBE","KRE","KIE","IAI","XHB","OIH","SMH","IYT","IBB","ITB","XRT","IYR"]
SN={"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLV":"Health Care","XLI":"Industrials","XLY":"Cons Disc","XLP":"Cons Staples","XLU":"Utilities","XLB":"Materials"}
MACRO_T={"equity_index","sovereign","central_bank","commodity","currency","rate_or_bond","economic_indicator","sector","market","exchange"}
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
        sym[j["entity_id"]]=j["symbol"]; name.setdefault(j["symbol"], j["entity_id"].split("__")[0].replace("_"," ").title())
docm={}; docmeta={}
for l in open(G/"lake/document.jsonl"):
    j=json.loads(l); docm[j["doc_id"]]=(j.get("published_at") or "")[:7]
    docmeta[j["doc_id"]]={"h":j.get("headline"),"s":j.get("source"),"u":j.get("url"),"d":(j.get("published_at") or "")[:10]}
freq=collections.Counter(); cause_m=collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(int))); dlabel={}
prov=collections.defaultdict(lambda: collections.defaultdict(list))   # (month,cause)->firm->[(sign,quote,mech,doc_id)]
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; cause_m[m][c][sym[e]]+=DIRV[d]; dlabel[c]=(c.split("__")[0].replace("_"," ").title(),c.split("__")[-1])
    pl=prov[(m,c)][sym[e]]
    if len(pl)<3: pl.append((DIRV[d], j.get("quote"), j.get("mechanism"), j.get("doc_id")))
years={"2010","2011","2012"}
def ret(t): return logret(yahoo(t,cache,p1,p2))
etf={e:ret(e) for e in BROAD+SUBIND}; etf={e:v for e,v in etf.items() if len([d for d in v if d[:4] in years])>300}
broad=[e for e in BROAD if e in etf]; subind=[e for e in SUBIND if e in etf]
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=150: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in years)>400: uni.append(s)
R={s:ret(s) for s in uni}
alld=sorted({d for s in uni for d in R[s] if d[:4] in years and d in Fmap and all(d in etf[e] for e in broad)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
def design(days,secs):
    cols=[[Fmap[d][i] for d in days] for i in mcol]+[[etf[e].get(d,np.nan) for d in days] for e in secs]
    X=np.array(cols,float).T; X=np.nan_to_num(X-np.nanmean(X,0)); return np.column_stack([np.ones(len(days)),X])
def residualize(secs):
    out={}
    for s in uni:
        days=[d for d in alld if d in R[s]]
        if len(days)<200: continue
        y=np.array([R[s][d] for d in days]); X=design(days,secs); b,*_=np.linalg.lstsq(X,y,rcond=None); out[s]=dict(zip(days,y-X@b))
    return out
R2=residualize(broad); R2s=residualize(broad+subind)
def sec_of(s):
    r=R[s]; best=("?",-1)
    for e in broad:
        cm=[d for d in alld if d in r and d in etf[e]]
        if len(cm)<40: continue
        x=np.array([r[d] for d in cm]);y=np.array([etf[e][d] for d in cm])
        if x.std() and y.std():
            c=abs(np.corrcoef(x,y)[0,1])
            if c>best[1]: best=(SN[e],c)
    return best[0]
SEC={s:sec_of(s) for s in R2}
by_sec=collections.defaultdict(list)
for s in R2: by_sec[SEC[s]].append(s)
mo=[f"{y}-{m:02d}" for y in (2010,2011,2012) for m in range(1,13)]; idx={m:i for i,m in enumerate(mo)}
def cc(D,a,b,days):
    cm=[d for d in days if d in D.get(a,{}) and d in D.get(b,{})]
    if len(cm)<20: return None
    x=np.array([D[a][d] for d in cm]);y=np.array([D[b][d] for d in cm])
    return float(np.corrcoef(x,y)[0,1]) if x.std() and y.std() else None
events=[]
for m in mo:
    i=idx[m]; win=[d for d in alld if d[:7] in mo[max(0,i-1):i+2]]
    for c,sgn in cause_m[m].items():
        if dlabel[c][1] in MACRO_T: continue
        ns=sorted(s for s in sgn if s in R2 and sgn[s]!=0)
        if len(ns)<2: continue
        pairs=[]; vals=[]; vsub=[]
        for x in range(len(ns)):
            for y in range(x+1,len(ns)):
                rc=cc(R2,ns[x],ns[y],win); rs=cc(R2s,ns[x],ns[y],win)
                if rc is not None: pairs.append([x,y,round(rc,3)]); vals.append(rc)
                if rs is not None: vsub.append(rs)
        if not vals: continue
        # evidence: the actual facts (verbatim quote + mechanism + source article) behind the links
        facts=[]; seen=set()
        for firm in ns:
            for fsgn,quote,mech,did in prov[(m,c)].get(firm,[]):
                k=(firm,quote)
                if k in seen or not quote: continue
                seen.add(k); dm=docmeta.get(did,{})
                facts.append({"t":firm,"dir":int(fsgn),"quote":quote,"mech":mech,
                              "h":dm.get("h"),"s":dm.get("s"),"u":dm.get("u"),"d":dm.get("d")})
                if len(facts)>=10: break
            if len(facts)>=10: break
        dom=collections.Counter(SEC[s] for s in ns).most_common(1)[0][0]
        # sector cohort: same-dominant-sector universe peers NOT in the event
        cohort=[s for s in by_sec[dom] if s not in ns][:14]
        base=[cc(R2s,a,p,win) for a in ns for p in cohort]; base=[v for v in base if v is not None]
        events.append({"month":m,"label":dlabel[c][0],"type":dlabel[c][1],"size":len(ns),
            "res":round(float(np.mean(vals)),3),"resmax":round(float(np.max(vals)),3),
            "res_sub":round(float(np.mean(vsub)),3) if vsub else None,       # survives sub-industry = specificity
            "dom":dom,"cohort_base":round(float(np.mean(base)),3) if base else None,
            "names":[{"t":s,"n":name.get(s,s),"sec":SEC[s],"dir":int(np.sign(sgn[s]))} for s in ns],
            "cohort":[{"t":s,"n":name.get(s,s),"sec":SEC[s]} for s in cohort],
            "facts":facts,"pairs":pairs})
events.sort(key=lambda e:(e["month"],-e["res"]))
Path("../data/eg_runs/eg100k_graph/events.json").write_text(json.dumps({"events":events,"sectors":list(SN.values())}))
print(f"-> events.json  {len(events)} events (with res_sub + sector cohort)")
ex=[e for e in events if e["res_sub"] and e["res_sub"]>=0.1 and e["cohort_base"] is not None]
print(f"   {len(ex)} events with specificity res_sub>=0.10 AND a sector cohort to contrast")
for e in sorted(ex,key=lambda e:-(e['res_sub']-(e['cohort_base'] or 0)))[:8]:
    print(f"   {e['month']} {e['label'][:28]:28} bound {e['size']} vs {len(e['cohort'])} {e['dom'][:4]} peers | res_sub {e['res_sub']:+.2f} vs cohort {e['cohort_base']:+.2f}")
