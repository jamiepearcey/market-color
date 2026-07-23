# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""PRICE-MOVE ATTRIBUTION (contemporaneous, first-order, direct). For each firm, decompose each month's
move additively into MACRO (9 factors) + SECTOR (9 SPDRs, orthogonal to macro) + IDIOSYNCRATIC, and attach
the named NEWS EVENTS (causal edges where the firm is the effect that month, with verbatim quote + article)
that explain the idiosyncratic part. No embedding, no prediction — pure explanation from the causal graph."""
import json, collections, csv, sys, bisect, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]
SN={"XLF":"Financials","XLK":"Technology","XLE":"Energy","XLV":"Health Care","XLI":"Industrials","XLY":"Cons Disc","XLP":"Cons Staples","XLU":"Utilities","XLB":"Materials"}
N_UNIV=90; DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}; name={}; inv=collections.defaultdict(list)
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
        sym[j["entity_id"]]=j["symbol"]; name.setdefault(j["symbol"], j["entity_id"].split("__")[0].replace("_"," ").title()); inv[j["symbol"]].append(j["entity_id"])
CANON=json.load(open(G/"catalyst_map.json"))
PRIOR=json.load(open(G/"mech_prior.json")); MECHP=PRIOR["mech"]; BASE=PRIOR["baseline"]
docmeta={json.loads(l)["doc_id"]:{"m":(json.loads(l).get("published_at") or "")[:7],"h":json.loads(l).get("headline"),"s":json.loads(l).get("source"),"u":json.loads(l).get("url"),"d":(json.loads(l).get("published_at") or "")[:10]} for l in open(G/"lake/document.jsonl")}
freq=collections.Counter(); ev_by=collections.defaultdict(list); _seen=collections.defaultdict(set)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); dm=docmeta.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not dm or dm["m"][:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    freq[sym[e]]+=1; ci=CANON.get(c)
    lab=ci["name"] if ci else c.split("__")[0].replace("_"," ").title(); typ=ci["type"] if ci else c.split("__")[-1]
    q=(j.get("quote") or "").strip()
    k=(sym[e],dm["m"])
    if q and q not in _seen[k]: _seen[k].add(q); ev_by[k].append({"cat":lab,"typ":typ,"dir":DIRV[d],"q":q,"mech":j.get("mechanism"),"h":dm["h"],"s":dm["s"],"u":dm["u"],"d":dm["d"]})
def ret(t): return logret(yahoo(t,cache,p1,p2))
spdr={e:ret(e) for e in SPDR}
years={"2010","2011","2012"}
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=N_UNIV: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in years)>400: uni.append(s)
alld=sorted({d for s in uni for d in ret(s) if d[:4] in years and d in Fmap and all(d in spdr[e] for e in SPDR)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
def secof(s,r):
    best=("?",-1)
    for e in SPDR:
        cm=[d for d in alld if d in r and d in spdr[e]]
        if len(cm)<40: continue
        x=np.array([r[d] for d in cm]);y=np.array([spdr[e][d] for d in cm])
        if x.std() and y.std():
            cc=abs(np.corrcoef(x,y)[0,1])
            if cc>best[1]: best=(SN[e],cc)
    return best[0]
mo=[f"{y}-{m:02d}" for y in (2010,2011,2012) for m in range(1,13)]
out=[]
for s in uni:
    r=ret(s); days=[d for d in alld if d in r]
    if len(days)<250: continue
    Y=np.array([r[d] for d in days])
    if np.max(np.abs(Y))>0.35 or np.std(Y)>0.06: continue   # drop penny/delisted/split-artifact series
    Xm=np.column_stack([np.ones(len(days))]+[[Fmap[d][i] for d in days] for i in mcol]); Xm=np.nan_to_num(Xm)
    bm,*_=np.linalg.lstsq(Xm,Y,rcond=None); macfit=Xm@bm; res1=Y-macfit
    Xs=np.column_stack([np.ones(len(days))]+[[spdr[e][d] for d in days] for e in SPDR])
    bs,*_=np.linalg.lstsq(Xs,res1,rcond=None); secfit=Xs@bs; idio=res1-secfit
    dm={d:(macfit[i],secfit[i],idio[i],Y[i]) for i,d in enumerate(days)}
    idio_d={d:idio[i] for i,d in enumerate(days)}
    moves=[]
    for m in mo:
        dd=[d for d in days if d[:7]==m]
        if len(dd)<10: continue
        tot=sum(dm[d][3] for d in dd); mac=sum(dm[d][0] for d in dd); sec=sum(dm[d][1] for d in dd); idi=sum(dm[d][2] for d in dd)
        evs=[dict(e) for e in ev_by.get((s,m),[])]
        newsdays=set()
        for e in evs:
            pr=MECHP.get(e.get("mech") or "?",{}); e["pmove"]=pr.get("move"); e["palign"]=pr.get("align"); e["trust"]=pr.get("trust","context")
            Dt=e.get("d")
            if not Dt: e["contrib"]=None; continue
            k=bisect.bisect_left(days,Dt)
            if k<len(days) and days[k][:7]==m:      # event-study: idio abnormal return on the article's trading day
                e["contrib"]=round(idio_d[days[k]],4); e["day"]=days[k]; newsdays.add(days[k])
            else: e["contrib"]=None
        idio_news=round(sum(idio_d[d] for d in newsdays),4)
        evs=sorted(evs,key=lambda x:-abs(x.get("contrib") or 0))[:6]
        moves.append({"m":m,"tot":round(tot,4),"macro":round(mac,4),"sector":round(sec,4),"idio":round(idi,4),
                      "idio_news":idio_news,"nev":len(ev_by.get((s,m),[])),"events":evs})
    # keep the most notable moves (by |total|) that have news
    notable=sorted([mv for mv in moves if mv["nev"]>0 and abs(mv["tot"])<0.6],key=lambda x:-abs(x["tot"]))[:8]
    if notable: out.append({"t":s,"n":name.get(s,s),"sec":secof(s,r),"moves":notable})
out.sort(key=lambda x:-max(abs(mv["tot"]) for mv in x["moves"]))
Path("../data/eg_runs/eg100k_graph/attribution.json").write_text(json.dumps({"firms":out,"sectors":list(SN.values())}))
print(f"-> attribution.json  {len(out)} firms with attributed moves")
ex=out[0]
print(f"  sample: {ex['n']} ({ex['sec']}) — biggest move:")
mv=max(ex["moves"],key=lambda x:abs(x["tot"]))
print(f"   {mv['m']}: total {mv['tot']:+.1%} = macro {mv['macro']:+.1%} + sector {mv['sector']:+.1%} + idio {mv['idio']:+.1%}")
for e in mv["events"][:3]: print(f"     [{e['cat']}] {'▲' if e['dir']>0 else '▼'} \"{e['q'][:56]}\"")
