# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""EMPIRICAL implied move BY NEWS TYPE. For every causal edge (firm effect, article date, mechanism/type),
measure the firm's IDIOSYNCRATIC (macro+sector-removed) abnormal return on the article day. Aggregate by
mechanism and by catalyst type: mean |abnormal move| (how much this news type moves prices) and mean signed
alignment (does the extracted direction match the realized move?). Answers 'what move does each news type imply'."""
import json, collections, csv, sys, bisect, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0,"scripts")
from cross_sectional_ic import FN, SKIP, yahoo, logret
SPDR=["XLF","XLK","XLE","XLV","XLI","XLY","XLP","XLU","XLB"]; N_UNIV=120; DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
G=Path("../data/eg_runs/eg100k_graph"); cache=G/"prices"
p1=int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2=int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())
Fmap={}
for row in csv.DictReader(open(G/"factor_snapshot_factor_returns.csv")):
    d=row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"]=np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP: sym[j["entity_id"]]=j["symbol"]
CANON=json.load(open(G/"catalyst_map.json"))
docd={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:10] for l in open(G/"lake/document.jsonl")}
def ret(t): return logret(yahoo(t,cache,p1,p2))
spdr={e:ret(e) for e in SPDR}; freq=collections.Counter()
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); e=j.get("effect_entity"); dd=docd.get(j.get("doc_id"))
    if e in sym and dd and dd[:4] in {"2010","2011","2012"} and j.get("effect_dir") in DIRV: freq[sym[e]]+=1
uni=[]
for s,_ in freq.most_common():
    if len(uni)>=N_UNIV: break
    r=ret(s)
    if sum(1 for d in r if d[:4] in {"2010","2011","2012"})>400: uni.append(s)
alld=sorted({d for s in uni for d in ret(s) if d[:4] in {"2010","2011","2012"} and d in Fmap and all(d in spdr[e] for e in SPDR)})
mcol=[i for i in range(len(FN)) if sum(np.isfinite(Fmap[d][i]) for d in alld)>=len(alld)*0.5]
# idiosyncratic (macro+sector removed) daily return per firm
idio={}
for s in uni:
    r=ret(s); days=[d for d in alld if d in r]
    if len(days)<250: continue
    Y=np.array([r[d] for d in days])
    if np.max(np.abs(Y))>0.35: continue
    Xm=np.nan_to_num(np.column_stack([np.ones(len(days))]+[[Fmap[d][i] for d in days] for i in mcol]))
    res1=Y-Xm@np.linalg.lstsq(Xm,Y,rcond=None)[0]
    Xs=np.column_stack([np.ones(len(days))]+[[spdr[e][d] for d in days] for e in SPDR])
    res2=res1-Xs@np.linalg.lstsq(Xs,res1,rcond=None)[0]
    idio[s]={d:res2[i] for i,d in enumerate(days)}
    idio[s]["_days"]=days
by_mech=collections.defaultdict(list); by_typ=collections.defaultdict(list)
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); e=j.get("effect_entity"); dd=docd.get(j.get("doc_id")); d=j.get("effect_dir"); c=j.get("cause_entity")
    if e not in sym or sym[e] not in idio or not dd or dd[:4] not in {"2010","2011","2012"} or d not in DIRV: continue
    s=sym[e]; days=idio[s]["_days"]; k=bisect.bisect_left(days,dd)
    if k>=len(days): continue
    ab=idio[s][days[k]]                                  # abnormal return on the article day
    mech=j.get("mechanism") or "?"; ci=CANON.get(c); typ=(ci["type"] if ci else (c.split("__")[-1] if c else "?"))
    by_mech[mech].append((abs(ab), DIRV[d]*ab)); by_typ[typ].append((abs(ab), DIRV[d]*ab))
def rep(dct,title,minn=40):
    print(f"\n=== implied |abnormal move| BY {title} (article-day event study) ===")
    print(f"  {'':26} {'mean |move|':>11} {'dir-aligned':>12} {'n':>7}")
    rows=[]
    for k,v in dct.items():
        if len(v)<minn: continue
        mag=np.mean([a for a,_ in v]); al=np.mean([s for _,s in v]); rows.append((mag,al,len(v),k))
    for mag,al,n,k in sorted(rows,reverse=True):
        print(f"  {k[:26]:26} {mag:10.2%} {al:+12.2%} {n:7}")
print(f"universe {len(idio)} firms; classifying article-day abnormal moves by news type")
rep(by_mech,"MECHANISM")
rep(by_typ,"CATALYST TYPE",minn=80)
allv=[a for v in by_mech.values() for a,_ in v]
print(f"\n  baseline: mean |abnormal move| on a RANDOM firm-day ~ {np.mean([abs(idio[s][d]) for s in list(idio)[:40] for d in idio[s]['_days'][::7]]):.2%}")
print("  => mean|move| >> baseline = this news type genuinely moves prices; dir-aligned>0 = extracted direction matches the move.")
