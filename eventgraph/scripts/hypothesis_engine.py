# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","httpx"]
# ///
"""UNIFIED DRIVER PIPELINE (replaces embedding ranking).
  activation (notable move)
    -> residual/attribution DECOMPOSITION (macro% / sector% / idio%)   <- the driver detector
    -> ROUTE the search by the decomposition (what kind of news to look for):
         macro-dominant  -> macro/monetary catalysts that month
         sector-dominant -> catalysts binding same-sector firms that month
         idio-dominant   -> the firm's own causal edges (event-study scored + type-trust)
    -> LLM REASONING (hypothesis engine): weigh decomposition + routed evidence + trust priors
    -> structured, evidence-grounded attribution hypothesis.
Contemporaneous & explanatory. No embedding ranking anywhere."""
import os, json, collections, sys, time
from pathlib import Path
import httpx
G=Path("../data/eg_runs/eg100k_graph")
GROQ="https://api.groq.com/openai/v1/chat/completions"; KEY=os.environ["GROQ_API_KEY"]; MODEL="openai/gpt-oss-120b"
ATTR=json.load(open(G/"attribution.json")); CANON=json.load(open(G/"catalyst_map.json"))
PRIOR=json.load(open(G/"mech_prior.json"))["mech"]
DIRV={"up":1,"down":-1,"widen":-1,"tighten":1}
BROAD={"macro","monetary","fiscal"}
sec_of={f["t"]:f["sec"] for f in ATTR["firms"]}
sym={}
for l in open(G/"entity_symbol.jsonl"):
    j=json.loads(l)
    if j["kind"]=="security": sym[j["entity_id"]]=j["symbol"]
docm={json.loads(l)["doc_id"]:(json.loads(l).get("published_at") or "")[:7] for l in open(G/"lake/document.jsonl")}
# month -> catalyst -> {firms, quote, type}
cat_m=collections.defaultdict(lambda: collections.defaultdict(lambda: {"firms":set(),"q":None,"typ":"?","n":0}))
for l in open(G/"lake/causal_event_edge.jsonl"):
    j=json.loads(l); m=docm.get(j.get("doc_id")); e=j.get("effect_entity"); c=j.get("cause_entity"); d=j.get("effect_dir")
    if not m or m[:4] not in {"2010","2011","2012"} or e not in sym or not c or d not in DIRV: continue
    ci=CANON.get(c); key=ci["name"] if ci else c.split("__")[0].replace("_"," ").title()
    typ=ci["type"] if ci else c.split("__")[-1]
    g=cat_m[m][key]; g["firms"].add(sym[e]); g["typ"]=typ; g["n"]+=1
    if not g["q"] and j.get("quote"): g["q"]=j["quote"][:140]
def sector_news(m,S,exclude):
    out=[]
    for k,g in cat_m.get(m,{}).items():
        if g["typ"] in BROAD: continue
        same=[f for f in g["firms"] if sec_of.get(f)==S and f!=exclude]
        if len(same)>=2: out.append((len(same),k,g["q"],sorted(same)[:5]))
    return sorted(out,reverse=True)[:4]
def macro_news(m):
    out=[(g["n"],k,g["q"]) for k,g in cat_m.get(m,{}).items() if g["typ"] in BROAD]
    return sorted(out,reverse=True)[:4]
def reason(firm,mv,S,idioev,secev,macev):
    tot,mac,sec,idi=mv["tot"],mv["macro"],mv["sector"],mv["idio"]
    shares=sorted([("macro",abs(mac)),("sector",abs(sec)),("idiosyncratic",abs(idi))],key=lambda x:-x[1])
    ev_i="\n".join(f"  - [{e.get('trust','?')}|{e.get('mech')}] contrib {('%+.1f%%'%(e['contrib']*100)) if e.get('contrib') is not None else 'n/a'} on {e.get('day','?')}: {e['cat']}: \"{e['q'][:100]}\"" for e in idioev[:5]) or "  (none captured)"
    ev_s="\n".join(f"  - {k} bound {len(fs)} {S} firms ({','.join(fs)}): \"{(q or '')[:100]}\"" for n,k,q,fs in secev) or "  (none captured)"
    ev_m="\n".join(f"  - {k} ({n} edges): \"{(q or '')[:100]}\"" for n,k,q in macev) or "  (none captured)"
    sysmsg=("You attribute a stock move to its driver. You are given a rigorous RESIDUAL DECOMPOSITION (which "
     "channel carried the move) and channel-routed evidence. Rules: weigh channels BY THE DECOMPOSITION shares; "
     "within the idiosyncratic evidence, trust 'driver'-type news (earnings/restructuring/guidance) over 'context' "
     "(regulation/factor mentions have ~no directional content); event-study contrib = the abnormal move on that "
     "article's day. Do not invent causes not in evidence; if evidence is thin say so. "
     'Return ONLY JSON {"channel":"macro|sector|idiosyncratic|mixed","primary_driver":str,"reasoning":str(<=60 words),"confidence":"high|med|low"}')
    user=(f"{firm} ({S}) moved {tot:+.1%} in {mv['m']}.\nDECOMPOSITION: macro {mac:+.1%}, sector {sec:+.1%}, idiosyncratic {idi:+.1%} "
     f"(dominant: {shares[0][0]}).\nFIRM-SPECIFIC news (event-study scored):\n{ev_i}\nSECTOR ({S}) news that month:\n{ev_s}\nMACRO news that month:\n{ev_m}")
    body={"model":MODEL,"temperature":0,"max_tokens":500,"response_format":{"type":"json_object"},
          "messages":[{"role":"system","content":sysmsg},{"role":"user","content":user}]}
    for a in range(4):
        try:
            r=httpx.post(GROQ,json=body,headers={"Authorization":f"Bearer {KEY}"},timeout=60)
            if r.status_code!=200: time.sleep(2*(a+1)); continue
            t=r.json()["choices"][0]["message"]["content"]; x,y=t.find("{"),t.rfind("}")
            return json.loads(t[x:y+1])
        except Exception: time.sleep(2*(a+1))
    return {}

def deterministic(mv,S,idioev,secev,macev):
    """Rule-based hypothesis when the LLM is unavailable: the decomposition picks the channel, the
    trust-ranked event-study evidence picks the driver."""
    shares={"macro":abs(mv["macro"]),"sector":abs(mv["sector"]),"idiosyncratic":abs(mv["idio"])}
    tot=sum(shares.values()) or 1e-9; ordered=sorted(shares.items(),key=lambda x:-x[1])
    dom=ordered[0][0]; channel="mixed" if ordered[1][1]/tot>0.33 else dom
    rank={"driver":2,"weak":1,"context":0}
    if dom=="idiosyncratic":
        sgn=1 if mv["idio"]>=0 else -1     # a driver must move the SAME direction as the idio move
        ev=sorted([e for e in idioev if e.get("contrib") is not None],
                  key=lambda e:((1 if e["contrib"]*sgn>0 else 0),rank.get(e.get("trust"),0),abs(e["contrib"])),reverse=True)
        if ev:
            pd=ev[0]["cat"]; tr=ev[0].get("trust","context")
            signok=ev[0]["contrib"]*sgn>0
            conf="high" if signok and tr=="driver" and abs(ev[0]["contrib"])>=0.01 else ("med" if signok and tr!="context" else "low")
            if not signok: why_sfx=" (WARNING: no captured event matches the move direction)"
            else: why_sfx=""
            why=f"idiosyncratic {mv['idio']:+.1%} dominates; top trust-ranked event: {pd} ({tr}, {ev[0]['contrib']:+.1%} on {ev[0].get('day','?')})"+why_sfx
        else: pd="unexplained (no captured firm-specific event)"; conf="low"; why=f"idiosyncratic {mv['idio']:+.1%} dominates but no captured event explains it"
    elif dom=="sector":
        pd=secev[0][1] if secev else f"{S} sector-wide move (no specific catalyst captured)"
        conf="med" if secev else "low"; why=f"sector {mv['sector']:+.1%} dominates; leading {S} catalyst: {pd}"
    else:
        pd=macev[0][1] if macev else "broad macro move (no specific catalyst captured)"
        conf="med" if macev else "low"; why=f"macro {mv['macro']:+.1%} dominates; leading macro catalyst: {pd}"
    return {"channel":channel,"primary_driver":pd,"reasoning":why,"confidence":conf,"method":"deterministic"}

# run on the biggest move of the first 8 firms (demo)
out=[]
for f in ATTR["firms"][:8]:
    mv=max(f["moves"],key=lambda x:abs(x["tot"])); S=f["sec"]
    se_=sector_news(mv["m"],S,f["t"]); me_=macro_news(mv["m"])
    h=reason(f["n"],mv,S,mv["events"],se_,me_)
    if not h: h=deterministic(mv,S,mv["events"],se_,me_)
    else: h["method"]="llm"
    rec={"firm":f["n"],"t":f["t"],"sec":S,"month":mv["m"],"total":mv["tot"],
         "decomp":{"macro":mv["macro"],"sector":mv["sector"],"idio":mv["idio"]},"hypothesis":h}
    out.append(rec)
    print(f"\n=== {f['n']} ({f['t']}) {mv['m']}  {mv['tot']:+.1%}  [macro {mv['macro']:+.1%} | sector {mv['sector']:+.1%} | idio {mv['idio']:+.1%}] ===")
    print(f"  channel: {h.get('channel','?')}  driver: {h.get('primary_driver','?')}  conf: {h.get('confidence','?')}")
    print(f"  reasoning: {h.get('reasoning','?')}")
json.dump(out,open(G/"unified_attributions.json","w"),indent=1)
print(f"\n-> unified_attributions.json ({len(out)} activations)")
