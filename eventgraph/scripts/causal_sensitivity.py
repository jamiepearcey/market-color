# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Causal-sensitivity model: how much does an instrument actually move given news of a
given TYPE (mechanism / cause-entity)? Learned from the co-history of causal edges
and realised abnormal returns.

  S_mag(T)  = E[ |AR_z| | news of type T ]          -> how much markets move on this news
  S_dir(T)  = E[ nsign*AR_z | T ] (+t-stat)         -> how RELIABLY, in the claimed direction
  S(I,T)    = per-instrument response (shrunk to the type prior for sparse names)

Then attribution: given instrument I moved AR_z at t, weight each candidate news n by
  R(n) = |S_dir(type(n))| * sign_consistency(n, AR_z) * recency  -> the leading driver.

Usage: uv run eventgraph/scripts/causal_sensitivity.py --graph-dir /tmp/eg_6k \
       --years 2010,2011,2012,2013 --price-from 2009 --price-to 2014
"""
import argparse, json, time, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

FORDER = ["market", "rates", "credit", "oil", "usd", "gold", "em"]
FP = {"market": "SPY", "rates": "IEF", "credit": "HYG", "oil": "USO", "usd": "UUP", "gold": "GLD", "em": "EEM"}
PROXIES = set(FP.values()); DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}

def yahoo(sym, cache, p1, p2):
    p = cache / f"{sym}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25); txt = r.text if r.status_code == 200 else ""
        except Exception: txt = ""
        p.write_text(txt); time.sleep(0.12)
    out = {}
    try:
        res = json.loads(txt)["chart"]["result"][0]; ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(res["timestamp"], cl):
            if c is not None: out[dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d")] = float(c)
    except Exception: pass
    return out

def logret(cl):
    ds = sorted(cl); return {ds[i]: math.log(cl[ds[i]] / cl[ds[i-1]]) for i in range(1, len(ds)) if cl[ds[i-1]] > 0 and cl[ds[i]] > 0}

def tstat(v):
    v = np.array(v); return (float(v.mean()), float(v.mean() / (v.std(ddof=1)/math.sqrt(len(v))+1e-12)), len(v)) if len(v) >= 5 else (float('nan'), float('nan'), len(v))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--years", default="2010,2011,2012,2013")
    ap.add_argument("--price-from", type=int, default=2009); ap.add_argument("--price-to", type=int, default=2014)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(",")); p1 = int(dt.datetime(a.price_from,1,1,tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to,12,31,tzinfo=dt.UTC).timestamp())

    import re
    name = {m.group(1): m.group(2).replace("''","'") for m in (re.search(r"INSERT INTO entity \([^)]*\) VALUES \('([^']+)','((?:[^']|'')*)','([^']*)'", l) for l in open(gd/"pg_upsert.sql")) if m}
    sym = {}
    for l in open(gd/"entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source","").endswith(("us","exact","fuzzy"))
        if ((j["kind"]=="security" and us) or j["kind"]=="etf_proxy") and not j["symbol"].startswith("^"): sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake/"document.jsonl")}

    edges = []
    for l in open(lake/"causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or sym[e] in PROXIES or j.get("effect_dir") not in DIR: continue
        if j.get("modality") not in ("happened","ongoing"): continue     # descriptive attribution
        edges.append({"eff": e, "sym": sym[e], "cause": c, "date": d[:10], "nsign": DIR[j["effect_dir"]],
                      "mech": j.get("mechanism") or "other", "quote": (j.get("quote") or "")[:90]})

    need = collections.Counter(x["sym"] for x in edges)
    fetch = list(dict.fromkeys(list(FP.values()) + [s for s,_ in need.most_common(400)]))
    print(f"prices: {len(fetch)} symbols ...")
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in fetch}; ok = {s for s,r in px.items() if len(r) > 250}
    fdates = sorted(set.intersection(*[set(px[FP[f]]) for f in FORDER]))
    F = np.array([[px[FP[f]][d] for f in FORDER] for d in fdates]); G = np.zeros_like(F)
    for j in range(len(FORDER)):
        if j == 0: G[:,0] = F[:,0]
        else:
            X = np.column_stack([np.ones(len(fdates)), G[:,:j]]); b,*_ = np.linalg.lstsq(X, F[:,j], rcond=None); G[:,j] = F[:,j] - X@b
    Gmap = {d: G[i] for i,d in enumerate(fdates)}
    def arz(symn, date):
        r = px.get(symn,{}); ds = sorted(r)
        if date not in r:
            nx=[d for d in ds if d>=date]; date=nx[0] if nx else None
        if not date or date not in r: return None
        i = ds.index(date)
        if i < 170: return None
        est=[d for d in ds[i-165:i-11] if d in Gmap]
        if len(est)<90 or date not in Gmap: return None
        y=np.array([r[d] for d in est]); Gc=np.array([Gmap[d] for d in est]); X=np.column_stack([np.ones(len(est)),Gc])
        beta,*_=np.linalg.lstsq(X,y,rcond=None); resid=y-X@beta; rstd=resid.std()
        pre=[ds[k] for k in range(i-5,i+1) if ds[k] in Gmap]
        if rstd==0 or not pre: return None
        return sum(r[d]-(beta[0]+beta[1:]@Gmap[d]) for d in pre)/(rstd*math.sqrt(len(pre)))

    for x in edges:
        x["z"] = arz(x["sym"], x["date"]) if x["sym"] in ok else None
    ev = [x for x in edges if x["z"] is not None]
    print(f"edges {len(edges)} (descriptive) · with realised AR_z {len(ev)}\n")

    # S by mechanism
    by_m = collections.defaultdict(lambda: {"mag": [], "dir": []})
    for x in ev: by_m[x["mech"]]["mag"].append(abs(x["z"])); by_m[x["mech"]]["dir"].append(x["nsign"]*x["z"])
    print("=== S(mechanism): market's causal sensitivity to news type ===")
    print(f"  {'mechanism':20} {'|AR_z| (size)':>13} {'signed (dir)':>13} {'t':>6} {'N':>5}")
    for m, d in sorted(by_m.items(), key=lambda kv: -np.mean(kv[1]['mag']) if kv[1]['mag'] else 0):
        if len(d["mag"]) < 8: continue
        mag = np.mean(d["mag"]); md, t, n = tstat(d["dir"])
        print(f"  {m:20} {mag:>13.2f} {md:>+13.2f} {t:>6.2f} {n:>5}")

    # S by cause entity (biggest movers)
    by_c = collections.defaultdict(list)
    for x in ev:
        if x["cause"] in name: by_c[x["cause"]].append(abs(x["z"]))
    print("\n=== S(cause entity): biggest realised market-movers (mean |AR_z| of their effects) ===")
    movers = [(np.mean(v), len(v), name[c]) for c, v in by_c.items() if len(v) >= 6]
    for mag, n, nm in sorted(movers, reverse=True)[:12]:
        print(f"  {nm[:34]:34} |AR_z|={mag:.2f}  N={n}")

    # attribution examples: biggest confirmed idiosyncratic moves + their news driver
    print("\n=== attribution: largest realised moves + the news the model says is leading ===")
    for x in sorted(ev, key=lambda x: -abs(x["z"]))[:10]:
        drv = name.get(x["cause"], x["cause"] or "?")
        print(f"  {x['eff'][:20]:20} {x['sym']:6} AR_z={x['nsign']*x['z']:+.1f} <- {drv[:22]:22} [{x['mech']}] {x['date']}  \"{x['quote'][:44]}\"")

if __name__ == "__main__":
    main()
