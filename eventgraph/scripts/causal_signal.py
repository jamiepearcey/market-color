# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Powerful causal test: pool all years and measure the MEAN SIGNED ABNORMAL RETURN
in the narrative direction  s = nsign * z  (z = event-time abnormal return, full
orthogonal factor model). If the graph carries real directional signal, mean(s) > 0
with a significant t-stat. Uses every edge (no threshold that discards 70%).

  DESCRIPTIVE : happened/ongoing edges, pre-window  [t-5..t0]
  PREDICTIVE  : forecast/hypothetical edges, forward [t+1..t+5]

Usage: uv run eventgraph/scripts/causal_signal.py --graph-dir /tmp/eg_6k --years 2010,2011,2012,2013
"""
import argparse, json, time, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

FORDER = ["market", "rates", "credit", "oil", "usd", "gold", "em"]
FACTOR_PROXY = {"market": "SPY", "rates": "IEF", "credit": "HYG", "oil": "USO",
                "usd": "UUP", "gold": "GLD", "em": "EEM"}
PROXIES = set(FACTOR_PROXY.values())
DIR_SIGN = {"up": 1, "down": -1, "widen": -1, "tighten": 1}

def yahoo(sym, cache, p1=1230768000, p2=1420070400):  # 2009 .. 2015
    p = cache / f"{sym}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            txt = r.text if r.status_code == 200 else ""
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
    ds = sorted(cl); r = {}
    for i in range(1, len(ds)):
        a, b = cl[ds[i - 1]], cl[ds[i]]
        if a > 0 and b > 0: r[ds[i]] = math.log(b / a)
    return r

def wls(y, X, w):
    XtW = X.T * w; beta = np.linalg.solve(XtW @ X, XtW @ y); return beta, (y - X @ beta)

def stat(v):
    v = np.array(v)
    if len(v) < 5: return (float("nan"), float("nan"), len(v))
    return (float(v.mean()), float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)) + 1e-12)), len(v))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--years", default="2010,2011,2012,2013")
    ap.add_argument("--max-symbols", type=int, default=450)
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = [y.strip() for y in a.years.split(",")]

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    edges = []
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or e not in sym or sym[e] in PROXIES or j.get("effect_dir") not in DIR_SIGN: continue
        if d[:4] in years:
            edges.append({"sym": sym[e], "date": d[:10], "year": d[:4], "dir": j["effect_dir"],
                          "modality": j.get("modality") or "happened"})

    need = collections.Counter(e["sym"] for e in edges)
    fetch = list(dict.fromkeys(list(FACTOR_PROXY.values()) + [s for s, _ in need.most_common(a.max_symbols)]))
    print(f"prices: {len(fetch)} symbols (cached) ...")
    px = {s: logret(yahoo(s, cache)) for s in fetch}
    ok = {s for s, r in px.items() if len(r) > 300}

    fdates = sorted(set.intersection(*[set(px[FACTOR_PROXY[f]]) for f in FORDER]))
    F = np.array([[px[FACTOR_PROXY[f]][d] for f in FORDER] for d in fdates])
    G = np.zeros_like(F)
    for j in range(len(FORDER)):
        if j == 0: G[:, 0] = F[:, 0]
        else:
            X = np.column_stack([np.ones(len(fdates)), G[:, :j]]); b, *_ = np.linalg.lstsq(X, F[:, j], rcond=None); G[:, j] = F[:, j] - X @ b
    Gmap = {d: G[i] for i, d in enumerate(fdates)}; hl = 252

    def windows(symn, date):
        r = px.get(symn, {}); ds = sorted(r)
        if date not in r:
            nx = [d for d in ds if d >= date]
            if not nx: return None
            date = nx[0]
        i = ds.index(date)
        if i < 170 or i + 5 >= len(ds): return None
        est = [d for d in ds[i - 165:i - 11] if d in Gmap]
        if len(est) < 90 or date not in Gmap: return None
        y = np.array([r[d] for d in est]); Gc = np.array([Gmap[d] for d in est])
        X = np.column_stack([np.ones(len(est)), Gc]); age = np.arange(len(est))[::-1]; w = 0.5 ** (age / hl)
        beta, resid = wls(y, X, w); rstd = resid.std()
        if rstd == 0: return None
        pred = lambda d: beta[0] + beta[1:] @ Gmap[d]
        pre = [ds[k] for k in range(i - 5, i + 1) if ds[k] in Gmap]
        fwd = [ds[k] for k in range(i + 1, i + 6) if ds[k] in Gmap]
        return (sum(r[d] - pred(d) for d in pre) / (rstd * math.sqrt(len(pre))),
                sum(r[d] - pred(d) for d in fwd) / (rstd * math.sqrt(max(len(fwd), 1))))

    S = collections.defaultdict(lambda: {"desc": [], "pred": []})
    for e in edges:
        if e["sym"] not in ok: continue
        res = windows(e["sym"], e["date"])
        if not res: continue
        zp, zf = res; ns = DIR_SIGN[e["dir"]]
        if e["modality"] in ("happened", "ongoing"): S[e["year"]]["desc"].append(ns * zp)
        elif e["modality"] in ("forecast", "hypothetical"): S[e["year"]]["pred"].append(ns * zf)

    print(f"\nmean signed abnormal return in narrative direction  (z units; t>1.96 = p<.05, t>2.58 = p<.01)")
    print(f"{'year':6} | {'DESCRIPTIVE mean':>17} {'t':>6} {'N':>5} | {'PREDICTIVE mean':>16} {'t':>6} {'N':>5}")
    alld, allp = [], []
    for y in years:
        d, p = S[y]["desc"], S[y]["pred"]; alld += d; allp += p
        md, td, nd = stat(d); mp, tp, npr = stat(p)
        star = lambda t: "**" if abs(t) > 2.58 else ("*" if abs(t) > 1.96 else "")
        print(f"{y:6} | {md:>+17.3f} {td:>6.2f}{star(td):2} {nd:>4} | {mp:>+16.3f} {tp:>6.2f}{star(tp):2} {npr:>4}")
    md, td, nd = stat(alld); mp, tp, npr = stat(allp)
    print(f"{'-'*66}")
    print(f"{'POOL':6} | {md:>+17.3f} {td:>6.2f}   {nd:>4} | {mp:>+16.3f} {tp:>6.2f}   {npr:>4}")

if __name__ == "__main__":
    main()
