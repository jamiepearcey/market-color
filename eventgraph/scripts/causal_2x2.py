# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Causal-side next swing: EVENT-TIME-AWARE abnormal returns off the full orthogonal
factor model, to separate description from prediction (and dodge reverse causality).

  - happened / ongoing  -> DESCRIPTIVE: score the move in the window UP TO the
    article  [t-5 .. t0]  (does the effect actually move as the news says it did?)
  - forecast / hypothetical -> PREDICTIVE: score the move AFTER the article
    [t+1 .. t+5]  (the genuine alpha test, no look-back leakage).

Run across years so a calm year (2013) can be compared to the crisis (2011), where
cross-asset correlation crushes the idiosyncratic residual.

Usage: uv run eventgraph/scripts/causal_2x2.py --graph-dir /tmp/eg_6k --years 2011,2013
"""
import argparse, json, time, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

FORDER = ["market", "rates", "credit", "oil", "usd", "gold", "em"]
FACTOR_PROXY = {"market": "SPY", "rates": "IEF", "credit": "HYG", "oil": "USO",
                "usd": "UUP", "gold": "GLD", "em": "EEM"}
PROXIES = set(FACTOR_PROXY.values())
DIR_SIGN = {"up": 1, "down": -1, "widen": -1, "tighten": 1}

def yahoo(sym, cache, p1=1230768000, p2=1388534400):
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
    XtW = X.T * w; XtWX = XtW @ X
    beta = np.linalg.solve(XtWX, XtW @ y)
    return beta, (y - X @ beta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--years", default="2011,2013")
    ap.add_argument("--max-symbols", type=int, default=350)
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = [y.strip() for y in a.years.split(",")]

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    causal = collections.defaultdict(list)
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or e not in sym or sym[e] in PROXIES or j.get("effect_dir") not in DIR_SIGN: continue
        yr = d[:4]
        if yr in years:
            causal[yr].append({"sym": sym[e], "date": d[:10], "dir": j["effect_dir"],
                               "modality": j.get("modality") or "happened", "quote": (j.get("quote") or "")[:90]})

    need = collections.Counter()
    for yr in years:
        for c in causal[yr]: need[c["sym"]] += 1
    fetch = list(dict.fromkeys(list(FACTOR_PROXY.values()) + [s for s, _ in need.most_common(a.max_symbols)]))
    print(f"prices: {len(fetch)} symbols (cached) ...")
    px = {s: logret(yahoo(s, cache)) for s in fetch}
    ok = {s for s, r in px.items() if len(r) > 250}

    fdates = sorted(set.intersection(*[set(px[FACTOR_PROXY[f]]) for f in FORDER]))
    F = np.array([[px[FACTOR_PROXY[f]][d] for f in FORDER] for d in fdates])
    G = np.zeros_like(F)
    for j in range(len(FORDER)):
        if j == 0: G[:, 0] = F[:, 0]
        else:
            X = np.column_stack([np.ones(len(fdates)), G[:, :j]]); b, *_ = np.linalg.lstsq(X, F[:, j], rcond=None); G[:, j] = F[:, j] - X @ b
    Gmap = {d: G[i] for i, d in enumerate(fdates)}
    hl = 252

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
        pre = [ds[k] for k in range(i - 5, i + 1) if ds[k] in Gmap]      # t-5..t0  (descriptive)
        fwd = [ds[k] for k in range(i + 1, i + 6) if ds[k] in Gmap]      # t+1..t+5 (predictive)
        car_pre = sum(r[d] - pred(d) for d in pre); car_fwd = sum(r[d] - pred(d) for d in fwd)
        return car_pre / (rstd * math.sqrt(len(pre))), car_fwd / (rstd * math.sqrt(max(len(fwd), 1)))

    def classify(z, nsign):
        return "over_attributed" if abs(z) < 1.0 else ("confirmed" if (z > 0) == (nsign > 0) else "contradiction")

    print(f"\n{'':10} {'DESCRIPTIVE (happened, t-5..t0)':>34}   {'PREDICTIVE (forecast, t+1..t+5)':>34}")
    print(f"{'year':10} {'conf':>6}{'over':>6}{'contra':>7}{'net':>6}{'N':>6}   {'conf':>6}{'over':>6}{'contra':>7}{'net':>6}{'N':>6}")
    for yr in years:
        desc, pred = collections.Counter(), collections.Counter()
        for c in causal[yr]:
            if c["sym"] not in ok: continue
            res = windows(c["sym"], c["date"])
            if not res: continue
            zp, zf = res; ns = DIR_SIGN[c["dir"]]
            if c["modality"] in ("happened", "ongoing"): desc[classify(zp, ns)] += 1
            elif c["modality"] in ("forecast", "hypothetical"): pred[classify(zf, ns)] += 1
        def row(b):
            n = sum(b.values()); net = (b["confirmed"] - b["contradiction"]) / max(n, 1)
            return f"{b['confirmed']:>6}{b['over_attributed']:>6}{b['contradiction']:>7}{net:>+6.2f}{n:>6}"
        print(f"{yr:10} {row(desc)}   {row(pred)}")
    print("\nreads: DESCRIPTIVE net>0 = the news describes real moves; PREDICTIVE net>0 = it")
    print("       anticipates them. 2013 (calm) vs 2011 (crisis) isolates the correlation effect.")

if __name__ == "__main__":
    main()
