# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Causal signal using the QUANT ENGINE's factors (factor_snapshot_factor_returns.csv
produced by quant-algos multi_factor_fit) instead of hand-rolled ETF factors.

Abnormal return = residual of the instrument's returns regressed (EWMA) on the
engine's 6 factors over a trailing window; then the signed-return t-test
(descriptive t-5..t0 for happened edges, predictive t+1..t+5 for forecast edges).

Usage: uv run causal_engine.py --graph-dir /tmp/eg_6k --years 2011,2012,2013
"""
import argparse, json, time, math, collections, csv, datetime as dt
from pathlib import Path
import httpx, numpy as np

FN = ["crypto", "fx_major", "fx_other", "bonds", "equity_idx", "metals"]
DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
# instruments that are macro proxies in OUR ticker space — skip as effects
SKIP = {"SPY", "IEF", "HYG", "USO", "UUP", "GLD", "EEM"}

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

def wls(y, X, w):
    XtW = X.T * w; return np.linalg.solve(XtW @ X, XtW @ y)

def tstat(v):
    v = np.array(v); return (float(v.mean()), float(v.mean() / (v.std(ddof=1)/math.sqrt(len(v))+1e-12)), len(v)) if len(v) >= 5 else (float('nan'), float('nan'), len(v))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--years", default="2011,2012,2013")
    ap.add_argument("--price-from", type=int, default=2009); ap.add_argument("--price-to", type=int, default=2014)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(",")); p1 = int(dt.datetime(a.price_from,1,1,tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to,12,31,tzinfo=dt.UTC).timestamp())

    # engine factors (keep NaNs; select live factors per window -> works pre-crypto too)
    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; ds = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        Fmap[ds] = np.array([float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])
    print(f"engine factors: {len(Fmap)} days, {len(FN)} factors (live factors selected per window)")

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"): sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    edges = []
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or sym[e] in SKIP or j.get("effect_dir") not in DIR: continue
        edges.append({"sym": sym[e], "date": d[:10], "year": d[:4], "nsign": DIR[j["effect_dir"]], "modality": j.get("modality") or "happened"})

    need = collections.Counter(e["sym"] for e in edges)
    fetch = [s for s, _ in need.most_common(450)]
    print(f"prices: {len(fetch)} instruments (cached) ...")
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in fetch}; ok = {s for s, r in px.items() if len(r) > 250}

    def windows(symn, date):
        r = px.get(symn, {}); ds = sorted(r)
        if date not in r:
            nx = [d for d in ds if d >= date]; date = nx[0] if nx else None
        if not date or date not in r or date not in Fmap: return None
        i = ds.index(date)
        if i < 170 or i + 5 >= len(ds): return None
        est = [d for d in ds[i-165:i-11] if d in Fmap]
        if len(est) < 90: return None
        Fc = np.array([Fmap[d] for d in est])                                  # (m,6)
        colok = np.where(np.isfinite(Fc).mean(axis=0) >= 0.9)[0]               # live factors this window
        if len(colok) == 0: return None
        rows = [k for k in range(len(est)) if np.all(np.isfinite(Fc[k, colok]))]
        if len(rows) < 90: return None
        y = np.array([r[est[k]] for k in rows]); Xf = Fc[np.ix_(rows, colok)]
        X = np.column_stack([np.ones(len(rows)), Xf]); age = np.arange(len(rows))[::-1]; w = 0.5 ** (age / 252)
        beta = wls(y, X, w); rstd = (y - X @ beta).std()
        if rstd == 0: return None
        def pred(d):
            f = Fmap[d][colok]
            return (beta[0] + beta[1:] @ f) if np.all(np.isfinite(f)) else None
        def car(days):
            s, c = 0.0, 0
            for d in days:
                p = pred(d)
                if p is not None: s += r[d] - p; c += 1
            return s, c
        pre = [ds[k] for k in range(i-5, i+1) if ds[k] in Fmap and ds[k] in r]
        fwd = [ds[k] for k in range(i+1, i+6) if ds[k] in Fmap and ds[k] in r]
        sp, cp = car(pre); sf, cf = car(fwd)
        if cp == 0: return None
        return (sp / (rstd * math.sqrt(cp)), sf / (rstd * math.sqrt(max(cf, 1))))

    S = collections.defaultdict(lambda: {"desc": [], "pred": []})
    for e in edges:
        if e["sym"] not in ok: continue
        res = windows(e["sym"], e["date"])
        if not res: continue
        zp, zf = res
        if e["modality"] in ("happened", "ongoing"): S[e["year"]]["desc"].append(e["nsign"] * zp)
        elif e["modality"] in ("forecast", "hypothetical"): S[e["year"]]["pred"].append(e["nsign"] * zf)

    print(f"\nsigned abnormal return in narrative direction — ENGINE factors  (t>1.96 = p<.05)")
    print(f"{'year':6} | {'DESCRIPTIVE mean':>17} {'t':>6} {'N':>5} | {'PREDICTIVE mean':>16} {'t':>6} {'N':>5}")
    ad, ap_ = [], []
    for y in sorted(S) if not years else sorted(years):
        d, pr = S[y]["desc"], S[y]["pred"]; ad += d; ap_ += pr
        md, td, nd = tstat(d); mp, tp, npr = tstat(pr); st = lambda t: "**" if abs(t) > 2.58 else ("*" if abs(t) > 1.96 else "")
        print(f"{y:6} | {md:>+17.3f} {td:>6.2f}{st(td):2} {nd:>4} | {mp:>+16.3f} {tp:>6.2f}{st(tp):2} {npr:>4}")
    md, td, nd = tstat(ad); mp, tp, npr = tstat(ap_)
    print("-" * 66)
    print(f"{'POOL':6} | {md:>+17.3f} {td:>6.2f}   {nd:>4} | {mp:>+16.3f} {tp:>6.2f}   {npr:>4}")

if __name__ == "__main__":
    main()
