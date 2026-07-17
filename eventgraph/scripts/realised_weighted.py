# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Appropriately-WEIGHTED realised sensitivities + a sharper causal 2x2.

Upgrades over realised_mvp.py (univariate, unweighted):
  1. ORTHOGONALIZED factors (market -> rates|mkt -> credit|{mkt,rates} -> ...),
     killing the multicollinearity that inflated the raw betas.
  2. MULTIVARIATE, EWMA-weighted (recent-regime) regression per asset -> each beta
     is the PARTIAL loading (controls for the other factors), with a real SE / t / R2.
  3. SECTOR (entity-type) empirical-Bayes shrinkage -> borrow strength for sparse names.
  4. BAYESIAN FUSION with the narrative prior (magnitude+basis+count -> prior mean &
     precision) -> posterior beta*  weighted by prior vs data precision.
  5. Abnormal returns re-derived off the FULL orthogonal factor model (not market-only)
     -> cleaner idiosyncratic residual -> the causal 2x2.

Output: lake/realised_sensitivity.jsonl {uni_beta, multi_beta, se, r2, sector_mean,
        narrative_prior, beta_star, post_precision, agrees}.

Usage: uv run eventgraph/scripts/realised_weighted.py --graph-dir /tmp/eg_6k --year 2011
"""
import argparse, json, time, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

FORDER = ["market", "rates", "credit", "oil", "usd", "gold", "em"]
FACTOR_PROXY = {"market": "SPY", "rates": "IEF", "credit": "HYG", "oil": "USO",
                "usd": "UUP", "gold": "GLD", "em": "EEM"}
SLUG2F = {"oil": "oil", "crude_oil": "oil", "crude": "oil", "oil_prices": "oil", "crude_oil_prices": "oil",
          "brent": "oil", "brent_crude": "oil", "energy": "oil", "rates": "rates", "interest_rate": "rates",
          "interest_rates": "rates", "rate": "rates", "yields": "rates", "treasury_yields": "rates",
          "bond": "rates", "bonds": "rates", "inflation": "rates", "credit": "credit", "credit_spread": "credit",
          "credit_spreads": "credit", "high_yield": "credit", "usd": "usd", "dollar": "usd", "us_dollar": "usd",
          "u_s_dollar": "usd", "currency": "usd", "gold": "gold", "equity": "market", "equities": "market",
          "equity_beta": "market", "stocks": "market", "market": "market", "em": "em", "emerging_markets": "em"}
DIR_SIGN = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
PROXIES = set(FACTOR_PROXY.values())   # entities that ARE a factor proxy: self-referential, exclude
MAG = {"high": 1.2, "medium": 0.7, "low": 0.3}
BASISW = {"stated_beta": 1.5, "historical": 1.3, "fundamental": 1.0}

def yahoo(sym, cache, p1=1230768000, p2=1388534400):
    p = cache / f"{sym}.json"
    txt = p.read_text() if p.exists() else ""
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
    XtW = X.T * w
    XtWX = XtW @ X
    beta = np.linalg.solve(XtWX, XtW @ y)
    resid = y - X @ beta
    dof = max(w.sum() - X.shape[1], 1)
    s2 = (w * resid**2).sum() / dof
    cov = s2 * np.linalg.inv(XtWX)
    ybar = np.average(y, weights=w)
    tss = (w * (y - ybar)**2).sum()
    r2 = 1 - (w * resid**2).sum() / max(tss, 1e-12)
    return beta, cov, r2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k"); ap.add_argument("--year", type=int, default=2011)
    ap.add_argument("--max-symbols", type=int, default=300); ap.add_argument("--halflife", type=int, default=252)
    ap.add_argument("--price-from", type=int, default=2009); ap.add_argument("--price-to", type=int, default=2014)
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp())
    p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
    etype = {}
    import re
    for m in (re.search(r"INSERT INTO entity \([^)]*\) VALUES \('([^']+)','(?:[^']|'')*','([^']*)'", l) for l in open(gd / "pg_upsert.sql")):
        if m: etype[m.group(1)] = m.group(2)
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    # aggregate sensitivity edges by (asset, factor)
    sagg = {}
    for l in open(lake / "sensitivity_edge.jsonl"):
        j = json.loads(l); ast = j.get("asset_entity"); f = SLUG2F.get((j.get("factor_id") or "").lower())
        if ast in sym and f and j.get("sign") is not None:
            d = sagg.setdefault((ast, f), {"signs": [], "mags": [], "basis": []})
            d["signs"].append(1 if j["sign"] > 0 else -1); d["mags"].append(j.get("magnitude_qual"))
            d["basis"].append(j.get("basis"))
    causal = []
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if d and d.startswith(str(a.year)) and e in sym and j.get("effect_dir") in DIR_SIGN:
            causal.append({"eff": e, "sym": sym[e], "date": d[:10], "dir": j["effect_dir"],
                           "modality": j.get("modality") or "happened", "quote": (j.get("quote") or "")[:120]})

    # fetch prices
    need = collections.Counter()
    for (ast, f) in sagg: need[sym[ast]] += 1
    for c in causal: need[c["sym"]] += 1
    fetch = list(dict.fromkeys(list(FACTOR_PROXY.values()) + [s for s, _ in need.most_common(a.max_symbols)]))
    print(f"prices: {len(fetch)} symbols (cached) ...")
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in fetch}
    ok = {s for s, r in px.items() if len(r) > 250}
    print(f"  {len(ok)}/{len(fetch)} usable\n")

    # orthogonal factor matrix on common factor dates
    fdates = sorted(set.intersection(*[set(px[FACTOR_PROXY[f]]) for f in FORDER]))
    F = np.array([[px[FACTOR_PROXY[f]][d] for f in FORDER] for d in fdates])
    G = np.zeros_like(F)
    for j in range(len(FORDER)):
        if j == 0: G[:, 0] = F[:, 0]
        else:
            X = np.column_stack([np.ones(len(fdates)), G[:, :j]])
            b, *_ = np.linalg.lstsq(X, F[:, j], rcond=None); G[:, j] = F[:, j] - X @ b
    Gmap = {d: G[i] for i, d in enumerate(fdates)}
    fcol = {f: i for i, f in enumerate(FORDER)}
    hl = a.halflife

    def asset_fit(symn, cols):  # multivariate EWMA WLS on orthogonal factor columns
        r = px[symn]; common = [d for d in fdates if d in r]
        if len(common) < 120: return None
        y = np.array([r[d] for d in common]); Gc = np.array([Gmap[d] for d in common])
        X = np.column_stack([np.ones(len(common))] + [Gc[:, fcol[f]] for f in cols])
        age = np.arange(len(common))[::-1]; w = 0.5 ** (age / hl)
        beta, cov, r2 = wls(y, X, w)
        return {f: (beta[k + 1], math.sqrt(max(cov[k + 1, k + 1], 1e-12))) for k, f in enumerate(cols)}, r2

    # per-asset multivariate fit on (market + its named factors)
    assets = {}
    for (ast, f) in sagg:
        assets.setdefault(ast, set()).add(f)
    rows, group = [], collections.defaultdict(list)
    for ast, facs in assets.items():
        symn = sym[ast]
        if symn not in ok or symn in PROXIES: continue
        cols = sorted({"market"} | facs)
        fit = asset_fit(symn, cols)
        if not fit: continue
        betas, r2 = fit
        for f in facs:
            if f == "market": continue
            b, se = betas[f]
            if se < 0.002: continue   # degenerate / near-collinear -> not a real loading
            # univariate (for contrast)
            uni = asset_fit(symn, ["market", f]) if f != "market" else None
            ub = uni[0][f][0] if uni else float("nan")
            agg = sagg[(ast, f)]
            nsign = 1 if sum(agg["signs"]) >= 0 else -1
            rows.append({"entity_id": ast, "symbol": symn, "type": etype.get(ast, "other"), "factor": f,
                         "uni_beta": round(ub, 3), "multi_beta": round(b, 3), "se": round(se, 3), "r2": round(r2, 3),
                         "n_assert": len(agg["signs"]), "consistency": round(abs(sum(agg["signs"])) / len(agg["signs"]), 2),
                         "mag": collections.Counter(x for x in agg["mags"] if x).most_common(1),
                         "basis": collections.Counter(x for x in agg["basis"] if x).most_common(1), "nsign": nsign})
            group[(etype.get(ast, "other"), f)].append((b, se))

    # sector empirical-Bayes shrinkage + Bayesian narrative fusion -> beta_star
    gmean = {}
    for k, vs in group.items():
        bs = np.array([b for b, _ in vs]); prec = np.array([1 / max(se, 1e-3)**2 for _, se in vs])
        gmean[k] = (float((bs * prec).sum() / prec.sum()), float(bs.var()) + 1e-3)
    agree = 0
    for r in rows:
        b, se = r["multi_beta"], max(r["se"], 1e-3)
        gm, gv = gmean[(r["type"], r["factor"])]
        # shrink toward sector mean
        bsh = (b / se**2 + gm / gv) / (1 / se**2 + 1 / gv); sesh = math.sqrt(1 / (1 / se**2 + 1 / gv))
        # narrative prior
        magv = MAG.get(r["mag"][0][0], 0.5) if r["mag"] else 0.5
        basisw = BASISW.get(r["basis"][0][0], 1.0) if r["basis"] else 1.0
        mu0 = r["nsign"] * magv
        tau0 = 0.4 * basisw * math.sqrt(r["n_assert"]) * r["consistency"]
        prec = 1 / sesh**2 + tau0
        bstar = (bsh / sesh**2 + mu0 * tau0) / prec
        r["sector_mean"] = round(gm, 3); r["narrative_prior"] = round(mu0, 3)
        r["beta_star"] = round(bstar, 3); r["post_precision"] = round(prec, 1)
        r["agrees"] = bool((bstar > 0) == (r["nsign"] > 0)); agree += r["agrees"]
        r["mag"] = r["mag"][0][0] if r["mag"] else None; r["basis"] = r["basis"][0][0] if r["basis"] else None
    rows.sort(key=lambda r: -r["post_precision"])
    with open(lake / "realised_sensitivity.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    print("=== (3) WEIGHTED SENSITIVITY (orthogonal multivariate + shrinkage + narrative fusion) ===")
    print(f"loadings {len(rows)}  ·  posterior sign agrees with narrative {agree}/{len(rows)} ({100*agree/max(len(rows),1):.0f}%)"
          f"  ·  mean R2 {np.mean([r['r2'] for r in rows]):.2f}")
    print(f"  {'asset':20} {'factor':6} {'uni_b':>6} {'multi_b':>7} {'se':>5} {'β*':>6} {'R2':>5} {'prec':>6}")
    for r in rows[:14]:
        print(f"  {r['entity_id'][:20]:20} {r['factor']:6} {r['uni_beta']:>6} {r['multi_beta']:>7} {r['se']:>5} {r['beta_star']:>6} {r['r2']:>5} {r['post_precision']:>6.0f}")

    # ---------- (4) 2x2 off the FULL orthogonal factor model ----------
    def abn(symn, date):
        r = px.get(symn, {}); ds = sorted(r)
        if date not in r:
            nx = [d for d in ds if d >= date]
            if not nx: return None
            date = nx[0]
        i = ds.index(date)
        if i < 150: return None
        est = [d for d in ds[i - 150:i - 5] if d in Gmap]
        if len(est) < 90: return None
        y = np.array([r[d] for d in est]); Gc = np.array([Gmap[d] for d in est])
        X = np.column_stack([np.ones(len(est)), Gc])
        age = np.arange(len(est))[::-1]; w = 0.5 ** (age / hl)
        beta, _, _ = wls(y, X, w)
        resid = y - X @ beta; rstd = resid.std()
        if date not in Gmap or rstd == 0: return None
        pred = lambda d: beta[0] + beta[1:] @ Gmap[d]
        ar0 = r[date] - pred(date)
        fwd = [ds[j] for j in range(i + 1, min(i + 6, len(ds))) if ds[j] in Gmap]
        car = sum(r[d] - pred(d) for d in fwd)
        return ar0 / rstd, car / (rstd * math.sqrt(max(len(fwd), 1)))
    box = collections.Counter(); mod = collections.defaultdict(collections.Counter)
    for c in causal:
        if c["sym"] not in ok or c["sym"] in PROXIES: continue
        res = abn(c["sym"], c["date"])
        if not res: continue
        z0, zf = res; z = z0 if c["modality"] == "happened" else zf; ns = DIR_SIGN[c["dir"]]
        cls = "over_attributed" if abs(z) < 1.0 else ("confirmed" if (z > 0) == (ns > 0) else "contradiction")
        box[cls] += 1; mod[c["modality"]][cls] += 1
    n = sum(box.values())
    print(f"\n=== (4) NARRATIVE x REALISED 2x2  ({a.year}, FULL {len(FORDER)}-factor abnormal returns) ===")
    for k in ["confirmed", "over_attributed", "contradiction"]:
        print(f"  {k:16} {box[k]:>4}  ({100*box[k]/max(n,1):.0f}%)")
    print(f"  evaluated edges: {n}")
    for m in ["happened", "forecast", "ongoing"]:
        c = mod[m]; t = sum(c.values())
        if t: print(f"    {m:10} confirmed {c['confirmed']}/{t} ({100*c['confirmed']/t:.0f}%)  contra {c['contradiction']} ({100*c['contradiction']/t:.0f}%)")

if __name__ == "__main__":
    main()
