# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Realised-plane MVP: turn the narrative graph's ASSERTED sensitivities/causes into
MEASURED ones, using real prices.

  1. Factor registry: canonical factor -> tradeable US proxy ETF; map the messy
     sensitivity `factor_id` slugs onto it.
  2. Prices: free daily bars from stooq (no key) for the factor proxies + the
     ticker-resolved entities. Cached.
  3. Realised betas (narrative-guided): for each asset that the graph says loads
     on factor F, regress its returns on F's proxy -> beta + R2; check whether the
     MEASURED sign agrees with the ASSERTED sign.  (Is sensitivity real?)
  4. Abnormal-return 2x2: for causal edges whose EFFECT is priced, market-model
     abnormal return around the article date -> CONFIRMED / OVER-ATTRIBUTED /
     CONTRADICTION, split by modality (happened=day0 descriptive, forecast=t+1..5
     predictive).  (Did the news actually move the market?)

Deliberate about method: log returns; market model estimated on a trailing window
with a gap to avoid event leakage; AR z-scored by residual vol; US-listed only
(stooq .us) to dodge FX/timezone bias; narrative selects the regressors.

Usage: uv run eventgraph/scripts/realised_mvp.py --graph-dir /tmp/eg_6k --year 2011
"""
import argparse, json, time, math, collections, datetime as dt
from pathlib import Path
import httpx, numpy as np

# canonical factor -> US proxy ETF (Yahoo ticker)
FACTOR_PROXY = {"market": "SPY", "rates": "IEF", "credit": "HYG",
                "oil": "USO", "usd": "UUP", "gold": "GLD", "em": "EEM"}
SLUG2F = {  # sensitivity_edge.factor_id slug -> canonical factor
    "oil": "oil", "crude_oil": "oil", "crude": "oil", "oil_prices": "oil", "crude_oil_prices": "oil",
    "brent": "oil", "brent_crude": "oil", "energy": "oil",
    "rates": "rates", "interest_rate": "rates", "interest_rates": "rates", "rate": "rates",
    "yields": "rates", "treasury_yields": "rates", "bond": "rates", "bonds": "rates", "inflation": "rates",
    "credit": "credit", "credit_spread": "credit", "credit_spreads": "credit", "high_yield": "credit",
    "usd": "usd", "dollar": "usd", "us_dollar": "usd", "u_s_dollar": "usd", "currency": "usd",
    "gold": "gold", "equity": "market", "equities": "market", "equity_beta": "market",
    "stocks": "market", "market": "market", "em": "em", "emerging_markets": "em",
}
DIR_SIGN = {"up": 1, "down": -1, "widen": -1, "tighten": 1}

def yahoo(sym, cache, p1=1230768000, p2=1388534400):  # 2009-01-01 .. 2014-01-01
    p = cache / f"{sym}.json"
    if p.exists():
        txt = p.read_text()
    else:
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
            txt = r.text if r.status_code == 200 else ""
        except Exception:
            txt = ""
        p.write_text(txt); time.sleep(0.12)
    out = {}
    try:
        res = json.loads(txt)["chart"]["result"][0]
        ts = res["timestamp"]
        ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose") if "adjclose" in ind else None) or ind["quote"][0]["close"]
        for t, c in zip(ts, cl):
            if c is not None:
                out[dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")] = float(c)
    except Exception:
        pass
    return out  # {date: close}

def logret(closes):
    ds = sorted(closes)
    r = {}
    for i in range(1, len(ds)):
        a, b = closes[ds[i - 1]], closes[ds[i]]
        if a > 0 and b > 0: r[ds[i]] = math.log(b / a)
    return r

def ols(y, X):  # X: (n,k) incl intercept col; returns beta, resid_std, r2
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid); tss = float(((y - y.mean()) ** 2).sum())
    return beta, math.sqrt(rss / max(len(y) - X.shape[1], 1)), (1 - rss / tss if tss > 0 else 0.0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--year", type=int, default=2011)
    ap.add_argument("--max-symbols", type=int, default=300)
    a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"
    cache = gd / "prices"; cache.mkdir(exist_ok=True)

    sym = {}  # entity_id -> ticker (US-listed only)
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if (j["kind"] == "security" and us) or j["kind"] == "etf_proxy":
            if not j["symbol"].startswith("^"):
                sym[j["entity_id"]] = j["symbol"]
    sym = {k: v for k, v in sym.items() if v}

    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    # causal edges in the target year with a priced effect
    causal = []
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id"))
        if not d or not d.startswith(str(a.year)): continue
        e = j.get("effect_entity")
        if e in sym and j.get("effect_dir") in DIR_SIGN:
            causal.append({"eff": e, "sym": sym[e], "date": d[:10], "dir": j["effect_dir"],
                           "modality": j.get("modality") or "happened", "quote": (j.get("quote") or "")[:120],
                           "cause": j.get("cause_entity")})
    # sensitivity edges with priced asset + mappable factor
    sens = []
    for l in open(lake / "sensitivity_edge.jsonl"):
        j = json.loads(l); ast = j.get("asset_entity"); fid = (j.get("factor_id") or "").lower()
        f = SLUG2F.get(fid)
        if ast in sym and f and f in FACTOR_PROXY and j.get("sign") is not None:
            sens.append({"asset": ast, "sym": sym[ast], "factor": f, "nsign": 1 if j["sign"] > 0 else -1,
                         "quote": (j.get("quote") or "")[:120]})

    # which symbols to fetch: factor proxies + most-referenced entity symbols
    need = collections.Counter()
    for c in causal: need[c["sym"]] += 1
    for s in sens: need[s["sym"]] += 1
    fetch = list(FACTOR_PROXY.values()) + [s for s, _ in need.most_common(a.max_symbols)]
    fetch = list(dict.fromkeys(fetch))
    print(f"fetching {len(fetch)} price series from stooq (cached) ...")
    px = {}
    for i, s in enumerate(fetch):
        px[s] = logret(yahoo(s, cache))
        if (i + 1) % 40 == 0: print(f"  {i+1}/{len(fetch)}")
    ok = {s for s, r in px.items() if len(r) > 200}
    print(f"  {len(ok)}/{len(fetch)} have usable history\n")

    # ---------- (3) realised betas: narrative-guided ----------
    fp = {f: px.get(p, {}) for f, p in FACTOR_PROXY.items()}
    beta_rows, agree, tot = [], 0, 0
    by_asset = collections.defaultdict(set)
    for s in sens:
        if s["sym"] in ok: by_asset[(s["asset"], s["sym"])].add((s["factor"], s["nsign"]))
    for (asset, symn), facs in by_asset.items():
        ar = px[symn]
        for f, nsign in facs:
            fr = fp[f]
            common = sorted(set(ar) & set(fr))
            if len(common) < 60: continue
            y = np.array([ar[d] for d in common]); x = np.array([fr[d] for d in common])
            X = np.column_stack([np.ones_like(x), x])
            b, _, r2 = ols(y, X); beta = b[1]
            match = bool((beta > 0) == (nsign > 0))
            tot += 1; agree += int(match)
            beta_rows.append({"entity_id": asset, "symbol": symn.upper(), "factor": f,
                              "narrative_sign": nsign, "realised_beta": round(beta, 3),
                              "r2": round(r2, 3), "sign_agrees": match, "n": len(common)})
    beta_rows.sort(key=lambda r: -abs(r["realised_beta"]))

    # ---------- (4) abnormal-return 2x2 ----------
    spy = px.get("SPY", {})
    def abn(symn, date):
        r = px.get(symn, {}); ds = sorted(r)
        if date not in r:  # snap to next trading day
            nxt = [d for d in ds if d >= date]
            if not nxt: return None
            date = nxt[0]
        i = ds.index(date)
        if i < 130: return None
        est = ds[i - 130:i - 5]
        common = [d for d in est if d in spy]
        if len(common) < 60: return None
        y = np.array([r[d] for d in common]); m = np.array([spy[d] for d in common])
        b, rstd, _ = ols(y, np.column_stack([np.ones_like(m), m]))
        if date not in spy or rstd == 0: return None
        ar0 = r[date] - (b[0] + b[1] * spy[date])
        fwd = [ds[j] for j in range(i + 1, min(i + 6, len(ds))) if ds[j] in spy]
        car = sum(r[d] - (b[0] + b[1] * spy[d]) for d in fwd)
        return ar0 / rstd, car / (rstd * math.sqrt(max(len(fwd), 1)))
    box = collections.Counter(); by_mod = collections.defaultdict(collections.Counter); examples = []
    for c in causal:
        if c["sym"] not in ok: continue
        res = abn(c["sym"], c["date"])
        if not res: continue
        z0, zf = res
        z = z0 if c["modality"] == "happened" else zf   # descriptive vs predictive
        nsign = DIR_SIGN[c["dir"]]
        if abs(z) < 1.0: cls = "over_attributed"
        elif (z > 0) == (nsign > 0): cls = "confirmed"
        else: cls = "contradiction"
        box[cls] += 1; by_mod[c["modality"]][cls] += 1
        if cls in ("confirmed", "contradiction") and len(examples) < 10:
            examples.append((cls, c["eff"], c["sym"].upper(), c["dir"], round(z, 1), c["date"], c["quote"]))

    # ---------- write + report ----------
    with open(lake / "realised_sensitivity.jsonl", "w") as f:
        for r in beta_rows: f.write(json.dumps(r) + "\n")
    print("=== (3) REALISED SENSITIVITY (narrative-guided betas) ===")
    print(f"measured {tot} asset-factor loadings; narrative sign agrees with realised beta: "
          f"{agree}/{tot} ({100*agree/max(tot,1):.0f}%)")
    print(f"  {'asset':22} {'factor':7} {'nsign':>5} {'beta':>7} {'R2':>5}")
    for r in beta_rows[:12]:
        print(f"  {r['entity_id'][:22]:22} {r['factor']:7} {r['narrative_sign']:>5} {r['realised_beta']:>7} {r['r2']:>5}")
    print(f"\n=== (4) NARRATIVE x REALISED 2x2  ({a.year}, market-model abnormal returns) ===")
    n = sum(box.values())
    for k in ["confirmed", "over_attributed", "contradiction"]:
        print(f"  {k:16} {box[k]:>4}  ({100*box[k]/max(n,1):.0f}%)")
    print(f"  evaluated edges: {n}")
    print("  modality gradient (confirm rate) -- the go/no-go signal:")
    for m in ["happened", "forecast", "ongoing", "hypothetical"]:
        c = by_mod[m]; t = sum(c.values())
        if t: print(f"    {m:12} confirmed {c['confirmed']}/{t} ({100*c['confirmed']/t:.0f}%)")
    print("\n  examples:")
    for cls, e, s, d, z, dat, q in examples:
        print(f"    [{cls:13}] {e[:26]:26} {s:8} says {d:5} realised z={z:+.1f} ({dat})  \"{q[:60]}\"")

if __name__ == "__main__":
    main()
