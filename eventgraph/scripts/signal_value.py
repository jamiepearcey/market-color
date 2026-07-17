# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
How much value is in the signal? — quantify BEFORE spending effort improving resolution.

The validated signal is CONTEMPORANEOUS (nowcast), and next-day is a reversal. So value
splits into two honest buckets:

  ATTRIBUTION value (robust) — of the idiosyncratic moves we can see, how much do we
    explain and how big are the events?
      * contemporaneous IC and cross-sectional R^2 (share of idiosyncratic variance)
      * per-event economics: mean |abnormal %| of an attributed big mover, and the
        expected SIGNED abnormal % in the claimed direction ("see this -> expect X%")
      * attribution yield: precision on big movers with a same-day directional edge

  TRADABLE value (the honest bound) — next-day, market-neutral, follow-the-signal:
      * per-event forward edge E[sign(signal) * next-day abnormal z] and its t
      * a daily long-top/short-bottom book -> annualized return & Sharpe
    Expected ~0 (efficient pricing); this bounds the "alpha" so we don't oversell.

Runs on the rigorous 9-engine-factor dataset (eg_6k, cached prices), split securities
vs ETF proxies. Usage: uv run eventgraph/scripts/signal_value.py --graph-dir /tmp/eg_6k
"""
import argparse, json, time, math, collections, csv, datetime as dt
from pathlib import Path
import httpx, numpy as np

FN = ["crypto", "fx_major", "fx_other", "bonds", "equity_idx", "metals", "em_fx_latam", "em_fx_emea", "em_fx_asia"]
DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
SKIP = {"SPY", "IEF", "HYG", "USO", "UUP", "GLD", "EEM", "FXI", "^TNX", "^VIX", "TLT", "DIA", "QQQ", "ACWI"}
TRADING_DAYS = 252


def yahoo(sym, cache, p1, p2):
    p = cache / f"{sym}.json"; txt = p.read_text() if p.exists() else ""
    if not p.exists():
        try:
            r = httpx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?period1={p1}&period2={p2}&interval=1d",
                          headers={"User-Agent": "Mozilla/5.0"}, timeout=25); txt = r.text if r.status_code == 200 else ""
        except Exception: txt = ""
        p.write_text(txt); time.sleep(0.1)
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


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 8: return (float("nan"), float("nan"), len(a))
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); d = math.sqrt((ra @ ra) * (rb @ rb))
    if d == 0: return (float("nan"), float("nan"), len(a))
    r = float((ra @ rb) / d); return (r, r * math.sqrt(max(len(a) - 2, 1) / max(1 - r * r, 1e-12)), len(a))


def tmean(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 5: return (float("nan"), float("nan"), len(v))
    return (float(v.mean()), float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)) + 1e-12)), len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--years", default="2010,2011,2012,2013")
    ap.add_argument("--price-from", type=int, default=2008); ap.add_argument("--price-to", type=int, default=2014)
    ap.add_argument("--fetch", type=int, default=600); a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"
    years = set(a.years.split(",")); p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = np.array([float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])

    sym = {}; sym_kind = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
            if sym_kind.get(j["symbol"]) != "security": sym_kind[j["symbol"]] = j["kind"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    sig = collections.Counter()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or sym[e] in SKIP or j.get("effect_dir") not in DIR: continue
        sig[(sym[e], d[:10])] += DIR[j["effect_dir"]]

    need = collections.Counter(k[0] for k in sig); fetch = [s for s, _ in need.most_common(a.fetch)]
    print(f"corpus {gd} years {sorted(years)} | {len(sig)} name-day signals, {len(need)} names")
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in fetch}; ok = {s for s, r in px.items() if len(r) > 250}
    print(f"prices: {len(ok)} usable\n")

    def abn(symn, date):
        r = px.get(symn, {}); ds = sorted(r)
        if not ds: return None
        if date not in r:
            nx = [d for d in ds if d >= date]; date = nx[0] if nx else None
        if not date or date not in r or date not in Fmap: return None
        i = ds.index(date)
        if i < 170 or i + 2 >= len(ds): return None
        est = [d for d in ds[i-165:i-11] if d in Fmap]
        if len(est) < 90: return None
        Fc = np.array([Fmap[d] for d in est]); colok = np.where(np.isfinite(Fc).mean(axis=0) >= 0.9)[0]
        if len(colok) == 0: return None
        rk = [k for k in range(len(est)) if np.all(np.isfinite(Fc[k, colok]))]
        if len(rk) < 90: return None
        y = np.array([r[est[k]] for k in rk]); X = np.column_stack([np.ones(len(rk)), Fc[np.ix_(rk, colok)]])
        w = 0.5 ** (np.arange(len(rk))[::-1] / TRADING_DAYS); beta = wls(y, X, w); rstd = (y - X @ beta).std()
        if rstd == 0: return None
        def resid(d):
            if d not in r or d not in Fmap: return None
            f = Fmap[d][colok]
            return (r[d] - (beta[0] + beta[1:] @ f)) if np.all(np.isfinite(f)) else None
        r0 = resid(ds[i]); r1 = resid(ds[i + 1]) if i + 1 < len(ds) else None
        return (r0, r1, rstd)

    # obs: (kind, signal, z0, z1, abn0_pct, abn1_pct)
    rows = []
    for (s, day) in sig:
        if s not in ok: continue
        res = abn(s, day)
        if not res or res[0] is None: continue
        r0, r1, rstd = res
        rows.append((sym_kind.get(s, "etf_proxy"), sig[(s, day)], r0 / rstd, (r1 / rstd if r1 is not None else np.nan), r0, (r1 if r1 is not None else np.nan)))
    print(f"{len(rows)} aligned observations\n")

    for kind in ["security", "etf_proxy", "ALL"]:
        R = [r for r in rows if kind == "ALL" or r[0] == kind]
        if len(R) < 20: print(f"### {kind}: too few ({len(R)})\n"); continue
        sg = np.array([r[1] for r in R], float); z0 = np.array([r[2] for r in R], float)
        z1 = np.array([r[3] for r in R], float); a0 = np.array([r[4] for r in R], float); a1 = np.array([r[5] for r in R], float)
        print(f"### {kind}  ({len(R)} obs) " + "=" * 40)
        # --- attribution value ---
        r0, t0, _ = spearman(sg, z0); r2 = r0 * r0
        big = np.abs(z0) > 2.0
        prec = float(np.mean(np.sign(sg[big]) == np.sign(z0[big]))) if big.sum() else float("nan")
        mag = float(np.mean(np.abs(a0[big]))) * 100 if big.sum() else float("nan")
        # expected signed move in claimed direction (nowcast economic value), all covered events
        dv, dt_, _ = tmean(np.sign(sg) * a0 * 100)
        print(f"  ATTRIBUTION  contemp IC {r0:+.3f} (t {t0:+.1f}) -> explains ~{r2*100:.1f}% of idiosyncratic-move variance")
        print(f"               big-mover precision {prec:.0%} (N={int(big.sum())}); attributed big move averages |{mag:.2f}%| abnormal")
        print(f"               expected SIGNED move in claimed direction: {dv:+.2f}% same-day (t {dt_:+.1f})  <- the nowcast value")
        # --- tradable value (next-day, honest) ---
        fedge, ft, _ = tmean(np.sign(sg) * a1 * 100)          # follow-the-signal next-day, in %
        # pooled market-neutral book: per-event MN return = sign(signal) * next-day abnormal %
        rets = (np.sign(sg) * a1)[np.isfinite(a1)]
        if len(rets) > 20:
            sharpe = float(rets.mean() / (rets.std(ddof=1) + 1e-12) * math.sqrt(TRADING_DAYS))  # if ~1 trade/day
            ann = float(rets.mean() * TRADING_DAYS * 100)
            print(f"  TRADABLE     next-day follow-signal edge {fedge:+.3f}%/event (t {ft:+.1f})")
            print(f"               pooled MN book ~ {ann:+.1f}%/yr, Sharpe {sharpe:+.2f}  (honest forward-alpha bound)")
        print()


if __name__ == "__main__":
    main()
