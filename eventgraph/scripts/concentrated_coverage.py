# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Concentrated coverage — recent DENSE news as a test set.

Instead of ingesting years of sparse history (2.3 docs/day -> 1.1% coverage), use a
short, dense recent window (eg_live ~150 docs/day) and measure coverage ONLY inside
that window. This is the cheap proof of what corpus density buys: where the news is
thick, what fraction of a name's big idiosyncratic moves do we actually explain, and
with what directional precision?

Everything is gated to the corpus's own date span [tmin, tmax] so pre-corpus movers
don't deflate the denominator. Abnormal return = residual on the engine's 9 factors
over a trailing window (same machinery as cross_sectional_ic / causal_engine).

Usage: uv run eventgraph/scripts/concentrated_coverage.py --graph-dir /tmp/eg_live
"""
import argparse, json, time, math, collections, csv, bisect, datetime as dt
from pathlib import Path
import httpx, numpy as np

FN = ["crypto", "fx_major", "fx_other", "bonds", "equity_idx", "metals",
      "em_fx_latam", "em_fx_emea", "em_fx_asia"]
DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
SKIP = {"SPY", "IEF", "HYG", "USO", "UUP", "GLD", "EEM", "FXI", "^TNX", "^VIX", "^GSPC", "TLT", "DIA", "QQQ", "ACWI"}


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
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b); a, b = a[m], b[m]
    if len(a) < 8: return (float("nan"), float("nan"), len(a))
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean(); denom = math.sqrt((ra @ ra) * (rb @ rb))
    if denom == 0: return (float("nan"), float("nan"), len(a))
    r = float((ra @ rb) / denom); return (r, r * math.sqrt(max(len(a) - 2, 1) / max(1 - r * r, 1e-12)), len(a))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_live")
    ap.add_argument("--price-from", type=int, default=2025); ap.add_argument("--price-to", type=int, default=2026)
    ap.add_argument("--z-thresh", type=float, default=1.5); ap.add_argument("--fetch", type=int, default=400)
    ap.add_argument("--factors", default="", help="factor_returns.csv (defaults to graph-dir, else /tmp/eg_6k)")
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())

    # The engine's 9-factor panel ends ~2026-06-23, so for a RECENT test window we neutralize
    # against liquid Yahoo factor proxies (available through today) instead of the stale CSV.
    MKT = ["ACWI", "TLT", "UUP", "GLD", "EEM"]   # global equity / bonds / dollar / gold / EM equity
    Fmap = {}  # populated after prices are fetched (below)

    if not (gd / "entity_symbol.jsonl").exists():
        print("entity_symbol.jsonl missing — run resolve_tickers.py on this graph-dir first."); return
    sym = {}; sym_kind = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
            # tag the symbol's kind; a real single-name security wins over a macro ETF proxy
            if sym_kind.get(j["symbol"]) != "security":
                sym_kind[j["symbol"]] = j["kind"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}
    alld = sorted(d[:10] for d in docdate.values() if d)
    tmin, tmax = alld[0], alld[-1]
    print(f"corpus {gd}  window {tmin} -> {tmax}  ({(dt.date.fromisoformat(tmax)-dt.date.fromisoformat(tmin)).days}d, {len(docdate)} docs)")

    sig = collections.Counter(); sym_edges = collections.defaultdict(list)
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or e not in sym or sym[e] in SKIP or j.get("effect_dir") not in DIR: continue
        s = sym[e]; day = d[:10]
        sym_edges[s].append(day)
        if tmin <= day <= tmax: sig[(s, day)] += DIR[j["effect_dir"]]
    for s in sym_edges: sym_edges[s].sort()

    need = collections.Counter(k[0] for k in sig)
    fetch = [s for s, _ in need.most_common(a.fetch)]
    print(f"{len(sig)} name-day signals over {len(need)} resolved names; fetching {len(fetch)} recent price series ...")
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in fetch}
    # market factor proxies (Yahoo, current)
    mkt = {m: logret(yahoo(m, cache, p1, p2)) for m in MKT}
    fdays = sorted(set().union(*[set(mkt[m]) for m in MKT]))
    for d in fdays:
        Fmap[d] = np.array([mkt[m].get(d, np.nan) for m in MKT])
    mcov = {m: sum(1 for d in fdays if d in mkt[m]) for m in MKT}
    ok = {s for s, r in px.items() if len(r) > 120}
    print(f"  {len(ok)} names with usable recent price history; market factors {MKT} (days {len(fdays)})\n")

    def resid_z_series(s):
        """full-window residual z per day for a name, regressed on the market factor proxies."""
        r = px.get(s, {}); ds = [d for d in sorted(r) if d in Fmap]
        if len(ds) < 100: return None
        F = np.array([Fmap[d] for d in ds]); colok = np.where(np.isfinite(F).mean(axis=0) >= 0.9)[0]
        rk = [k for k in range(len(ds)) if np.all(np.isfinite(F[k, colok]))]
        if len(rk) < 100: return None
        dd = [ds[k] for k in rk]; y = np.array([r[d] for d in dd])
        X = np.column_stack([np.ones(len(dd)), F[np.ix_(rk, colok)]])
        beta = np.linalg.lstsq(X, y, rcond=None)[0]; resid = y - X @ beta; rstd = resid.std()
        if rstd == 0: return None
        return {dd[k]: resid[k] / rstd for k in range(len(dd))}

    zser = {s: resid_z_series(s) for s in ok}
    zser = {s: z for s, z in zser.items() if z}

    # coverage + IC, split by instrument kind; securities further split US-listed vs foreign
    # (foreign local listings misalign with US-session news timing + the US-proxy factor model)
    KINDS = ["security_us", "security_fx", "etf_proxy", "ALL"]
    cov = {k: [0, 0] for k in KINDS}          # [explained, total]
    prec = {k: [0, 0] for k in KINDS}         # [sign-ok, same-day-directional]
    icsig = {k: [] for k in KINDS}; icz0 = {k: [] for k in KINDS}; icz1 = {k: [] for k in KINDS}
    percov = collections.Counter(); pertot = collections.Counter()
    def kind_of(s):
        k = sym_kind.get(s, "etf_proxy")
        if k == "security": return "security_us" if "." not in s else "security_fx"
        return "etf_proxy"
    for s, z in zser.items():
        kind = kind_of(s); tags = [kind, "ALL"]
        days = sorted(d for d in z if tmin <= d <= tmax); edates = sym_edges[s]; zk = sorted(z)
        for d in days:
            if abs(z[d]) <= a.z_thresh: continue
            pertot[s] += 1
            lo = (dt.date.fromisoformat(d) - dt.timedelta(days=2)).isoformat()
            hit = bisect.bisect_right(edates, d) - bisect.bisect_left(edates, lo) > 0
            if hit: percov[s] += 1
            nd = sig.get((s, d), 0)
            for t in tags:
                cov[t][1] += 1; cov[t][0] += 1 if hit else 0
                if nd != 0:
                    prec[t][1] += 1; prec[t][0] += 1 if (nd > 0) == (z[d] > 0) else 0
        for (ss, dd_), v in sig.items():
            if ss != s or dd_ not in z: continue
            i = bisect.bisect_left(zk, dd_); z1 = z[zk[i + 1]] if i + 1 < len(zk) else np.nan
            for t in tags:
                icsig[t].append(v); icz0[t].append(z[dd_]); icz1[t].append(z1)

    print(f"=== CONCENTRATED COVERAGE (|z|>{a.z_thresh}, within {tmin}..{tmax}), split by instrument kind ===")
    print(f"  {'kind':10} {'coverage':>14} {'same-day precision':>20} {'contemp IC':>14} {'lag+1 IC':>12}")
    for k in KINDS:
        c, ct = cov[k]; p, pt = prec[k]
        r0, t0, n0 = spearman(icsig[k], icz0[k]); r1, t1, _ = spearman(icsig[k], np.array(icz1[k], float))
        covs = f"{c}/{ct} ({c/ct:.0%})" if ct else "-"
        precs = f"{p}/{pt} ({p/max(pt,1):.0%})" if pt else "-"
        print(f"  {k:10} {covs:>14} {precs:>20} {r0:>+8.3f} (t{t0:>+4.1f}) {r1:>+7.3f} (t{t1:>+4.1f})")
    print(f"\n=== best-covered names (news explains their moves) ===")
    for s in sorted(percov, key=lambda x: -percov[x])[:14]:
        print(f"  {s:8} [{sym_kind.get(s,'?'):9}] {percov[s]}/{pertot[s]} big moves have news")


if __name__ == "__main__":
    main()
