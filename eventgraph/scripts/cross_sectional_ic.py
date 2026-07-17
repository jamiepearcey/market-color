# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
Cross-sectional Information Coefficient + attribution coverage — the decisive test
of the reframe.

The aggregate news-conditioned model looked null because the market portfolio
DIVERSIFIES AWAY idiosyncratic news by construction. News is a cross-sectional,
name-level force. So instead of pooling news into 9 factors and reading the net,
we ask, at the (event -> name) level:

  IC   — does the graph's per-name directional signal rank-correlate with each
         name's ABNORMAL (factor-neutral) return? contemporaneous (t0) and
         lagged (t+1, the honest forecast test).
  hit  — sign agreement between signal and abnormal move.
  coverage/precision — of the big movers (|abnormal z|>2), how many does the
         graph explain, and when it attributes, is the sign right?
  per-mechanism name-level reliability — mean(dir * abnormal z), the CONDITIONAL
         S_dir we averaged away in the pooled model.

Abnormal return = residual of the name's return regressed (EWMA-WLS) on the
engine's 9 factors over a trailing window (same machinery as causal_engine.py).

Usage: uv run eventgraph/scripts/cross_sectional_ic.py --graph-dir /tmp/eg_6k --years 2010,2011,2012,2013
"""
import argparse, json, time, math, collections, csv, datetime as dt
from pathlib import Path
import httpx, numpy as np

FN = ["crypto", "fx_major", "fx_other", "bonds", "equity_idx", "metals",
      "em_fx_latam", "em_fx_emea", "em_fx_asia"]
DIR = {"up": 1, "down": -1, "widen": -1, "tighten": 1}
# pooled directional reliability (for the weighted-signal variant only)
S_DIR = {"earnings": 0.72, "supply_shock": 0.29, "other": 0.22, "rating_action": 0.21, "guidance": 0.20,
         "demand_change": 0.17, "monetary_policy": 0.15, "rate_decision": 0.10, "regulation": 0.08,
         "contagion": 0.04, "geopolitics": 0.03, "mergers_acquisitions": -0.02, "default": -0.11, "data_surprise": -0.39}
# macro/factor proxies — exclude as "names" so the signal can't trivially predict its own factor
SKIP = {"SPY", "IEF", "HYG", "USO", "UUP", "GLD", "EEM", "FXI", "^TNX", "^VIX", "^GSPC", "TLT", "DIA", "QQQ"}


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


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 8: return (float("nan"), float("nan"), len(a))
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = math.sqrt((ra @ ra) * (rb @ rb))
    if denom == 0: return (float("nan"), float("nan"), len(a))
    r = float((ra @ rb) / denom)
    t = r * math.sqrt(max(len(a) - 2, 1) / max(1 - r * r, 1e-12))
    return (r, t, len(a))


def tstat(v):
    v = np.asarray(v, float)
    if len(v) < 5: return (float("nan"), float("nan"), len(v))
    return (float(v.mean()), float(v.mean() / (v.std(ddof=1) / math.sqrt(len(v)) + 1e-12)), len(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--years", default="2010,2011,2012,2013")
    ap.add_argument("--price-from", type=int, default=2008); ap.add_argument("--price-to", type=int, default=2014)
    ap.add_argument("--fetch", type=int, default=600)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(",")); p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; ds = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        Fmap[ds] = np.array([float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l); us = j.get("source", "").endswith(("us", "exact", "fuzzy"))
        if ((j["kind"] == "security" and us) or j["kind"] == "etf_proxy") and not j["symbol"].startswith("^"):
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    # edges -> per (symbol, date) signal, plus per-edge records for the mechanism breakdown
    sig_raw = collections.Counter()     # net direction count
    sig_wtd = collections.Counter()     # S_dir-weighted
    edge_recs = collections.defaultdict(list)  # (sym,date) -> [(dir, mech)]
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or sym[e] in SKIP or j.get("effect_dir") not in DIR: continue
        s = sym[e]; day = d[:10]; mech = j.get("mechanism") or "other"
        sig_raw[(s, day)] += DIR[j["effect_dir"]]
        sig_wtd[(s, day)] += DIR[j["effect_dir"]] * S_DIR.get(mech, 0.05)
        edge_recs[(s, day)].append((DIR[j["effect_dir"]], mech))

    need = collections.Counter(k[0] for k in sig_raw)
    fetch = [s for s, _ in need.most_common(a.fetch)]
    print(f"corpus {a.graph_dir}  years {sorted(years)}  |  {len(sig_raw)} name-day signals over {len(need)} names")
    print(f"prices: fetching/loading {len(fetch)} names ...")
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in fetch}
    ok = {s for s, r in px.items() if len(r) > 250}
    print(f"  {len(ok)} names with usable price history")

    def abn(symn, date):
        """single-day abnormal-return z on the event day (z0) and next day (z1)."""
        r = px.get(symn, {}); ds = sorted(r)
        if not ds: return None
        if date not in r:
            nx = [d for d in ds if d >= date]; date = nx[0] if nx else None
        if not date or date not in r or date not in Fmap: return None
        i = ds.index(date)
        if i < 170 or i + 2 >= len(ds): return None
        est = [d for d in ds[i-165:i-11] if d in Fmap]
        if len(est) < 90: return None
        Fc = np.array([Fmap[d] for d in est])
        colok = np.where(np.isfinite(Fc).mean(axis=0) >= 0.9)[0]
        if len(colok) == 0: return None
        rows = [k for k in range(len(est)) if np.all(np.isfinite(Fc[k, colok]))]
        if len(rows) < 90: return None
        y = np.array([r[est[k]] for k in rows]); Xf = Fc[np.ix_(rows, colok)]
        X = np.column_stack([np.ones(len(rows)), Xf]); age = np.arange(len(rows))[::-1]; w = 0.5 ** (age / 252)
        beta = wls(y, X, w); rstd = (y - X @ beta).std()
        if rstd == 0: return None
        def z(d):
            if d not in r or d not in Fmap: return None
            f = Fmap[d][colok]
            if not np.all(np.isfinite(f)): return None
            return (r[d] - (beta[0] + beta[1:] @ f)) / rstd
        z0 = z(ds[i]); z1 = z(ds[i+1]) if i + 1 < len(ds) else None
        return z0, z1

    # assemble aligned observations
    rows = []  # (day, sym, sraw, swtd, z0, z1)
    for (s, day) in sig_raw:
        if s not in ok: continue
        res = abn(s, day)
        if not res: continue
        z0, z1 = res
        if z0 is None: continue
        rows.append((day, s, sig_raw[(s, day)], sig_wtd[(s, day)], z0, z1))
    print(f"  {len(rows)} aligned (name, day) observations\n")
    if len(rows) < 30:
        print("too few observations"); return

    sraw = np.array([r[2] for r in rows], float); swtd = np.array([r[3] for r in rows], float)
    z0 = np.array([r[4] for r in rows], float); z1 = np.array([r[5] if r[5] is not None else np.nan for r in rows], float)

    print("=== NAME-LEVEL cross-sectional alignment (pooled) ===")
    for label, sig in [("signal_raw (net direction)", sraw), ("signal_wtd (x S_dir)", swtd)]:
        r0, t0, n0 = spearman(sig, z0)
        m1 = np.isfinite(z1); r1, t1, n1 = spearman(sig[m1], z1[m1])
        hit0 = float(np.mean(np.sign(sig) == np.sign(z0)))
        hit1 = float(np.mean(np.sign(sig[m1]) == np.sign(z1[m1])))
        print(f"  {label:26}  contemp IC {r0:+.3f} (t {t0:+.1f}, hit {hit0:.0%})   lag+1 IC {r1:+.3f} (t {t1:+.1f}, hit {hit1:.0%})")

    # directional abnormal return in the CLAIMED direction (name-level S_dir)
    print("\n=== per-mechanism NAME-LEVEL reliability  mean(dir * abnormal z), contemporaneous ===")
    permech = collections.defaultdict(list)
    for r in rows:
        for (d, mech) in edge_recs[(r[1], r[0])]:
            permech[mech].append(d * r[4])  # dir * z0
    print(f"  {'mechanism':16} {'mean':>7} {'t':>6} {'N':>5}   (pooled S_dir in parens)")
    for mech in sorted(permech, key=lambda m: -tstat(permech[m])[1] if len(permech[m]) >= 5 else 99):
        m, t, n = tstat(permech[mech])
        if n < 8: continue
        star = "**" if abs(t) > 2.58 else ("*" if abs(t) > 1.96 else "")
        print(f"  {mech:16} {m:>+7.3f} {t:>+6.2f}{star:2} {n:>5}   (pooled {S_DIR.get(mech, 0):+.2f})")

    # attribution coverage/precision on big movers
    print("\n=== attribution coverage on big idiosyncratic movers (|abnormal z| > 2) ===")
    big = [r for r in rows if abs(r[4]) > 2.0]
    if big:
        prec = float(np.mean([np.sign(r[2]) == np.sign(r[4]) for r in big]))
        print(f"  {len(big)} big-mover name-days have a graph edge; sign-precision (edge dir == move dir) = {prec:.0%}")
        print(f"  (coverage vs ALL big movers needs the full price panel — this is precision on the covered set)")

    # cross-sectional IC per day where the cross-section is wide enough -> IR
    byday = collections.defaultdict(list)
    for r in rows: byday[r[0]].append(r)
    ics = []
    for day, rs in byday.items():
        if len(rs) < 8: continue
        r_, t_, n_ = spearman([x[2] for x in rs], [x[4] for x in rs])
        if not math.isnan(r_): ics.append(r_)
    if len(ics) >= 5:
        ics = np.array(ics); ir = ics.mean() / (ics.std(ddof=1) / math.sqrt(len(ics)) + 1e-12)
        print(f"\n=== daily cross-sectional IC (days with >=8 names) ===")
        print(f"  mean daily IC {ics.mean():+.3f}  over {len(ics)} days  ->  IR {ir:+.2f}  (breadth is thin in this corpus)")
    else:
        print(f"\n  (too few wide-cross-section days for a daily-IC/IR estimate — pooled IC above is the read)")


if __name__ == "__main__":
    main()
