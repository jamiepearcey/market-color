# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
NEWS FACTORS — a set-of-factors model built from the graph, tested CONTEMPORANEOUSLY
(not as a price predictor; news explains same-day, doesn't forecast -- established).

Reframe: news is idiosyncratic pooled into one macro factor (the aggregate null), but
a SET of THEMATIC news factors captures transient, event-driven co-movement that static
factor models (fixed market/size/value loadings) miss -- a "sovereign contagion" factor
binding banks in 2011, a "supply shock" factor across energy, etc. We residualize the 9
macro factors first (abnormal return), then ask what the news themes explain ON TOP.

For each theme k we build a factor-mimicking daily return: long the positive-news names,
short the negative-news names, equal weight, and read that day's mean ABNORMAL return in
the news direction. Series over days -> the theme's contemporaneous premium (mean, t). The
inter-theme correlation of these return series says whether they are DISTINCT factors.

Prediction-market carry-over: CONVICTION (net dir / flow = consensus strength, like a
market price) as a loading, and DISPERSION (opposing mass) as a non-directional UNCERTAINTY
factor tested against realized |z| (does conflicting news explain volatility?).

Usage: uv run eventgraph/scripts/news_factors.py --graph-dir /tmp/eg100k_graph --years 2010,2011,2012
"""
import argparse, json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP

# mechanism -> economic THEME (the factor set). Unmapped -> excluded ('other').
THEME = {
    "earnings": "earnings", "guidance": "earnings", "forecast": "earnings", "valuation": "earnings",
    "monetary_policy": "monetary", "rate_decision": "monetary",
    "fiscal_policy": "fiscal", "debt_auction": "fiscal", "budget": "fiscal", "election": "fiscal",
    "geopolitics": "geopolitics",
    "regulation": "regulation", "rating_action": "regulation", "rating_review": "regulation", "fraud": "regulation",
    "mergers_acquisitions": "m_and_a", "ipo": "m_and_a", "investment": "m_and_a",
    "demand_change": "demand_supply", "supply_shock": "demand_supply", "supply_change": "demand_supply",
    "growth": "demand_supply", "weather": "demand_supply", "weather_event": "demand_supply",
    "contagion": "credit", "default": "credit", "fund_flows": "credit",
    "competition": "product",
    "appointment": "management", "officer_of": "management", "leadership": "management",
    "data_surprise": "macro_data", "data_release": "macro_data",
}


def norm_mech(m):
    return THEME.get((m or "").strip().lower().replace(" ", "_").replace("-", "_"), "other")


def tstat(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    if len(x) < 5 or x.std() == 0: return (float(x.mean()) if len(x) else float("nan"), float("nan"), len(x))
    return float(x.mean()), float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))), len(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--min-names", type=int, default=3)
    ap.add_argument("--price-from", type=int, default=2008); ap.add_argument("--price-to", type=int, default=2014)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(","))
    p1 = int(dt.datetime(a.price_from, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(a.price_to, 12, 31, tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = np.array([float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    # per (symbol, day): theme -> net dir; plus aggregate flow (count) & up/dn mass
    theme_dir = collections.defaultdict(lambda: collections.defaultdict(float))  # (s,day) -> {theme: net}
    flow = collections.Counter(); up = collections.Counter(); dn = collections.Counter()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        s = sym[e]; day = d[:10]; sd = DIR[j["effect_dir"]]; th = norm_mech(j.get("mechanism"))
        theme_dir[(s, day)][th] += sd
        flow[(s, day)] += 1; up[(s, day)] += (sd > 0); dn[(s, day)] += (sd < 0)
    cells = list(theme_dir)
    print(f"{len(cells)} (name,day) news cells over {len(set(k[0] for k in cells))} US names, {years}")

    # abnormal return + z for each news cell (rolling EWMA-WLS on the 9 macro factors)
    def abn(s, day, pxr):
        r = pxr.get(s); ds = sorted(r) if r else []
        if day not in r or day not in Fmap: return None
        i = ds.index(day)
        if i < 170 or i + 1 >= len(ds): return None
        est = [d for d in ds[i-165:i-11] if d in Fmap]
        if len(est) < 90: return None
        Fc = np.array([Fmap[d] for d in est]); colok = np.where(np.isfinite(Fc).mean(axis=0) >= 0.9)[0]
        if len(colok) == 0: return None
        rows = [k for k in range(len(est)) if np.all(np.isfinite(Fc[k, colok]))]
        if len(rows) < 90: return None
        y = np.array([r[est[k]] for k in rows]); X = np.column_stack([np.ones(len(rows)), Fc[np.ix_(rows, colok)]])
        w = 0.5 ** (np.arange(len(rows))[::-1] / 252); beta = wls(y, X, w); rstd = (y - X @ beta).std()
        if rstd == 0: return None
        f = Fmap[day][colok]
        if not np.all(np.isfinite(f)): return None
        ar = r[day] - (beta[0] + beta[1:] @ f)
        return ar, ar / rstd

    names = sorted(set(k[0] for k in cells))
    px = {s: logret(yahoo(s, cache, p1, p2)) for s in names}
    px = {s: r for s, r in px.items() if len(r) > 250}
    ab = {}
    for (s, day) in cells:
        if s in px:
            res = abn(s, day, px)
            if res: ab[(s, day)] = res
    print(f"{len(ab)} cells with a clean abnormal return\n")

    THEMES = sorted(set(THEME.values()))
    # A) per-theme contemporaneous premium = mean(sign(net_dir) * z), and factor-mimicking daily return
    print("=== A. news-theme factors: contemporaneous pricing ===")
    print(f"  {'theme':14} {'premium(dir*z)':>14} {'t':>7} {'N':>6} | {'factor ret/day':>14} {'t':>7} {'days':>5} {'hit':>5}")
    day_ret = collections.defaultdict(lambda: collections.defaultdict(list))  # theme -> day -> [dir*ar]
    prem = collections.defaultdict(list)
    for (s, day), td in theme_dir.items():
        if (s, day) not in ab: continue
        ar, z = ab[(s, day)]
        for th, net in td.items():
            if net == 0: continue
            prem[th].append(np.sign(net) * z)
            day_ret[th][day].append(np.sign(net) * ar)
    factor_series = {}
    for th in THEMES:
        pm, pt, pn = tstat(prem[th])
        drets = {d: np.mean(v) for d, v in day_ret[th].items() if len(v) >= a.min_names}
        factor_series[th] = drets
        fr = list(drets.values())
        fm, ft, fn = tstat(fr)
        hit = np.mean([v > 0 for v in fr]) if fr else float("nan")
        star = "**" if abs(pt) > 2.58 else ("*" if abs(pt) > 1.96 else "")
        print(f"  {th:14} {pm:+14.3f} {pt:+7.1f}{star:2} {pn:6} | {fm*1e4:+12.1f}bp {ft:+7.1f} {fn:5} {hit:5.0%}")

    # B) inter-factor correlation on common days -> distinct dimensions?
    print("\n=== B. inter-factor correlation (are these DISTINCT co-movement dimensions?) ===")
    common = sorted(set.intersection(*[set(factor_series[th]) for th in THEMES if factor_series[th]]) )
    big = [th for th in THEMES if len(factor_series[th]) >= 40]
    if len(big) >= 2:
        alldays = sorted(set().union(*[set(factor_series[th]) for th in big]))
        M = np.array([[factor_series[th].get(d, np.nan) for th in big] for d in alldays])
        print("     " + " ".join(f"{th[:5]:>6}" for th in big))
        for i, th in enumerate(big):
            cors = []
            for j in range(len(big)):
                m = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
                cors.append(np.corrcoef(M[m, i], M[m, j])[0, 1] if m.sum() > 20 else float("nan"))
            print(f"  {th[:5]:>5} " + " ".join(f"{c:+6.2f}" for c in cors))
        offdiag = [np.corrcoef(M[np.isfinite(M[:, i]) & np.isfinite(M[:, j]), i], M[np.isfinite(M[:, i]) & np.isfinite(M[:, j]), j])[0, 1]
                   for i in range(len(big)) for j in range(i + 1, len(big)) if (np.isfinite(M[:, i]) & np.isfinite(M[:, j])).sum() > 20]
        print(f"  mean |off-diagonal correlation| = {np.nanmean(np.abs(offdiag)):.2f}  (low = distinct factors)")

    # C) ATTENTION/DISPERSION (prediction-market carry-over): non-directional news -> realized vol
    print("\n=== C. non-directional factors (news as attention / uncertainty) ===")
    lf, az, disp = [], [], []
    for (s, day), fl in flow.items():
        if (s, day) not in ab: continue
        _, z = ab[(s, day)]
        lf.append(math.log(fl)); az.append(abs(z))
        n = up[(s, day)] + dn[(s, day)]; disp.append(2 * min(up[(s, day)], dn[(s, day)]) / n if n else 0.0)
    az = np.array(az)
    if len(az) > 50:
        rflow = np.corrcoef(lf, az)[0, 1]
        rdisp = np.corrcoef(disp, az)[0, 1]
        print(f"  corr(log news-flow, |abnormal z|)   = {rflow:+.3f}   [attention -> bigger moves]")
        print(f"  corr(narrative dispersion, |abn z|) = {rdisp:+.3f}   [conflicting news -> bigger moves]")
        print(f"  mean |z| all news cells {az.mean():.2f} vs high-flow(>=4) {np.array([az[k] for k in range(len(az)) if lf[k]>=math.log(4)]).mean():.2f}")


if __name__ == "__main__":
    main()
