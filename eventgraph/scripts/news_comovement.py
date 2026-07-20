# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
NEWS FACTORS as CO-MOVEMENT (the modern multi-factor payoff): a news factor isn't
just a directional premium -- it's a COMMON source of covariance. Static models
(market/size/value, our 9 macro) have FIXED loadings and miss transient, event-driven
co-movement: European banks cratering together on sovereign contagion in H2-2011,
energy names on a supply shock. We residualize the 9 macro factors first (abnormal
return), then ask: do names sharing a news THEME co-move in ABNORMAL returns MORE than
random pairs? If yes, news defines residual covariance = a factor the static model missed.
This is where CONTAGION (null on direction, because a systemic hit's per-name sign is
ambiguous) should finally show up -- as CORRELATION, not direction.

A. Weekly theme factor returns (beats daily sparsity) + full inter-factor correlation.
B. Co-exposure co-movement: mean pairwise abnormal-return correlation among names sharing
   a theme in a month, vs a random-pair baseline (macro already removed -> baseline ~0).

Usage: uv run eventgraph/scripts/news_comovement.py --graph-dir /tmp/eg100k_graph --years 2010,2011,2012
"""
import argparse, json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_factors import THEME, norm_mech


def isoweek(day): y, w, _ = dt.date.fromisoformat(day).isocalendar(); return f"{y}-W{w:02d}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--min-common", type=int, default=12)
    ap.add_argument("--max-pairs", type=int, default=4000)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(",")); rng = np.random.RandomState(0)
    p1 = int(dt.datetime(2008, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(2014, 12, 31, tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = np.array([float(row[f]) if row[f] not in ("", "NaN", "nan") else np.nan for f in FN])
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    # theme exposure: (symbol, month) -> {theme: net_dir}, (symbol, week) -> {theme: net}
    monthexp = collections.defaultdict(lambda: collections.defaultdict(float))
    weekexp = collections.defaultdict(lambda: collections.defaultdict(float))
    newsnames = set()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        s = sym[e]; day = d[:10]; sd = DIR[j["effect_dir"]]; th = norm_mech(j.get("mechanism"))
        if th == "other": continue
        monthexp[(s, day[:7])][th] += sd; weekexp[(s, isoweek(day))][th] += sd; newsnames.add(s)
    print(f"{len(newsnames)} US names carry news; building abnormal-return series ...")

    # full daily abnormal-return series per news name (macro-residualized)
    def ar_series(s):
        r = logret(yahoo(s, cache, p1, p2))
        if len(r) < 300: return {}
        ds = sorted(r); out = {}
        for i in range(len(ds)):
            day = ds[i]
            if day[:4] not in years or i < 170 or day not in Fmap: continue
            est = [d for d in ds[i-165:i-11] if d in Fmap]
            if len(est) < 90: continue
            Fc = np.array([Fmap[d] for d in est]); colok = np.where(np.isfinite(Fc).mean(axis=0) >= 0.9)[0]
            if len(colok) == 0: continue
            rows = [k for k in range(len(est)) if np.all(np.isfinite(Fc[k, colok]))]
            if len(rows) < 90: continue
            y = np.array([r[est[k]] for k in rows]); X = np.column_stack([np.ones(len(rows)), Fc[np.ix_(rows, colok)]])
            w = 0.5 ** (np.arange(len(rows))[::-1] / 252); beta = wls(y, X, w)
            f = Fmap[day][colok]
            if np.all(np.isfinite(f)): out[day] = r[day] - (beta[0] + beta[1:] @ f)
        return out
    AR = {s: ar_series(s) for s in sorted(newsnames)}
    AR = {s: v for s, v in AR.items() if len(v) > 100}
    print(f"{len(AR)} names with abnormal-return series\n")

    THEMES = sorted(set(THEME.values()))
    # A) WEEKLY theme factor returns + inter-factor correlation
    print("=== A. weekly news-theme factor returns (long +news / short -news, abnormal) ===")
    wk_fac = collections.defaultdict(dict)  # theme -> week -> factor return
    for (s, wk), td in weekexp.items():
        if s not in AR: continue
        war = sum(v for d, v in AR[s].items() if isoweek(d) == wk)
        for th, net in td.items():
            if net == 0: continue
            wk_fac[th].setdefault(wk, []).append(np.sign(net) * war)
    series = {}
    print(f"  {'theme':14} {'ret/wk':>10} {'t':>6} {'weeks':>6}")
    for th in THEMES:
        s2 = {wk: np.mean(v) for wk, v in wk_fac[th].items() if len(v) >= 3}
        series[th] = s2; arr = np.array(list(s2.values()))
        if len(arr) >= 5:
            t = arr.mean() / (arr.std(ddof=1) / np.sqrt(len(arr)))
            print(f"  {th:14} {arr.mean()*1e4:+8.0f}bp {t:+6.1f} {len(arr):6}")
    big = [th for th in THEMES if len(series[th]) >= 30]
    print(f"\n  inter-factor correlation ({len(big)} themes with >=30 weeks):")
    allw = sorted(set().union(*[set(series[th]) for th in big]))
    M = np.array([[series[th].get(w, np.nan) for th in big] for w in allw])
    print("        " + " ".join(f"{th[:6]:>7}" for th in big))
    offd = []
    for i, th in enumerate(big):
        row = []
        for j in range(len(big)):
            m = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
            c = np.corrcoef(M[m, i], M[m, j])[0, 1] if m.sum() > 15 else np.nan
            row.append(c)
            if i < j and np.isfinite(c): offd.append(abs(c))
        print(f"  {th[:7]:>7} " + " ".join(f"{c:+7.2f}" for c in row))
    print(f"  mean |off-diagonal| = {np.mean(offd):.2f}  (low => distinct factors)")

    # B) CO-EXPOSURE CO-MOVEMENT: pairwise abnormal-return correlation, exposed vs random
    print("\n=== B. co-exposure co-movement (abnormal-return corr: shared theme vs random) ===")
    def pair_corr(s1, s2, month):
        d1 = {d: v for d, v in AR[s1].items() if d[:7] == month}; d2 = AR[s2]
        common = [d for d in d1 if d in d2]
        if len(common) < a.min_common: return None
        x = np.array([d1[d] for d in common]); y = np.array([d2[d] for d in common])
        if x.std() == 0 or y.std() == 0: return None
        return np.corrcoef(x, y)[0, 1]
    # exposed pairs per theme
    theme_month_names = collections.defaultdict(lambda: collections.defaultdict(set))
    for (s, mo), td in monthexp.items():
        if s not in AR: continue
        for th in td: theme_month_names[th][mo].add(s)
    print(f"  {'theme':14} {'exposed corr':>13} {'n_pairs':>8} {'baseline':>9} {'EXCESS':>8}")
    allnames = sorted(AR)
    for th in THEMES:
        exc = []
        for mo, ns in theme_month_names[th].items():
            ns = sorted(ns)
            if len(ns) < 3: continue
            pairs = [(ns[i], ns[j]) for i in range(len(ns)) for j in range(i + 1, len(ns))]
            if len(pairs) > 60: pairs = [pairs[k] for k in rng.choice(len(pairs), 60, replace=False)]
            for s1, s2 in pairs:
                c = pair_corr(s1, s2, mo)
                if c is not None: exc.append(c)
                if len(exc) >= a.max_pairs: break
            if len(exc) >= a.max_pairs: break
        # matched random baseline: same months, random name pairs
        base = []
        months = [mo for mo, ns in theme_month_names[th].items() if len(ns) >= 3]
        for _ in range(min(len(exc), a.max_pairs)):
            mo = months[rng.randint(len(months))] if months else None
            if not mo: break
            s1, s2 = allnames[rng.randint(len(allnames))], allnames[rng.randint(len(allnames))]
            if s1 == s2: continue
            c = pair_corr(s1, s2, mo)
            if c is not None: base.append(c)
        if len(exc) >= 30:
            em, bm = np.mean(exc), (np.mean(base) if base else float("nan"))
            print(f"  {th:14} {em:+13.3f} {len(exc):8} {bm:+9.3f} {em-bm:+8.3f}")


if __name__ == "__main__":
    main()
