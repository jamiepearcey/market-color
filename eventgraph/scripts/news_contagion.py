# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
CONTAGION CASE STUDY (H2-2011) — turn the pooled +0.075 credit co-movement excess into
a dated picture. Claim: the news CONTAGION factor is a TRANSIENT common component that a
static factor model misses. We measure it with the ABSORPTION RATIO (Kritzman et al.
systemic-risk measure): the fraction of ABNORMAL-return variance captured by the first
principal component of a name set in a month. Calm month -> names idiosyncratic -> PC1
low (~1/N). Contagion -> all names load on ONE factor -> PC1 spikes.

We compute the monthly absorption ratio of the CONTAGION-news-exposed names and overlay
contagion news intensity; contrast with an EARNINGS-exposed (idiosyncratic) control set
that should NOT spike. If contagion absorption spikes in H2-2011 (EU sovereign crisis +
US downgrade) while earnings' does not, news named a transient co-movement factor in real
time -- in the residual, after the 9 macro factors are removed.

Usage: uv run eventgraph/scripts/news_contagion.py --graph-dir /tmp/eg100k_graph
"""
import argparse, json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_factors import norm_mech

CREDIT = {"credit"}; IDIO = {"earnings"}  # theme buckets to contrast


def ar_series(s, cache, Fmap, years, p1, p2):
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


def absorption(AR, names, month, min_days=12, min_names=5):
    series = {s: {d: v for d, v in AR[s].items() if d[:7] == month} for s in names if s in AR}
    series = {s: v for s, v in series.items() if len(v) >= min_days}
    if len(series) < min_names: return None
    days = sorted(set.intersection(*[set(v) for v in series.values()]))
    if len(days) < min_days: return None
    M = np.array([[series[s][d] for d in days] for s in series])  # names x days
    M = M - M.mean(axis=1, keepdims=True)
    C = np.corrcoef(M)
    if not np.all(np.isfinite(C)): return None
    ev = np.linalg.eigvalsh(C)[::-1]
    return ev[0] / ev.sum(), len(series)  # PC1 variance fraction, n names


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--min-edges", type=int, default=3)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(","))
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

    credit_edges = collections.Counter(); idio_edges = collections.Counter()
    contagion_month = collections.Counter()  # month -> credit-edge count
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        th = norm_mech(j.get("mechanism"))
        if th in CREDIT: credit_edges[sym[e]] += 1; contagion_month[d[:7]] += 1
        elif th in IDIO: idio_edges[sym[e]] += 1
    contagion_set = sorted(s for s, n in credit_edges.items() if n >= a.min_edges)
    control_set = sorted(s for s, n in idio_edges.items() if n >= a.min_edges and credit_edges[s] < a.min_edges)
    print(f"contagion-exposed names (>= {a.min_edges} credit edges): {len(contagion_set)}")
    print(f"  {', '.join(contagion_set[:30])}{' ...' if len(contagion_set)>30 else ''}")
    print(f"earnings-control names: {len(control_set)}\n")

    need = sorted(set(contagion_set) | set(control_set))
    AR = {s: ar_series(s, cache, Fmap, years, p1, p2) for s in need}
    AR = {s: v for s, v in AR.items() if len(v) > 80}
    print(f"{len(AR)} names with abnormal-return series\n")

    months = [f"{y}-{m:02d}" for y in sorted(years) for m in range(1, 13)]
    print(f"  {'month':8} {'contagion_abs':>14} {'(n)':>5} {'earnings_abs':>13} {'(n)':>5} {'credit_news':>12}")
    peak = (0, None)
    for mo in months:
        ca = absorption(AR, contagion_set, mo); ia = absorption(AR, control_set, mo)
        cs = f"{ca[0]:.2f}" if ca else "  -"; cn = ca[1] if ca else 0
        is_ = f"{ia[0]:.2f}" if ia else "  -"; inn = ia[1] if ia else 0
        news = contagion_month.get(mo, 0)
        bar = "#" * int((ca[0] if ca else 0) * 40)
        print(f"  {mo:8} {cs:>14} {cn:5} {is_:>13} {inn:5} {news:12}  {bar}")
        if ca and ca[0] > peak[0]: peak = (ca[0], mo)
    print(f"\n  PEAK contagion absorption: {peak[0]:.2f} in {peak[1]}")

    # summary: H2-2011 vs rest
    h2 = [absorption(AR, contagion_set, f"2011-{m:02d}") for m in range(7, 13)]
    h2 = [x[0] for x in h2 if x]
    rest = [absorption(AR, contagion_set, mo) for mo in months if not mo.startswith("2011-1") and mo not in (f"2011-0{m}" for m in range(7,10))]
    rest = [x[0] for x in rest if x]
    if h2 and rest:
        print(f"  contagion absorption  H2-2011 mean {np.mean(h2):.2f}  vs  other months {np.mean(rest):.2f}")
        ih2 = [absorption(AR, control_set, f"2011-{m:02d}") for m in range(7, 13)]; ih2 = [x[0] for x in ih2 if x]
        if ih2: print(f"  earnings-control absp H2-2011 mean {np.mean(ih2):.2f}  (should NOT spike)")


if __name__ == "__main__":
    main()
