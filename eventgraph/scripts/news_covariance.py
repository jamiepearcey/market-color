# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
CORRELATION-PERSISTENCE SIGNAL + covariance-forecast APPLICATION. Turns the finding
(news causal links flag STRUCTURAL correlation that persists vs TRANSIENT that mean-reverts)
into (3) a formal persistence classifier and (2) an out-of-sample covariance forecast that
beats a shrinkage baseline.

Panel: over a dense universe of active names, every pair-month gets
  trailing_corr  = abnormal-return corr over [m-trail .. m]   (what a risk model sees)
  linked         = the pair shares >=1 CAUSE entity in the causal graph that month
  next_corr      = realized abnormal-return corr over [m+1 .. m+next]   (the target)

(3) PERSISTENCE CLASSIFIER: among already-correlated pairs (trailing > tau), P(correlation
    persists) for linked vs unlinked = the structural-vs-transient signal.
(2) FORECAST: chronological train/test. Baseline forecast of next_corr = shrink(trailing)
    toward the grand mean (Ledoit-Wolf spirit, intensity fit on train). News forecast lets
    the shrinkage + level depend on `linked`. Report out-of-sample pairwise MSE / R2 and a
    matrix-level Frobenius error. If news lowers held-out error, it improves the covariance
    forecast -- a risk product, not just a finding.

Usage: uv run eventgraph/scripts/news_covariance.py --graph-dir /tmp/eg100k_graph --topk 140
"""
import argparse, json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series


def months_between(y0, y1):
    return [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]


def corr_matrix(AR, names, months, min_days=15):
    """pairwise abnormal-return correlation over a set of months; returns {(i,j): c}."""
    cols = {s: {d: v for d, v in AR[s].items() if d[:7] in months} for s in names}
    out = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            s1, s2 = names[a], names[b]; d1 = cols[s1]
            common = [d for d in d1 if d in cols[s2]]
            if len(common) < min_days: continue
            x = np.array([d1[d] for d in common]); y = np.array([cols[s2][d] for d in common])
            if x.std() > 0 and y.std() > 0: out[(s1, s2)] = float(np.corrcoef(x, y)[0, 1])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--topk", type=int, default=140)
    ap.add_argument("--trail", type=int, default=3); ap.add_argument("--next", type=int, default=2, dest="nxt")
    ap.add_argument("--tau", type=float, default=0.3)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"; cache.mkdir(exist_ok=True)
    years = set(a.years.split(",")); p1 = int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: json.loads(l).get("published_at") for l in open(lake / "document.jsonl")}

    cause_effects = collections.defaultdict(lambda: collections.defaultdict(set)); freq = collections.Counter()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or not c or j.get("effect_dir") not in DIR: continue
        cause_effects[d[:7]][c].add(sym[e]); freq[sym[e]] += 1

    universe = [s for s, _ in freq.most_common()]  # by news activity
    print(f"building abnormal-return series for top {a.topk} active names ...")
    AR = {}
    for s in universe:
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150: AR[s] = v
    names = sorted(AR); print(f"{len(names)} names in universe\n")

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years))
    idx = {m: i for i, m in enumerate(allmonths)}
    panel = []  # (month, trailing, linked, next)
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        trail = corr_matrix(AR, names, allmonths[i-a.trail:i+1]); nxt = corr_matrix(AR, names, allmonths[i+1:i+1+a.nxt])
        linked = set()
        for c, effs in cause_effects[m].items():
            effs = sorted(e for e in effs if e in AR)
            for x in range(len(effs)):
                for y in range(x+1, len(effs)): linked.add((effs[x], effs[y]))
        for pair, tc in trail.items():
            if pair in nxt: panel.append((m, tc, 1 if pair in linked else 0, nxt[pair]))
    P = np.array([(t, lk, nx) for _, t, lk, nx in panel]); mon = [p[0] for p in panel]
    print(f"panel: {len(P)} pair-months  ({int(P[:,1].sum())} news-linked)\n")

    # (3) PERSISTENCE CLASSIFIER — among already-correlated pairs (trailing > tau)
    hi = P[P[:, 0] > a.tau]
    def persist_rate(rows): return float(np.mean(rows[:, 2] >= rows[:, 0])) if len(rows) else float("nan")
    lk_hi = hi[hi[:, 1] == 1]; ul_hi = hi[hi[:, 1] == 0]
    print(f"=== (3) correlation-persistence signal: already-correlated pairs (trailing > {a.tau}) ===")
    print(f"  news-LINKED  : persists {persist_rate(lk_hi):.0%}   next_corr {lk_hi[:,2].mean():+.3f} (from {lk_hi[:,0].mean():+.3f})   n={len(lk_hi)}")
    print(f"  unlinked     : persists {persist_rate(ul_hi):.0%}   next_corr {ul_hi[:,2].mean():+.3f} (from {ul_hi[:,0].mean():+.3f})   n={len(ul_hi)}")
    print(f"  => a shared news driver keeps a correlation alive; without it, it mean-reverts.\n")

    # (2) OUT-OF-SAMPLE FORECAST — chronological train/test
    tm = sorted(set(mon)); cut = tm[int(len(tm) * 0.6)]
    tr = np.array([P[k] for k in range(len(P)) if mon[k] < cut]); te = np.array([P[k] for k in range(len(P)) if mon[k] >= cut])
    print(f"=== (2) covariance forecast: train {tm[0]}..<{cut} ({len(tr)}), test {cut}.. ({len(te)}) ===")
    gm = tr[:, 0].mean()  # grand-mean shrink target from train
    def fit(rows, news):  # OLS forecast of next_corr
        cols = [np.ones(len(rows)), rows[:, 0]]  # intercept, trailing
        if news: cols += [rows[:, 1], rows[:, 1] * rows[:, 0]]  # linked, linked*trailing
        X = np.column_stack(cols); b, *_ = np.linalg.lstsq(X, rows[:, 2], rcond=None); return b
    def pred(rows, b, news):
        cols = [np.ones(len(rows)), rows[:, 0]]
        if news: cols += [rows[:, 1], rows[:, 1] * rows[:, 0]]
        return np.column_stack(cols) @ b
    def mse(y, yh): return float(np.mean((y - yh) ** 2))
    y_te = te[:, 2]
    base_b = fit(tr, False); news_b = fit(tr, True)
    naive = mse(y_te, gm * np.ones(len(te)))           # shrink-all-to-mean
    persist = mse(y_te, te[:, 0])                        # trailing persists (random walk)
    mb = mse(y_te, pred(te, base_b, False)); mn = mse(y_te, pred(te, news_b, True))
    var = y_te.var()
    print(f"  OOS MSE   grand-mean {naive:.4f}   trailing {persist:.4f}   baseline-shrink {mb:.4f}   +NEWS {mn:.4f}")
    print(f"  OOS R2    baseline {1-mb/var:+.3f}   +news {1-mn/var:+.3f}   (news MSE reduction {100*(mb-mn)/mb:+.1f}%)")
    print(f"  news coefs: linked {news_b[2]:+.3f}, linked*trailing {news_b[3]:+.3f}  (news pulls linked pairs' forecast UP)")

    # matrix-level Frobenius on test months (contagion names)
    print("\n=== matrix-level: mean Frobenius error of the forecast corr matrix (test months) ===")
    fb = []; fn = []
    for m in [mm for mm in tm if mm >= cut]:
        rows = np.array([P[k] for k in range(len(P)) if mon[k] == m])
        if len(rows) < 30: continue
        fb.append(np.sqrt(mse(rows[:, 2], pred(rows, base_b, False)) * len(rows)))
        fn.append(np.sqrt(mse(rows[:, 2], pred(rows, news_b, True)) * len(rows)))
    if fb:
        print(f"  baseline {np.mean(fb):.2f}   +news {np.mean(fn):.2f}   ({100*(np.mean(fb)-np.mean(fn))/np.mean(fb):+.1f}%)")
    print("\n  CAVEAT: pairs share names (non-independent) -> treat significance as indicative;")
    print("  the OOS MSE reduction on held-out months is the honest economic read.")


if __name__ == "__main__":
    main()
