# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
NOVELTY / COMPLEMENTARITY TEST: does the extracted CAUSAL structure (shared-cause link)
carry information beyond a plain CO-MENTION baseline (Schwenkler-Zheng-style) for predicting
correlation PERSISTENCE?

Two pair-month links, deliberately different in construction:
  co_mention(i,j,m)   = i,j appear in the SAME document that month (symmetric, WITHIN-article).
  shared_cause(i,j,m) = i,j are both EFFECTS of the SAME cause entity that month -- possibly in
                        DIFFERENT articles (graph-transitive, ACROSS-article: the causal graph
                        links assets co-mention cannot).

Target: does an already-elevated correlation (trailing > tau) PERSIST (next_corr >= trailing)?
We partition pairs into neither / co-mention-only / cause-only / BOTH and read persistence,
then regress the correlation RISE on both links + their interaction (controls trailing level).

Outcomes: cause redundant w/ co-mention -> no novelty. Cause adds beyond co-mention (and/or
they are complementary, BOTH strongest) -> a genuine, publishable delta.

Usage: uv run eventgraph/scripts/comention_vs_cause.py --graph-dir /tmp/eg100k_graph --topk 160
"""
import argparse, json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix

MENTION_FACTS = [("causal_event_edge.jsonl", ("cause_entity", "effect_entity")),
                 ("event.jsonl", ("issuer_entity",)), ("sensitivity_edge.jsonl", ("asset_entity",)),
                 ("sentiment_annotation.jsonl", ("target_entity",))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--topk", type=int, default=160)
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

    freq = collections.Counter()
    cause_effects = collections.defaultdict(lambda: collections.defaultdict(set))
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        freq[sym[e]] += 1
        if c: cause_effects[d[:7]][c].add(sym[e])

    print(f"building abnormal-return series for top {a.topk} active names ...")
    AR = {}
    for s, _ in freq.most_common():
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150: AR[s] = v
    names = sorted(AR); U = set(names); print(f"{len(names)} names\n")

    # doc -> {universe symbols mentioned} (any fact node) -> co-mention pairs per month
    doc_ents = collections.defaultdict(set)
    for fname, flds in MENTION_FACTS:
        fp = lake / fname
        if not fp.exists(): continue
        for l in open(fp):
            j = json.loads(l); did = j.get("doc_id")
            if not did: continue
            for f in flds:
                e = j.get(f)
                if e in sym and sym[e] in U: doc_ents[did].add(sym[e])
    comention_month = collections.defaultdict(set)  # month -> {(i,j)}
    for did, ents in doc_ents.items():
        d = docdate.get(did)
        if not d or d[:4] not in years: continue
        es = sorted(ents)
        for x in range(len(es)):
            for y in range(x+1, len(es)): comention_month[d[:7]].add((es[x], es[y]))
    sharedcause_month = collections.defaultdict(set)
    for m, ce in cause_effects.items():
        for c, effs in ce.items():
            effs = sorted(e for e in effs if e in U)
            for x in range(len(effs)):
                for y in range(x+1, len(effs)): sharedcause_month[m].add((effs[x], effs[y]))

    # overlap of the two link sets (pooled)
    allc = set().union(*comention_month.values()) if comention_month else set()
    alls = set().union(*sharedcause_month.values()) if sharedcause_month else set()
    inter = len(allc & alls); uni = len(allc | alls)
    print(f"link sets (pooled distinct pairs): co-mention {len(allc)}  shared-cause {len(alls)}  "
          f"overlap {inter}  Jaccard {inter/uni:.2f}\n")

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years)); idx = {m: i for i, m in enumerate(allmonths)}
    panel = []  # (trailing, next, com, cause)
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        trail = corr_matrix(AR, names, allmonths[i-a.trail:i+1]); nxt = corr_matrix(AR, names, allmonths[i+1:i+1+a.nxt])
        com = comention_month.get(m, set()); cau = sharedcause_month.get(m, set())
        for pair, tc in trail.items():
            if pair in nxt: panel.append((tc, nxt[pair], 1 if pair in com else 0, 1 if pair in cau else 0))
    P = np.array(panel); print(f"panel {len(P)} pair-months  (co-mention {int(P[:,2].sum())}, shared-cause {int(P[:,3].sum())})\n")

    # (I) persistence by category among already-correlated pairs
    hi = P[P[:, 0] > a.tau]
    def stat(rows): return (float(np.mean(rows[:,1] >= rows[:,0])) if len(rows) else float("nan"),
                            float(rows[:,1].mean()) if len(rows) else float("nan"),
                            float(rows[:,0].mean()) if len(rows) else float("nan"), len(rows))
    cats = {"neither": hi[(hi[:,2]==0)&(hi[:,3]==0)], "co-mention only": hi[(hi[:,2]==1)&(hi[:,3]==0)],
            "shared-cause only": hi[(hi[:,2]==0)&(hi[:,3]==1)], "BOTH": hi[(hi[:,2]==1)&(hi[:,3]==1)]}
    print(f"=== persistence of elevated correlations (trailing > {a.tau}) by link type ===")
    print(f"  {'category':18} {'persists':>9} {'next_corr':>10} {'from':>7} {'n':>7}")
    for k, r in cats.items():
        pr, nc, fr, n = stat(r)
        print(f"  {k:18} {pr:9.0%} {nc:+10.3f} {fr:+7.3f} {n:7}")

    # (II) incremental regression: does each link add beyond the other? RISE ~ trail + com + cause + com*cause
    y = P[:,1] - P[:,0]
    X = np.column_stack([np.ones(len(P)), P[:,0], P[:,2], P[:,3], P[:,2]*P[:,3]])
    b, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X@b
    XtXi = np.linalg.inv(X.T@X); se = np.sqrt((resid@resid)/(len(y)-X.shape[1]) * np.diag(XtXi))
    labels = ["intercept", "trailing", "co_mention", "shared_cause", "com*cause"]
    print(f"\n=== incremental: RISE ~ trailing + co_mention + shared_cause + interaction ===")
    for lab, coef, s in zip(labels, b, se):
        star = "**" if abs(coef/s)>2.58 else ("*" if abs(coef/s)>1.96 else "")
        print(f"  {lab:14} {coef:+.4f}  t={coef/s:+5.1f}{star}")
    print("\n  READ: co_mention coef = co-mention's own effect; shared_cause coef = what CAUSAL")
    print("  structure adds BEYOND co-mention. Both>0 => complementary (a two-signal process).")
    print("  CAVEAT: dyadic non-independence inflates t; category persistence rates are the robust read.")


if __name__ == "__main__":
    main()
