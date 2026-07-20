# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
RIGOR ROBUSTNESS for the core persistence result (owed per external review):

A. MONTH-CLUSTERED SEs. 2010-12 has huge common shocks; dyadic clustering handles
   name-sharing but not time-sharing. Report OLS / dyadic / MONTH-clustered t for the
   core regression (RISE ~ trailing + linked-tier dummies); inference = worst of the
   cluster dimensions (conservative two-way approximation).

B. STRICT LINK TIMING. Known endogeneity: links form INSIDE the trailing window, and
   journalists may write about pairs BECAUSE they already co-move. Strict design:
     links     from months m-3..m-2   (formation window)
     trailing  over months m-1..m     (measurement window, NO overlap with formation)
     target    months m+1..m+2        (forward)
   If linked pairs still persist more, the link is not a re-description of the same-
   window co-movement -- the news preceded the correlation measurement entirely.

Usage: uv run scripts/timing_robustness.py --graph-dir ../data/eg_runs/eg100k_graph
"""
import argparse, json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
from news_latent import dyadic_ses


def cluster_ses(X, y, beta, gidx):
    """one-way cluster-robust SEs over integer group ids."""
    u = y - X @ beta; s = X * u[:, None]; k = X.shape[1]
    G = np.zeros((int(gidx.max()) + 1, k)); np.add.at(G, gidx, s)
    bread = np.linalg.inv(X.T @ X)
    ng = G.shape[0]
    adj = (ng / (ng - 1)) * ((len(y) - 1) / (len(y) - k))
    V = adj * bread @ (G.T @ G) @ bread
    d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--topk", type=int, default=160)
    ap.add_argument("--tau", type=float, default=0.3)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"
    years = set(a.years.split(",")); p1 = int(dt.datetime(2008,1,1,tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime(2014,12,31,tzinfo=dt.UTC).timestamp())

    Fmap = {}
    for row in csv.DictReader(open(gd / "factor_snapshot_factor_returns.csv")):
        d = row["date"]; Fmap[f"{d[:4]}-{d[4:6]}-{d[6:8]}"] = np.array([float(row[f]) if row[f] not in ("","NaN","nan") else np.nan for f in FN])
    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in SKIP:
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: (json.loads(l).get("published_at") or "")[:10] for l in open(lake / "document.jsonl")}
    freq = collections.Counter()
    cause_m = collections.defaultdict(lambda: collections.defaultdict(set))  # month -> cause -> syms
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        freq[sym[e]] += 1
        if c: cause_m[d[:7]][c].add(sym[e])

    print(f"building series for top {a.topk} names ...")
    AR = {}
    for s, _ in freq.most_common():
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150: AR[s] = v
    names = sorted(AR); U = set(names); print(f"{len(names)} names\n")

    def links_in(months):
        out = set()
        for m in months:
            for c, ss in cause_m.get(m, {}).items():
                ss = sorted(s for s in ss if s in U)
                for x in range(len(ss)):
                    for y in range(x+1, len(ss)): out.add((ss[x], ss[y]))
        return out

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years)); idx = {m: i for i, m in enumerate(allmonths)}

    # ---- A. month-clustered SEs on the STANDARD design (links m-3..m, trailing m-3..m)
    rowsA, plistA, mlistA = [], [], []
    for m in allmonths:
        i = idx[m]
        if i < 3 or i + 2 >= len(allmonths): continue
        wm = allmonths[i-3:i+1]
        trail = corr_matrix(AR, names, wm); nxt = corr_matrix(AR, names, allmonths[i+1:i+3])
        lk = links_in(wm)
        for pair, tc in trail.items():
            if pair not in nxt: continue
            rowsA.append((tc, nxt[pair], 1 if pair in lk else 0)); plistA.append(pair); mlistA.append(i)
    A = np.array(rowsA)
    nid, pid = {}, {}
    ii = np.array([nid.setdefault(p[0], len(nid)) for p in plistA]); jj = np.array([nid.setdefault(p[1], len(nid)) for p in plistA])
    pp = np.array([pid.setdefault(p, len(pid)) for p in plistA]); mm = np.array(mlistA); mm -= mm.min()
    y = A[:, 1] - A[:, 0]
    X = np.column_stack([np.ones(len(A)), A[:, 0], A[:, 2]])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta
    ols = np.sqrt((resid @ resid) / (len(y) - 3) * np.diag(np.linalg.inv(X.T @ X)))
    dy = dyadic_ses(X, y, beta, ii, jj, pp)
    mo = cluster_ses(X, y, beta, mm)
    print(f"=== A. core regression RISE ~ trailing + linked (window links, {len(A)} pair-months, {mm.max()+1} months) ===")
    print(f"  {'coef':10} {'value':>8} {'OLS t':>8} {'DYADIC t':>9} {'MONTH t':>8}   [inference = worst]")
    for lab, b, so, sd, sm in zip(["intercept","trailing","linked"], beta, ols, dy, mo):
        print(f"  {lab:10} {b:+8.4f} {b/so:+8.1f} {b/sd:+9.1f} {b/sm:+8.1f}")

    # ---- B. strict timing: links m-3..m-2 | trailing m-1..m | target m+1..m+2
    rowsB = []
    for m in allmonths:
        i = idx[m]
        if i < 3 or i + 2 >= len(allmonths): continue
        lk = links_in(allmonths[i-3:i-1])                       # formation: m-3, m-2
        trail = corr_matrix(AR, names, allmonths[i-1:i+1])      # measurement: m-1, m (no overlap)
        nxt = corr_matrix(AR, names, allmonths[i+1:i+3])
        for pair, tc in trail.items():
            if pair not in nxt: continue
            rowsB.append((tc, nxt[pair], 1 if pair in lk else 0))
    B = np.array(rowsB)
    hi = B[B[:, 0] > a.tau]
    lkm = hi[:, 2] == 1
    print(f"\n=== B. STRICT TIMING (links precede corr-measurement window entirely) ===")
    print(f"  elevated pairs (trailing > {a.tau}): linked {int(lkm.sum())}, unlinked {int((~lkm).sum())}")
    for lbl, r in [("linked (formed BEFORE meas. window)", hi[lkm]), ("unlinked", hi[~lkm])]:
        if len(r):
            print(f"  {lbl:38} persists {float(np.mean(r[:,1] >= r[:,0])):.0%}   next {r[:,1].mean():+.3f} (from {r[:,0].mean():+.3f})   n={len(r)}")
    print("\n  If the linked persistence advantage survives strict timing, the link is not a")
    print("  re-description of same-window co-movement (news preceded the measured correlation).")


if __name__ == "__main__":
    main()
