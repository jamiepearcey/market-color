# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
DOES NEWS CAUSAL STRUCTURE LEAD THE CORRELATION MATRIX? — the one non-tautological
test. Conventional risk models estimate covariance from TRAILING returns, so they're
structurally lagged. News describing first-moment moves can't mechanically produce a
LEADING signal on future second-moment (pairwise) correlation. So: does the news
causal network at month m predict which pairs become correlated at m+1, ABOVE the
trailing correlation a risk model already has?

  predictor : news_link(i,j,m) = the pair shares >=1 CAUSE entity in the causal graph
              that month (the emerging co-driver network -- e.g. both driven by "ECB"
              / "Greek debt").
  target    : realized abnormal-return corr(i,j) over months [m+1 .. m+next].
  control   : trailing corr(i,j) over [m-trail .. m] (what a covariance risk model sees).
  clean stat: RISE = target - trailing. If linked pairs RISE more than matched unlinked
              pairs, news anticipated the correlation before trailing covariance did.

All correlations on ABNORMAL returns (9 macro factors already removed) -> a rise the
factor risk model cannot see. Reverse causality is ruled out: predictor at m, target at
m+1, controlling for through-m correlation.

Usage: uv run eventgraph/scripts/news_leads_correlation.py --graph-dir /tmp/eg100k_graph
"""
import argparse, json, collections, csv, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series


def months_between(y0, y1):
    return [f"{y}-{m:02d}" for y in range(y0, y1 + 1) for m in range(1, 13)]


def win_corr(a1, a2, months):
    d1 = {d: v for d, v in a1.items() if d[:7] in months}
    common = [d for d in d1 if d in a2]
    if len(common) < 15: return None
    x = np.array([d1[d] for d in common]); y = np.array([a2[d] for d in common])
    if x.std() == 0 or y.std() == 0: return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--trail", type=int, default=3)
    ap.add_argument("--next", type=int, default=2, dest="nxt"); ap.add_argument("--max-pairs", type=int, default=1200)
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

    # emerging causal network: cause_effects[month][cause_entity] = {effect symbols}
    cause_effects = collections.defaultdict(lambda: collections.defaultdict(set))
    newsnames = set()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or not c or j.get("effect_dir") not in DIR: continue
        cause_effects[d[:7]][c].add(sym[e]); newsnames.add(sym[e])
    print(f"{len(newsnames)} news names; building abnormal-return series ...")
    AR = {s: ar_series(s, cache, Fmap, years, p1, p2) for s in sorted(newsnames)}
    AR = {s: v for s, v in AR.items() if len(v) > 120}
    allnames = sorted(AR)
    print(f"{len(AR)} names with abnormal-return series\n")

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years))
    idx = {m: i for i, m in enumerate(allmonths)}
    link, unlink = [], []  # (trailing, target, rise)
    news_density = {};
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        trail_ms = allmonths[i - a.trail:i + 1]; next_ms = allmonths[i + 1:i + 1 + a.nxt]
        # linked pairs: share a cause entity this month
        linked = set()
        for c, effs in cause_effects[m].items():
            effs = sorted(e for e in effs if e in AR)
            for x in range(len(effs)):
                for y in range(x + 1, len(effs)):
                    linked.add((effs[x], effs[y]))
        news_density[m] = len(linked)
        if not linked: continue
        lp = sorted(linked)
        if len(lp) > a.max_pairs: lp = [lp[k] for k in rng.choice(len(lp), a.max_pairs, replace=False)]
        for s1, s2 in lp:
            tc = win_corr(AR[s1], AR[s2], trail_ms); nc = win_corr(AR[s1], AR[s2], next_ms)
            if tc is not None and nc is not None: link.append((tc, nc, nc - tc))
        # matched random UNLINKED pairs, same count, same month
        got = 0; tries = 0
        while got < len(lp) and tries < len(lp) * 20:
            tries += 1
            s1, s2 = allnames[rng.randint(len(allnames))], allnames[rng.randint(len(allnames))]
            if s1 == s2 or (min(s1, s2), max(s1, s2)) in linked: continue
            tc = win_corr(AR[s1], AR[s2], trail_ms); nc = win_corr(AR[s1], AR[s2], next_ms)
            if tc is not None and nc is not None: unlink.append((tc, nc, nc - tc)); got += 1

    L = np.array(link); U = np.array(unlink)
    print(f"=== news-linked pairs: {len(L)}   matched unlinked: {len(U)} ===\n")

    def rpt(name, A):
        print(f"  {name:16} trailing_corr {A[:,0].mean():+.3f}   next_corr {A[:,1].mean():+.3f}   RISE {A[:,2].mean():+.4f}")
    rpt("news-linked", L); rpt("unlinked", U)
    # the decisive number: excess RISE of linked over unlinked (both start ~same trailing?)
    dl, du = L[:, 2], U[:, 2]
    se = np.sqrt(dl.var(ddof=1)/len(dl) + du.var(ddof=1)/len(du))
    t = (dl.mean() - du.mean()) / se
    print(f"\n  EXCESS RISE (linked - unlinked) = {dl.mean()-du.mean():+.4f}   t = {t:+.1f}")
    print(f"  next_corr excess (linked - unlinked) = {L[:,1].mean()-U[:,1].mean():+.4f}")
    # control for trailing level via a simple partial: regress rise on [1, linked, trailing]
    X = np.column_stack([np.ones(len(L)+len(U)), np.r_[np.ones(len(L)), np.zeros(len(U))], np.r_[L[:,0], U[:,0]]])
    yv = np.r_[L[:,2], U[:,2]]
    beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ beta; se_b = np.sqrt((resid@resid)/(len(yv)-3) * np.linalg.inv(X.T@X)[1,1])
    print(f"  controlling for trailing level: linked coef on RISE = {beta[1]:+.4f}  t = {beta[1]/se_b:+.1f}")

    # DECISIVE CONTROL: stratify by trailing correlation. The confound is "linked pairs
    # are already related (same sector) -> already correlated". Trailing corr is the direct
    # observable of that relatedness (subsumes sector, supply-chain, common-factor). If
    # linked pairs RISE more than unlinked WITHIN a trailing-corr bin, news adds beyond the
    # current correlation structure -- it is not just a dressed-up relatedness signal.
    print("\n=== DECISIVE: excess RISE within trailing-correlation bins (controls relatedness/sector) ===")
    print(f"  {'trailing bin':14} {'linked rise':>12} {'nL':>5} {'unlinked rise':>14} {'nU':>5} {'EXCESS':>8}")
    edges = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.45, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        lm = L[(L[:, 0] >= lo) & (L[:, 0] < hi)]; um = U[(U[:, 0] >= lo) & (U[:, 0] < hi)]
        if len(lm) < 20 or len(um) < 20: continue
        exc = lm[:, 2].mean() - um[:, 2].mean()
        se = np.sqrt(lm[:, 2].var(ddof=1)/len(lm) + um[:, 2].var(ddof=1)/len(um))
        star = "**" if abs(exc/se) > 2.58 else ("*" if abs(exc/se) > 1.96 else "")
        print(f"  [{lo:+.2f},{hi:+.2f}) {lm[:,2].mean():+12.4f} {len(lm):5} {um[:,2].mean():+14.4f} {len(um):5} {exc:+8.4f}{star}")

    # systemic illustration: does news link density(m) lead next-month absorption of linked names?
    print("\n=== systemic: news-network density(m) vs next-month realized co-movement ===")
    dens = [(m, news_density.get(m, 0)) for m in allmonths if m in news_density]
    top = sorted(dens, key=lambda x: -x[1])[:6]
    print("  highest news-link-density months (candidate early warnings):")
    for m, n in top: print(f"    {m}: {n} shared-cause pairs")


if __name__ == "__main__":
    main()
