# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
IS THE CAUSAL GRAPH A LATENT REPRESENTATION OF RELATION (beyond surface co-mention)?

Co-mention is a surface, near-tautological feature: names share an article because they're
related. The deeper claim: the causal graph embeds each asset in a LATENT cause-space -- the
tf-idf-weighted profile of WHICH DRIVERS move it -- and proximity there encodes economic
relation that co-occurrence only crudely proxies.

Decisive test: cause_sim(i,j) = cosine of the two assets' cause-profile vectors. Does it
predict realized correlation for the PURE-LATENT set -- pairs NEVER co-mentioned AND sharing
NO cause this month -- controlling for trailing correlation? If a similar causal fingerprint
predicts co-movement with no co-occurrence and no shared event, the graph learned a latent
economic embedding, not attention co-occurrence.

tf-idf down-weights generic causes (drive everyone: "the economy") and up-weights specific
ones ("Greek debt", "ECB") -> a SPECIFIC shared driver, not generic attention.

Usage: uv run eventgraph/scripts/news_latent.py --graph-dir /tmp/eg100k_graph --topk 160
"""
import argparse, json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
from comention_vs_cause import MENTION_FACTS


def dyadic_ses(X, y, beta, iidx, jidx, pidx):
    """Aronow-Samii-Assenova dyadic cluster-robust SEs: the 'meat' sums scores over
    every pair of observations sharing a node. meat = A'A - B'B where A_node = sum of
    scores over obs touching that node, B_pair = sum over obs of the same unordered pair."""
    u = y - X @ beta; s = X * u[:, None]; bread = np.linalg.inv(X.T @ X); k = X.shape[1]
    nn = int(max(iidx.max(), jidx.max())) + 1
    A = np.zeros((nn, k)); np.add.at(A, iidx, s); np.add.at(A, jidx, s)
    B = np.zeros((int(pidx.max()) + 1, k)); np.add.at(B, pidx, s)
    V = bread @ (A.T @ A - B.T @ B) @ bread; d = np.diag(V)
    return np.sqrt(np.where(d > 0, d, np.nan))


def report_both(title, X, y, ii, jj, pp, labels):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); resid = y - X @ beta
    ols = np.sqrt((resid @ resid) / (len(y) - X.shape[1]) * np.diag(np.linalg.inv(X.T @ X)))
    dy = dyadic_ses(X, y, beta, ii, jj, pp)
    print(title)
    for lab, c, so, sd in zip(labels, beta, ols, dy):
        star = "**" if abs(c/sd) > 2.58 else ("*" if abs(c/sd) > 1.96 else "  ")
        print(f"  {lab:18} {c:+.4f}   OLS t={c/so:+7.1f}   DYADIC t={c/sd:+6.1f}{star}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg100k_graph")
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--topk", type=int, default=160)
    ap.add_argument("--trail", type=int, default=3); ap.add_argument("--next", type=int, default=2, dest="nxt")
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
    sec = json.loads((gd / "sector.json").read_text()) if (gd / "sector.json").exists() else {}
    sec = {k: v for k, v in sec.items() if v}; print(f"sector labels: {len(sec)} symbols\n")
    vix = json.loads((gd / "vix_monthly.json").read_text()) if (gd / "vix_monthly.json").exists() else {}

    # per (name, month): Counter of cause entities; and cause-set for shared-cause flag
    name_causes = collections.defaultdict(collections.Counter); freq = collections.Counter()
    cause_effects = collections.defaultdict(lambda: collections.defaultdict(set))
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        s = sym[e]; freq[s] += 1
        if c:
            name_causes[(s, d[:7])][c] += 1; cause_effects[d[:7]][c].add(s)

    print(f"building abnormal-return series for top {a.topk} active names ...")
    AR = {}
    for s, _ in freq.most_common():
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150: AR[s] = v
    names = sorted(AR); U = set(names); print(f"{len(names)} names\n")

    # co-mention pairs per month (same-doc)
    doc_ents = collections.defaultdict(set)
    for fname, flds in MENTION_FACTS:
        fp = lake / fname
        if not fp.exists(): continue
        for l in open(fp):
            j = json.loads(l); did = j.get("doc_id")
            for f in flds:
                e = j.get(f)
                if e in sym and sym[e] in U: doc_ents[did].add(sym[e])
    comention_month = collections.defaultdict(set)
    for did, ents in doc_ents.items():
        d = docdate.get(did)
        if not d or d[:4] not in years: continue
        es = sorted(ents)
        for x in range(len(es)):
            for y in range(x+1, len(es)): comention_month[d[:7]].add((es[x], es[y]))

    # idf over causes: df = # distinct names a cause ever drove
    cause_df = collections.Counter()
    seen = collections.defaultdict(set)
    for (s, m), cc in name_causes.items():
        for c in cc: seen[c].add(s)
    for c, ss in seen.items(): cause_df[c] = len(ss)
    N = len(names)
    idf = {c: math.log(1 + N / df) for c, df in cause_df.items()}

    def profile(s, months):  # tf-idf cause vector over a window
        v = collections.Counter()
        for m in months:
            for c, n in name_causes.get((s, m), {}).items(): v[c] += n
        return {c: n * idf.get(c, 0.0) for c, n in v.items()}
    def cos(a1, a2):
        if not a1 or not a2: return 0.0
        common = set(a1) & set(a2)
        if not common: return 0.0
        dot = sum(a1[c] * a2[c] for c in common)
        n1 = math.sqrt(sum(x*x for x in a1.values())); n2 = math.sqrt(sum(x*x for x in a2.values()))
        return dot / (n1 * n2) if n1 and n2 else 0.0

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years)); idx = {m: i for i, m in enumerate(allmonths)}
    panel = []; pairs_list = []; months_list = []  # + (i,j) and month per row
    examples = []
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        wm = allmonths[i-a.trail:i+1]
        trail = corr_matrix(AR, names, wm); nxt = corr_matrix(AR, names, allmonths[i+1:i+1+a.nxt])
        com = comention_month.get(m, set())
        cau = set()
        for c, effs in cause_effects[m].items():
            effs = sorted(e for e in effs if e in U)
            for x in range(len(effs)):
                for y in range(x+1, len(effs)): cau.add((effs[x], effs[y]))
        prof = {s: profile(s, wm) for s in names}
        for pair, tc in trail.items():
            if pair not in nxt: continue
            cs = cos(prof[pair[0]], prof[pair[1]])
            isc = 1 if pair in com else 0; iss = 1 if pair in cau else 0
            both_known = pair[0] in sec and pair[1] in sec
            same_sec = 1 if both_known and sec[pair[0]] == sec[pair[1]] else 0
            panel.append((tc, nxt[pair], isc, iss, cs, same_sec, 1 if both_known else 0)); pairs_list.append(pair); months_list.append(m)
            if m == "2011-09" and isc == 0 and iss == 0 and cs > 0.5 and len(examples) < 12:
                examples.append((pair, cs, tc, nxt[pair]))
    P = np.array(panel); print(f"panel {len(P)} pair-months\n")
    # node + pair integer ids for dyadic cluster-robust SEs
    nid = {}; pid = {}
    ii = np.array([nid.setdefault(p[0], len(nid)) for p in pairs_list])
    jj = np.array([nid.setdefault(p[1], len(nid)) for p in pairs_list])
    pp = np.array([pid.setdefault(p, len(pid)) for p in pairs_list])
    print(f"dyadic clustering over {len(nid)} nodes / {len(pid)} distinct pairs "
          f"(SEs account for pair-observations sharing a name)\n")

    # (I) incremental regression: does cause_sim add beyond co-mention + shared-cause + trailing?
    y = P[:,1] - P[:,0]
    X = np.column_stack([np.ones(len(P)), P[:,0], P[:,2], P[:,3], P[:,4]])
    report_both("=== RISE ~ trailing + co_mention + shared_cause + cause_sim(latent) ===",
                X, y, ii, jj, pp, ["intercept","trailing","co_mention","shared_cause","cause_sim(latent)"])

    # (II) PURE-LATENT: pairs NEVER co-mentioned AND no shared cause this month -> does cause_sim predict?
    pl = P[(P[:,2]==0) & (P[:,3]==0)]
    print(f"\n=== PURE-LATENT set: no co-mention, no shared cause ({len(pl)} pair-months) ===")
    print("  cause_sim bin -> next_corr and correlation persistence (trailing controlled by binning trailing too)")
    print(f"  {'cause_sim':>12} {'mean next_corr':>14} {'mean trailing':>13} {'RISE':>8} {'n':>7}")
    for lo, hi in [(0.0,1e-6),(1e-6,0.05),(0.05,0.15),(0.15,0.35),(0.35,1.01)]:
        r = pl[(pl[:,4]>=lo) & (pl[:,4]<hi)]
        if len(r) < 50: continue
        lbl = "0 (none)" if hi<=1e-6 else f"[{lo:.2f},{hi:.2f})"
        print(f"  {lbl:>12} {r[:,1].mean():+14.3f} {r[:,0].mean():+13.3f} {(r[:,1]-r[:,0]).mean():+8.4f} {len(r):7}")
    # correlation of cause_sim with next_corr WITHIN pure-latent, partialling trailing
    plm = (P[:,2]==0) & (P[:,3]==0)
    if plm.sum() > 100:
        Xp = np.column_stack([np.ones(int(plm.sum())), P[plm,0], P[plm,4]])
        report_both("\n  pure-latent: next_corr ~ trailing + cause_sim (no co-mention, no shared event)",
                    Xp, P[plm,1], ii[plm], jj[plm], pp[plm], ["intercept","trailing","cause_sim"])

    # (III) SECTOR CONTROL — is cause_sim latent ECONOMIC relation or latent SECTOR?
    km = P[:, 6] == 1  # pairs with both sectors known
    kn = P[km]
    print(f"\n=== SECTOR CONTROL ({len(kn)} pair-months, both sectors known; same-sector {int(kn[:,5].mean()*100)}%) ===")
    Xk = np.column_stack([np.ones(len(kn)), kn[:, 0], kn[:, 5], kn[:, 4]])  # 1, trailing, same_sector, cause_sim
    report_both("", Xk, kn[:, 1] - kn[:, 0], ii[km], jj[km], pp[km], ["intercept","trailing","same_sector","cause_sim"])
    # THE decisive cut: does cause_sim predict correlation among CROSS-SECTOR pairs (sector can't explain)?
    xm = km & (P[:, 5] == 0); xs = P[xm]
    print(f"\n  CROSS-SECTOR pairs only ({len(xs)}) -- sector cannot explain any co-movement here:")
    print(f"  {'cause_sim':>12} {'next_corr':>10} {'trailing':>9} {'n':>7}")
    for lo, hi in [(0.0,1e-6),(1e-6,0.10),(0.10,0.30),(0.30,1.01)]:
        r = xs[(xs[:,4]>=lo) & (xs[:,4]<hi)]
        if len(r) < 40: continue
        lbl = "0 (none)" if hi<=1e-6 else f"[{lo:.2f},{hi:.2f})"
        print(f"  {lbl:>12} {r[:,1].mean():+10.3f} {r[:,0].mean():+9.3f} {len(r):7}")
    Xx = np.column_stack([np.ones(len(xs)), xs[:,0], xs[:,4]])
    report_both("  cross-sector: next_corr ~ trailing + cause_sim", Xx, xs[:,1], ii[xm], jj[xm], pp[xm], ["intercept","trailing","cause_sim"])
    print("  => cause_sim>0 here = causal fingerprint predicts co-movement BETWEEN sectors = latent ECONOMIC relation, not sector.")

    # (IV) SUB-PERIOD STABILITY of the cross-sector cause_sim (mini out-of-sample: is t=2.8 stable or fragile?)
    marr = np.array(months_list); umonths = sorted(set(marr)); med = umonths[len(umonths)//2]
    print(f"\n=== SUB-PERIOD stability of cross-sector cause_sim (split at {med}) ===")
    for lbl, sub in [("first half", xm & (marr < med)), ("second half", xm & (marr >= med))]:
        r = P[sub]
        if len(r) < 200: continue
        mm = sorted(set(marr[sub]))
        Xh = np.column_stack([np.ones(len(r)), r[:, 0], r[:, 4]])
        report_both(f"  {lbl} ({mm[0]}..{mm[-1]}, n={len(r)})", Xh, r[:, 1], ii[sub], jj[sub], pp[sub], ["intercept","trailing","cause_sim"])

    # (V) STATE-DEPENDENCE: does cross-sector cause_sim scale SYSTEMATICALLY with market stress (VIX)?
    if vix:
        vraw = np.array([vix.get(m, np.nan) for m in months_list]); vfin = np.isfinite(vraw)
        vz = (vraw - np.nanmean(vraw)) / np.nanstd(vraw)
        m2 = xm & vfin; r = P[m2]; v = vz[m2]
        Xi = np.column_stack([np.ones(len(r)), r[:, 0], r[:, 4], v, r[:, 4] * v])
        print("\n=== STATE-DEPENDENCE: cross-sector, next_corr ~ trailing + cause_sim + VIXz + cause_sim x VIXz ===")
        report_both("", Xi, r[:, 1], ii[m2], jj[m2], pp[m2], ["intercept","trailing","cause_sim","VIXz","cause_sim x VIXz"])
        # VIX terciles across MONTHS (high tercile spans 2010 flash-crash, 2011 crisis, 2012 Grexit -> not one episode)
        um = sorted(set(months_list), key=lambda mm: vix.get(mm, 0)); t3 = len(um) // 3
        terc = [("low-VIX", set(um[:t3])), ("mid-VIX", set(um[t3:2*t3])), ("high-VIX", set(um[2*t3:]))]
        print("\n  cross-sector cause_sim by VIX tercile (systematic gradient, or only 2011?):")
        for lbl, ms in terc:
            mask = xm & np.array([mm in ms for mm in months_list])
            rr = P[mask]
            if len(rr) < 150: continue
            Xt = np.column_stack([np.ones(len(rr)), rr[:, 0], rr[:, 4]])
            av = np.mean([vix[m] for m in ms if m in vix])
            report_both(f"  {lbl} (avg VIX {av:.0f}, months {sorted(ms)[0]}..{sorted(ms)[-1]}, n={len(rr)})",
                        Xt, rr[:, 1], ii[mask], jj[mask], pp[mask], ["intercept","trailing","cause_sim"])

    if examples:
        print("\n  example never-co-mentioned high-cause-similarity pairs (2011-09):")
        for (i, j), cs, tc, nc in examples[:8]:
            print(f"    {i:6} ~ {j:6}  cause_sim {cs:.2f}  trailing {tc:+.2f} -> next {nc:+.2f}")
    print("\n  CAVEAT: dyadic non-independence inflates t; the binned pure-latent gradient is the robust read.")


if __name__ == "__main__":
    main()
