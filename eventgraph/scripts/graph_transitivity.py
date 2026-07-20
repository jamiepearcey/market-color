# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
GRAPH TRANSITIVITY — does MULTI-HOP structure predict co-movement the embedding cannot
see? The cause-profile embedding is blind to pairs with DISJOINT but CONNECTED driver
sets (cosine = 0). Features per pair-month, from the entity causal graph over the same
trailing window as the correlation (design hardened by external review):

  bridge2  : an edge connects a driver of i to a driver of j (undirected — either
             orientation implies shock propagation).
  parent2  : DIRECTED common cause: some c0 -> (driver of i) AND c0 -> (driver of j).
  child2   : PLACEBO — common EFFECT of the two driver sets (collider; should be ~0).

Review-mandated safeguards:
  - hubs: WITHIN-WINDOW degree; entities in the top --hub-pct by degree are excluded
    from bridge/parent formation (greece is a hub in 2011, not 2010). KILL CONDITION:
    effect must survive hub exclusion. Prevalence reported with/without.
  - document leakage: a bridging/parent edge only counts if at least one of its source
    docs mentions NEITHER leg (else it's disguised co-mention).
  - set-size confound: controls for log driver-set sizes + product + logfreq (bridge
    probability rises mechanically with |D_i| x |D_j|).
  - PRE-REGISTERED: coefficients positive but BELOW cause_sim's +0.17; directed
    parent2 (hub-corrected) is the most trustworthy; child2 ~ 0.

PRIMARY: pure-latent pairs (no shared cause in window, never co-mentioned in window,
cause_sim = 0): RISE ~ trailing + same_sector + sizes + bridge2 + parent2 + child2.
SECONDARY: full-panel ordering with link1 + cause_sim included. Dyadic SEs throughout.

Usage: uv run scripts/graph_transitivity.py --graph-dir ../data/eg_runs/eg100k_graph
"""
import argparse, json, collections, csv, math, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
from news_latent import dyadic_ses, report_both
from comention_vs_cause import MENTION_FACTS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--topk", type=int, default=160)
    ap.add_argument("--trail", type=int, default=3); ap.add_argument("--next", type=int, default=2, dest="nxt")
    ap.add_argument("--hub-pct", type=float, default=2.0)
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
    sec = json.loads((gd / "sector.json").read_text()) if (gd / "sector.json").exists() else {}
    sec = {k: v for k, v in sec.items() if v}
    docdate = {json.loads(l)["doc_id"]: (json.loads(l).get("published_at") or "")[:10] for l in open(lake / "document.jsonl")}

    # entity-graph edges (cause -> effect over ALL entities) + per-edge doc mention-sets;
    # driver sets D_month(sym); asset news freq
    freq = collections.Counter()
    drivers_m = collections.defaultdict(lambda: collections.defaultdict(set))  # month -> sym -> {cause}
    egraph_m = collections.defaultdict(lambda: collections.defaultdict(set))   # month -> cause -> {effect}
    edgedocs = collections.defaultdict(set)                                    # (month, c, e) -> {doc}
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or not c or not e: continue
        m = d[:7]
        if e in sym and j.get("effect_dir") in DIR:
            freq[sym[e]] += 1; drivers_m[m][sym[e]].add(c)
        egraph_m[m][c].add(e); edgedocs[(m, c, e)].add(j.get("doc_id"))

    # doc -> mentioned resolved symbols (for leakage exclusion) and co-mention pairs
    doc_syms = collections.defaultdict(set)
    for fname, flds in MENTION_FACTS:
        fp = lake / fname
        if not fp.exists(): continue
        for l in open(fp):
            j = json.loads(l)
            for f in flds:
                e = j.get(f)
                if e in sym: doc_syms[j.get("doc_id")].add(sym[e])
    comention_m = collections.defaultdict(set)
    for did, ents in doc_syms.items():
        d = docdate.get(did)
        if not d or d[:4] not in years: continue
        es = sorted(ents)
        for x in range(len(es)):
            for y in range(x+1, len(es)): comention_m[d[:7]].add((es[x], es[y]))

    print(f"building series for top {a.topk} names ...")
    AR = {}
    for s, _ in freq.most_common():
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150: AR[s] = v
    names = sorted(AR); U = set(names); print(f"{len(names)} names\n")

    def window_structs(wmonths, hub_pct):
        """driver sets, adjacency (undirected + directed), hubs, comention, per name over a window."""
        D = collections.defaultdict(set); fwd = collections.defaultdict(set); rev = collections.defaultdict(set)
        emention = collections.defaultdict(set)  # (c,e) -> union of mentioned syms across its docs (min over docs would be ideal; keep per-doc)
        edocs = collections.defaultdict(list)    # (c,e) -> [docs]
        com = set()
        for m in wmonths:
            for s_, cs in drivers_m.get(m, {}).items():
                if s_ in U: D[s_] |= cs
            for c, effs in egraph_m.get(m, {}).items():
                for e in effs:
                    fwd[c].add(e); rev[e].add(c)
                    for doc in edgedocs.get((m, c, e), ()): edocs[(c, e)].append(doc)
            com |= comention_m.get(m, set())
        deg = collections.Counter()
        for c, effs in fwd.items():
            deg[c] += len(effs)
            for e in effs: deg[e] += 1
        ranked = sorted(deg, key=lambda n: -deg[n])
        hubs = set(ranked[:max(1, int(len(ranked) * hub_pct / 100))])
        def clean_edge(c, e, i_, j_):
            """edge counts only if some source doc mentions neither leg (anti-leakage)."""
            return any(not ({i_, j_} & doc_syms.get(doc, set())) for doc in edocs.get((c, e), ()))
        return D, fwd, rev, hubs, com, clean_edge

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years)); idx = {m: i for i, m in enumerate(allmonths)}
    rows, plist = [], []
    prev_bridge_nohub = prev_bridge_hub = 0
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        wm = allmonths[i-a.trail:i+1]
        trail = corr_matrix(AR, names, wm); nxt = corr_matrix(AR, names, allmonths[i+1:i+1+a.nxt])
        D, fwdA, revA, hubs, com, clean_edge = window_structs(wm, a.hub_pct)
        # per-name derived sets with hub exclusion
        neigh, parents, children = {}, {}, {}
        for s_ in names:
            ds = {c for c in D.get(s_, ()) if c not in hubs}
            nb, pa, ch = set(), set(), set()
            for c in ds:
                nb |= {x for x in fwdA.get(c, ()) if x not in hubs}
                nb |= {x for x in revA.get(c, ()) if x not in hubs}
                pa |= {x for x in revA.get(c, ()) if x not in hubs}   # c0 -> c : common CAUSE candidates
                ch |= {x for x in fwdA.get(c, ()) if x not in hubs}   # c -> c0 : common EFFECT (placebo)
            neigh[s_], parents[s_], children[s_] = nb, pa, ch
        for pair, tc in trail.items():
            if pair not in nxt: continue
            i_, j_ = pair
            Di = {c for c in D.get(i_, ()) if c not in hubs}; Dj = {c for c in D.get(j_, ()) if c not in hubs}
            shared = bool(D.get(i_, set()) & D.get(j_, set()))
            # bridge2: driver of i adjacent to driver of j, with a leakage-clean edge
            b2 = 0
            for c1 in Di:
                hits = (fwdA.get(c1, set()) | revA.get(c1, set())) & Dj
                for c2 in hits:
                    if c2 in hubs: continue
                    if clean_edge(c1, c2, i_, j_) or clean_edge(c2, c1, i_, j_): b2 = 1; break
                if b2: break
            p2_ = 1 if (parents.get(i_, set()) & parents.get(j_, set())) else 0
            ch2 = 1 if (children.get(i_, set()) & children.get(j_, set())) else 0
            szi, szj = len(Di), len(Dj)
            lf = math.log(1 + freq[i_]) + math.log(1 + freq[j_])
            same_sec = 1 if (sec.get(i_) and sec.get(i_) == sec.get(j_)) else 0
            comen = 1 if pair in com else 0
            rows.append((tc, nxt[pair], b2, p2_, ch2, shared, comen, same_sec, lf,
                         math.log(1+szi)+math.log(1+szj), szi*szj))
            plist.append(pair)
    P = np.array(rows)
    nid, pid = {}, {}
    ii = np.array([nid.setdefault(p[0], len(nid)) for p in plist]); jj = np.array([nid.setdefault(p[1], len(nid)) for p in plist])
    pp = np.array([pid.setdefault(p, len(pid)) for p in plist])
    print(f"panel {len(P)} pair-months | bridge2 {int(P[:,2].sum())} parent2 {int(P[:,3].sum())} child2 {int(P[:,4].sum())} "
          f"(hub-excluded top {a.hub_pct}%)")

    # PRIMARY: pure-latent subsample (no shared cause, no co-mention in window)
    pl = (P[:, 5] == 0) & (P[:, 6] == 0)
    Q = P[pl]
    print(f"pure-latent subsample: {len(Q)} pair-months | bridge2 {int(Q[:,2].sum())} parent2 {int(Q[:,3].sum())} child2 {int(Q[:,4].sum())}\n")
    y = Q[:, 1] - Q[:, 0]
    X = np.column_stack([np.ones(len(Q)), Q[:, 0], Q[:, 7], Q[:, 8], Q[:, 9], Q[:, 2], Q[:, 3], Q[:, 4]])
    report_both("=== PRIMARY (pure-latent): RISE ~ trailing + same_sector + logfreq + set_sizes + bridge2 + parent2 + child2(placebo) ===",
                X, y, ii[pl], jj[pl], pp[pl],
                ["intercept","trailing","same_sector","logfreq","log_set_sizes","bridge2","parent2","child2_placebo"])

    # SECONDARY: full-panel ordering
    y2 = P[:, 1] - P[:, 0]
    X2 = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 7], P[:, 8], P[:, 9], P[:, 5], P[:, 6], P[:, 2], P[:, 3]])
    report_both("\n=== SECONDARY (full panel): RISE ~ trailing + sector + freq + sizes + shared_cause + co_mention + bridge2 + parent2 ===",
                X2, y2, ii, jj, pp,
                ["intercept","trailing","same_sector","logfreq","log_set_sizes","shared_cause","co_mention","bridge2","parent2"])
    print("\n  PRE-REGISTERED: bridge2/parent2 > 0 but < cause_sim's +0.17; child2 ~ 0 (collider placebo);")
    print("  KILL CONDITION applied: hubs (top-degree, within-window) excluded from all features.")


if __name__ == "__main__":
    main()
