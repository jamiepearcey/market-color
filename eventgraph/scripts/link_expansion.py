# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
LINK-SOURCE EXPANSION + DIRECTION-AGREEMENT (blind-spots #1-revised and #4), designs
hardened per external review:

A. New link sources beyond shared-cause: SENSITIVITY (both assets extracted as sensitive
   to the same factor entity that month) and RELATION (extracted competes_with/supplies/
   customer_of/counterparty; split STATIC ever-related vs MONTH-ACTIVE). Review-mandated
   controls: same_sector dummy (SIC-2; sensitivity links may proxy sector — KILL CONDITION:
   sensitivity coef must survive it), continuous trailing, log news-frequency of both legs,
   mutually exclusive hierarchy causal > relation > sensitivity. Pre-registered ORDERED
   prediction: unlinked < sensitivity <= causal on persistence, sensitivity surviving the
   sector control; else declared a sector proxy.

B. Direction agreement among shared-cause pairs (both legs net-directional): B1 same-sign
   vs opposite-sign predicts SIGNED next corr (continuous trailing control; trailing
   distributions reported). B2 hedge integrity among negative-trailing pairs — primary
   spec = link x trailing interaction (attenuated reversion), secondary = sign
   preservation P(next<0); PRE-COMMITTED ABANDON if N < 100.

Usage: uv run scripts/link_expansion.py --graph-dir ../data/eg_runs/eg100k_graph
"""
import argparse, json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
from news_latent import dyadic_ses, report_both

DIRS = {"up": 1, "down": -1, "widen": -1, "tighten": 1}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--years", default="2010,2011,2012"); ap.add_argument("--topk", type=int, default=160)
    ap.add_argument("--trail", type=int, default=3); ap.add_argument("--next", type=int, default=2, dest="nxt")
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
    sec = json.loads((gd / "sector.json").read_text()) if (gd / "sector.json").exists() else {}
    sec = {k: v for k, v in sec.items() if v}
    docdate = {json.loads(l)["doc_id"]: (json.loads(l).get("published_at") or "")[:10] for l in open(lake / "document.jsonl")}

    freq = collections.Counter()
    cause_net = collections.defaultdict(lambda: collections.defaultdict(int))   # (m,c) -> sym -> net dir
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIRS: continue
        freq[sym[e]] += 1
        if c: cause_net[(d[:7], c)][sym[e]] += DIRS[j["effect_dir"]]

    print(f"building series for top {a.topk} names ...")
    AR = {}
    for s, _ in freq.most_common():
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150: AR[s] = v
    names = sorted(AR); U = set(names); print(f"{len(names)} names\n")

    # sensitivity links: (m, factor) -> syms
    sens_m = collections.defaultdict(set)
    for l in open(lake / "sensitivity_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); s_ = j.get("asset_entity"); f = j.get("factor_entity") or j.get("factor_id")
        if not d or d[:4] not in years or s_ not in sym or sym[s_] not in U or not f: continue
        sens_m[(d[:7], f)].add(sym[s_])
    sens_pairs = collections.defaultdict(set)   # month -> pairs
    for (m, f), ss in sens_m.items():
        ss = sorted(ss)
        for x in range(len(ss)):
            for y in range(x+1, len(ss)): sens_pairs[m].add((ss[x], ss[y]))

    # relation links: month-active + static (ever in window)
    rel_month = collections.defaultdict(set); rel_static = set()
    for l in open(lake / "relation_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id"))
        s1, s2 = j.get("source_entity"), j.get("target_entity")
        if not d or d[:4] not in years or s1 not in sym or s2 not in sym: continue
        a1, a2 = sym[s1], sym[s2]
        if a1 == a2 or a1 not in U or a2 not in U: continue
        pr = tuple(sorted((a1, a2)))
        rel_month[d[:7]].add(pr); rel_static.add(pr)

    # causal links + signed nets
    caus_pairs = collections.defaultdict(dict)  # month -> pair -> (net_i, net_j)
    for (m, c), effs in cause_net.items():
        ss = sorted(s for s in effs if s in U)
        for x in range(len(ss)):
            for y in range(x+1, len(ss)):
                pr = (ss[x], ss[y]); prev = caus_pairs[m].get(pr, (0, 0))
                caus_pairs[m][pr] = (prev[0] + effs[ss[x]], prev[1] + effs[ss[y]])

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years)); idx = {m: i for i, m in enumerate(allmonths)}
    rows, plist = [], []
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        trail = corr_matrix(AR, names, allmonths[i-a.trail:i+1]); nxt = corr_matrix(AR, names, allmonths[i+1:i+1+a.nxt])
        cp = caus_pairs.get(m, {}); sp = sens_pairs.get(m, set()); rm = rel_month.get(m, set())
        for pair, tc in trail.items():
            if pair not in nxt: continue
            # mutually exclusive hierarchy: causal > relation(month) > sensitivity > static-relation-only
            if pair in cp: cat = 1
            elif pair in rm: cat = 2
            elif pair in sp: cat = 3
            elif pair in rel_static: cat = 4
            else: cat = 0
            nets = cp.get(pair, (0, 0))
            same_sec = 1 if (sec.get(pair[0]) and sec.get(pair[0]) == sec.get(pair[1])) else 0
            lf = math.log(1 + freq[pair[0]]) + math.log(1 + freq[pair[1]])
            rows.append((tc, nxt[pair], cat, same_sec, lf, nets[0], nets[1])); plist.append(pair)
    P = np.array(rows)
    nid, pid = {}, {}
    ii = np.array([nid.setdefault(p[0], len(nid)) for p in plist]); jj = np.array([nid.setdefault(p[1], len(nid)) for p in plist])
    pp = np.array([pid.setdefault(p, len(pid)) for p in plist])
    cnt = collections.Counter(P[:, 2].astype(int))
    print(f"panel {len(P)}: unlinked {cnt[0]} | causal {cnt[1]} | relation-month {cnt[2]} | sensitivity {cnt[3]} | relation-static {cnt[4]}\n")

    # A. persistence table (with mean trailing per category, per review)
    hi = P[P[:, 0] > a.tau]
    print(f"=== A: persistence of elevated corr (trailing > {a.tau}) by link source ===")
    print(f"  {'category':18} {'persists':>9} {'next':>7} {'trail':>7} {'same_sec%':>9} {'n':>7}")
    for cat, lbl in [(0,"unlinked"),(1,"causal"),(2,"relation-month"),(3,"sensitivity"),(4,"relation-static")]:
        r = hi[hi[:, 2] == cat]
        if not len(r): continue
        print(f"  {lbl:18} {float(np.mean(r[:,1]>=r[:,0])):9.0%} {r[:,1].mean():+7.3f} {r[:,0].mean():+7.3f} {r[:,3].mean():9.0%} {len(r):7}")

    # regression with review-mandated controls; then the sector KILL-CONDITION check
    y = P[:, 1] - P[:, 0]
    d_c = (P[:,2]==1).astype(float); d_rm = (P[:,2]==2).astype(float); d_s = (P[:,2]==3).astype(float); d_rs = (P[:,2]==4).astype(float)
    X1 = np.column_stack([np.ones(len(P)), P[:,0], P[:,4], d_c, d_rm, d_s, d_rs])
    report_both("\n=== RISE ~ trailing + logfreq + causal + relation-month + sensitivity + relation-static (NO sector) ===",
                X1, y, ii, jj, pp, ["intercept","trailing","logfreq","causal","relation_month","sensitivity","relation_static"])
    X2 = np.column_stack([X1, P[:,3]])
    report_both("\n=== same + same_sector (KILL CONDITION: sensitivity must survive) ===",
                X2, y, ii, jj, pp, ["intercept","trailing","logfreq","causal","relation_month","sensitivity","relation_static","same_sector"])

    # B1. direction agreement among causal-linked pairs with both legs net-directional
    bl = (P[:,2]==1) & (P[:,5]!=0) & (P[:,6]!=0)
    B = P[bl]; agree = (np.sign(B[:,5])==np.sign(B[:,6])).astype(float)
    print(f"\n=== B1: signed prediction among causal pairs, both legs directional (n={len(B)}; same-sign {int(agree.sum())}, opposite {int(len(B)-agree.sum())}) ===")
    print(f"  trailing dist: same-sign mean {B[agree==1,0].mean():+.3f}, opposite {B[agree==0,0].mean():+.3f}")
    Xb = np.column_stack([np.ones(len(B)), B[:,0], agree])
    report_both("  next_corr ~ trailing + same_sign", Xb, B[:,1], ii[bl], jj[bl], pp[bl], ["intercept","trailing","same_sign"])

    # B2. hedge integrity among negative-trailing pairs (abandon if N<100)
    for thr in (-0.15, -0.10):
        neg = P[P[:,0] < thr]
        linked_opp = ((neg[:,2]==1) & (np.sign(neg[:,5])!=np.sign(neg[:,6])) & (neg[:,5]!=0) & (neg[:,6]!=0))
        linked_any = neg[:,2] > 0
        print(f"\n=== B2: negative pairs trailing < {thr}: N={len(neg)} (linked-any {int(linked_any.sum())}, opposite-sign causal {int(linked_opp.sum())}) ===")
        if int(linked_any.sum()) < 100:
            print("  PRE-COMMITTED ABANDON: linked-N < 100 -- underpowered, not testing."); continue
        m = np.ones(len(neg), bool)
        Xn = np.column_stack([np.ones(len(neg)), neg[:,0], linked_any.astype(float), linked_any*neg[:,0]])
        idxs = np.where(P[:,0] < thr)[0]
        report_both("  next ~ trailing + linked + linked*trailing (attenuated reversion = interaction toward +)",
                    Xn, neg[:,1], ii[idxs], jj[idxs], pp[idxs], ["intercept","trailing","linked","linked_x_trailing"])
        print(f"  sign preservation P(next<0): linked {float(np.mean(neg[linked_any,1]<0)):.0%} vs unlinked {float(np.mean(neg[~linked_any,1]<0)):.0%}")
        break


if __name__ == "__main__":
    main()
