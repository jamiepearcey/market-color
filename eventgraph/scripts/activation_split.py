# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
ACTIVATION SPLIT — pre-registered test of the 'activation, not structure' thesis.

The graph treats every extracted cause as meaningful; markets don't. Hypothesis: the
surviving persistence signal is carried by ACTIVATED links (the market visibly reacted
on the edge day: volume spike / abnormal move) and DORMANT links are noise.

  activation(edge s,day) = vol_z(s,day) >= 2  OR  |ar_z(s,day)| >= 2
    vol_z: log-volume z vs trailing 60 obs (volume arrays already in the cached Yahoo
    chart JSONs). ar_z: abnormal return / trailing 60d abnormal-return std.
  link tag: act1 = at least one leg confirmed on a contributing edge day; act2 = both.

PRE-REGISTERED PREDICTIONS (before running):
  P1 persistence(linked-activated) >> persistence(linked-dormant) ~= persistence(unlinked)
  P2 direction hit-rate higher on volume-confirmed news days (volume is direction-blind
     -> non-circular for direction)

Primary target = NEXT-month correlation (activation measured day-of, month m -> no
target contamination). Dyadic SEs on the regression.

Usage: uv run eventgraph/scripts/activation_split.py --graph-dir /tmp/eg100k_graph --topk 160
"""
import argparse, json, collections, csv, sys, math, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cross_sectional_ic import yahoo, logret, wls, FN, DIR, SKIP
from news_contagion import ar_series
from news_covariance import months_between, corr_matrix
from news_latent import dyadic_ses, report_both


def volume_z_series(symn, cache):
    """day -> log-volume z vs trailing 60 obs, from the already-cached Yahoo chart JSON."""
    p = cache / f"{symn}.json"
    if not p.exists(): return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]; vol = (res["indicators"]["quote"][0].get("volume") or [])
    except Exception:
        return {}
    days, lv = [], []
    for t, v in zip(ts, vol):
        if v:
            days.append(dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d")); lv.append(math.log(v))
    out = {}
    for i in range(60, len(days)):
        w = np.array(lv[i-60:i]); sd = w.std()
        if sd > 0: out[days[i]] = (lv[i] - w.mean()) / sd
    return out


def ar_z_series(ar):
    """day -> abnormal-return z vs trailing 60 abnormal obs."""
    days = sorted(ar); vals = [ar[d] for d in days]; out = {}
    for i in range(60, len(days)):
        w = np.array(vals[i-60:i]); sd = w.std()
        if sd > 0: out[days[i]] = vals[i] / sd
    return out


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

    # edges: (month, cause) -> {effect sym: set(days)}; plus per-(sym,day) net direction for P2
    ce_days = collections.defaultdict(lambda: collections.defaultdict(set))
    sig = collections.Counter(); freq = collections.Counter()
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); d = docdate.get(j.get("doc_id")); e = j.get("effect_entity"); c = j.get("cause_entity")
        if not d or d[:4] not in years or e not in sym or j.get("effect_dir") not in DIR: continue
        s = sym[e]; freq[s] += 1
        sig[(s, d[:10])] += DIR[j["effect_dir"]]
        if c: ce_days[d[:7]][c].add((s, d[:10]))

    print(f"building series for top {a.topk} names ...")
    AR, VZ, ARZ = {}, {}, {}
    for s, _ in freq.most_common():
        if len(AR) >= a.topk: break
        v = ar_series(s, cache, Fmap, years, p1, p2)
        if len(v) > 150:
            AR[s] = v; VZ[s] = volume_z_series(s, cache); ARZ[s] = ar_z_series(v)
    names = sorted(AR); U = set(names)
    nvol = sum(1 for s in names if VZ[s]); print(f"{len(names)} names ({nvol} with volume)\n")

    def leg_confirmed(s, day):
        vz = VZ.get(s, {}).get(day); az = ARZ.get(s, {}).get(day)
        if vz is None and az is None: return None  # unknown
        return (vz is not None and vz >= 2) or (az is not None and abs(az) >= 2)

    # month -> pair -> activation level: for each shared-cause pair record per-leg confirmation
    pair_act = collections.defaultdict(dict)  # month -> (i,j) -> (leg_i_conf, leg_j_conf) best-of
    for m, ce in ce_days.items():
        for c, sd in ce.items():
            by_sym = collections.defaultdict(set)
            for s, day in sd:
                if s in U: by_sym[s].add(day)
            ss = sorted(by_sym)
            for x in range(len(ss)):
                for y in range(x+1, len(ss)):
                    i, j = ss[x], ss[y]
                    ci = any((leg_confirmed(i, d) or False) for d in by_sym[i])
                    cj = any((leg_confirmed(j, d) or False) for d in by_sym[j])
                    prev = pair_act[m].get((i, j), (False, False))
                    pair_act[m][(i, j)] = (prev[0] or ci, prev[1] or cj)

    allmonths = months_between(min(int(y) for y in years), max(int(y) for y in years)); idx = {m: i for i, m in enumerate(allmonths)}
    rows = []; plist = []
    for m in allmonths:
        i = idx[m]
        if i < a.trail or i + a.nxt >= len(allmonths): continue
        trail = corr_matrix(AR, names, allmonths[i-a.trail:i+1]); nxt = corr_matrix(AR, names, allmonths[i+1:i+1+a.nxt])
        pa = pair_act.get(m, {})
        for pair, tc in trail.items():
            if pair not in nxt: continue
            act = pa.get(pair)
            if act is None: cat = 0            # unlinked
            elif act[0] and act[1]: cat = 3    # both legs confirmed
            elif act[0] or act[1]: cat = 2     # one leg confirmed
            else: cat = 1                      # linked but dormant
            rows.append((tc, nxt[pair], cat)); plist.append(pair)
    P = np.array(rows)
    nid, pid = {}, {}
    ii = np.array([nid.setdefault(p[0], len(nid)) for p in plist])
    jj = np.array([nid.setdefault(p[1], len(nid)) for p in plist])
    pp = np.array([pid.setdefault(p, len(pid)) for p in plist])
    cnt = collections.Counter(P[:, 2].astype(int))
    print(f"panel {len(P)} pair-months: unlinked {cnt[0]}, dormant {cnt[1]}, act1 {cnt[2]}, act2 {cnt[3]}\n")

    # P1 — persistence among elevated pairs
    hi = P[P[:, 0] > a.tau]
    print(f"=== P1: persistence of elevated correlations (trailing > {a.tau}) ===")
    print(f"  {'category':22} {'persists':>9} {'next_corr':>10} {'from':>8} {'n':>7}")
    for cat, lbl in [(0, "unlinked"), (1, "linked-DORMANT"), (2, "linked-ACT (one leg)"), (3, "linked-ACT (both)")]:
        r = hi[hi[:, 2] == cat]
        if not len(r): continue
        print(f"  {lbl:22} {float(np.mean(r[:,1] >= r[:,0])):9.0%} {r[:,1].mean():+10.3f} {r[:,0].mean():+8.3f} {len(r):7}")

    # regression with dyadic SEs: RISE ~ trailing + dormant + act(one) + act(both)
    y = P[:, 1] - P[:, 0]
    X = np.column_stack([np.ones(len(P)), P[:, 0], (P[:,2]==1).astype(float), (P[:,2]==2).astype(float), (P[:,2]==3).astype(float)])
    report_both("\n=== RISE ~ trailing + dormant + activated(one leg) + activated(both legs) ===",
                X, y, ii, jj, pp, ["intercept","trailing","linked_dormant","linked_act1","linked_act2"])

    # P2 — direction hit-rate by volume confirmation (volume is direction-blind)
    print("\n=== P2: direction hit-rate on news days, by volume confirmation ===")
    buckets = {"vol-confirmed (vz>=2)": [], "unconfirmed (vz<2)": []}
    for (s, day), net in sig.items():
        if s not in U or net == 0: continue
        az = ARZ.get(s, {}).get(day); vz = VZ.get(s, {}).get(day)
        if az is None or vz is None: continue
        k = "vol-confirmed (vz>=2)" if vz >= 2 else "unconfirmed (vz<2)"
        buckets[k].append((np.sign(net) == np.sign(az), np.sign(net) * az))
    for k, v in buckets.items():
        if not v: continue
        hits = np.array([h for h, _ in v]); dz = np.array([z for _, z in v])
        t = dz.mean() / (dz.std(ddof=1) / math.sqrt(len(dz))) if len(dz) > 5 else float("nan")
        print(f"  {k:24} hit {hits.mean():.0%}   mean dir*z {dz.mean():+.3f} (t {t:+.1f})   n={len(v)}")


if __name__ == "__main__":
    main()
