# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
HOBERG-PHILLIPS TNIC — a validated product-market network, and what it is good for.

WHAT TNIC IS, AND WHAT IT IS NOT. TNIC (Text-based Network Industry Classification,
Hoberg & Phillips, JPE 2016) scores every pair of US public firms on the vocabulary
overlap of their 10-K product descriptions. Pairs above a threshold are PRODUCT-MARKET
RIVALS. It is rebuilt every year, so the network moves as firms reposition.

It is NOT customer-supplier data. Cohen & Frazzini (2008) is about VERTICAL links —
your customer's news predicts your return because investors do not track the
relationship. TNIC is HORIZONTAL: competitors. Testing lead-lag along TNIC links
tests a different (and weaker-prior) hypothesis, because rivals are exactly the
relationship industry classification already makes obvious. This file does not
claim to test Cohen-Frazzini, and the register says so.

WHAT IT IS GENUINELY GOOD FOR HERE — the taxonomy question. Our sector layer has
been through three revisions: residual-overlap clustering (spurious), a ~140-name
hand-made GICS map (12% coverage), and SEC-assigned SIC (74%, but a 1987 vintage
that lumps all software into 7372). TNIC is the literature's answer, and it is
directly testable: if it is a better description of who competes with whom, then
TNIC rivals should co-move MORE in idiosyncratic space than same-SIC peers do.
That is measurable, and it is measured below.

IDENTIFIER CHAIN, with its caveat. TNIC ships gvkey only. We chain
    ticker -> CIK   (SEC company_tickers.json, already stored in sic_map.json)
    CIK    -> gvkey (public cik_gvkey crosswalk, Wenzhi-Ding/Std_Security_Code)
which links 507 of 547 names (93%). That crosswalk is DERIVED FROM WRDS TABLES and
its own README warns it is for exploration rather than publication without
independent verification. Spot checks pass (AAPL->1690, AAL->1045), but any result
built on it inherits that provenance caveat.

Usage:
    uv run scripts/tnic_links.py --build     # build the linked rival network
    uv run scripts/tnic_links.py --compare   # TNIC vs SIC vs correlation neighbours
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg100k_graph"
TNIC_TSV = Path("/tmp/tnic3_universe.tsv")
LINKMAP = Path("/tmp/ticker_gvkey.json")
OUT = G / "tnic_links.json"
TRAIL = 60
MIN_OVERLAP = 400
RNG = np.random.default_rng(59)


def build() -> None:
    tg = {t: int(v) for t, v in json.loads(LINKMAP.read_text()).items()}
    gt = collections.defaultdict(list)
    for t, g in tg.items():
        gt[g].append(t)
    rows = collections.defaultdict(list)   # year -> [(a,b,score)]
    n_self = n_bad = 0
    with open(TNIC_TSV) as fh:
        next(fh)
        for line in fh:
            p = line.rstrip("\r\n").split("\t")
            if len(p) != 4:
                n_bad += 1
                continue
            y, g1, g2, s = p
            # TNIC carries SELF-PAIRS (gvkey1 == gvkey2) with a BLANK score, as
            # presence markers for that firm-year. They are not rival links; parsing
            # them as 0.0 would seed the network with self-edges.
            if g1 == g2 or not s.strip():
                n_self += 1
                continue
            for a in gt.get(int(g1), ()):
                for b in gt.get(int(g2), ()):
                    if a != b:
                        rows[int(y)].append((a, b, float(s)))
    print(f"skipped {n_self} self/blank-score rows, {n_bad} malformed")
    payload = {"years": {str(y): [[a, b, round(s, 4)] for a, b, s in v]
                         for y, v in sorted(rows.items())},
               "n_tickers_linked": len(tg)}
    OUT.write_text(json.dumps(payload))
    print(f"wrote {OUT}")
    for y in sorted(rows):
        deg = collections.Counter()
        for a, b, _ in rows[y]:
            deg[a] += 1
        print(f"  {y}: {len(rows[y]):6} directed rival pairs, "
              f"{len(deg):3} names with >=1 rival, median degree {int(np.median(list(deg.values()) or [0]))}")


def load_idio():
    C = json.loads((G / "mcp_cache.json").read_text())
    days = C["days"]; n = len(days)
    idi = {}
    for tk, dec in C["decomp"].items():
        v = np.full(n, np.nan)
        for k, arr in dec.items():
            v[int(k)] = arr[3]
        if np.isfinite(v).sum() >= 400:
            idi[tk] = v
    return C, days, idi


def corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < MIN_OVERLAP:
        return None
    x, y = a[m], b[m]
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return None
    return float(((x - x.mean()) @ (y - y.mean())) / (len(x) * sx * sy))


def compare() -> None:
    """THE TAXONOMY TEST. If TNIC describes competition better than SIC, its pairs
    should co-move more in IDIOSYNCRATIC space — after market and sector are already
    removed, which is a demanding bar."""
    C, days, idi = load_idio()
    T = json.loads(OUT.read_text())
    sect = C["sectors"]

    tnic_pairs, tnic_score = set(), {}
    for y, rows in T["years"].items():
        if not (2008 <= int(y) <= 2014):
            continue
        for a, b, s in rows:
            if a in idi and b in idi:
                k = (min(a, b), max(a, b))
                tnic_pairs.add(k)
                tnic_score[k] = max(tnic_score.get(k, 0.0), s)

    names = sorted(idi)
    ind = {t: sect.get(t, {}).get("industry", "UNK") for t in names}
    mg = {t: sect.get(t, {}).get("major_group", "UNK") for t in names}
    corr_nb = set()
    for tk, nb in C["neighbours"].items():
        for t, _r, _s in nb[:10]:
            if tk in idi and t in idi:
                corr_nb.add((min(tk, t), max(tk, t)))

    # sample random pairs for a baseline
    rand = set()
    while len(rand) < 4000:
        a, b = names[RNG.integers(len(names))], names[RNG.integers(len(names))]
        if a != b:
            rand.add((min(a, b), max(a, b)))

    groups = {
        "TNIC rivals": tnic_pairs,
        "TNIC rivals (top-quartile score)": {k for k in tnic_pairs
                                             if tnic_score[k] >= np.percentile(
                                                 list(tnic_score.values()), 75)},
        "same SIC 4-digit industry": {(a, b) for a, b in rand | tnic_pairs | corr_nb
                                      if ind[a] == ind[b] and ind[a] != "UNK"},
        "same SIC major group": {(a, b) for a, b in rand | tnic_pairs | corr_nb
                                 if mg[a] == mg[b] and mg[a] != "UNK"},
        "our correlation neighbours": corr_nb,
        "random pairs": rand,
    }
    print(f"{'pair set':36}{'n':>7}{'mean |r|':>10}{'mean r':>9}{'>0.2':>8}")
    res = []
    cache = {}
    for label, S in groups.items():
        vals = []
        for k in sorted(S):
            if k not in cache:
                cache[k] = corr(idi[k[0]], idi[k[1]])
            c = cache[k]
            if c is not None:
                vals.append(c)
        if len(vals) < 30:
            continue
        v = np.array(vals)
        print(f"{label:36}{len(v):7}{np.abs(v).mean():10.4f}{v.mean():9.4f}"
              f"{(np.abs(v) > 0.2).mean():8.1%}")
        res.append({"set": label, "n": len(v), "mean_abs": float(np.abs(v).mean()),
                    "mean": float(v.mean()), "frac_gt_02": float((np.abs(v) > 0.2).mean())})

    # how much do TNIC and SIC actually disagree?
    both = tnic_pairs
    same_ind = sum(1 for a, b in both if ind[a] == ind[b] and ind[a] != "UNK")
    same_mg = sum(1 for a, b in both if mg[a] == mg[b] and mg[a] != "UNK")
    print(f"\nOf {len(both)} TNIC rival pairs in our universe:")
    print(f"  {same_ind:5} ({same_ind/len(both):.0%}) are also same 4-digit SIC industry")
    print(f"  {same_mg:5} ({same_mg/len(both):.0%}) are also same SIC major group")
    print(f"  -> TNIC identifies {len(both)-same_mg} rival pairs ({1-same_mg/len(both):.0%}) "
          f"that SIC puts in DIFFERENT major groups")
    ov = len(tnic_pairs & corr_nb)
    print(f"  {ov} TNIC pairs ({ov/max(len(corr_nb),1):.0%} of our correlation neighbours) "
          f"overlap our own link graph")
    out = G / "tnic_compare.json"
    out.write_text(json.dumps({"groups": res, "tnic_pairs": len(both),
                               "tnic_same_sic_industry": same_ind,
                               "tnic_same_sic_major": same_mg,
                               "overlap_with_corr_neighbours": ov}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--compare", action="store_true")
    a = ap.parse_args()
    if a.build:
        build()
    elif a.compare:
        compare()
    else:
        ap.print_help()
