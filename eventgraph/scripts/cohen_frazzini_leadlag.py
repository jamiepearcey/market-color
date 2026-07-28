# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
COHEN & FRAZZINI (2008) — does news about A predict a LINKED firm B tomorrow?

THE HYPOTHESIS, from the literature. "Economic Links and Predictable Returns"
(JF 2008): investors are inattentive to firms connected to the one in the news, so
information diffuses with a LAG along economic links. Menzly & Ozbas (2010) extend
it to industries. The published effect is large — roughly 1.5%/month long-short on
a customer-momentum portfolio.

WHY THIS IS THE RIGHT NEXT TEST.
  - It is a LEAD-LAG test. Everything this project has measured is contemporaneous,
    and the contemporaneous result is the one that survived. Diffusion is a
    different axis entirely and has never been tried here.
  - It uses the link graph we already built and validated, and Cohen-Frazzini's
    mechanism is INATTENTION — so the effect should be STRONGER on links that are
    not obvious, i.e. the 29% that cross a SIC industry boundary. That gives a
    directional prediction, not just a level test, which is much harder to fake.
  - Power is far better than the Savor test: each news event generates a
    prediction for every neighbour, so n multiplies rather than divides.

A DESIGN NOTE THAT BUYS POWER. This test only needs days where A HAS news. A
positive news label is reliable on any day — only ABSENCE is unreliable, because a
thin day cannot distinguish "no news" from "not sampled". So unlike the Savor and
predictive tests, the full panel is usable here, not just the 96 deep days.

DESIGN.
    trigger    (A, t) with linked news AND |z_A| >= THR   (a real move to diffuse)
    target     each neighbour B with NO news at t         (diffusion, not co-news)
    y          sign(z_A) * forward idiosyncratic return of B over t+1..t+h,
               in units of B's own trailing vol
    control    the SAME triggers paired with RANDOM non-neighbour firms, which
               absorbs any market-wide drift after news days
    split      same-industry vs CROSS-industry links (the inattention prediction)

Block-bootstrapped over dates: many pairs share a trigger day and are strongly
cross-correlated, so treating them as independent would badly understate the SE.

MDE is reported first, because a null from an underpowered test is not a null —
that lesson cost this project a nearly-filed false rejection on Savor.

Usage:
    uv run scripts/cohen_frazzini_leadlag.py --market us
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
GRAPHS = {"us": "eg100k_graph", "india": "india2021"}
TRAIL = 60
HORIZONS = (1, 5, 21)
THR = 1.5            # A must actually have moved for there to be anything to diffuse
N_BOOT = 2000
N_RANDOM_CTRL = 6    # random non-neighbour partners per trigger
RNG = np.random.default_rng(41)


def boot_mean(by_date: dict, n_boot: int = N_BOOT):
    d = np.array(sorted(by_date), dtype=object)
    if len(d) < 5:
        return None
    ms = []
    for _ in range(n_boot):
        pick = d[RNG.integers(0, len(d), len(d))]
        v = [x for k in pick for x in by_date.get(k, ())]
        if v:
            ms.append(float(np.mean(v)))
    if not ms:
        return None
    m = np.array(ms)
    return {"mean": float(m.mean()), "lo": float(np.percentile(m, 2.5)),
            "hi": float(np.percentile(m, 97.5)),
            "se": float((np.percentile(m, 97.5) - np.percentile(m, 2.5)) / 3.92)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", choices=sorted(GRAPHS), default="us")
    ap.add_argument("--links", choices=("corr", "tnic"), default="corr",
                    help="corr = our idiosyncratic-correlation neighbours; "
                         "tnic = Hoberg-Phillips product-market RIVALS (a different "
                         "hypothesis: rivals are horizontal, Cohen-Frazzini is vertical)")
    a = ap.parse_args()
    G = ROOT / GRAPHS[a.market]
    C = json.loads((G / "mcp_cache.json").read_text())
    days: list[str] = C["days"]; n = len(days)
    di = {d: i for i, d in enumerate(days)}
    print(f"market={a.market}  {n} trading days")

    news_cells = set()
    for k in C["events"]:
        tk, d = k.split("|")
        news_cells.add((tk, d))

    # signed idiosyncratic series + trailing vol per name
    idi, sdv = {}, {}
    for tk, dec in C["decomp"].items():
        v = np.full(n, np.nan)
        for k, arr in dec.items():
            v[int(k)] = arr[3]
        if np.isfinite(v).sum() < 400:
            continue
        sd = np.full(n, np.nan)
        for i in range(TRAIL, n):
            w = v[i - TRAIL:i]; w = w[np.isfinite(w)]
            if len(w) >= 40:
                s = float(w.std())
                if s > 1e-6:
                    sd[i] = s
        idi[tk] = v; sdv[tk] = sd
    print(f"{len(idi)} names with idiosyncratic series")

    sect = C["sectors"]
    if a.links == "tnic":
        # TNIC rivals, per YEAR — the network is rebuilt annually, so a link is only
        # used on days inside the year it was estimated for. Using a pooled network
        # would leak later product-market structure into earlier dates.
        T = json.loads((G / "tnic_links.json").read_text())
        by_year = collections.defaultdict(lambda: collections.defaultdict(list))
        for y, rows in T["years"].items():
            for x, b, sc in rows:
                if x in idi and b in idi:
                    by_year[int(y)][x].append((b, float(sc),
                                               sect.get(x, {}).get("industry") ==
                                               sect.get(b, {}).get("industry")))
        def neigh_on(tk, date):
            return by_year.get(int(date[:4]), {}).get(tk, ())[:10]
        n_links = sum(len(v) for yr in by_year.values() for v in yr.values())
        print(f"TNIC rival links: {n_links} across {len(by_year)} years")
    else:
        _nb = {tk: [(t, r, bool(same)) for t, r, same in v[:10]]
               for tk, v in C["neighbours"].items() if tk in idi}
        def neigh_on(tk, date):
            return _nb.get(tk, ())
    neigh = None  # use neigh_on(ticker, date) instead
    allnames = sorted(idi)


    def fwd(tk, i, h):
        v, sd = idi[tk], sdv[tk]
        if not np.isfinite(sd[i]):
            return None
        seg = v[i + 1:i + 1 + h]
        if len(seg) < h or np.isfinite(seg).sum() < h:
            return None
        return float(np.nansum(seg)) / (sd[i] * np.sqrt(h))

    # buckets[(arm, h)][date] -> list of aligned forward returns
    buckets = collections.defaultdict(lambda: collections.defaultdict(list))
    n_trig = 0
    for (A, d) in sorted(news_cells):
        if A not in idi or d not in di:
            continue
        i = di[d]
        if i < TRAIL + 5 or i + max(HORIZONS) + 1 >= n:
            continue
        if not np.isfinite(idi[A][i]) or not np.isfinite(sdv[A][i]):
            continue
        zA = float(idi[A][i]) / float(sdv[A][i])
        if abs(zA) < THR:
            continue
        n_trig += 1
        s = np.sign(zA)
        for (B, r, same) in neigh_on(A, d):
            if B not in idi or (B, d) in news_cells:   # B must NOT have its own news
                continue
            for h in HORIZONS:
                f = fwd(B, i, h)
                if f is None:
                    continue
                buckets[("linked", h)][d].append(s * f)
                buckets[("same-ind" if same else "cross-ind", h)][d].append(s * f)
        # random non-neighbour controls, same trigger day
        for _ in range(N_RANDOM_CTRL):
            B = allnames[int(RNG.integers(0, len(allnames)))]
            if B == A or B in {x for x, _, _ in neigh_on(A, d)} or (B, d) in news_cells:
                continue
            for h in HORIZONS:
                f = fwd(B, i, h)
                if f is not None:
                    buckets[("random", h)][d].append(s * f)
    print(f"{n_trig} trigger events (news on A, |z_A| >= {THR})\n")

    print(f"{'arm':>10} {'h':>3} {'pairs':>7} {'dates':>6} {'aligned fwd':>12} "
          f"{'95% CI':>20} {'MDE':>7}   interpretation")
    res = []
    for h in HORIZONS:
        for arm in ("linked", "same-ind", "cross-ind", "random"):
            bd = buckets.get((arm, h))
            if not bd:
                continue
            npairs = sum(len(v) for v in bd.values())
            st = boot_mean(bd)
            if not st:
                continue
            mde = 2.80 * st["se"]
            tag = ("DIFFUSION (B follows A)" if st["lo"] > 0 else
                   "B moves AGAINST A" if st["hi"] < 0 else "indistinguishable from 0")
            print(f"{arm:>10} {h:3} {npairs:7} {len(bd):6} {st['mean']:12.4f} "
                  f"[{st['lo']:+.4f}, {st['hi']:+.4f}] {mde:7.3f}   {tag}")
            res.append({"arm": arm, "h": h, "pairs": npairs, "dates": len(bd),
                        **st, "mde": mde, "tag": tag})
        # linked minus random, bootstrapped jointly
        L, Rn = buckets.get(("linked", h)), buckets.get(("random", h))
        if L and Rn:
            alld = np.array(sorted(set(L) | set(Rn)), dtype=object)
            ds = []
            for _ in range(N_BOOT):
                pick = alld[RNG.integers(0, len(alld), len(alld))]
                lv = [x for k in pick for x in L.get(k, ())]
                rv = [x for k in pick for x in Rn.get(k, ())]
                if lv and rv:
                    ds.append(float(np.mean(lv)) - float(np.mean(rv)))
            if ds:
                dm = np.array(ds); lo, hi = np.percentile(dm, [2.5, 97.5])
                v = ("LINK EFFECT" if lo > 0 else "reverse" if hi < 0 else "no link effect")
                print(f"{'LINK-RAND':>10} {h:3} {'':7} {'':6} {dm.mean():12.4f} "
                      f"[{lo:+.4f}, {hi:+.4f}] {2.80*(hi-lo)/3.92:7.3f}   <-- {v}")
                res.append({"arm": "linked_minus_random", "h": h, "mean": float(dm.mean()),
                            "lo": float(lo), "hi": float(hi), "tag": v})
        # the inattention prediction: cross-industry should be STRONGER
        Sm, Cr = buckets.get(("same-ind", h)), buckets.get(("cross-ind", h))
        if Sm and Cr:
            alld = np.array(sorted(set(Sm) | set(Cr)), dtype=object)
            ds = []
            for _ in range(N_BOOT):
                pick = alld[RNG.integers(0, len(alld), len(alld))]
                cv = [x for k in pick for x in Cr.get(k, ())]
                sv = [x for k in pick for x in Sm.get(k, ())]
                if cv and sv:
                    ds.append(float(np.mean(cv)) - float(np.mean(sv)))
            if ds:
                dm = np.array(ds); lo, hi = np.percentile(dm, [2.5, 97.5])
                v = ("INATTENTION PATTERN" if lo > 0 else
                     "opposite" if hi < 0 else "no difference")
                print(f"{'CROSS-SAME':>10} {h:3} {'':7} {'':6} {dm.mean():12.4f} "
                      f"[{lo:+.4f}, {hi:+.4f}] {2.80*(hi-lo)/3.92:7.3f}   <-- {v}")
                res.append({"arm": "cross_minus_same", "h": h, "mean": float(dm.mean()),
                            "lo": float(lo), "hi": float(hi), "tag": v})
        print()

    out = G / f"cohen_frazzini_leadlag_{a.links}.json"
    out.write_text(json.dumps({"market": a.market, "links": a.links, "n_triggers": n_trig,
                               "threshold": THR, "results": res}, indent=1))
    print(f"wrote {out}")
    print("\naligned fwd = sign(A's move) x B's forward idiosyncratic return, in B's own")
    print("trailing-vol units. POSITIVE = B follows A with a lag (information diffusion).")
    print("'random' is the same trigger days paired with NON-linked firms, so it absorbs")
    print("any market-wide drift after news days. The link effect is LINKED minus RANDOM.")


if __name__ == "__main__":
    main()
