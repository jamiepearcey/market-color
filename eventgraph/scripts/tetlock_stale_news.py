# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
TETLOCK (2011) — do investors overreact to STALE news?

THE HYPOTHESIS. "All the News That's Fit to Reprint: Do Investors React to Stale
Information?" (RFS 2011). Some stories mostly repeat information already published
about the same firm. If investors were processing information correctly, a story
carrying no new content should move the price very little. Tetlock finds they do
react — and that the reaction then REVERSES, because there was nothing new in it.
Fresh news, by contrast, should not reverse.

WHY THIS ONE IS WORTH RUNNING. It is the only remaining candidate on the reading
list that needs NO new data — just the chunk text we already store — and it probes
a failure mode nothing else here has: OVERREACTION. Every previous test asked
whether news carries information. This asks whether the market misprices news that
carries none, which can be true even when all the information tests are null.

MEASURE.
    staleness   max cosine similarity between this story's text and the text of
                any PRIOR story about the SAME firm within LOOKBACK trading days,
                on hashed word n-grams
    fresh       low staleness; stale = high staleness (tercile split)
    outcome     sign(idiosyncratic move at t) x forward idiosyncratic return,
                in the name's own trailing-vol units
                POSITIVE = the reaction persisted.  NEGATIVE = it reversed.

Tetlock predicts the STALE tercile reverses (negative) and the FRESH tercile does
not, so the estimand is STALE minus FRESH, expected NEGATIVE.

Two things reported before the verdict, both learned the hard way here:
  - MDE against the published effect, so a null from an underpowered test is not
    mistaken for evidence of absence;
  - the CONTEMPORANEOUS reaction by tercile, because if stale and fresh news
    produce different-sized same-day moves then the groups differ in something
    other than staleness and the comparison is confounded.

Usage:
    uv run scripts/tetlock_stale_news.py --market us
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel  # noqa: E402

LOOKBACK = 125       # trading days to search for a prior story about the same firm
HORIZONS = (1, 5, 21)
DIM = 2 ** 15
N_BOOT = 2000
RNG = np.random.default_rng(77)
WORD = re.compile(r"[a-z][a-z']+")
STOP = set("""the a an and or but if of to in on for with at by from as is are was were be been
being this that these those it its his her their our your not no than then there here have has
had do does did will would could should may might must can said says say new inc corp co ltd
company companies percent said reuters bloomberg""".split())


def vec(text: str) -> dict:
    """Hashed bag of unigrams+bigrams, L2-normalised. Cheap, and adequate: we only
    need relative similarity between stories about the same firm."""
    toks = [w for w in WORD.findall(text.lower()) if w not in STOP and len(w) > 2]
    if len(toks) < 12:
        return {}
    counts: dict[int, float] = collections.defaultdict(float)
    for t in toks:
        counts[hash(t) % DIM] += 1.0
    for i in range(len(toks) - 1):
        counts[hash(toks[i] + "_" + toks[i + 1]) % DIM] += 1.0
    norm = np.sqrt(sum(v * v for v in counts.values()))
    if norm <= 0:
        return {}
    return {k: v / norm for k, v in counts.items()}


def cos(a: dict, b: dict) -> float:
    if not a or not b:
        return 0.0
    if len(a) > len(b):
        a, b = b, a
    return float(sum(v * b.get(k, 0.0) for k, v in a.items()))


def boot(by_date: dict, n_boot: int = N_BOOT):
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
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(m.mean()), "lo": float(lo), "hi": float(hi),
            "se": float((hi - lo) / 3.92)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default="us", choices=("us", "india"))
    a = ap.parse_args()
    P = panel.load(a.market)
    n = len(P.days)
    print(f"market={a.market}  {n} trading days")

    # signed idiosyncratic series + trailing vol
    idi, sdv = {}, {}
    for tk, dec in P.decomp.items():
        v = np.full(n, np.nan)
        for k, arr in dec.items():
            v[int(k)] = arr[3]
        if np.isfinite(v).sum() < 400:
            continue
        sd = np.full(n, np.nan)
        for i in range(60, n):
            w = v[i - 60:i]; w = w[np.isfinite(w)]
            if len(w) >= 40:
                s = float(w.std())
                if s > 1e-6:
                    sd[i] = s
        idi[tk], sdv[tk] = v, sd

    # story text per (ticker, date), from the cached documents
    C = json.loads((P.graph / "mcp_cache.json").read_text())
    stories = collections.defaultdict(list)   # ticker -> [(day_index, vector)]
    cell_vec = {}
    for k, v in C["events"].items():
        tk, d = k.split("|")
        i = P.day_index.get(d)
        if i is None or tk not in idi:
            continue
        txt = " ".join(" ".join(C["docs"][x]["chunks"]) for x in v["docs"]
                       if x in C["docs"])
        w = vec(txt)
        if not w:
            continue
        cell_vec[(tk, d)] = w
        stories[tk].append((i, w))
    for v in stories.values():
        v.sort()
    print(f"{len(cell_vec)} event cells with usable text")

    rows = []
    for (tk, d), w in sorted(cell_vec.items()):
        i = P.day_index[d]
        prior = [(j, wv) for j, wv in stories[tk] if 0 < i - j <= LOOKBACK]
        if not prior:
            continue
        stale = max(cos(w, wv) for _, wv in prior)
        if not np.isfinite(idi[tk][i]) or not np.isfinite(sdv[tk][i]):
            continue
        z = float(idi[tk][i]) / float(sdv[tk][i])
        f = {}
        for h in HORIZONS:
            seg = idi[tk][i + 1:i + 1 + h]
            if len(seg) == h and np.isfinite(seg).sum() == h:
                f[h] = float(np.nansum(seg)) / (float(sdv[tk][i]) * np.sqrt(h))
        if f:
            rows.append({"tk": tk, "d": d, "stale": stale, "z": z,
                         "n_prior": len(prior), "fwd": f})
    print(f"{len(rows)} events with a prior story within {LOOKBACK}d\n")
    if len(rows) < 200:
        raise SystemExit("too few to test")

    sv = np.array([r["stale"] for r in rows])
    q1, q2 = np.percentile(sv, [33.3, 66.7])
    print(f"staleness terciles: fresh < {q1:.3f} <= mid < {q2:.3f} <= stale")
    groups = {"FRESH": [r for r in rows if r["stale"] < q1],
              "MID": [r for r in rows if q1 <= r["stale"] < q2],
              "STALE": [r for r in rows if r["stale"] >= q2]}

    # CONFOUND CHECK — do the terciles differ in same-day reaction size?
    print(f"\n{'tercile':>8}{'n':>6}{'mean staleness':>16}{'|same-day z|':>14}"
          f"{'mean n_prior':>14}")
    for g, rs in groups.items():
        print(f"{g:>8}{len(rs):6}{np.mean([r['stale'] for r in rs]):16.3f}"
              f"{np.mean([abs(r['z']) for r in rs]):14.3f}"
              f"{np.mean([r['n_prior'] for r in rs]):14.1f}")

    print(f"\n{'tercile':>8}{'h':>3}{'n':>6}{'aligned fwd':>13}{'95% CI':>22}{'MDE':>8}"
          "   interpretation")
    res = []
    for h in HORIZONS:
        st = {}
        for g, rs in groups.items():
            bd = collections.defaultdict(list)
            for r in rs:
                if h in r["fwd"]:
                    bd[r["d"]].append(np.sign(r["z"]) * r["fwd"][h])
            b = boot(bd)
            if not b:
                continue
            st[g] = (b, bd)
            tag = ("PERSISTS" if b["lo"] > 0 else "REVERSES" if b["hi"] < 0
                   else "indistinguishable from 0")
            print(f"{g:>8}{h:3}{sum(len(v) for v in bd.values()):6}{b['mean']:13.4f}"
                  f"  [{b['lo']:+.4f}, {b['hi']:+.4f}]{2.80*b['se']:8.3f}   {tag}")
            res.append({"tercile": g, "h": h, **b})
        if "STALE" in st and "FRESH" in st:
            sd_, fd_ = st["STALE"][1], st["FRESH"][1]
            alld = np.array(sorted(set(sd_) | set(fd_)), dtype=object)
            ds = []
            for _ in range(N_BOOT):
                pick = alld[RNG.integers(0, len(alld), len(alld))]
                x = [v for k in pick for v in sd_.get(k, ())]
                y = [v for k in pick for v in fd_.get(k, ())]
                if x and y:
                    ds.append(float(np.mean(x)) - float(np.mean(y)))
            if ds:
                dm = np.array(ds); lo, hi = np.percentile(dm, [2.5, 97.5])
                v = ("TETLOCK EFFECT (stale reverses relative to fresh)" if hi < 0 else
                     "opposite of Tetlock" if lo > 0 else "no difference")
                print(f"{'ST-FR':>8}{h:3}{'':6}{dm.mean():13.4f}  [{lo:+.4f}, {hi:+.4f}]"
                      f"{2.80*(hi-lo)/3.92:8.3f}   <-- {v}")
                res.append({"tercile": "stale_minus_fresh", "h": h,
                            "mean": float(dm.mean()), "lo": float(lo), "hi": float(hi),
                            "verdict": v})
        print()

    out = P.graph / "tetlock_stale_news.json"
    out.write_text(json.dumps({"market": a.market, "lookback": LOOKBACK,
                               "n_rows": len(rows), "results": res}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
