# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
SAVOR (2012) — do price shocks WITH news continue, and shocks WITHOUT news reverse?

THE HYPOTHESIS, from the literature rather than from us. Savor, "Stock returns
after major price shocks: The impact of information" (JFE 2012), and Chan (2003)
before it: a large price move accompanied by identifiable news is an informed
move and CONTINUES; a large move with no news is uninformed order flow and
REVERSES. This is the form of anomaly our own tests never had: CONDITIONAL on the
news/no-news split, not a level effect of news on returns.

WHY IT IS THE RIGHT TEST FOR THIS DATASET. Every null in this project asked "does
news predict?" unconditionally, and the literature (Roll 1988; Cutler, Poterba &
Summers 1989) says that has always failed. Savor asks something we are unusually
well set up to answer, because 97.7% of our >4-sigma moves have NO linked news —
the supposedly-reversing group is the large one.

DESIGN.
    shock      signed idiosyncratic return at t, standardised by that name's own
               trailing 60-day idiosyncratic vol:  z_t = idi_t / sd_t
    forward    cumulative idiosyncratic return over t+1 .. t+k, in the same units,
               scaled by sqrt(k) so horizons are comparable
    aligned    y = sign(z_t) * forward   ->  POSITIVE = continuation
                                             NEGATIVE = reversal
    split      shocks with linked news vs shocks without
    estimand   the DIFFERENCE between those two groups

CONTROLS, each of which this project learned the hard way:
  - DEEP-COVERAGE DAYS ONLY. On a thinly-sampled day a "no news" label means "not
    sampled", so the no-news arm would be contaminated with news days and the two
    groups would converge by construction. The full-panel version is reported
    alongside purely to show that effect.
  - BLOCK BOOTSTRAP OVER DATES. Shocks on the same day are cross-sectionally
    correlated; treating them as independent would understate the standard error
    badly. Resampling whole dates keeps that dependence intact.
  - Nothing at or before t enters the forward window; sd_t uses only t-60..t-1.

Usage:
    uv run scripts/savor_news_noews.py --market us
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
THRESHOLDS = (2.0, 3.0, 4.0)
N_BOOT = 2000
RNG = np.random.default_rng(23)


def block_bootstrap(by_date: dict, dates: list, n_boot: int) -> tuple:
    """Resample whole DATES with replacement. Shocks on one day share a common
    component; treating them as independent observations would understate the
    standard error."""
    means = []
    d_arr = np.array(dates, dtype=object)
    for _ in range(n_boot):
        pick = d_arr[RNG.integers(0, len(d_arr), len(d_arr))]
        vals = [v for d in pick for v in by_date.get(d, ())]
        if vals:
            means.append(float(np.mean(vals)))
    if not means:
        return float("nan"), float("nan"), float("nan")
    m = np.array(means)
    return float(m.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", choices=sorted(GRAPHS), default="us")
    a = ap.parse_args()
    G = ROOT / GRAPHS[a.market]
    C = json.loads((G / "mcp_cache.json").read_text())
    days: list[str] = C["days"]
    n = len(days)
    print(f"market={a.market}  {n} trading days")

    dpd = collections.Counter()
    for l in open(G / "lake" / "document.jsonl"):
        d = (json.loads(l).get("published_at") or "")[:10]
        if d:
            dpd[d] += 1
    deep = {d for d in days if dpd.get(d, 0) >= 100}
    print(f"{len(deep)} deep-coverage days (>=100 documents)")

    has_news = set()
    for k in C["events"]:
        tk, d = k.split("|")
        has_news.add((tk, d))

    # signed idiosyncratic series per name
    obs = []          # (date, tier, has_news, z, {k: forward})
    for tk, dec in C["decomp"].items():
        v = np.full(n, np.nan)
        for k, arr in dec.items():
            v[int(k)] = arr[3]
        fin = np.isfinite(v)
        if fin.sum() < 400:
            continue
        for i in range(TRAIL, n - max(HORIZONS) - 1):
            if not fin[i]:
                continue
            w = v[i - TRAIL:i]
            w = w[np.isfinite(w)]
            if len(w) < 40:
                continue
            sd = float(w.std())
            if sd <= 1e-6:
                continue
            z = float(v[i]) / sd
            if abs(z) < min(THRESHOLDS):
                continue
            fwd = {}
            for h in HORIZONS:
                seg = v[i + 1:i + 1 + h]
                if np.isfinite(seg).sum() < h:
                    continue
                # scale by sqrt(h) so a k-day sum is comparable to a 1-day move
                fwd[h] = float(np.nansum(seg)) / (sd * np.sqrt(h))
            if not fwd:
                continue
            d = days[i]
            obs.append({"d": d, "deep": d in deep, "news": (tk, d) in has_news,
                        "z": z, "fwd": fwd})
    print(f"{len(obs)} shocks at |z| >= {min(THRESHOLDS)}\n")

    results = []
    for sample_name, keep in (("DEEP-coverage days only", lambda o: o["deep"]),
                              ("full panel (labels unreliable)", lambda o: True)):
        sub = [o for o in obs if keep(o)]
        if len(sub) < 200:
            continue
        print(f"=== {sample_name} — {len(sub)} shocks ===")
        print(f"{'|z|>':>5} {'h':>3} {'group':>8} {'n':>6} {'aligned fwd':>12} "
              f"{'95% CI':>20}   interpretation")
        for thr in THRESHOLDS:
            for h in HORIZONS:
                row = {"sample": sample_name, "thr": thr, "h": h}
                stats = {}
                for grp, want in (("news", True), ("no-news", False)):
                    sel = [o for o in sub if abs(o["z"]) >= thr and o["news"] is want
                           and h in o["fwd"]]
                    if len(sel) < 40:
                        stats[grp] = None
                        continue
                    by_date = collections.defaultdict(list)
                    for o in sel:
                        by_date[o["d"]].append(np.sign(o["z"]) * o["fwd"][h])
                    dts = sorted(by_date)
                    m, lo, hi = block_bootstrap(by_date, dts, N_BOOT)
                    stats[grp] = {"n": len(sel), "mean": m, "lo": lo, "hi": hi,
                                  "n_dates": len(dts)}
                    tag = ("CONTINUATION" if lo > 0 else
                           "REVERSAL" if hi < 0 else "indistinguishable from 0")
                    print(f"{thr:5.0f} {h:3} {grp:>8} {len(sel):6} {m:12.4f} "
                          f"[{lo:+.4f}, {hi:+.4f}]   {tag}")
                # difference in means, bootstrapped jointly over dates
                if stats.get("news") and stats.get("no-news"):
                    nd = collections.defaultdict(list); od = collections.defaultdict(list)
                    for o in sub:
                        if abs(o["z"]) < thr or h not in o["fwd"]:
                            continue
                        (nd if o["news"] else od)[o["d"]].append(
                            np.sign(o["z"]) * o["fwd"][h])
                    alld = sorted(set(nd) | set(od))
                    diffs = []
                    d_arr = np.array(alld, dtype=object)
                    for _ in range(N_BOOT):
                        pick = d_arr[RNG.integers(0, len(d_arr), len(d_arr))]
                        nv = [v for d in pick for v in nd.get(d, ())]
                        ov = [v for d in pick for v in od.get(d, ())]
                        if nv and ov:
                            diffs.append(float(np.mean(nv)) - float(np.mean(ov)))
                    if diffs:
                        dm = np.array(diffs)
                        dlo, dhi = np.percentile(dm, 2.5), np.percentile(dm, 97.5)
                        verdict = ("SAVOR EFFECT" if dlo > 0 else
                                   "OPPOSITE of Savor" if dhi < 0 else "no difference")
                        print(f"{'':5} {'':3} {'DIFF':>8} {'':6} {dm.mean():12.4f} "
                              f"[{dlo:+.4f}, {dhi:+.4f}]   <-- {verdict}")
                        row["diff"] = {"mean": float(dm.mean()), "lo": float(dlo),
                                       "hi": float(dhi), "verdict": verdict}
                row["stats"] = stats
                results.append(row)
            print()
        print()

    out = G / "savor_news_noews.json"
    out.write_text(json.dumps({"market": a.market, "n_shocks": len(obs),
                               "n_deep_days": len(deep), "results": results},
                              indent=1, default=str))
    print(f"wrote {out}")
    print("\naligned fwd = sign(shock) x forward idiosyncratic return, in trailing-vol")
    print("units. POSITIVE = the move continued. NEGATIVE = it reversed.")
    print("Savor predicts news shocks continue MORE than no-news shocks, i.e. DIFF > 0.")


if __name__ == "__main__":
    main()
