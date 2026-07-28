# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
STEP 2 — score the LLM's exposure hypotheses against what actually happened.

This is the FIRST script in the test to load a price. `llm_exposure_dump.py`
emitted the events and the universe with no prices; the predictions in
`data/llm_exposure_predictions.json` were written from those alone, by a model
whose knowledge cutoff (May 2026) predates the event window (2026-06-27..07-19).
So the reasoning could not have been fitted to the outcome.

MEASURE. For each name, the daily return residualised on the equal-weight
universe return (a one-factor market removal), standardised by that name's own
trailing 60-day residual vol. |z| is therefore "how unusual was this move FOR THIS
NAME", which is the right scale when the candidate lists differ in raw volatility.

TESTS.
  1. most_exposed vs least_exposed  (the pre-registered prediction)
  2. most_exposed vs the whole universe
  3. permutation null: draw random same-size baskets from the universe and see
     where the observed gap falls. Non-parametric, so the micro-cap skew in the
     least_exposed list cannot manufacture significance.

Usage:
    uv run scripts/llm_exposure_score.py
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg_live2"
PRED = Path(__file__).resolve().parent.parent / "data" / "llm_exposure_predictions.json"
RNG = np.random.default_rng(11)
TRAIL = 60


def load_prices():
    out = {}
    pdir = G / "prices"
    for f in sorted(os.listdir(pdir)):
        if not f.endswith(".json"):
            continue
        try:
            j = json.load(open(pdir / f))
        except Exception:
            continue
        r = (j.get("chart") or {}).get("result")
        if not r:
            continue
        ts = r[0].get("timestamp") or []
        cl = r[0]["indicators"]["quote"][0].get("close") or []
        pts = [(dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat(), c)
               for t, c in zip(ts, cl) if c is not None]
        if len(pts) < 400:
            continue
        tail = [c for _, c in pts[-60:]]
        z = sum(1 for i in range(1, len(tail)) if abs(tail[i] - tail[i - 1]) < 1e-9) / max(1, len(tail) - 1)
        if z >= 0.15:
            continue
        out[f[:-5]] = dict(pts)
    return out


def main() -> None:
    P = json.load(open(PRED))
    px = load_prices()
    days = sorted({d for s in px.values() for d in s})
    idx = {d: i for i, d in enumerate(days)}

    # returns matrix
    rets = {}
    for t, s in px.items():
        ds = sorted(s)
        r = {}
        for i in range(1, len(ds)):
            a, b = s[ds[i - 1]], s[ds[i]]
            if a and b:
                r[ds[i]] = (b - a) / a
        rets[t] = r
    uni = sorted(rets)

    # equal-weight market, then residual, then standardise by own trailing vol
    mkt = {}
    for d in days:
        v = [rets[t][d] for t in uni if d in rets[t]]
        if len(v) >= 30:
            mkt[d] = float(np.mean(v))
    resid = {}
    for t in uni:
        ds = [d for d in sorted(rets[t]) if d in mkt]
        if len(ds) < TRAIL + 10:
            continue
        y = np.array([rets[t][d] for d in ds])
        x = np.array([mkt[d] for d in ds])
        beta = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else 0.0
        e = y - beta * x
        z = {}
        for i in range(TRAIL, len(ds)):
            sd = e[i - TRAIL:i].std()
            if sd > 0:
                z[ds[i]] = abs(e[i]) / sd
        resid[t] = z

    w0, w1 = P["window"]
    win = [d for d in days if w0 <= d <= w1]
    print(f"universe {len(resid)} names · window {w0}..{w1} ({len(win)} sessions)")

    def score(tickers):
        v = [resid[t][d] for t in tickers if t in resid for d in win if d in resid[t]]
        return (float(np.mean(v)), len(v)) if v else (float("nan"), 0)

    most = [x["t"] for x in P["most_exposed"]]
    least = [x["t"] for x in P["least_exposed"]]
    miss = [t for t in most + least if t not in resid]
    if miss:
        print(f"  (no usable residual for: {', '.join(miss)})")
    most = [t for t in most if t in resid]
    least = [t for t in least if t in resid]

    m_s, m_n = score(most)
    l_s, l_n = score(least)
    u_s, u_n = score(list(resid))
    print(f"\n=== MEAN |standardised abnormal move| over the window")
    print(f"  most_exposed  ({len(most):>2} names, {m_n:>4} obs)   {m_s:.3f}")
    print(f"  least_exposed ({len(least):>2} names, {l_n:>4} obs)   {l_s:.3f}")
    print(f"  whole universe({len(resid):>3} names, {u_n:>4} obs)   {u_s:.3f}")
    print(f"\n  gap most-least   {m_s - l_s:+.3f}")
    print(f"  gap most-universe{m_s - u_s:+.3f}")

    # permutation null on the gap
    pool = list(resid)
    null = []
    for _ in range(4000):
        a = list(RNG.choice(pool, len(most), replace=False))
        b = list(RNG.choice([x for x in pool if x not in a], len(least), replace=False))
        null.append(score(a)[0] - score(b)[0])
    null = np.array(null)
    obs = m_s - l_s
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    print(f"\n=== PERMUTATION NULL (4000 random basket pairs of the same sizes)")
    print(f"  observed gap {obs:+.3f}   null mean {null.mean():+.3f}   null p95 {np.percentile(null,95):+.3f}"
          f"   p = {p:.3f}{'  *' if p < 0.05 else '   (not significant)'}")

    print(f"\n=== PER-NAME, most_exposed (was the reasoning right for the right names?)")
    rows = sorted(((np.mean([resid[t][d] for d in win if d in resid[t]]), t) for t in most), reverse=True)
    for v, t in rows:
        why = next(x["why"] for x in P["most_exposed"] if x["t"] == t)
        print(f"  {t:6} {v:5.2f}   {why[:78]}")
    print(f"\n=== PER-NAME, least_exposed")
    for v, t in sorted(((np.mean([resid[t][d] for d in win if d in resid[t]]), t) for t in least), reverse=True):
        print(f"  {t:6} {v:5.2f}")


if __name__ == "__main__":
    main()
