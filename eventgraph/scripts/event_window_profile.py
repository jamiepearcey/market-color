# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
THE EVENT WINDOW — what happens BEFORE the news day, not just on and after it.

THE GAP THIS FILLS. Every test in this project measured the news day (t=0) or the
days after it. Nothing looked at t<0. That is a real omission for three reasons:

  1. TRADABILITY. A t=0 effect cannot be traded — you cannot act on news you have
     not seen. A t-1 or t-2 effect can be, IF you know the event is coming. That
     is the whole argument for supplementing with an event calendar.
  2. CONTROL CONTAMINATION. Our control samples random NON-event days. If the days
     immediately before news carry elevated risk, some of them are sitting in the
     control pool, inflating it — which would make every measured effect an
     UNDERSTATEMENT.
  3. LEAKAGE vs ANTICIPATION. Elevated risk before an announcement is either
     information leaking or the market anticipating a scheduled event. The two are
     distinguishable only with a calendar, which is the next piece of work.

WHAT IT MEASURES. For every event cell, the standardised idiosyncratic move at
each offset from -10 to +10 trading days, against the same control the gate uses.
Reported as a ratio to control, so 1.0 means "an ordinary day".

Bootstrapped over DATES, because event cells on the same day share a common
component and treating them as independent understates the error badly.

Usage:
    uv run scripts/event_window_profile.py --market us
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel  # noqa: E402

OFFSETS = list(range(-10, 11))
N_BOOT = 1500
RNG = np.random.default_rng(101)


def boot(by_date: dict):
    d = np.array(sorted(by_date), dtype=object)
    if len(d) < 5:
        return None
    ms = []
    for _ in range(N_BOOT):
        pick = d[RNG.integers(0, len(d), len(d))]
        v = [x for k in pick for x in by_date.get(k, ())]
        if v:
            ms.append(float(np.mean(v)))
    if not ms:
        return None
    m = np.array(ms)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(m.mean()), "lo": float(lo), "hi": float(hi)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", default="us", choices=("us", "india"))
    ap.add_argument("--classes", default="earnings,guidance,m_and_a,rating_action",
                    help="also profile these classes separately")
    a = ap.parse_args()
    P = panel.load(a.market)
    days, di = P.days, P.day_index
    print(f"market={a.market}  {len(days)} trading days")

    def sig(tk, i):
        v = P.sigma.get(tk, {}).get(str(i))
        return None if v is None else float(v)

    # CONTROL: random cells with no news anywhere within +/-10 days of them, so the
    # control itself cannot be a pre- or post-event day. This is stricter than the
    # gate's control and is the point of the exercise.
    ev_by_tk = collections.defaultdict(set)
    for (tk, d) in P.events:
        i = di.get(d)
        if i is not None:
            ev_by_tk[tk].add(i)
    tks = P.tickers()
    ctrl_vals, ctrl_loose = [], []
    for _ in range(60000):
        tk = tks[int(RNG.integers(len(tks)))]
        ks = list(P.sigma.get(tk, {}))
        if not ks:
            continue
        k = int(ks[int(RNG.integers(len(ks)))])
        m = sig(tk, k)
        if m is None:
            continue
        near = any(abs(k - j) <= 10 for j in ev_by_tk.get(tk, ()))
        if not near:
            ctrl_vals.append(m)
        if k not in ev_by_tk.get(tk, ()):
            ctrl_loose.append(m)
    C_strict = float(np.mean(ctrl_vals))
    C_loose = float(np.mean(ctrl_loose))
    print(f"control (clean, no news within +/-10d): {C_strict:.4f}σ   n={len(ctrl_vals)}")
    print(f"control (the gate's: merely not an event day): {C_loose:.4f}σ   n={len(ctrl_loose)}")
    infl = (C_loose / C_strict - 1) * 100
    print(f"  -> the gate's control is {infl:+.1f}% vs a clean one. Positive means it is "
          f"contaminated by\n     near-event days, and every measured effect is an UNDERSTATEMENT.\n")

    want = [c.strip() for c in a.classes.split(",") if c.strip()]
    profiles = {}
    for label in ["ALL"] + want:
        buckets = {o: collections.defaultdict(list) for o in OFFSETS}
        for (tk, d), types in P.events.items():
            if label != "ALL" and label not in types:
                continue
            i = di.get(d)
            if i is None:
                continue
            for o in OFFSETS:
                m = sig(tk, i + o)
                if m is not None:
                    buckets[o][days[i + o] if 0 <= i + o < len(days) else d].append(m)
        prof = {}
        for o in OFFSETS:
            b = boot(buckets[o])
            if b:
                prof[o] = {"n": sum(len(v) for v in buckets[o].values()),
                           "ratio": b["mean"] / C_strict,
                           "lo": b["lo"] / C_strict, "hi": b["hi"] / C_strict}
        profiles[label] = prof

    for label, prof in profiles.items():
        if not prof:
            continue
        print(f"=== {label} — idiosyncratic move as a multiple of a clean control day ===")
        print(f"{'offset':>7}{'n':>7}{'x control':>11}{'95% CI':>20}")
        for o in OFFSETS:
            p = prof.get(o)
            if not p:
                continue
            star = " <-- news day" if o == 0 else (" *" if p["lo"] > 1.0 else "")
            print(f"{o:>7}{p['n']:7}{p['ratio']:11.3f}   [{p['lo']:.3f}, {p['hi']:.3f}]{star}")
        pre = [prof[o]["ratio"] for o in range(-5, 0) if o in prof]
        pre_sig = [o for o in range(-5, 0) if o in prof and prof[o]["lo"] > 1.0]
        if pre:
            print(f"  mean t-5..t-1 = {np.mean(pre):.3f}x control; "
                  f"days significantly above 1.0: {pre_sig if pre_sig else 'none'}")
        print()

    out = P.graph / "event_window_profile.json"
    out.write_text(json.dumps({"market": a.market, "control_strict": C_strict,
                               "control_loose": C_loose,
                               "profiles": {k: {str(o): v for o, v in p.items()}
                                            for k, p in profiles.items()}}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
