# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
THE SAME EVENT-WINDOW PROFILE, BUT ON CALENDAR DATES — is t-1 real or an artifact?

THE QUESTION THIS SETTLES. Profiling the news graph found risk elevated at t-1
(1.12x a clean control) — but t+1 was elevated by almost exactly as much (1.14x).
Leakage builds up and resolves; it is not symmetric. The symmetry instead suggests
TIMESTAMP SMEARING: news dates are `published_at`, so a story filed after the
close is dated one day and moves the market the next.

An 8-K Item 2.02 filing carries `acceptanceDateTime`, so the ambiguity is
removable. A release accepted at or after 16:00 ET is assigned to the NEXT
session; one accepted before is assigned to the filing date. If the t-1 elevation
was smearing, it should COLLAPSE under this alignment. If it survives, it is
something real — anticipation of a known date, or genuine leakage.

Three arms, so the answer is attributable:
    NEWS-DATED       the news graph's earnings cells (the original measurement)
    CALENDAR-RAW     8-K filing date, no time adjustment
    CALENDAR-ALIGNED 8-K shifted to the session it could first have affected

Usage:
    uv run scripts/calendar_window_profile.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
CAL = ROOT / "earnings_calendar.json"
OFFSETS = list(range(-6, 7))
N_BOOT = 1500
CLOSE_ET = 16
RNG = np.random.default_rng(303)


def boot(bd):
    d = np.array(sorted(bd), dtype=object)
    if len(d) < 5:
        return None
    ms = []
    for _ in range(N_BOOT):
        pick = d[RNG.integers(0, len(d), len(d))]
        v = [x for k in pick for x in bd.get(k, ())]
        if v:
            ms.append(float(np.mean(v)))
    if not ms:
        return None
    m = np.array(ms)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return m.mean(), lo, hi


def main() -> None:
    P = panel.load("us")
    days, di = P.days, P.day_index
    cal = json.loads(CAL.read_text())
    print(f"calendar: {len(cal)} tickers, "
          f"{sum(len(v) for v in cal.values())} Item 2.02 releases")

    def sig(tk, i):
        v = P.sigma.get(tk, {}).get(str(i))
        return None if v is None else float(v)

    # clean control: no NEWS and no CALENDAR event within +/-10 sessions
    busy = collections.defaultdict(set)
    for (tk, d) in P.events:
        i = di.get(d)
        if i is not None:
            busy[tk].add(i)
    for tk, rows in cal.items():
        for r in rows:
            i = di.get(r["filed"])
            if i is not None:
                busy[tk].add(i)
    tks = P.tickers()
    cv = []
    for _ in range(60000):
        tk = tks[int(RNG.integers(len(tks)))]
        ks = list(P.sigma.get(tk, {}))
        if not ks:
            continue
        k = int(ks[int(RNG.integers(len(ks)))])
        m = sig(tk, k)
        if m is not None and not any(abs(k - j) <= 10 for j in busy.get(tk, ())):
            cv.append(m)
    C = float(np.mean(cv))
    print(f"clean control (no news AND no filing within +/-10d): {C:.4f}σ  n={len(cv)}\n")

    def next_session(idx: int) -> int:
        return idx + 1 if idx + 1 < len(days) else idx

    # build the three event sets
    sets: dict[str, list[tuple[str, int]]] = {"NEWS-DATED": [], "CALENDAR-RAW": [],
                                              "CALENDAR-ALIGNED": []}
    for (tk, d), types in P.events.items():
        if "earnings" in types and di.get(d) is not None and tk in P.sigma:
            sets["NEWS-DATED"].append((tk, di[d]))
    n_after = 0
    for tk, rows in cal.items():
        if tk not in P.sigma:
            continue
        for r in rows:
            i = di.get(r["filed"])
            if i is None:
                continue
            sets["CALENDAR-RAW"].append((tk, i))
            acc = r.get("accepted") or ""
            after = False
            if len(acc) >= 13:
                try:
                    after = int(acc[11:13]) >= CLOSE_ET
                except ValueError:
                    after = False
            n_after += after
            sets["CALENDAR-ALIGNED"].append((tk, next_session(i) if after else i))
    tot_cal = len(sets["CALENDAR-RAW"])
    print(f"{n_after}/{tot_cal} releases filed at/after {CLOSE_ET}:00 ET "
          f"({n_after/max(tot_cal,1):.0%}) -> shifted to the next session\n")

    print(f"{'arm':>18}" + "".join(f"{o:>7}" for o in OFFSETS))
    print(f"{'':>18}" + "".join(f"{('t'+str(o)) if o else 't0':>7}" for o in OFFSETS))
    res = {}
    for name, evs in sets.items():
        bd = {o: collections.defaultdict(list) for o in OFFSETS}
        for tk, i in evs:
            for o in OFFSETS:
                j = i + o
                if 0 <= j < len(days):
                    m = sig(tk, j)
                    if m is not None:
                        bd[o][days[j]].append(m)
        row, sigs, prof = f"{name:>18}", f"{'':>18}", {}
        for o in OFFSETS:
            b = boot(bd[o])
            if not b:
                row += f"{'-':>7}"; sigs += f"{'':>7}"; continue
            r, lo, hi = b[0] / C, b[1] / C, b[2] / C
            prof[o] = {"ratio": r, "lo": lo, "hi": hi,
                       "n": sum(len(v) for v in bd[o].values())}
            row += f"{r:7.2f}"
            sigs += f"{'    *  ' if lo > 1.0 else '       '}"
        print(row)
        print(sigs + "  (* = 95% CI above a clean day)")
        res[name] = prof
    print(f"\n{'arm':>18}{'n at t0':>9}{'t-1':>8}{'t0':>8}{'t+1':>8}   t-1 verdict")
    for name, prof in res.items():
        if 0 not in prof:
            continue
        v = ("SIGNIFICANT" if prof.get(-1, {}).get("lo", 0) > 1.0 else "not significant")
        print(f"{name:>18}{prof[0]['n']:9}{prof.get(-1,{}).get('ratio',float('nan')):8.2f}"
              f"{prof[0]['ratio']:8.2f}{prof.get(1,{}).get('ratio',float('nan')):8.2f}   {v}")
    out = ROOT / "eg100k_graph" / "calendar_window_profile.json"
    out.write_text(json.dumps({"control": C, "n_after_close": n_after,
                               "n_calendar": tot_cal,
                               "profiles": {k: {str(o): x for o, x in v.items()}
                                            for k, v in res.items()}}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
