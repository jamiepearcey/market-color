# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
PAIR REGIMES — when does a correlation start, and when is it over?

Everything built so far is cross-sectional: correlation measured once over the
whole sample. That hides the question actually worth asking — relationships form
and decay, and a pair that co-moved in 2010 may be unrelated by 2012.

WHAT THIS MEASURES. For each tracked pair, the rolling correlation of their
IDIOSYNCRATIC residuals (market and sector already removed), on a 120-day window
stepped monthly.

THE HARD PART IS THE NULL. "This correlation ended" is only meaningful against a
band of what |correlation| looks like when there is NO relationship. Two residual
series of finite length correlate spuriously all the time, and the smaller the
window the wider that band. So the band is measured, not assumed:

    for each pair, CIRCULARLY SHIFT one series by a random offset and recompute
    the same rolling correlation. That destroys the contemporaneous relationship
    while preserving each series' own length, variance and autocorrelation. Do it
    many times; the 95th percentile of |shifted correlation| is the band.

A window is only called ACTIVE if |corr| exceeds that pair's own measured band.

STATES, all defined against the band rather than an arbitrary threshold:
    ENDED      active for >=MIN_RUN consecutive windows, then inactive for
               >=MIN_RUN consecutive windows, and still inactive at the sample end
    EMERGING   the mirror image: inactive, then active, still active at the end
    PERSISTENT active in most windows throughout
    EPISODIC   flips repeatedly — active but not stable, and worth distrusting
    NEVER      never clears its own band; it was noise in the pooled number

WHAT THIS DOES NOT DO. It does not establish why a correlation changed, and a
detected transition is not evidence of a cause. The news attached downstream is a
CANDIDATE ACCOUNT to be assessed, exactly as with single-name moves.

Usage:
    uv run scripts/pair_regimes.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
OUT = G / "pair_regimes.json"

WIN = 120          # rolling correlation window (trading days)
STEP = 21          # monthly
MIN_OBS = 80       # usable observations required inside a window
N_SHIFT = 24       # circular shifts per pair for the empirical band
BAND_Q = 95        # percentile of |shifted corr| that defines "no relationship"
MIN_RUN = 3        # consecutive windows required to call a state change
# A transition needs time to be CONFIRMED. With MIN_RUN=3 a change detected near the
# end of the sample rests on the weakest evidence the rule allows and has had no
# opportunity to revert — so transitions pile up at the boundary. First cut produced
# 15 EMERGING and 10 ENDED in 2014-10 alone (the last months of data), which is an
# artifact of the window, not a burst of regime change. Require the new state to hold
# for MIN_CONFIRM windows before it is reported at all.
MIN_CONFIRM = 6    # ~6 months of the new state after the transition
N_PAIRS = 500
RNG = np.random.default_rng(11)


def rolling_corr(a: np.ndarray, b: np.ndarray, starts: list[int]) -> np.ndarray:
    out = np.full(len(starts), np.nan)
    for k, s in enumerate(starts):
        x, y = a[s:s + WIN], b[s:s + WIN]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < MIN_OBS:
            continue
        xx, yy = x[m], y[m]
        sx, sy = xx.std(), yy.std()
        if sx <= 0 or sy <= 0:
            continue
        out[k] = float(((xx - xx.mean()) @ (yy - yy.mean())) / (len(xx) * sx * sy))
    return out


def classify(active: np.ndarray, valid: np.ndarray) -> tuple[str, int | None]:
    """Reduce a boolean activity series to a state plus the transition index.

    `valid` marks windows where the correlation could actually be COMPUTED. A
    window with no data is UNKNOWN, not inactive — conflating the two invents
    transitions at the edges of the sample. (First cut of this did exactly that:
    284 of 500 pairs came back "EMERGING" on the very first window, and pairs were
    labelled ENDED while their final correlation sat far above their own band.)
    Only valid windows take part, and a state change must be interior to them.
    """
    idx = np.flatnonzero(valid)
    if len(idx) < 2 * MIN_RUN:
        return "INSUFFICIENT", None
    a = active[idx].astype(bool)
    n = len(a)
    if a.sum() == 0:
        return "NEVER", None
    runs, start = [], 0
    for i in range(1, n + 1):
        if i == n or a[i] != a[start]:
            runs.append((bool(a[start]), start, i - 1))
            start = i
    long = [r for r in runs if r[2] - r[1] + 1 >= MIN_RUN]
    if len(long) >= 2:
        last, prev = long[-1], long[-2]
        # the transition must be interior: a run that merely starts at the first
        # valid window is the series beginning, not a regime change
        #
        # AND the CURRENT state must agree with the label. Deciding purely from the
        # last long run let a pair be called ENDED while its final correlation sat
        # at 0.478 against a 0.19 band — it had gone quiet, then come back for two
        # windows, too short to form a run of its own. If it is active now it has
        # not ended.
        confirmed = (n - last[1]) >= MIN_CONFIRM
        if prev[1] > 0 and confirmed:
            if prev[0] and not last[0] and not a[-1]:
                return "ENDED", int(idx[last[1]])
            if not prev[0] and last[0] and a[-1]:
                return "EMERGING", int(idx[last[1]])
        if prev[1] > 0 and not confirmed:
            # a change is visible but has not yet held long enough to be called
            return "UNCONFIRMED", int(idx[last[1]])
    if a.mean() >= 0.7:
        return "PERSISTENT", None
    if len(runs) >= 5:
        return "EPISODIC", None
    return "PERSISTENT" if a[-1] else "NEVER", None


def main() -> None:
    C = json.loads((G / "mcp_cache.json").read_text())
    days: list[str] = C["days"]
    n = len(days)

    # signed idiosyncratic returns from the shared cache
    idi = {}
    for tk, dec in C["decomp"].items():
        v = np.full(n, np.nan)
        for k, arr in dec.items():
            v[int(k)] = arr[3]
        if np.isfinite(v).sum() >= 500:
            idi[tk] = v
    print(f"{len(idi)} names with idiosyncratic series")

    # candidate pairs: the neighbour lists, deduped
    cand = set()
    for tk, nb in C["neighbours"].items():
        if tk not in idi:
            continue
        for t, r, _same in nb:
            if t in idi:
                cand.add((min(tk, t), max(tk, t)))
    cand = sorted(cand)
    print(f"{len(cand)} candidate pairs from neighbour lists")

    starts = list(range(0, n - WIN, STEP))
    wdates = [days[s + WIN - 1] for s in starts]

    scored = []
    for a, b in cand:
        c = rolling_corr(idi[a], idi[b], starts)
        if np.isfinite(c).sum() < 12:
            continue
        scored.append((float(np.nanmean(np.abs(c))), a, b, c))
    scored.sort(reverse=True)
    scored = scored[:N_PAIRS]
    print(f"{len(scored)} pairs with enough history; computing empirical bands "
          f"({N_SHIFT} circular shifts each)")

    sect = C["sectors"]
    out = []
    state_count = collections.Counter()
    for j, (mabs, a, b, c) in enumerate(scored):
        # empirical band: circular shifts destroy the contemporaneous link only
        null = []
        for _ in range(N_SHIFT):
            off = int(RNG.integers(WIN, n - WIN))
            null.append(rolling_corr(idi[a], np.roll(idi[b], off), starts))
        null = np.abs(np.concatenate(null))
        null = null[np.isfinite(null)]
        if len(null) < 50:
            continue
        band = float(np.percentile(null, BAND_Q))
        valid = np.isfinite(c)
        active = np.zeros(len(c), dtype=bool)
        active[valid] = np.abs(c[valid]) > band
        state, tidx = classify(active, valid)
        state_count[state] += 1
        # trend over the final third
        tail = c[-max(4, len(c) // 3):]
        tail_ok = np.isfinite(tail)
        slope = None
        if tail_ok.sum() >= 4:
            x = np.arange(len(tail))[tail_ok]
            slope = float(np.polyfit(x, tail[tail_ok], 1)[0])
        out.append({
            "a": a, "b": b,
            "a_ind": sect.get(a, {}).get("industry", "UNK"),
            "b_ind": sect.get(b, {}).get("industry", "UNK"),
            "a_mg": sect.get(a, {}).get("major_group", "UNK"),
            "b_mg": sect.get(b, {}).get("major_group", "UNK"),
            "cross_industry": sect.get(a, {}).get("industry") != sect.get(b, {}).get("industry"),
            "band": round(band, 3),
            "mean_abs": round(mabs, 3),
            "corr": [None if not np.isfinite(x) else round(float(x), 3) for x in c],
            "active": [bool(x) for x in active],
            "state": state,
            "transition_index": tidx,
            "transition_date": wdates[tidx] if tidx is not None else None,
            "tail_slope_per_window": None if slope is None else round(slope, 4),
            # last VALID correlation — c[-1] may simply be a window with no data
            "last_corr": (round(float(c[valid][-1]), 3) if valid.any() else None),
            "last_window": (wdates[int(np.flatnonzero(valid)[-1])] if valid.any() else None),
        })
        if (j + 1) % 100 == 0:
            print(f"  {j+1}/{len(scored)}")

    payload = {"window_days": WIN, "step_days": STEP, "band_percentile": BAND_Q,
               "min_run": MIN_RUN, "min_confirm": MIN_CONFIRM, "n_shifts": N_SHIFT,
               "window_dates": wdates, "pairs": out}
    OUT.write_text(json.dumps(payload))
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB), {len(out)} pairs")
    print("states:", dict(state_count))
    print(f"median empirical band |r| = {np.median([p['band'] for p in out]):.3f} "
          f"(this is the level below which a correlation is indistinguishable from none)")
    for st in ("ENDED", "EMERGING"):
        ex = [p for p in out if p["state"] == st][:4]
        if ex:
            print(f"\n  example {st}:")
            for p in ex:
                print(f"    {p['a']:6}~{p['b']:6} band {p['band']:.2f} "
                      f"at {p['transition_date']}  last r={p['last_corr']}  "
                      f"{'cross-industry' if p['cross_industry'] else 'same industry'}")


if __name__ == "__main__":
    main()
