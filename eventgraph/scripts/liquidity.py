"""
LIQUIDITY GUARD — shared, because getting this wrong silently corrupted every
sigma this project has produced.

THE TRAP. ~17% of the resolved ticker universe (123 of 670) are illiquid OTC
foreign ADRs (GECFF, QUCCF, KAKKF, MEJHF ...) whose prices are STALE: >20% of
their days print an exactly-zero return, the worst at 99.4%. Any measure that
normalises a move by a TRAILING volatility divides, for these names, by something
approaching zero — so the one day they actually trade explodes to an absurd
sigma. A 267-sigma "control" was traced to exactly this.

WHY THE OBVIOUS GUARD FAILS. `if w.std() <= 0: skip` does NOT catch it. A window
that is 99% stale zeros has a tiny but strictly POSITIVE standard deviation, so
it sails through. The test has to be on the ZERO-RETURN FRACTION, plus an
absolute floor on the volatility itself.

A SECOND, INDEPENDENT TRAP. Several scripts drew a control sample with a seeded
RNG over `list(...)` built from `{t for t, _ in cells}` — a SET. Python randomises
string hashing per process, so set iteration order differs every run, the seeded
draw picks different names each time, and results are not reproducible (the pooled
control was observed swinging 0.86 -> 1.02 sigma between identical builds). This
also MASKED the staleness bug, because most runs happened to miss the worst names.
ALWAYS sort a collection before sampling from it with a seeded RNG.

Reassuringly, the headline topic->risk map was robust: earnings measured 2.28x
control before the fix and 2.27x after, because the contamination inflated the
control and the event classes by the same factor. Only ABSOLUTE sigmas were wrong.
"""
from __future__ import annotations

import numpy as np

MAX_ZERO_FRAC = 0.10    # reject a name (or window) that is more than 10% stale
MIN_RESID_VOL = 0.002   # 0.2%/day floor on trailing residual vol
MIN_HISTORY = 600


def tradeable(returns) -> bool:
    """True if a name trades often enough to support a normalised move.

    `returns` may be a dict of date->log-return or any sequence of returns.
    """
    v = np.abs(np.asarray(list(returns.values()) if isinstance(returns, dict)
                          else list(returns), dtype=float))
    if len(v) < MIN_HISTORY:
        return False
    return float((v < 1e-12).mean()) <= MAX_ZERO_FRAC


def usable_window(w, min_len: int = 40) -> bool:
    """True if a trailing window can serve as a volatility denominator.

    `min_len` is exposed because some callers build SHORT sub-windows (e.g. five
    consecutive 20-day blocks used as log-vol lags in an AR model). Those are just
    as vulnerable: a stale block yields log(~0), a huge negative outlier that
    silently poisons every regressor built from it.
    """
    w = np.asarray(w, dtype=float)
    w = w[np.isfinite(w)]
    if len(w) < min_len:
        return False
    if float((np.abs(w) < 1e-12).mean()) > MAX_ZERO_FRAC:
        return False
    return float(w.std()) >= MIN_RESID_VOL


def filter_universe(rets: dict) -> tuple[dict, list]:
    """Split a ticker->returns mapping into (tradeable, dropped-as-stale)."""
    keep, drop = {}, []
    for tk in sorted(rets):
        (keep.__setitem__(tk, rets[tk]) if tradeable(rets[tk]) else drop.append(tk))
    return keep, drop
