# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
THE GATE — one implementation, trailing priors, shared by every view and the MCP.

WHY THIS MODULE EXISTS. Two problems it fixes at once:

  1. STATIC PRIORS WERE WRONG. Gate decisions compare an observed idiosyncratic
     move against a measured prior for its news class. Those priors were pooled
     over 2010-2012 — and a rolling estimate shows that is unsafe. Classes drift
     twice as much as the baseline (mean spread 0.38 vs 0.19 sigma) and the
     RANKING reorders: legal_regulatory occupies every position from 1st to 7th
     riskiest across the sample; six of seven classes traverse most of the
     ordering. Only `earnings` is stable (rank 1-2). A pooled prior is therefore
     too strict in low-prior regimes (real surprises marked insufficient) and too
     loose in high-prior regimes (ordinary days waved through).
     -> priors and the control are estimated on a TRAILING window as of the event
        date, never pooled.

  2. THE GATE WAS RE-IMPLEMENTED PER PAGE. Every builder had its own copy, which
     is how they drifted apart (one used a 500-observation filter and produced a
     1.796-sigma control against the 0.72 measured everywhere else). One module,
     one verdict.

WHAT THE GATE IS — and is NOT. It is a NARRATIVE CONFIDENCE TEST: is a candidate
explanation strong enough to be treated as the PRIMARY account of a move? It is
not a test of truth or causality.
  - It never asserts what DID cause a move. Rejecting one explanation is not
    evidence for another.
  - A failing verdict does not mean the news is false or that its impact was
    zero. It means the observed reaction falls within the normal historical range
    for that class, so the narrative is CONTRIBUTORY RATHER THAN EXPLANATORY.

THE THREE CHECKS (all against trailing estimates as of the date):
  a. the idiosyncratic move exceeds an ordinary non-event day (the control)
  b. the class itself produces reactions distinguishable from ordinary days in
     the trailing window
  c. surprise = observed - class_prior > 0

Usage:
    uv run scripts/gate.py --build      # compute + cache the rolling prior table
    from gate import Gate; g = Gate(); g.evaluate("2011-03-23", "earnings", 1.29)
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
# Each market gets its OWN priors and its OWN control. A US prior tells you nothing
# about what is normal for an Indian name: different volatility regime, different
# news cadence, different coverage. Sharing them would silently import one market's
# baseline into the other's verdicts.
GRAPHS = {"us": "eg100k_graph", "india": "india2021"}
G = ROOT / GRAPHS["us"]
PRIORS = G / "classification" / "rolling_priors.json"
BETA_WIN, TRAIL = 250, 60
ROLL = 250          # trailing window for each estimate
STEP = 21
MIN_EV = 18         # minimum events of a class in-window to trust its prior
MIN_CTRL = 200
MAX_ZERO_FRAC = 0.10   # reject stale/illiquid names and windows
MIN_RESID_VOL = 0.002  # 0.2%/day floor on trailing residual vol


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------
class Gate:
    """Reads the cached rolling table and returns a verdict for one event."""

    def __init__(self, path: Path = PRIORS):
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run: uv run scripts/gate.py --build")
        d = json.loads(path.read_text())
        self.dates: list[str] = d["dates"]
        self.control: list[float] = d["control"]
        # each entry is [mean, sd] — the sd is what makes a surprise interpretable
        self.priors: dict[str, dict[str, list[float]]] = d["priors"]
        self.pooled_full: dict[str, list[float]] = d["pooled"]
        self.pooled: dict[str, float] = {k: v[0] for k, v in d["pooled"].items()}
        self.pooled_control: float = d["pooled_control"]

    def _idx(self, date: str) -> int:
        """Index of the latest estimate at or before `date` (never look ahead)."""
        lo, hi = 0, len(self.dates) - 1
        if date < self.dates[0]:
            return 0
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.dates[mid] <= date:
                lo = mid
            else:
                hi = mid - 1
        return lo

    def as_of(self, date: str, event_class: str) -> dict:
        """Trailing prior + control in force on `date`, with provenance."""
        i = self._idx(date)
        p = self.priors.get(event_class, {}).get(self.dates[i])
        fallback = p is None
        if fallback:
            p = self.pooled_full.get(event_class)
        return {"asof": self.dates[i], "control": self.control[i],
                "prior": p[0] if p else None, "prior_sd": p[1] if p else None,
                "prior_is_pooled_fallback": fallback}

    def evaluate(self, date: str, event_class: str, observed_sigma: float) -> dict:
        """The verdict. `observed_sigma` = |idiosyncratic move| / trailing residual vol."""
        ctx = self.as_of(date, event_class)
        ctrl, prior = ctx["control"], ctx["prior"]
        reasons: list[str] = []
        if prior is None:
            return {"status": "INSUFFICIENT", "gate_failed": "class_not_measurable",
                    "observed_sigma": round(observed_sigma, 2), **ctx,
                    "surprise": None, "do_not_narrate": True,
                    "reasons": [f"no trailing estimate for '{event_class}' as of {ctx['asof']}"]}
        surprise = observed_sigma - prior
        if observed_sigma < ctrl:
            reasons.append(
                f"the {observed_sigma:.2f}σ idiosyncratic move is within the range of an "
                f"ordinary non-event day ({ctrl:.2f}σ as of {ctx['asof']})")
        if prior <= ctrl:
            reasons.append(
                f"'{event_class}' events were not, in the trailing window to {ctx['asof']}, "
                f"producing reactions distinguishable from ordinary days "
                f"(prior {prior:.2f}σ vs control {ctrl:.2f}σ)")
        if surprise <= 0:
            reasons.append(
                f"the observed reaction ({observed_sigma:.2f}σ) is within the normal historical "
                f"range for {event_class} events as of {ctx['asof']} "
                f"(prior {prior:.2f}σ; surprise {surprise:+.2f}σ)")
        ok = not reasons
        # GRADED CONFIDENCE. A hard threshold makes a +0.05σ clearance read exactly
        # like a +1.0σ one, which is how "sufficient to explain the price action"
        # gets attached to a move that only barely cleared its class prior. Scale
        # the surprise by the class's OWN dispersion: exceeding the prior by a tenth
        # of a standard deviation is not the same evidence as exceeding it by one.
        sd = ctx["prior_sd"] or 0.0
        z = surprise / sd if sd > 0 else 0.0
        tier = ("INSUFFICIENT" if not ok else
                "WEAK" if z < 0.35 else "MODERATE" if z < 0.9 else "STRONG")
        strength = {
            "INSUFFICIENT": "the narrative does not clear the bar",
            "WEAK": ("MARGINAL — the move exceeds the class prior by only "
                     f"{z:.2f} of that class's own standard deviation. Treat this as weak "
                     "corroboration, not a confident explanation. Hedge the language "
                     "explicitly; do not write that the news 'explains' or 'drove' the move."),
            "MODERATE": (f"the move exceeds the class prior by {z:.2f} class standard "
                         "deviations — reasonable support, state it with measured confidence"),
            "STRONG": (f"the move exceeds the class prior by {z:.2f} class standard "
                       "deviations — well outside the normal range for this class"),
        }[tier]
        return {
            "status": "SUFFICIENT" if ok else "INSUFFICIENT",
            "confidence": tier,
            "surprise_z": round(z, 2),
            "strength_note": strength,
            "gate_failed": None if ok else ("below_control" if observed_sigma < ctrl
                                            else "class_not_distinguishable" if prior <= ctrl
                                            else "no_surprise"),
            "observed_sigma": round(observed_sigma, 2),
            "surprise": round(surprise, 2), **ctx,
            # machine-readable instruction for a calling LLM: generating a causal
            # story from the attached evidence would be confabulation.
            "do_not_narrate": not ok,
            "reasons": reasons,
            "framing": ("the narrative may be treated as the primary explanation, at the stated "
                        "confidence" if ok else
                        "the narrative is CONTRIBUTORY RATHER THAN EXPLANATORY: it is not shown "
                        "to be false, nor its impact zero, only insufficient to account for the "
                        "magnitude of the observed move. This does not establish what did cause it."),
        }


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------
def build(market: str = "us") -> None:
    """Estimate the rolling priors from the SHARED cache.

    This used to re-derive the whole residual pipeline itself — and drifted: its
    residual removed only the market factor, while mcp_precompute's decomposition
    removed market AND sector. The gate was therefore testing a different quantity
    from the "idiosyncratic" shown next to it in the UI, and every improvement to the
    sector factor bypassed the gate. Reading `sigma` straight from mcp_cache.json
    deletes the duplicate pipeline and makes that impossible by construction.
    """
    global G, PRIORS
    G = ROOT / GRAPHS[market]
    PRIORS = G / "classification" / "rolling_priors.json"
    cache = G / "mcp_cache.json"
    if not cache.exists():
        raise SystemExit(f"{cache} missing — run: uv run scripts/mcp_precompute.py "
                         f"--market {market}")
    C = json.loads(cache.read_text())
    days: list[str] = C["days"]
    sigma = C["sigma"]

    ev = collections.defaultdict(set)
    for k, v in C["events"].items():
        tk, d = k.split("|")
        ev[(tk, d)] |= set(v["types"])

    # (day_index, class, sigma) for every measurable event cell
    tks = sorted(sigma)
    ti = {d: i for i, d in enumerate(days)}
    evrows = []

    # AUTHORITATIVE EARNINGS as its own class. The news-derived `earnings` class is
    # dated by publication, which puts 49% of releases (those filed at or after the
    # close) one session early and splits the reaction across two days. The 8-K
    # Item 2.02 calendar is exact, exogenous and known ex ante; aligned to the
    # session it could first affect, the same event measures 3.2x a clean control
    # against 2.2x news-dated. Kept as a SEPARATE class rather than overwriting
    # `earnings`, so the two remain comparable and nothing silently changes meaning.
    cal_cells = set()
    for tk, recs in (C.get("earnings_calendar") or {}).items():
        if tk not in sigma:
            continue
        for r in recs:
            i = ti.get(r["session"])
            if i is None:
                continue
            m = sigma[tk].get(str(i))
            if m is not None:
                evrows.append((i, "earnings_8k", float(m)))
                cal_cells.add((tk, r["session"]))
    if cal_cells:
        print(f"calendar earnings: {len(cal_cells)} measurable (ticker, session) cells")
    for (tk, d), types in sorted(ev.items()):
        i = ti.get(d)
        if i is None or tk not in sigma:
            continue
        m = sigma[tk].get(str(i))
        if m is None:
            continue
        for t in sorted(types):
            evrows.append((i, t, float(m)))

    # control: random (ticker, day) cells with NO linked news
    rng = np.random.default_rng(5)
    evset = set(ev) | cal_cells   # a scheduled earnings day is not a control day
    ctrl = []
    for _ in range(40000):
        tk = tks[int(rng.integers(len(tks)))]
        keys = sigma[tk]
        if not keys:
            continue
        ks = sorted(keys)
        k = ks[int(rng.integers(len(ks)))]
        d = days[int(k)]
        if (tk, d) in evset:
            continue
        ctrl.append((int(k), float(keys[k])))
    classes = sorted({t for _, t, _ in evrows})
    print(f"{len(tks)} names, {len(evrows)} event-class cells, "
          f"{len(ctrl)} control draws, {len(classes)} classes")

    dates, controls = [], []
    priors: dict[str, dict[str, list[float]]] = {c: {} for c in classes}
    lo_i = min(i for i, _ in ctrl)
    start = lo_i + ROLL
    for end in range(start, len(days) - 2, STEP):
        lo = end - ROLL
        cw = [m for i, m in ctrl if lo <= i < end]
        if len(cw) < MIN_CTRL:
            continue
        d = days[end]
        dates.append(d)
        controls.append(round(float(np.mean(cw)), 4))
        for c in classes:
            v = [m for i, t, m in evrows if t == c and lo <= i < end]
            if len(v) >= MIN_EV:
                priors[c][d] = [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
    pooled = {c: [round(float(np.mean([m for _, t, m in evrows if t == c])), 4),
                  round(float(np.std([m for _, t, m in evrows if t == c])), 4)]
              for c in classes if sum(1 for _, t, _ in evrows if t == c) >= 30}
    out = {"dates": dates, "control": controls, "priors": priors, "pooled": pooled,
           "pooled_control": round(float(np.mean([m for _, m in ctrl])), 4),
           "meta": {"roll_days": ROLL, "step_days": STEP, "min_events": MIN_EV,
                    "n_events": len(evrows), "n_control_draws": len(ctrl),
                    "source": "mcp_cache.json sigma (market AND sector removed)",
                    "note": "priors/control are TRAILING as of each date; never pooled"}}
    PRIORS.parent.mkdir(parents=True, exist_ok=True)
    PRIORS.write_text(json.dumps(out, indent=1))
    print(f"wrote {PRIORS}")
    print(f"  {len(dates)} estimate dates {dates[0]}..{dates[-1]}")
    print(f"  control range {min(controls):.2f}..{max(controls):.2f}σ "
          f"(pooled {out['pooled_control']:.2f}σ)")
    for c in ("earnings", "guidance", "m_and_a", "rating_action"):
        if c in pooled:
            print(f"  {c:16} pooled {pooled[c][0]:.3f}σ = "
                  f"{pooled[c][0]/out['pooled_control']:.2f}x control")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--market", choices=sorted(GRAPHS), default="us")
    ap.add_argument("--test", nargs=3, metavar=("DATE", "CLASS", "SIGMA"))
    a = ap.parse_args()
    if a.build:
        build(a.market)
    elif a.test:
        g = Gate(ROOT / GRAPHS[a.market] / "classification" / "rolling_priors.json")
        print(json.dumps(g.evaluate(a.test[0], a.test[1], float(a.test[2])), indent=2))
    else:
        ap.print_help()
