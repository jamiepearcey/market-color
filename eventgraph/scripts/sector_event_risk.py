# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
SECTOR x EVENT-CLASS IMMEDIATE RISK — what the news flow says once it is
aggregated into the standardised taxonomy rather than the raw LLM vocab.

WHY RE-DERIVE THE CLASSES HERE. mcp_cache.json was built before taxonomy.py grew
its substring fallback, so its event tags carry the old exact-match-only vocab
(1,039 of ~4,000 tags were `other`). Prices come from the cache; CLASSES are
recomputed from the graph through the current taxonomy, so this reflects the
corrected aggregation.

THE CONTROL IS SECTOR-MATCHED, AND THAT IS THE POINT. Energy names are simply
more volatile than Consumer Staples names. A single pooled control therefore
reports sector beta as if it were event-class risk -- the same mistake the IMF
jump-hazard work made until the baseline was localised to each event's own
neighbourhood. Every lift here is against the SAME SECTOR's non-event days, so a
number above 1.0x means "this class moves this sector more than an ordinary day
in that sector", not "this sector is volatile".

READ THE OUTPUT AS COVERAGE + CONDITIONAL RISK, NOT ALPHA. F17/F18 put the news
channel at a 0-3% incremental sliver and F24/F25/F28 killed direction, so the
question this answers is "which event classes carry the immediate risk, and does
that differ by sector" -- a risk-description question, not a forecast.

Usage:
    uv run scripts/sector_event_risk.py
    uv run scripts/sector_event_risk.py --min-n 15 --group
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402
from panel import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
RNG = np.random.default_rng(4242)
N_BOOT = 2000


def rebuild_cells(graph: Path):
    """(ticker, date) -> {std_event_type}, using the CURRENT taxonomy."""
    tick, sector = {}, {}
    for l in open(graph / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t:
            tick[j["entity_id"]] = t
            if j.get("std_sector") not in (None, "UNK"):
                sector[t] = j["std_sector"]

    doctypes = collections.defaultdict(set)
    unmapped = collections.Counter()
    for l in open(graph / "lake" / "event.jsonl"):
        j = json.loads(l)
        raw = j.get("event_type")
        std, _ = tax.std_event_type(raw)
        if j.get("doc_id"):
            doctypes[j["doc_id"]].add(std)
        if std == "other":
            unmapped[(raw or "").strip().lower()] += 1

    docday = {}
    for l in open(graph / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d

    cells = collections.defaultdict(set)
    for l in open(graph / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        tk = tick.get(j.get("effect_entity"))
        day = docday.get(j.get("doc_id"))
        if tk and day:
            cells[(tk, day)] |= doctypes.get(j["doc_id"], set())
    return cells, sector, unmapped


def boot_mean(vals: np.ndarray) -> tuple[float, float]:
    """Bootstrap 95% CI on the mean -- n is small in most cells and a bare mean
    would imply precision that is not there."""
    if len(vals) < 3:
        return float("nan"), float("nan")
    draws = vals[RNG.integers(0, len(vals), (N_BOOT, len(vals)))].mean(axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-n", type=int, default=10)
    ap.add_argument("--group", action="store_true", help="aggregate to event_group")
    a = ap.parse_args()

    P = load("us")
    graph = ROOT / "eg100k_graph"
    cells, sector_of, unmapped = rebuild_cells(graph)

    # sector-matched control: every (ticker, day) with a sigma and NO event
    evset = set(cells)
    ctrl = collections.defaultdict(list)
    for tk, series in P.sigma.items():
        sec = sector_of.get(tk)
        if not sec:
            continue
        for istr, m in series.items():
            if m is None:
                continue
            day = P.days[int(istr)]
            if (tk, day) in evset:
                continue
            ctrl[sec].append(float(m))
    ctrl_mean = {s: float(np.mean(v)) for s, v in ctrl.items() if len(v) > 200}
    overall_ctrl = float(np.mean([m for v in ctrl.values() for m in v]))

    # event cells, keyed by (sector, class)
    obs = collections.defaultdict(list)
    by_class = collections.defaultdict(list)
    for (tk, day), classes in cells.items():
        sec = sector_of.get(tk)
        m = P.sigma_at(tk, day)
        if not sec or m is None:
            continue
        for c in classes:
            key = tax.EVENT_GROUP.get(c, "other") if a.group else c
            obs[(sec, key)].append(float(m))
            by_class[key].append((float(m), sec))

    print(f"panel: {len(P.days)} days, {len(P.sigma)} tickers | "
          f"event cells rebuilt: {len(cells)} | sectors with control: {len(ctrl_mean)}")
    print(f"pooled control sigma: {overall_ctrl:.3f}\n")
    print("control sigma BY SECTOR (why a pooled control would mislead):")
    for s, v in sorted(ctrl_mean.items(), key=lambda x: -x[1]):
        print(f"   {s:26} {v:.3f}   n={len(ctrl[s]):>7}")

    # ---- class level, sector-matched -------------------------------------
    print(f"\n=== EVENT CLASS x SECTOR-MATCHED CONTROL  (min n={a.min_n})")
    print(f"  {'class':18} {'n':>5} {'mean s':>7} {'ctrl':>6} {'lift':>6}  {'95% CI on lift':>18}")
    rows = []
    for c, vals in by_class.items():
        if len(vals) < a.min_n:
            continue
        m = np.array([v for v, _ in vals])
        # each observation divided by ITS OWN sector's control, then averaged
        lifts = np.array([v / ctrl_mean[s] for v, s in vals if s in ctrl_mean])
        if len(lifts) < a.min_n:
            continue
        lo, hi = boot_mean(lifts)
        rows.append((c, len(lifts), m.mean(), float(np.mean([ctrl_mean[s] for _, s in vals if s in ctrl_mean])),
                     lifts.mean(), lo, hi))
    for c, n, ms, cs, lift, lo, hi in sorted(rows, key=lambda r: -r[4]):
        sig = " *" if lo > 1.0 else ""
        print(f"  {c:18} {n:>5} {ms:>7.3f} {cs:>6.3f} {lift:>5.2f}x  [{lo:>5.2f}, {hi:>5.2f}]{sig}")

    # ---- the sector x class grid ------------------------------------------
    print(f"\n=== SECTOR x CLASS lift (cells with n>={a.min_n}; blank = too thin)")
    secs = [s for s in ctrl_mean if any((s, c) in obs and len(obs[(s, c)]) >= a.min_n
                                        for c in {k for _, k in obs})]
    classes = [c for c, n, *_ in sorted(rows, key=lambda r: -r[1])][:9]
    hdr = "  " + f"{'sector':24}" + "".join(f"{c[:11]:>12}" for c in classes)
    print(hdr)
    for s in sorted(secs):
        line = f"  {s:24}"
        for c in classes:
            v = obs.get((s, c), [])
            line += f"{(np.mean(v) / ctrl_mean[s]):>11.2f}x" if len(v) >= a.min_n else f"{'':>12}"
        print(line)

    print(f"\n=== residual `other` after the corrected taxonomy")
    tot = sum(unmapped.values())
    llm = sum(v for k, v in unmapped.items() if k in ("other", "null", ""))
    print(f"  {tot} events still `other`; {llm} ({llm/tot:.0%}) are the EXTRACTOR "
          f"emitting 'other'/null — irreducible without re-extraction.")
    print("  largest remaining mappable strings:")
    for k, v in unmapped.most_common(12):
        if k not in ("other", "null", ""):
            print(f"    {v:>6}  {k[:50]}")


if __name__ == "__main__":
    main()
