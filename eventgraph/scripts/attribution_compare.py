# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
IS THE EVENT-CLASS ATTRIBUTION WRONG? Doc-union vs issuer-entity, measured.

THE BUG THIS TESTS. Every consumer of event classes in this repo
(build_ticker_desk, build_news_desk, sector_event_risk, ...) attributes classes
like this:

    doctypes[doc_id] = {every event_type appearing anywhere in the document}
    cell[(effect_entity_ticker, date)] |= doctypes[doc_id]

so EVERY entity on the receiving end of ANY causal edge in a document inherits
EVERY event class in that document. A "Latin America Equity Preview" column that
mentions one merger tags all five named companies with `m_and_a`. Inspecting a
blind sample of 320 `m_and_a` cells, the great majority were not M&A events for
the tagged name at all -- bond and country ETFs (HYG, TLT, FEZ, RSX) tagged to
deals, roundup columns, executive hires, patent suits.

THE CORRECT PATH EXISTS AND IS UNUSED. `event.issuer_entity` names the entity an
event is ABOUT. `causal_event_edge.event_id` would be even better but is 0%
populated (134,366 rows, none linked), so issuer_entity is the available fix:
32.5% of events carry one, 10,214 of them resolve to a ticker.

THE PREDICTION, stated before measuring: if doc-union dilution is real, the
issuer-attributed classes should show HIGHER lift on far FEWER cells, and the
effect should be largest for `m_and_a` -- the class whose documents most often
name many uninvolved companies.

Usage:
    uv run scripts/attribution_compare.py
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
RNG = np.random.default_rng(99)


def boot(v, n=2000):
    if len(v) < 3:
        return float("nan"), float("nan")
    a = np.asarray(v)
    d = a[RNG.integers(0, len(a), (n, len(a)))].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="eg100k_graph")
    ap.add_argument("--min-n", type=int, default=12)
    a = ap.parse_args()
    G = ROOT / a.graph
    P = load("us")

    tick, sector = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t:
            tick[j["entity_id"]] = t
            if j.get("std_sector") not in (None, "UNK"):
                sector[t] = j["std_sector"]

    docday = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d

    doctypes = collections.defaultdict(set)
    issuer_cells = collections.defaultdict(set)
    for l in open(G / "lake" / "event.jsonl"):
        j = json.loads(l)
        std, _ = tax.std_event_type(j.get("event_type"))
        doc = j.get("doc_id")
        if doc:
            doctypes[doc].add(std)
        tk = tick.get(j.get("issuer_entity"))
        day = docday.get(doc)
        if tk and day:
            issuer_cells[(tk, day)].add(std)          # CORRECT: event -> its issuer

    docunion_cells = collections.defaultdict(set)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        tk, doc = tick.get(j.get("effect_entity")), j.get("doc_id")
        day = docday.get(doc)
        if tk and day:
            docunion_cells[(tk, day)] |= doctypes.get(doc, set())   # CURRENT

    # sector-matched control over non-event days (union of both cell sets)
    evset = set(docunion_cells) | set(issuer_cells)
    ctrl = collections.defaultdict(list)
    for tk, series in P.sigma.items():
        sec = sector.get(tk)
        if not sec:
            continue
        for istr, m in series.items():
            if m is None:
                continue
            if (tk, P.days[int(istr)]) in evset:
                continue
            ctrl[sec].append(float(m))
    cmean = {s: float(np.mean(v)) for s, v in ctrl.items() if len(v) > 200}

    def measure(cells):
        out = collections.defaultdict(list)
        for (tk, day), classes in cells.items():
            sec, m = sector.get(tk), P.sigma_at(tk, day)
            if not sec or m is None or sec not in cmean:
                continue
            for c in classes:
                out[c].append(float(m) / cmean[sec])
        return out

    A, B = measure(docunion_cells), measure(issuer_cells)
    print(f"cells — doc-union: {len(docunion_cells)}   issuer-attributed: {len(issuer_cells)}\n")
    print(f"  {'class':18} {'union n':>8} {'lift':>6} | {'issuer n':>8} {'lift':>6} "
          f"{'95% CI':>16}  {'delta':>7}")
    keys = sorted(set(A) | set(B), key=lambda c: -len(A.get(c, [])))
    for c in keys:
        av, bv = A.get(c, []), B.get(c, [])
        if len(av) < a.min_n and len(bv) < a.min_n:
            continue
        al = np.mean(av) if len(av) >= a.min_n else float("nan")
        bl = np.mean(bv) if len(bv) >= a.min_n else float("nan")
        lo, hi = boot(bv) if len(bv) >= a.min_n else (float("nan"),) * 2
        ci = f"[{lo:.2f}, {hi:.2f}]" if len(bv) >= a.min_n else ""
        d = bl - al if (bl == bl and al == al) else float("nan")
        print(f"  {c:18} {len(av):>8} {al:>5.2f}x | {len(bv):>8} {bl:>5.2f}x "
              f"{ci:>16}  {d:>+6.2f}")


if __name__ == "__main__":
    main()
