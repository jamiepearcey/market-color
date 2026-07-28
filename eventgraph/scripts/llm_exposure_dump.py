# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
STEP 1 — the BLIND dump for the LLM-as-activation-pathway test.

THE HYPOTHESIS. Lexical and embedding matching can only find what is textually
co-present, which is why every variant died (F54/F55/F56): the vocabulary carries
period topic, not transferable exposure. An LLM can supply a link that is written
down NOWHERE in the corpus — "Greek default -> peripheral sovereign holdings ->
French banks" — by reasoning rather than matching. That is the untested channel.

WHY THIS CORPUS AND NOT eg100k. eg100k is 2010-2012, saturated in any model's
training data. F55 showed the failure mode is TEMPORAL TRANSFER, so hindsight
manufactures exactly the appearance of success. eg_live2 runs 2026-06-27 to
2026-07-19 — after the reasoning model's May 2026 cutoff — so the outcomes cannot
have been memorised. It is small (23 days), so this is a PILOT, not a verdict.

BLINDNESS IS MECHANICAL. This script loads no price data of any kind. It emits the
event, the firms already named in its coverage (which are excluded from scoring),
and the candidate universe. `llm_exposure_score.py` does the join afterwards.

Usage:
    uv run scripts/llm_exposure_dump.py > /tmp/exposure_blind.txt
"""
from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg_live2"
MIN_BARS, FRESH = 400, "2026-07-10"


def usable_universe():
    """Liquid, fresh, US-listed names. NO prices are returned — only the symbols."""
    import datetime as dt
    out = []
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
        pts = [(t, c) for t, c in zip(ts, cl) if c is not None]
        if len(pts) < MIN_BARS:
            continue
        last = dt.datetime.fromtimestamp(pts[-1][0], dt.timezone.utc).date().isoformat()
        if last < FRESH:
            continue
        tail = [c for _, c in pts[-60:]]
        z = sum(1 for i in range(1, len(tail)) if abs(tail[i] - tail[i - 1]) < 1e-9) / max(1, len(tail) - 1)
        if z < 0.15:
            out.append(f[:-5])
    return sorted(out)


def main() -> None:
    uni = set(usable_universe())
    tick, sector, name_of = {}, {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t and t in uni:
            tick[j["entity_id"]] = t
            if j.get("std_sector") not in (None, "UNK"):
                sector.setdefault(t, j["std_sector"])
            name_of.setdefault(t, j.get("name") or t)

    docday, headline = {}, {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
            headline[j["doc_id"]] = (j.get("headline") or "").strip()

    # group causal edges by (date, cause entity) -> an "event"
    ev = collections.defaultdict(lambda: {"firms": set(), "quotes": [], "docs": set()})
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        c, e, doc = j.get("cause_entity"), j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        if not c or not d:
            continue
        k = (d, c)
        ev[k]["docs"].add(doc)
        if e in tick:
            ev[k]["firms"].add(tick[e])
        q = (j.get("quote") or "").strip()
        if q and len(ev[k]["quotes"]) < 3:
            ev[k]["quotes"].append(q[:170])

    rows = [(k, v) for k, v in ev.items() if len(v["docs"]) >= 2 and v["quotes"]]
    rows.sort(key=lambda x: -len(x[1]["docs"]))
    rows = rows[:12]

    print("CANDIDATE UNIVERSE — rank exposure within this set only")
    print(f"({len(uni)} liquid US-listed names, prices deliberately not shown)\n")
    bysec = collections.defaultdict(list)
    for t in sorted(uni):
        bysec[sector.get(t, "—")].append(t)
    for sec in sorted(bysec):
        print(f"  {sec:24} {' '.join(bysec[sec])}")

    print(f"\n\n{'='*100}\nEVENTS ({len(rows)}) — for each, name the firms you expect to move")
    print(f"{'='*100}")
    for (d, c), v in rows:
        print(f"\n▶ EVENT  {d}  ·  cause: {c}")
        hs = [headline[x] for x in list(v['docs'])[:3] if headline.get(x)]
        for h in hs:
            print(f"    headline: {h[:120]}")
        for q in v["quotes"]:
            print(f"    quote:    \"{q}\"")
        print(f"    ALREADY NAMED in this coverage (excluded from scoring): "
              f"{', '.join(sorted(v['firms'])) if v['firms'] else '(none)'}")
    print(f"\n\n# NO PRICE DATA APPEARS IN THIS FILE BY DESIGN.", file=sys.stderr)


if __name__ == "__main__":
    main()
