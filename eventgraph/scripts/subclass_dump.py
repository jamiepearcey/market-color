# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
STEP 1 of the LLM sub-classification pilot: the BLIND dump.

WHY THIS IS A SEPARATE SCRIPT. The whole value of sub-classifying a bucket by LLM
depends on the labeller never seeing the outcome. If the labels are chosen while
looking at sigma, "we found a high-signal sub-bucket" is guaranteed and
meaningless. So blindness is enforced mechanically rather than by good intentions:
this script emits ONLY (cell_id, ticker, date, headlines) and deliberately does
not load the price panel at all. `subclass_measure.py` does the join afterwards.

The bucket under test is `m_and_a`: 907 cells at 1.21x sector-matched lift -- the
largest and nearly the flattest class in F36. It is the right target because the
prior is mechanical rather than hopeful: announced takeover targets gap 20-40%,
so if that is inside this bucket, a 1.21x average is concealing it.

PRE-REGISTERED SUB-SCHEME (fixed before any label is written):
    target_announced  name is the target of an agreed/announced bid
    target_rumour     name is a rumoured/speculated target, unconfirmed
    acquirer          name is the buyer
    deal_progress     regulatory/closing/completion step in an existing deal
    deal_failed       deal collapsed, withdrawn or blocked
    third_party       name only mentioned in someone else's deal (peer,
                      adviser, sector read-through)

PRE-REGISTERED PREDICTIONS (so the test can fail):
    1. target_announced is the highest sub-class, >=2.5x
    2. target_rumour is elevated but below target_announced
    3. third_party ~1.0x AND is numerous -- this is the dilution hypothesis for
       why the pooled bucket reads 1.21x
    4. acquirer only mildly above 1.0x

Usage:
    uv run scripts/subclass_dump.py --class m_and_a --n 320 > /tmp/mna_blind.tsv
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="eg100k_graph")
    ap.add_argument("--class", dest="klass", default="m_and_a")
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--seed", type=int, default=20260726)
    a = ap.parse_args()
    G = ROOT / a.graph

    tick = {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t:
            tick[j["entity_id"]] = t

    doct = collections.defaultdict(set)
    for l in open(G / "lake" / "event.jsonl"):
        j = json.loads(l)
        if j.get("doc_id"):
            doct[j["doc_id"]].add(tax.std_event_type(j.get("event_type"))[0])

    meta = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            meta[j["doc_id"]] = (d, (j.get("headline") or "").strip())

    cells = collections.defaultdict(set)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        tk, doc = tick.get(j.get("effect_entity")), j.get("doc_id")
        m = meta.get(doc)
        if tk and m and a.klass in doct.get(doc, set()):
            cells[(tk, m[0])].add(m[1])

    rows = [(tk, d, sorted(h for h in hs if h)) for (tk, d), hs in cells.items()]
    rows = [r for r in rows if r[2]]
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.n]

    print("cell_id\tticker\tdate\theadlines")
    for tk, d, hs in sorted(rows, key=lambda r: (r[1], r[0])):
        print(f"{tk}|{d}\t{tk}\t{d}\t" + " ||| ".join(h[:150] for h in hs[:2]))
    print(f"# {len(rows)} cells dumped for class={a.klass} "
          f"(of {len(cells)} total) — NO sigma in this file by design",
          file=sys.stderr)


if __name__ == "__main__":
    main()
