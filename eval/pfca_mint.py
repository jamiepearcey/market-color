#!/usr/bin/env python3
"""PFCA — Prospective Forced-Choice Attribution question minting.

Clean-question gates (all mechanical, pre-registered):
  G1 consensus ground truth: >=2 independent source_names assert
     semantically-agreeing causes (cos >= 0.70) for the same
     (subject, direction, event-day) move.
  G2 prospective: first publication of any consensus assertion is
     AFTER the event day T (the link is printed later, not at T).
  G3 leak-free at T: no fact published <= T has a claim OR cause
     within cos 0.60 of the ground-truth attribution.
  G4 answerable at T: >=2 facts published <= T mention the root
     cause entities (the ingredients are in print pre-T).
  G5 distractors: >=3 same-window candidate causes over the same
     subject/desk that are NOT the truth (cos < 0.55).

  mint  -> counts per gate + writes eval/pfca_questions.jsonl + examples
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from strat_blind import _embed_all  # noqa: E402
from weighted_walk_test import norm_event_time  # noqa: E402

CONSENSUS_COS = 0.70
LEAK_COS = 0.60
DISTRACTOR_COS = 0.55
MIN_LAG_H = 2.0  # attribution must trail the first move report by >= this

# Docs still date-floored (no recoverable time-of-day) get ASYMMETRIC
# treatment: for leak scanning they count as published at 00:00 (could be a
# leak -> question discarded), for evidence/answerability at 23:59:59 (can't
# be trusted as pre-cutoff -> not usable). Both errors discard questions;
# neither contaminates one.


def _valid_cause(f):
    c = str(f.get("cause") or "")
    return c and c.lower() not in ("none", "null") and len(c.split()) >= 4


def cmd_mint(args):
    import duckdb
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    pub_ts = {d: u for d, u in duckdb.sql(
        f"select doc_id, published_utc from "
        f"read_parquet('{ROOT}/data/news_corpus/dt=*/part-*.parquet')").fetchall()}
    for f in rows:
        f["_etime"] = norm_event_time(f.get("time"), f.get("published_date"))
        try:
            f["_pub"] = dt.date.fromisoformat(str(f.get("published_date"))[:10])
        except Exception:
            f["_pub"] = None
        ts = None
        raw = pub_ts.get(f["doc_id"])
        if raw:
            try:
                ts = dt.datetime.fromisoformat(str(raw)).replace(tzinfo=None)
            except Exception:
                ts = None
        floored = ts is None or (ts.hour, ts.minute, ts.second) == (0, 0, 0)
        base = ts if ts is not None else (
            dt.datetime.combine(f["_pub"], dt.time()) if f["_pub"] else None)
        # asymmetric bounds for floored docs (see note above)
        f["_ts_leak"] = base  # earliest it could have appeared
        f["_ts_use"] = (base.replace(hour=23, minute=59, second=59)
                        if (base is not None and floored) else base)
    print("[mint] embedding causes + claims", file=sys.stderr)
    cause_v = _embed_all([str(f.get("cause") or "") if _valid_cause(f) else "-"
                          for f in rows])
    claim_v = _embed_all([str(f.get("claim") or "") for f in rows])

    # candidate relationships: same subject+direction+event-day, causal facts;
    # mentions = ALL facts on that move (incl. cause-less move reports)
    groups = collections.defaultdict(list)
    mentions = collections.defaultdict(list)
    for i, f in enumerate(rows):
        if not (f["_etime"] and f["_pub"]):
            continue
        subj = str(f.get("subject") or "").lower().strip()
        d = str(f.get("direction") or "")
        if len(subj) >= 4 and d in ("up", "down"):
            mentions[(subj, d, f["_etime"])].append(i)
            if _valid_cause(f):
                groups[(subj, d, f["_etime"])].append(i)

    pubs = [f["_pub"] for f in rows if f["_pub"]]
    corpus_lo, corpus_hi = min(pubs), max(pubs)
    stats = collections.Counter()
    questions = []
    for (subj, d, T), idxs in groups.items():
        # G0: the EVENT itself must sit inside the corpus window (otherwise
        # "published <= T" is empty and later gates pass vacuously)
        if not (corpus_lo <= T < corpus_hi):
            continue
        stats["G0_in_window"] += 1
        # G1: >=2 distinct sources with agreeing causes
        by_src = collections.defaultdict(list)
        for i in idxs:
            by_src[str(rows[i].get("source_name") or "?")].append(i)
        if len(by_src) < 2:
            continue
        stats["multi_source_groups"] += 1
        reps = [v[0] for v in by_src.values()]
        best_pair = None
        for a in range(len(reps)):
            for b in range(a + 1, len(reps)):
                cos = float(cause_v[reps[a]] @ cause_v[reps[b]])
                if cos >= CONSENSUS_COS and (best_pair is None or cos > best_pair[0]):
                    best_pair = (cos, reps[a], reps[b])
        if not best_pair:
            continue
        stats["G1_consensus"] += 1
        _, ia, ib = best_pair
        truth_i = ia if rows[ia]["_pub"] <= rows[ib]["_pub"] else ib
        truth = str(rows[truth_i]["cause"])

        # cutoff = earliest appearance of ANY consensus attribution. Clusters
        # containing a date-floored attribution have UNMEASURABLE timing —
        # discard them (can't place the cutoff) rather than fake midnight.
        attr_ts = [rows[i]["_ts_leak"] for i in idxs if rows[i]["_ts_leak"]]
        if not attr_ts:
            continue
        if any(rows[i]["_ts_leak"] != rows[i]["_ts_use"] for i in idxs):
            stats["G2_ambiguous_timing"] += 1
            continue
        cutoff = min(attr_ts)

        # G2 (hour-level prospective): the move itself must be in print
        # >= MIN_LAG_H before the first attribution
        move_ts = [rows[j]["_ts_use"] for j in mentions[(subj, d, T)]
                   if rows[j]["_ts_use"] and not _valid_cause(rows[j])]
        if not move_ts or (cutoff - min(move_ts)) < dt.timedelta(hours=MIN_LAG_H):
            lag = (cutoff - min(move_ts)) if move_ts else None
            print(f"  G2-fail {subj[:28]:<28} T={T} cutoff={cutoff} "
                  f"bare-move-reports={len(move_ts)} lag={lag}", file=sys.stderr)
            continue
        stats["G2_prospective"] += 1

        # G3: leak scan over everything possibly visible before the cutoff
        tv = cause_v[truth_i]
        pre_leak = [j for j, g in enumerate(rows)
                    if g["_ts_leak"] and g["_ts_leak"] < cutoff]
        leak = any(max(float(claim_v[j] @ tv),
                       float(cause_v[j] @ tv) if _valid_cause(rows[j]) else 0.0)
                   >= LEAK_COS for j in pre_leak)
        if leak:
            continue
        stats["G3_leakfree"] += 1
        # evidence pool = facts DEFINITELY published before the cutoff
        pre = [j for j, g in enumerate(rows)
               if g["_ts_use"] and g["_ts_use"] < cutoff]

        # G4: root ingredients in print pre-T
        ce = [e.lower() for e in (rows[truth_i].get("cause_entities") or [])
              if len(e) > 3]
        if not ce:
            print(f"  G4-fail {subj[:30]} T={T}: NO cause_entities "
                  f"(truth: {truth[:80]})", file=sys.stderr)
            continue
        support = sum(1 for j in pre if any(
            e in (" ".join([str(rows[j].get("claim") or ""),
                            " ".join(rows[j].get("entities") or [])]).lower())
            for e in ce))
        if support < 2:
            print(f"  G4-fail {subj[:30]} T={T}: support={support} "
                  f"ce={ce[:3]} (truth: {truth[:80]})", file=sys.stderr)
            continue
        stats["G4_answerable"] += 1

        # G5: same-window plausible distractors (other causes, same desk/subject-adjacent)
        desk = rows[truth_i].get("desk")
        cand = [j for j in pre if _valid_cause(rows[j])
                and rows[j].get("desk") == desk
                and float(cause_v[j] @ tv) < DISTRACTOR_COS]
        seen, distractors = set(), []
        for j in sorted(cand, key=lambda j: -float(cause_v[j] @ claim_v[truth_i])):
            key = str(rows[j]["cause"])[:60].lower()
            if key not in seen:
                seen.add(key)
                distractors.append(str(rows[j]["cause"]))
            if len(distractors) >= 4:
                break
        if len(distractors) < 3:
            continue
        stats["G5_distractors"] += 1
        questions.append({
            "id": f"P{len(questions):03d}", "subject": subj, "direction": d,
            "event_date": T.isoformat(), "cutoff_utc": cutoff.isoformat(),
            "move_first_seen": min(move_ts).isoformat(),
            "query": f"As of {cutoff.isoformat()}, {subj} has moved {d} and no "
                     f"published explanation exists yet. Which candidate driver "
                     f"most likely explains it?",
            "truth": truth, "distractors": distractors,
            "truth_sources": sorted(by_src.keys()),
        })

    (HERE / "pfca_questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions) + ("\n" if questions else ""))
    print(json.dumps(stats, indent=2))
    print(f"minted {len(questions)} clean questions -> eval/pfca_questions.jsonl")
    for q in questions[:3]:
        print(f"\n{q['id']} {q['subject']} {q['direction']} T={q['event_date']} "
              f"cutoff {q['cutoff_utc']}")
        print(f"  TRUTH: {q['truth'][:120]}")
        print(f"  D1:    {q['distractors'][0][:120]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mint"])
    cmd_mint(p.parse_args())


if __name__ == "__main__":
    main()
