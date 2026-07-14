#!/usr/bin/env python3
"""PFCA v2 — hourly PRICE-anchored prospective question minting.

The event anchor is exogenous (|zscore_1h| >= Z on hourly bars), so the move
never needs to be in print. Gates (mechanical, pre-registered):
  G0 event inside the fact-corpus window
  G1 attribution consensus: >=2 independent sources publish semantically
     agreeing causes (cos >= 0.70) for this instrument, event-window +/-1d
  G2 prospective: earliest attribution published >= MIN_LAG_H after the
     spike bar closes (real time-of-day required on attribution docs)
  G3 leak-free: nothing possibly visible before the cutoff semantically
     matches the attribution (cos >= 0.60), incl. floored docs at 00:00
  G4 answerable: root cause entities appear in >=2 facts definitely
     published before the cutoff
  G5 >=3 plausible same-window distractor causes (cos < 0.55 to truth)

Zero LLM cost: local embeddings only. Uses existing facts.parquet + corpus
published_utc + data/prices/prices_hourly.parquet.

  uv run ... python eval/pfca_hourly.py mint
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

from newslag import INSTRUMENT_KEYWORDS  # noqa: E402
from strat_blind import _embed_all  # noqa: E402
from weighted_walk_test import norm_event_time  # noqa: E402

Z_MIN = 3.0
MIN_LAG_H = 2.0
CONSENSUS_COS = 0.70
# Leak = semantic match AND shared cause entity. Calibration: p99 of max
# cosine from a 10k-fact corpus to arbitrary truth text is ~0.70-0.71
# (MiniLM domain noise); cosine alone cannot mark leaks. 0.75 + entity
# overlap separates narrative continuation from topical noise.
LEAK_COS = 0.75
DISTRACTOR_COS = 0.55


def _valid_cause(f):
    c = str(f.get("cause") or "")
    return c and c.lower() not in ("none", "null") and len(c.split()) >= 4


def _load_facts():
    import duckdb
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    pub_ts = {d: u for d, u in duckdb.sql(
        f"select doc_id, published_utc from "
        f"read_parquet('{ROOT}/data/news_corpus/dt=*/part-*.parquet')").fetchall()}
    for f in rows:
        f["_etime"] = norm_event_time(f.get("time"), f.get("published_date"))
        ts = None
        raw = pub_ts.get(f["doc_id"])
        if raw:
            try:
                ts = dt.datetime.fromisoformat(str(raw))
                ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None) \
                    if ts.tzinfo else ts
            except Exception:
                ts = None
        floored = ts is None or (ts.hour, ts.minute, ts.second) == (0, 0, 0)
        if ts is None:
            try:
                ts = dt.datetime.fromisoformat(str(f.get("published_date"))[:10])
            except Exception:
                ts = None
        f["_ts_leak"] = ts
        f["_ts_use"] = (ts.replace(hour=23, minute=59, second=59)
                        if (ts is not None and floored) else ts)
        hay = (str(f.get("subject") or "") + " " + str(f.get("claim") or "")[:80]
               + " " + " ".join(f.get("entities") or [])).lower()
        f["_inst"] = None
        for sym, kws in INSTRUMENT_KEYWORDS.items():
            if any(k in hay for k in kws):
                f["_inst"] = sym
                break
    return rows


def _load_events():
    import duckdb
    px = duckdb.sql(
        f"select symbol, ts_utc, ret_1h, zscore_1h from "
        f"'{ROOT}/data/prices/prices_hourly.parquet' "
        f"where abs(zscore_1h) >= {Z_MIN}").fetchall()
    out = []
    for sym, ts, ret, z in px:
        t = ts.astimezone(dt.timezone.utc).replace(tzinfo=None) \
            if ts.tzinfo else ts
        out.append({"sym": sym, "t": t, "ret": ret, "z": z})
    return out


def cmd_mint(args):
    global Z_MIN, MIN_LAG_H
    Z_MIN = args.z
    MIN_LAG_H = args.lag
    rows = _load_facts()
    events = _load_events()
    taxonomy = collections.defaultdict(collections.Counter)
    print(f"[mint] {len(events)} hourly spike events, {len(rows)} facts",
          file=sys.stderr)
    print("[mint] embedding causes + claims", file=sys.stderr)
    cause_v = _embed_all([str(f.get("cause") or "") if _valid_cause(f) else "-"
                          for f in rows])
    claim_v = _embed_all([str(f.get("claim") or "") for f in rows])

    pubs = [f["_ts_leak"] for f in rows if f["_ts_leak"]]
    lo, hi = min(pubs), max(pubs)
    stats = collections.Counter()
    questions = []
    for ev in events:
        sym, t = ev["sym"], ev["t"]
        if not (lo <= t <= hi - dt.timedelta(hours=6)):
            continue
        stats["G0_in_window"] += 1
        d = "up" if ev["ret"] > 0 else "down"

        # attribution candidates: instrument-matched causal facts, direction
        # compatible, event time within +/-1d of the spike, REAL timestamps,
        # published AFTER the spike bar
        cands = []
        for i, f in enumerate(rows):
            if f["_inst"] != sym or not _valid_cause(f):
                continue
            if str(f.get("direction")) in ("up", "down") and f["direction"] != d:
                continue
            if not f["_etime"] or abs((f["_etime"]
                    - t.date()).days) > 1:
                continue
            if f["_ts_leak"] is None or f["_ts_leak"] != f["_ts_use"]:
                continue  # floored: timing unmeasurable
            if f["_ts_leak"] <= t:
                continue
            cands.append(i)
        if not cands:
            continue
        stats["has_post_attribution"] += 1

        # G1 consensus across >=2 sources
        by_src = collections.defaultdict(list)
        for i in cands:
            by_src[str(rows[i].get("source_name") or "?")].append(i)
        if len(by_src) < 2:
            continue
        best = None
        reps = [v[0] for v in by_src.values()]
        for a in range(len(reps)):
            for b in range(a + 1, len(reps)):
                cos = float(cause_v[reps[a]] @ cause_v[reps[b]])
                if cos >= CONSENSUS_COS and (best is None or cos > best[0]):
                    best = (cos, reps[a], reps[b])
        if not best:
            continue
        stats["G1_consensus"] += 1
        _, ia, ib = best
        truth_i = ia if rows[ia]["_ts_leak"] <= rows[ib]["_ts_leak"] else ib
        truth = str(rows[truth_i]["cause"])

        # cutoff = FIRST PRINT of this episode's cause anywhere in the corpus
        # (consensus above only VERIFIES the truth; the earliest matching
        # statement is the information event)
        tv = cause_v[truth_i]
        truth_ce = [e.lower() for e in
                    (rows[truth_i].get("cause_entities") or []) if len(e) > 3]

        def _states_cause(j):
            sim = max(float(claim_v[j] @ tv),
                      float(cause_v[j] @ tv) if _valid_cause(rows[j]) else 0.0)
            if sim < LEAK_COS:
                return False
            hay = (" ".join([str(rows[j].get("claim") or ""),
                             str(rows[j].get("cause") or ""),
                             " ".join(rows[j].get("entities") or [])])).lower()
            if not any(e in hay for e in truth_ce):
                return False
            et = rows[j]["_etime"]
            return et is None or abs((et - t.date()).days) <= 1

        stated = [j for j, g in enumerate(rows)
                  if g["_ts_leak"] and _states_cause(j)]
        first_print = min((rows[j]["_ts_leak"] for j in stated),
                          default=rows[truth_i]["_ts_leak"])
        if first_print <= t:
            stats["news_led"] += 1  # cause in print BEFORE the spike
            taxonomy[sym]["news_led"] += 1
            continue
        if (first_print - t) < dt.timedelta(hours=MIN_LAG_H):
            stats["near_simultaneous"] += 1
            taxonomy[sym]["near_simultaneous"] += 1
            continue
        cutoff = first_print
        stats["G2_market_led"] += 1
        taxonomy[sym]["market_led"] += 1

        # G3 leak scan: semantic match AND shared cause entity
        tv = cause_v[truth_i]
        truth_ce = [e.lower() for e in
                    (rows[truth_i].get("cause_entities") or []) if len(e) > 3]
        pre_leak = [j for j, g in enumerate(rows)
                    if g["_ts_leak"] and g["_ts_leak"] < cutoff]

        def _is_leak(j):
            sim = max(float(claim_v[j] @ tv),
                      float(cause_v[j] @ tv) if _valid_cause(rows[j]) else 0.0)
            if sim < LEAK_COS:
                return False
            hay = (" ".join([str(rows[j].get("claim") or ""),
                             str(rows[j].get("cause") or ""),
                             " ".join(rows[j].get("entities") or [])])).lower()
            if not any(e in hay for e in truth_ce):
                return False
            # same narrative from a PRIOR episode is history, not a leak of
            # this spike's cause; leak requires the matching fact's own event
            # to sit in the spike window (unknown event time = leak, conservative)
            et = rows[j]["_etime"]
            return et is None or abs((et - t.date()).days) <= 1

        if any(_is_leak(j) for j in pre_leak):
            stats["G3_narrative_leak"] += 1
            continue
        stats["G3_leakfree"] += 1

        pre = [j for j, g in enumerate(rows)
               if g["_ts_use"] and g["_ts_use"] < cutoff]
        ce = [e.lower() for e in (rows[truth_i].get("cause_entities") or [])
              if len(e) > 3]
        if not ce:
            continue
        support = sum(1 for j in pre if any(
            e in (" ".join([str(rows[j].get("claim") or ""),
                            " ".join(rows[j].get("entities") or [])]).lower())
            for e in ce))
        if support < 2:
            continue
        stats["G4_answerable"] += 1

        # plausibility-matched distractors: same INSTRUMENT's other candidate
        # causes first (a guesser can't pick "the only oil-shaped option"),
        # in a moderate similarity band to the truth
        def _distractor_pool(same_inst):
            return [j for j in pre if _valid_cause(rows[j])
                    and (rows[j]["_inst"] == sym) == same_inst
                    and 0.30 <= float(cause_v[j] @ tv) < DISTRACTOR_COS]
        seen, distractors = set(), []
        for pool in (_distractor_pool(True), _distractor_pool(False)):
            for j in sorted(pool, key=lambda j: -float(cause_v[j] @ tv)):
                key = str(rows[j]["cause"])[:60].lower()
                if key not in seen:
                    seen.add(key)
                    distractors.append(str(rows[j]["cause"]))
                if len(distractors) >= 4:
                    break
            if len(distractors) >= 4:
                break
        if len(distractors) < 3:
            continue
        stats["G5_minted"] += 1
        questions.append({
            "id": f"H{len(questions):03d}", "symbol": sym,
            "spike_utc": t.isoformat(), "z": round(ev["z"], 2),
            "direction": d, "cutoff_utc": cutoff.isoformat(),
            "lag_hours": round((cutoff - t).total_seconds() / 3600, 1),
            "query": f"{sym} moved {d} sharply (|z|={abs(ev['z']):.1f}) in the "
                     f"hour ending {t.isoformat()}Z. No published explanation "
                     f"exists yet. Which candidate driver most likely explains it?",
            "truth": truth, "distractors": distractors,
            "truth_sources": sorted(by_src.keys()),
        })

    (HERE / "pfca_hourly_questions.jsonl").write_text(
        "\n".join(json.dumps(q) for q in questions) + ("\n" if questions else ""))
    print(json.dumps({"z": Z_MIN, "lag_h": MIN_LAG_H, **stats}, indent=2))
    if taxonomy:
        print("taxonomy by instrument:")
        for sym in sorted(taxonomy, key=lambda s: -sum(taxonomy[s].values())):
            c = taxonomy[sym]
            print(f"  {sym:<10} news_led={c['news_led']} "
                  f"simult={c['near_simultaneous']} market_led={c['market_led']}")
    print(f"minted {len(questions)} -> eval/pfca_hourly_questions.jsonl")
    for q in questions[:5]:
        print(f"\n{q['id']} {q['symbol']} z={q['z']} spike={q['spike_utc']} "
              f"lag={q['lag_hours']}h")
        print(f"  TRUTH: {q['truth'][:130]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mint"])
    p.add_argument("--z", type=float, default=3.0)
    p.add_argument("--lag", type=float, default=2.0)
    cmd_mint(p.parse_args())


if __name__ == "__main__":
    main()
