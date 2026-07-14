#!/usr/bin/env python3
"""Recompute train_chains endpoint/root row-indices against the CURRENT
facts.parquet (row indices go stale whenever the parquet is rebuilt).
Keeps chain ids/queries so the reprojection cache stays valid."""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from fusion_train import _hay, OUT  # noqa: E402
from weighted_walk_test import norm_event_time  # noqa: E402

WINDOW_DAYS = 2  # supports must sit within +/- this of the canonical event


def main():
    import datetime as dt
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    hays = [_hay(g) for g in rows]
    etimes = [norm_event_time(g.get("time"), g.get("published_date")) for g in rows]
    chains = [json.loads(l) for l in open(OUT / "train_chains.jsonl") if l.strip()]
    kept, dropped = [], 0
    for c in chains:
        claim = c["query"].removeprefix("Explain why this happened: ").rstrip(".")
        canon = None
        for i, f in enumerate(rows):
            if f["doc_id"] == c["exclude_doc"] \
                    and str(f.get("claim") or "").rstrip(".") == claim:
                canon = (i, f)
                break
        if not canon:
            dropped += 1
            continue
        fi, f = canon
        ct = etimes[fi]
        if not ct:
            dropped += 1
            continue
        lo, hi = (ct - dt.timedelta(days=WINDOW_DAYS),
                  ct + dt.timedelta(days=WINDOW_DAYS))
        subj = str(f.get("subject") or "").lower().strip()
        ce = [e for e in (f.get("cause_entities") or []) if len(e) > 3]
        endpoint = [i for i, g in enumerate(rows)
                    if g["doc_id"] != f["doc_id"] and subj in hays[i]
                    and etimes[i] and lo <= etimes[i] <= hi]
        root = [i for i, g in enumerate(rows)
                if g["doc_id"] != f["doc_id"] and any(e in hays[i] for e in ce)
                and etimes[i] and lo <= etimes[i] <= hi]
        if len(endpoint) < 2 or len(root) < 2:
            dropped += 1
            continue
        c["canon_idx"] = fi
        c["endpoint_ids"] = endpoint[:400]
        c["root_ids"] = root[:400]
        kept.append(c)
    (OUT / "train_chains.jsonl").write_text(
        "\n".join(json.dumps(c) for c in kept) + "\n")
    print(f"relabeled {len(kept)} chains against {len(rows)} facts "
          f"(dropped {dropped}: canonical missing or supports thinned)")


if __name__ == "__main__":
    main()
