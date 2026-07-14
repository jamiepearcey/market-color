#!/usr/bin/env python3
"""Chain retest: was the cause-walk discounted early on a biased question mix?

15 canonical-exclusion chain questions (eval/blind_questions_chains.jsonl).
Arms:
  v2      hybrid v2 as fielded in the stratified benchmark (answers for the
          5 A-* questions are REUSED from eval/strat_out/).
  v2walk  identical, plus the restored cause-walk: for top facts with a
          DRIVER, embed the driver text and attach the best cross-document
          upstream facts (exclusions respected) as UPSTREAM FACT lines.

Same budget, double-judged (win = both A/B orders).
Stages: prep -> (OUT=chain_out run_ablation.sh gen) -> judge-prep ->
(OUT=chain_out run_ablation.sh judge) -> score.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402
from chunk_baseline import GEN_TMPL, JUDGE_TMPL, _fact_line  # noqa: E402
from strat_blind import _embed_all, _resolve_exclusions, _v2_context  # noqa: E402

import os
OUT = HERE / os.environ.get("CHAIN_OUT", "chain_out")
TEMPORAL = os.environ.get("TEMPORAL") == "1"
STRAT = HERE / "strat_out"
WALK_SEEDS = 8   # top facts whose DRIVER we walk
WALK_PER = 2     # upstream facts attached per driver


def _walk_lines(query_vec, fact_hits, facts, fmat, excluded, used_fids, start_idx):
    """Cause-walk: embed DRIVER texts of top facts, attach best cross-doc
    upstream facts. Returns list of rendered UPSTREAM FACT lines."""
    drivers = []
    for rank, (_, f) in enumerate(fact_hits[:WALK_SEEDS]):
        c = f.get("cause")
        if c and str(c).lower() not in ("none", "null"):
            drivers.append((rank, f, str(c)))
    if not drivers:
        return []
    dvecs = _embed_all([c for _, _, c in drivers])
    lines = []
    for (rank, parent, _), dv in zip(drivers, dvecs, strict=True):
        sims = fmat @ dv
        added = 0
        for i in np.argsort(-sims):
            f = facts[int(i)]
            if (f["fact_id"] in used_fids or f["doc_id"] in excluded
                    or f["doc_id"] == parent["doc_id"]):
                continue
            if TEMPORAL:
                pe, fe = parent.get("published_epoch") or 0, f.get("published_epoch") or 0
                if pe and fe and fe > pe:
                    continue
            cause = (f" DRIVER: {f['cause']}"
                     if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else "")
            lines.append(f"[{start_idx + len(lines)}] UPSTREAM FACT (behind item on "
                         f"'{parent.get('subject')}'): {f.get('claim')} "
                         f"(source: {f.get('source_name')}, {f.get('published_date')}){cause}")
            used_fids.add(f["fact_id"])
            added += 1
            if added >= WALK_PER:
                break
    return lines


def cmd_prep(args):
    import pyarrow.parquet as pq
    from qdrant_client.http import models  # noqa: F401  (strat imports need qdrant up)
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    extracted_docs = {f["doc_id"] for f in facts}
    by_doc: dict[str, list] = {}
    for f in facts:
        by_doc.setdefault(f["doc_id"], []).append(f)
    print(f"[chain] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions_chains.jsonl") if l.strip()]
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in questions:
        excluded = _resolve_exclusions(q, facts)
        items, budget = _v2_context(q["query"], facts, fmat, client, by_doc,
                                    extracted_docs, excluded)
        (OUT / f"gen_{q['id']}_v2.txt").write_text(GEN_TMPL.format(query=q["query"], items=items))
        # reuse existing v2 answers when the gen prompt is byte-identical
        # (strat_out for A-* questions, chain_out for the rest)
        reused = False
        for src in (STRAT, HERE / "chain_out"):
            ans = src / f"ans_{q['id']}_v2.txt"
            gen = src / f"gen_{q['id']}_v2.txt"
            if (ans.exists() and gen.exists()
                    and gen.read_text() == (OUT / f"gen_{q['id']}_v2.txt").read_text()):
                shutil.copy(ans, OUT / f"ans_{q['id']}_v2.txt")
                reused = True
                break

        # v2walk: rebuild v2 context but reserve ~25% of budget for walk lines
        qv = _embed_all([q["query"]])[0]
        sims = fmat @ qv
        order = [int(i) for i in np.argsort(-sims)
                 if facts[int(i)]["doc_id"] not in excluded][:50]
        fact_hits = [(float(sims[i]), facts[i]) for i in order]
        used_fids: set = set()
        walk = _walk_lines(qv, fact_hits, facts, fmat, excluded, used_fids, start_idx=1)
        walk_chars = sum(len(l) for l in walk)
        # v2 context truncated to leave room, then walk lines appended, renumbered
        base_items, _ = _v2_context(q["query"], facts, fmat, client, by_doc,
                                    extracted_docs, excluded)
        base_lines = base_items.split("\n")
        kept, used = [], 0
        for line in base_lines:
            if used + len(line) > budget - walk_chars and kept:
                break
            kept.append(line)
            used += len(line)
        renum = []
        for i, l in enumerate(kept + walk):
            renum.append(re.sub(r"^\[\d+\]", f"[{i + 1}]", l))
        (OUT / f"gen_{q['id']}_v2walk.txt").write_text(
            GEN_TMPL.format(query=q["query"], items="\n".join(renum)))
        manifest.append({"id": q["id"], "query": q["query"], "budget": budget,
                         "excluded_docs": sorted(excluded), "walk_lines": len(walk),
                         "v2_answer_reused": reused})
        print(f"  {q['id']:<18} excluded={len(excluded)} walk_lines={len(walk)} "
              f"reused_v2_ans={reused}", file=sys.stderr)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(manifest)} questions")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        a_p = OUT / f"ans_{m['id']}_v2walk.txt"
        b_p = OUT / f"ans_{m['id']}_v2.txt"
        if not a_p.exists() or not b_p.exists():
            print(f"missing: {m['id']}", file=sys.stderr)
            continue
        a, b = a_p.read_text().strip(), b_p.read_text().strip()
        (OUT / f"judge_{m['id']}_o1.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=a, b=b))
        (OUT / f"judge_{m['id']}_o2.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=b, b=a))
        n += 1
    print(f"prepared {n} pairs x 2 orders")


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    wins = losses = ties = 0
    for m in manifest:
        vs = []
        for order, walk_is_a in (("o1", True), ("o2", False)):
            vp = OUT / f"verdict_{m['id']}_{order}.txt"
            match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text()) if vp.exists() else None
            vs.append(None if not match or match.group(1) == "TIE"
                      else (match.group(1) == "A") == walk_is_a)
        both = [v for v in vs if v is not None]
        if len(both) == 2 and all(both):
            res = "WALK_WIN"; wins += 1
        elif len(both) == 2 and not any(both):
            res = "V2_WIN"; losses += 1
        else:
            res = "TIE"; ties += 1
        print(f"  {m['id']:<18} {res}   (orders: {vs})")
    print(json.dumps({"walk_wins": wins, "v2_wins": losses, "ties": ties}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
