#!/usr/bin/env python3
"""E2E test of the extras-supply hypothesis: do route-recovered root
candidates (evidence dense@50 misses), fed to the SAME analyst-note
reasoning prompt at the SAME budget, produce better root-cause answers?

Arms (matched 50-fact budget, identical instructions, answer-key graded):
  dense   = dense top-50
  extras  = dense top-40 + up to 10 route-recovered missed-root facts
            (lex_ent/lex_full/walk/reproj interleaved by rank), flagged as
            supplementary alternative-route retrievals

Uses the temporally-relabeled chains (±2d event windows) and the existing
rankings2.json — no re-embedding. Answer key = canonical fact's cause.

  prep   -> writes gen_T###_{dense,extras}.txt into eval/xe_out/
  (OUT=eval/xe_out bash eval/run_ablation.sh gen)   opus generation
  grade  -> haiku answer-key grading + paired comparison
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from chunk_baseline import _fact_line  # noqa: E402
from twopass import GRADE_TMPL, _grade_arm, SYNTH_TMPL  # noqa: E402

OUT = HERE / "xe_out"
N_SAMPLE = 30
N_EXTRAS = 10
ROUTES = ["lex_ent", "lex_full", "walk", "reproj"]

NOTE_TMPL = """You are a market analyst. Below are numbered atomic facts retrieved as evidence. Structure your answer as an analyst note (3-8 sentences):
1. The most likely explanation, weighed from the evidence.
2. Where the evidence points to competing explanations, weigh them explicitly.
3. Any non-consensus or tail signal that might matter more than the consensus, and why.
Cite item numbers in brackets. News is aggregated collective reasoning — distinguishing what consensus believes from what might actually be relevant is the value of the note.

QUESTION: {query}

FACTS:
{items}
"""


def _load():
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    chains = {c["id"]: c for c in
              (json.loads(l) for l in open(HERE / "fusion_out" / "train_chains.jsonl")
               if l.strip())}
    ranked = {r["id"]: r for r in
              json.loads((HERE / "fusion_out" / "rankings2.json").read_text())
              if r["id"] in chains}
    return rows, chains, ranked


def _extras_for(c, r, k=50):
    dense50 = set(r["routes"]["dense"][:k])
    missed = set(c["root_ids"]) - dense50
    picked, pos = [], 0
    while len(picked) < N_EXTRAS:
        advanced = False
        for rt in ROUTES:
            lst = r["routes"].get(rt, [])
            if pos < len(lst):
                advanced = True
                i = lst[pos]
                if i in missed and i not in picked:
                    picked.append(i)
                    if len(picked) >= N_EXTRAS:
                        break
        if not advanced:
            break
        pos += 1
    return picked


def _questions(tag):
    return json.loads((OUT / f"questions{tag}.json").read_text())


def cmd_prep(args):
    rows, chains, ranked = _load()
    eligible = []
    for cid, c in chains.items():
        r = ranked.get(cid)
        if not r:
            continue
        cause = str(rows[c["canon_idx"]].get("cause") or "")
        if cause.lower() in ("", "none", "null"):
            continue
        if args.max_root_docs:
            nd = len({rows[i]["doc_id"] for i in c["root_ids"]})
            if nd > args.max_root_docs:
                continue
        if len(_extras_for(c, r)) >= (2 if args.max_root_docs else 5):
            eligible.append(cid)
    rng = random.Random(11)
    sample = sorted(rng.sample(sorted(eligible), min(N_SAMPLE, len(eligible))))
    OUT.mkdir(exist_ok=True)
    questions = []
    for cid in sample:
        c, r = chains[cid], ranked[cid]
        extras = _extras_for(c, r)
        dense = [i for i in r["routes"]["dense"] if i not in set(extras)]
        arm_a = dense[:50]
        arm_b = dense[:50 - len(extras)]
        block_a = "\n".join(_fact_line(n + 1, rows[i]) for n, i in enumerate(arm_a))
        lines_b = [_fact_line(n + 1, rows[i]) for n, i in enumerate(arm_b)]
        lines_b.append("--- SUPPLEMENTARY (retrieved via alternative routes; "
                       "candidate upstream drivers to weigh) ---")
        lines_b += [_fact_line(len(arm_b) + n + 1, rows[i])
                    for n, i in enumerate(extras)]
        (OUT / f"gen_{cid}_dense.txt").write_text(
            NOTE_TMPL.format(query=c["query"], items=block_a))
        (OUT / f"gen_{cid}_extras.txt").write_text(
            NOTE_TMPL.format(query=c["query"], items="\n".join(lines_b)))
        questions.append({"id": cid, "query": c["query"],
                          "canonical_cause": str(rows[c["canon_idx"]]["cause"]),
                          "n_extras": len(extras)})
        print(f"  {cid} extras={len(extras)}", file=sys.stderr)
    (OUT / f"questions{args.tag}.json").write_text(json.dumps(questions, indent=2))
    print(f"prepared {len(questions)} question pairs "
          f"({len(eligible)} eligible chains)")


def cmd_grade(args):
    qs = _questions(args.tag)
    res = {}
    for arm in ("dense", "extras"):
        sc = _grade_arm(qs, lambda q, a=arm: OUT / f"ans_{q['id']}_{a}.txt",
                        OUT / f"grades_{arm}")
        vals = [v for v in sc.values() if v is not None]
        res[arm] = sc
        print(f"{arm:<7} n={len(vals)} mean={sum(vals)/len(vals):.2f} "
              f"dist={{0: {vals.count(0)}, 1: {vals.count(1)}, 2: {vals.count(2)}}}")
    better = worse = same = 0
    for q in qs:
        a, b = res["extras"].get(q["id"]), res["dense"].get(q["id"])
        if a is None or b is None:
            continue
        better += a > b
        worse += a < b
        same += a == b
    print(json.dumps({"paired": {"extras_better": better,
                                 "dense_better": worse, "same": same}}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "grade"])
    p.add_argument("--max-root-docs", type=int, default=0,
                   help="only chains whose root evidence spans <= N docs")
    p.add_argument("--tag", default="", help="suffix for questions file")
    args = p.parse_args()
    {"prep": cmd_prep, "grade": cmd_grade}[args.stage](args)


if __name__ == "__main__":
    main()
