#!/usr/bin/env python3
"""Forced-choice answering over PFCA questions — objective scoring.

Arms (per question, options shuffled with a per-question seed):
  evidence   top-K facts DEFINITELY published before the cutoff, rendered
             as context; model picks the letter
  closedbook no context at all — controls for guessable distractors; a
             clean question set scores near-chance here

Score = accuracy vs the truth letter. No judge, no prose grading.

  uv run ... python eval/pfca_answer.py run [--questions FILE]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

from chunk_baseline import _fact_line  # noqa: E402
from pfca_hourly import _load_facts  # noqa: E402
from strat_blind import _embed_all  # noqa: E402

OUT = HERE / "pfca_out"
CLAUDE = "/opt/homebrew/bin/claude"
MODEL = "claude-opus-4-8"
K = 40

PICK_TMPL = """{context}QUESTION: {query}

CANDIDATE DRIVERS:
{options}

NOTE: the move happened BEFORE any explanation was published, so the true
driver may not be directly stated in the evidence. Weigh precursors and
escalation dynamics in the evidence (what was building), not just what is
explicitly asserted.

Reply with exactly one line: ANSWER: <letter>"""


def _ask(prompt):
    r = subprocess.run([CLAUDE, "-p", "--model", MODEL],
                       input=prompt, capture_output=True, text=True, timeout=300)
    m = re.search(r"ANSWER:\s*([A-E])", r.stdout or "", re.I)
    return m.group(1).upper() if m else None


def cmd_run(args):
    qs = [json.loads(l) for l in open(args.questions) if l.strip()]
    if not qs:
        sys.exit("no questions")
    rows = _load_facts()
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}"
                       for f in rows])
    OUT.mkdir(exist_ok=True)
    import numpy as np
    results = {"evidence": [], "closedbook": []}
    for q in qs:
        rng = random.Random(q["id"])
        opts = [q["truth"]] + q["distractors"]
        rng.shuffle(opts)
        letters = "ABCDE"[:len(opts)]
        truth_letter = letters[opts.index(q["truth"])]
        opt_block = "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts))

        cutoff = dt.datetime.fromisoformat(q["cutoff_utc"])
        qv = _embed_all([q["query"]])[0]
        sims = fmat @ qv
        order = [int(i) for i in np.argsort(-sims)
                 if rows[int(i)]["_ts_use"] and rows[int(i)]["_ts_use"] < cutoff][:K]
        ctx = "EVIDENCE (all published before the question time):\n" + "\n".join(
            _fact_line(n + 1, rows[i]) for n, i in enumerate(order)) + "\n\n"

        for arm, context in (("evidence", ctx), ("closedbook", "")):
            cf = OUT / f"{q['id']}_{arm}.txt"
            if cf.exists() and cf.read_text().strip():
                pick = cf.read_text().strip()
            else:
                pick = _ask(PICK_TMPL.format(context=context, query=q["query"],
                                             options=opt_block)) or "?"
                cf.write_text(pick)
            ok = pick == truth_letter
            results[arm].append(ok)
            print(f"  {q['id']} {arm:<10} pick={pick} truth={truth_letter} "
                  f"{'OK' if ok else 'X'}", file=sys.stderr)
    n = len(qs)
    print(json.dumps({
        "n": n,
        "evidence_acc": round(sum(results["evidence"]) / n, 3),
        "closedbook_acc": round(sum(results["closedbook"]) / n, 3),
        "chance": round(1 / 5, 3),
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["run"])
    p.add_argument("--questions",
                   default=str(HERE / "pfca_hourly_questions.jsonl"))
    cmd_run(p.parse_args())


if __name__ == "__main__":
    main()
