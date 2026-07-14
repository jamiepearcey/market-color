#!/usr/bin/env python3
"""Counterfactual ablation: does asm's attached context change the ANSWER?

Stage `prep`: for each eval case, build two 10-fact contexts (dense@10 vs asm@10)
and write generation prompts + a manifest. Cases where asm == dense (trigger
never fired / attachments empty) are skipped — no assist to measure.
Stage `judge-prep`: after answers are generated, write blind pairwise judge
prompts with randomized A/B order (seeded per case id).
Stage `score`: parse judge verdicts + citation use of attached facts.

Generation/judging themselves run via the claude CLI (see eval/run_ablation.sh).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

OUT = HERE / "ablation_out"

GEN_TMPL = """You are a market analyst. Answer the question using ONLY the numbered facts below. Cite every claim with the fact number in brackets, e.g. [3]. If the facts do not fully answer the question, answer as far as they allow and say what is missing. 2-6 sentences.

QUESTION: {query}

FACTS:
{facts}
"""

JUDGE_TMPL = """You are judging two answers to a market question. Both were written from retrieved facts. Judge which answer better explains the CAUSAL CHAIN — the specific root events and mechanism, not vague gestures at "the conflict" or "market conditions". Groundedness matters: prefer specific, cited causes. Ignore style and length.

QUESTION: {query}

ANSWER A:
{a}

ANSWER B:
{b}

Reply with exactly one line: VERDICT: A or VERDICT: B or VERDICT: TIE
Then one sentence of justification."""


def fact_line(i: int, f: dict) -> str:
    cause = f" DRIVER: {f['cause']}" if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else ""
    return f"[{i}] {f.get('claim')} (source: {f.get('source_name')}, {f.get('published_date')}){cause}"


def cmd_prep(args):
    import ppr_experiment as px

    rt = px.Retriever(argparse.Namespace(
        parquet=str(px.DEFAULT_PARQUET), qdrant_url="http://localhost:6333",
        embedding_provider="fastembed",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        seed_k=8, hub_gamma=0.0))
    cases = []
    for path in (HERE / "ppr_cases_hard.jsonl", HERE / "ppr_cases.jsonl"):
        cases += [json.loads(l) for l in open(path) if l.strip()]

    OUT.mkdir(exist_ok=True)
    manifest = []
    for c in cases:
        dense = [f for _, f in rt.run("dense", c["query"], 10)]
        asm = [f for _, f in rt.run("asm", c["query"], 10)]
        dense_ids = [f["fact_id"] for f in dense]
        asm_ids = [f["fact_id"] for f in asm]
        if dense_ids == asm_ids:
            manifest.append({"id": c["id"], "skipped": "identical_context"})
            continue
        attach_idx = [i + 1 for i, fid in enumerate(asm_ids) if fid not in dense_ids]
        for arm, facts in (("dense", dense), ("asm", asm)):
            prompt = GEN_TMPL.format(
                query=c["query"],
                facts="\n".join(fact_line(i + 1, f) for i, f in enumerate(facts)))
            (OUT / f"gen_{c['id']}_{arm}.txt").write_text(prompt)
        manifest.append({"id": c["id"], "query": c["query"],
                         "attachment_indices": attach_idx})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    live = [m for m in manifest if "skipped" not in m]
    print(f"prepared {len(live)} differing cases / {len(cases)} total "
          f"({len(cases) - len(live)} identical, skipped)")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        if "skipped" in m:
            continue
        ans = {}
        for arm in ("dense", "asm"):
            p = OUT / f"ans_{m['id']}_{arm}.txt"
            if not p.exists() or not p.read_text().strip():
                print(f"missing answer: {p}", file=sys.stderr)
                break
            ans[arm] = p.read_text().strip()
        else:
            # blind order, deterministic per case id
            asm_is_a = int(hashlib.sha256(m["id"].encode()).hexdigest(), 16) % 2 == 0
            a, b = (ans["asm"], ans["dense"]) if asm_is_a else (ans["dense"], ans["asm"])
            (OUT / f"judge_{m['id']}.txt").write_text(
                JUDGE_TMPL.format(query=m["query"], a=a, b=b))
            m["asm_is_a"] = asm_is_a
            n += 1
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {n} judge prompts")


def cmd_score(args):
    import re
    manifest = json.loads((OUT / "manifest.json").read_text())
    rows, wins, losses, ties, cited = [], 0, 0, 0, 0
    for m in manifest:
        if "skipped" in m or "asm_is_a" not in m:
            continue
        vp = OUT / f"verdict_{m['id']}.txt"
        if not vp.exists():
            continue
        text = vp.read_text()
        match = re.search(r"VERDICT:\s*(A|B|TIE)", text)
        if not match:
            rows.append((m["id"], "unparsed", None))
            continue
        v = match.group(1)
        asm_won = (v == "A") == m["asm_is_a"] if v != "TIE" else None
        # citation use: does the asm answer cite any attached fact index?
        ans = (OUT / f"ans_{m['id']}_asm.txt").read_text()
        used = any(re.search(rf"\[{i}\]", ans) for i in m["attachment_indices"])
        cited += used
        if asm_won is True:
            wins += 1
        elif asm_won is False:
            losses += 1
        else:
            ties += 1
        rows.append((m["id"], {True: "ASM_WIN", False: "DENSE_WIN", None: "TIE"}[asm_won], used))
    for cid, verdict, used in rows:
        print(f"  {cid:<36} {verdict:<10} attachments_cited={used}")
    n = wins + losses + ties
    print(json.dumps({
        "judged": n, "asm_wins": wins, "dense_wins": losses, "ties": ties,
        "meaningful_assist_rate": round(wins / n, 3) if n else None,
        "attachment_citation_rate": round(cited / n, 3) if n else None,
    }, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
