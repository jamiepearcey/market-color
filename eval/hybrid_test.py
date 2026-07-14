#!/usr/bin/env python3
"""Fact-anchored chunk hybrid (fchunk): facts are the retrieval key, the
parent passage is the payload.

Context = top asm facts as ANCHORS (one per source doc, up to 5), each
rendered as its best-matching parent chunk (claim-embedding match within the
same doc) plus the fact's DRIVER line; remaining budget filled with atomic
fact lines for breadth. Char budget matched to the facts@50 context.

Stages: prep (gen prompts in eval/blind_out_hybrid/), judge-prep (pairwise
judge prompts vs the restricted-chunk answers AND vs the pure-facts answers
from eval/blind_out/), score. Generation/judging via run_ablation.sh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402
from chunk_baseline import CHUNK_COLLECTION, GEN_TMPL, JUDGE_TMPL, K_FACTS, _fact_line  # noqa: E402

OUT = HERE / "blind_out_hybrid"
BASE = HERE / "blind_out"  # restricted run: ans_*_facts.txt / ans_*_chunks.txt
ANCHORS = 5


def cmd_prep(args):
    import ppr_experiment as px
    from qdrant_client.http import models
    rt = px.Retriever(argparse.Namespace(
        parquet=str(px.DEFAULT_PARQUET), qdrant_url="http://localhost:6333",
        embedding_provider="fastembed",
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        seed_k=8, hub_gamma=0.0))
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions.jsonl") if l.strip()]
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in questions:
        hits = rt.run("asm", q["query"], K_FACTS)
        budget = len("\n".join(_fact_line(i + 1, f) for i, (_, f) in enumerate(hits)))

        anchors, seen_docs = [], set()
        for _, f in hits:
            if f.get("doc_id") in seen_docs:
                continue
            seen_docs.add(f.get("doc_id"))
            anchors.append(f)
            if len(anchors) >= ANCHORS:
                break
        avecs = ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL,
                         [str(a.get("claim") or "") for a in anchors],
                         ic.DEFAULT_OLLAMA_URL, None)
        lines, used, n_anchor = [], 0, 0
        anchor_fids = {a["fact_id"] for a in anchors}
        for a, av in zip(anchors, avecs, strict=True):
            pts = client.query_points(
                collection_name=CHUNK_COLLECTION, query=av, limit=1,
                with_payload=True,
                query_filter=models.Filter(must=[models.FieldCondition(
                    key="doc_id", match=models.MatchValue(value=a["doc_id"]))])).points
            if not pts:
                continue
            pl = pts[0].payload or {}
            cause = (f" DRIVER: {a['cause']}"
                     if a.get("cause") and str(a["cause"]).lower() not in ("none", "null") else "")
            line = (f"[{len(lines) + 1}] PASSAGE: {pl['text']} "
                    f"(source: {pl.get('source_name')}, {pl.get('published_date')}) "
                    f"KEY FACT: {a.get('claim')}{cause}")
            if used + len(line) > budget and lines:
                break
            lines.append(line)
            used += len(line)
            n_anchor += 1
        for _, f in hits:  # breadth fill with atomic facts
            if f["fact_id"] in anchor_fids:
                continue
            line = _fact_line(len(lines) + 1, f)
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
        (OUT / f"gen_{q['id']}_hybrid.txt").write_text(
            GEN_TMPL.format(query=q["query"], items="\n".join(lines)))
        manifest.append({"id": q["id"], "query": q["query"], "budget_chars": budget,
                         "used_chars": used, "anchors": n_anchor,
                         "atomic_facts": len(lines) - n_anchor})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for m in manifest:
        print(f"  {m['id']:<22} anchors={m['anchors']} atomic={m['atomic_facts']} "
              f"used={m['used_chars']}/{m['budget_chars']}")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    for opp in ("chunks", "facts"):
        d = HERE / f"blind_hyb_vs_{opp}"
        d.mkdir(exist_ok=True)
        n = 0
        for m in manifest:
            hyb = OUT / f"ans_{m['id']}_hybrid.txt"
            base = BASE / f"ans_{m['id']}_{opp}.txt"
            if not hyb.exists() or not base.exists():
                print(f"missing answers for {m['id']} vs {opp}", file=sys.stderr)
                continue
            hyb_is_a = int(hashlib.sha256(f"{m['id']}|{opp}".encode()).hexdigest(), 16) % 2 == 0
            a, b = ((hyb.read_text().strip(), base.read_text().strip()) if hyb_is_a
                    else (base.read_text().strip(), hyb.read_text().strip()))
            (d / f"judge_{m['id']}.txt").write_text(
                JUDGE_TMPL.format(query=m["query"], a=a, b=b))
            m[f"hyb_is_a_vs_{opp}"] = hyb_is_a
            n += 1
        print(f"prepared {n} judge prompts vs {opp} in {d.name}/")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    for opp in ("chunks", "facts"):
        d = HERE / f"blind_hyb_vs_{opp}"
        wins = losses = ties = 0
        print(f"== hybrid vs {opp}")
        for m in manifest:
            key = f"hyb_is_a_vs_{opp}"
            vp = d / f"verdict_{m['id']}.txt"
            if key not in m or not vp.exists():
                continue
            match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text())
            if not match:
                print(f"  {m['id']:<22} UNPARSED")
                continue
            v = match.group(1)
            won = (v == "A") == m[key] if v != "TIE" else None
            print(f"  {m['id']:<22} " + {True: "HYBRID_WIN", False: f"{opp.upper()}_WIN",
                                          None: "TIE"}[won])
            if won is True:
                wins += 1
            elif won is False:
                losses += 1
            else:
                ties += 1
        print(json.dumps({"vs": opp, "hybrid_wins": wins,
                          f"{opp}_wins": losses, "ties": ties}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
