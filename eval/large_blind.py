#!/usr/bin/env python3
"""Large-sample blind comparison: hybrid v2 vs full-corpus chunk-RAG.

24 fresh causal questions (eval/blind_questions_large.jsonl), matched char
budgets, and DOUBLE JUDGING with flipped A/B order per pair: a system wins a
question only if it wins BOTH orders; split or double-tie counts as a tie.
Kills position bias and halves single-verdict noise.

Stages: prep -> (OUT=large_out run_ablation.sh gen) -> judge-prep ->
(OUT=large_out run_ablation.sh judge) -> score.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402
from chunk_baseline import CHUNK_COLLECTION, GEN_TMPL, JUDGE_TMPL, _fact_line  # noqa: E402

OUT = HERE / "large_out"
N_PASSAGES = 5
FACT_K = 50


def _embed_all(texts):
    out = []
    for chunk in ic._chunked(texts, 256):
        out.extend(ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL, chunk,
                            ic.DEFAULT_OLLAMA_URL, None))
    m = np.asarray(out, dtype="float32")
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def _v2_context(q, facts, fmat, client, by_doc, extracted_docs):
    """Hybrid v2 context (contextual embeddings + dual nomination). Returns
    (items_text, budget) with budget = char length of the facts@50 rendering."""
    from qdrant_client.http import models
    qv = _embed_all([q])[0]
    sims = fmat @ qv
    order = np.argsort(-sims)[:FACT_K]
    fact_hits = [(float(sims[i]), facts[int(i)]) for i in order]
    budget = len("\n".join(_fact_line(i + 1, f) for i, (_, f) in enumerate(fact_hits)))

    passages: dict[str, dict] = {}
    seen_docs = set()
    for s, f in fact_hits:
        if f["doc_id"] in seen_docs:
            continue
        seen_docs.add(f["doc_id"])
        cv = _embed_all([str(f.get("claim") or "")])[0]
        pts = client.query_points(
            collection_name=CHUNK_COLLECTION, query=cv, limit=1, with_payload=True,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchValue(value=f["doc_id"]))])).points
        if not pts:
            continue
        pl = pts[0].payload or {}
        key = f"{pl['doc_id']}|{pl['chunk_index']}"
        entry = passages.setdefault(key, {"pl": pl, "score": 0.0, "facts": []})
        entry["score"] = max(entry["score"], s)
        entry["facts"].append(f)
        if len(passages) >= N_PASSAGES + 2:
            break
    pts = client.query_points(collection_name=CHUNK_COLLECTION, query=qv,
                              limit=100, with_payload=True).points
    for p in pts:
        pl = p.payload or {}
        if pl.get("doc_id") not in extracted_docs:
            continue
        key = f"{pl['doc_id']}|{pl['chunk_index']}"
        entry = passages.setdefault(key, {"pl": pl, "score": 0.0, "facts": []})
        entry["score"] = max(entry["score"], float(p.score))
        if len(passages) >= N_PASSAGES * 3:
            break

    ranked = sorted(passages.values(), key=lambda e: -e["score"])
    lines, used, used_fids = [], 0, set()
    n_pass = 0
    for e in ranked:
        if n_pass >= N_PASSAGES:
            break
        pl = e["pl"]
        key_facts = e["facts"]
        if not key_facts:
            cands = by_doc.get(pl["doc_id"], [])
            if cands:
                cvs = _embed_all([str(c.get("claim") or "") for c in cands[:20]])
                key_facts = [cands[int(np.argmax(cvs @ qv))]]
        ann = ""
        for kf in key_facts[:2]:
            cause = (f" DRIVER: {kf['cause']}"
                     if kf.get("cause") and str(kf["cause"]).lower() not in ("none", "null") else "")
            ann += f" KEY FACT: {kf.get('claim')}{cause}"
            used_fids.add(kf["fact_id"])
        line = (f"[{len(lines) + 1}] PASSAGE: {pl['text']} "
                f"(source: {pl.get('source_name')}, {pl.get('published_date')}){ann}")
        if used + len(line) > budget and lines:
            break
        lines.append(line)
        used += len(line)
        n_pass += 1
    for _, f in fact_hits:
        if f["fact_id"] in used_fids:
            continue
        line = _fact_line(len(lines) + 1, f)
        if used + len(line) > budget:
            break
        lines.append(line)
        used += len(line)
        used_fids.add(f["fact_id"])
    return "\n".join(lines), budget


def _chunk_context(q, client, budget):
    qv = _embed_all([q])[0]
    pts = client.query_points(collection_name=CHUNK_COLLECTION, query=qv,
                              limit=40, with_payload=True).points
    lines, used = [], 0
    for p in pts:
        pl = p.payload or {}
        line = (f"[{len(lines) + 1}] {pl['text']} "
                f"(source: {pl.get('source_name')}, {pl.get('published_date')})")
        if used + len(line) > budget and lines:
            break
        lines.append(line)
        used += len(line)
    return "\n".join(lines)


def cmd_prep(args):
    import pyarrow.parquet as pq
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    extracted_docs = {f["doc_id"] for f in facts}
    by_doc: dict[str, list] = {}
    for f in facts:
        by_doc.setdefault(f["doc_id"], []).append(f)
    print(f"[large] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions_large.jsonl") if l.strip()]
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in questions:
        items, budget = _v2_context(q["query"], facts, fmat, client, by_doc, extracted_docs)
        (OUT / f"gen_{q['id']}_v2.txt").write_text(GEN_TMPL.format(query=q["query"], items=items))
        chunk_items = _chunk_context(q["query"], client, budget)
        (OUT / f"gen_{q['id']}_chunks.txt").write_text(
            GEN_TMPL.format(query=q["query"], items=chunk_items))
        manifest.append({"id": q["id"], "query": q["query"], "budget": budget})
        print(f"  {q['id']:<24} budget={budget}", file=sys.stderr)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(manifest)} question pairs")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        v2 = OUT / f"ans_{m['id']}_v2.txt"
        ch = OUT / f"ans_{m['id']}_chunks.txt"
        if not v2.exists() or not ch.exists():
            print(f"missing answers: {m['id']}", file=sys.stderr)
            continue
        a, b = v2.read_text().strip(), ch.read_text().strip()
        # order 1: v2 = A;  order 2: v2 = B — position bias cancels by design
        (OUT / f"judge_{m['id']}_o1.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=a, b=b))
        (OUT / f"judge_{m['id']}_o2.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=b, b=a))
        n += 1
    print(f"prepared {n} pairs x 2 orders = {2 * n} judge prompts")


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    wins = losses = ties = 0
    for m in manifest:
        vs = []
        for order, v2_is_a in (("o1", True), ("o2", False)):
            vp = OUT / f"verdict_{m['id']}_{order}.txt"
            if not vp.exists():
                vs.append(None)
                continue
            match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text())
            if not match:
                vs.append(None)
                continue
            v = match.group(1)
            vs.append(None if v == "TIE" else (v == "A") == v2_is_a)
        both = [v for v in vs if v is not None]
        if len(both) == 2 and all(both):
            res = "V2_WIN"; wins += 1
        elif len(both) == 2 and not any(both):
            res = "CHUNKS_WIN"; losses += 1
        else:
            res = "TIE/SPLIT"; ties += 1
        print(f"  {m['id']:<24} {res}   (orders: {vs})")
    n = wins + losses + ties
    print(json.dumps({"questions": n, "v2_wins": wins, "chunk_wins": losses,
                      "tie_or_split": ties}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
