#!/usr/bin/env python3
"""Stratified blind benchmark: hybrid v2 vs full-corpus chunk-RAG on the query
types the product actually serves.

Categories (eval/blind_questions_strat.jsonl):
  chain    hard causal inference with CANONICAL-DOC EXCLUSION: the doc(s)
           stating the endpoint<-root link are resolved by pattern and removed
           from BOTH systems' retrieval — the chain provably exists but must
           be reassembled from surviving pieces.
  agg      cross-document aggregation (briefs).
  screen   structured screening ("which X did Y, and why each").
  pivot    temporal evolution across the window.
  control  single-story why questions (expected parity).

Matched char budgets; double judging with flipped A/B order (win = both).
Stages: prep -> (OUT=strat_out run_ablation.sh gen) -> judge-prep ->
(OUT=strat_out run_ablation.sh judge) -> score.
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

OUT = HERE / "strat_out"
N_PASSAGES = 5
FACT_K = 50

GEN_COVERAGE_TMPL = """You are a market analyst. Answer the question using ONLY the numbered context items below. Cite every claim with the item number in brackets, e.g. [3]. Be comprehensive: include every distinct item the context supports, organized clearly (short bullets are fine). Do not invent items the context does not support; if coverage seems incomplete, say so.

QUESTION: {query}

CONTEXT:
{items}
"""

JUDGE_COVERAGE_TMPL = """You are judging two answers to a market research question, each written from retrieved context. Judge which answer is BETTER RESEARCH OUTPUT: more distinct correct items covered, more specific attributed detail per item (figures, named actors, stated causes), better organized. Penalize vagueness and padding. Ignore style otherwise.

QUESTION: {query}

ANSWER A:
{a}

ANSWER B:
{b}

Reply with exactly one line: VERDICT: A or VERDICT: B or VERDICT: TIE
Then one sentence of justification."""


def _embed_all(texts):
    out = []
    for chunk in ic._chunked(texts, 256):
        out.extend(ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL, chunk,
                            ic.DEFAULT_OLLAMA_URL, None))
    m = np.asarray(out, dtype="float32")
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def _resolve_exclusions(q, facts):
    ex = q.get("exclude")
    if not ex:
        return set()
    docs = set()
    for f in facts:
        cl = str(f.get("claim") or "").lower()
        ca = str(f.get("cause") or "").lower()
        if any(e in cl for e in ex["endpoint"]) and any(r in ca for r in ex["cause"]):
            docs.add(f["doc_id"])
    return docs


def _v2_context(query, facts, fmat, client, by_doc, extracted_docs, excluded):
    from qdrant_client.http import models
    qv = _embed_all([query])[0]
    sims = fmat @ qv
    order = [int(i) for i in np.argsort(-sims)
             if facts[int(i)]["doc_id"] not in excluded][:FACT_K]
    fact_hits = [(float(sims[i]), facts[i]) for i in order]
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
                              limit=150, with_payload=True).points
    for p in pts:
        pl = p.payload or {}
        if pl.get("doc_id") not in extracted_docs or pl.get("doc_id") in excluded:
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


def _chunk_context(query, client, budget, excluded):
    qv = _embed_all([query])[0]
    pts = client.query_points(collection_name=CHUNK_COLLECTION, query=qv,
                              limit=80, with_payload=True).points
    lines, used = [], 0
    for p in pts:
        pl = p.payload or {}
        if pl.get("doc_id") in excluded:
            continue
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
    print(f"[strat] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions_strat.jsonl") if l.strip()]
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in questions:
        excluded = _resolve_exclusions(q, facts)
        gen_tmpl = GEN_TMPL if q["category"] in ("chain", "control") else GEN_COVERAGE_TMPL
        items, budget = _v2_context(q["query"], facts, fmat, client, by_doc,
                                    extracted_docs, excluded)
        (OUT / f"gen_{q['id']}_v2.txt").write_text(gen_tmpl.format(query=q["query"], items=items))
        chunk_items = _chunk_context(q["query"], client, budget, excluded)
        (OUT / f"gen_{q['id']}_chunks.txt").write_text(
            gen_tmpl.format(query=q["query"], items=chunk_items))
        manifest.append({"id": q["id"], "query": q["query"], "category": q["category"],
                         "budget": budget, "excluded_docs": sorted(excluded)})
        print(f"  {q['id']:<22} [{q['category']:<7}] budget={budget} "
              f"excluded={len(excluded)}", file=sys.stderr)
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
        tmpl = JUDGE_TMPL if m["category"] in ("chain", "control") else JUDGE_COVERAGE_TMPL
        a, b = v2.read_text().strip(), ch.read_text().strip()
        (OUT / f"judge_{m['id']}_o1.txt").write_text(tmpl.format(query=m["query"], a=a, b=b))
        (OUT / f"judge_{m['id']}_o2.txt").write_text(tmpl.format(query=m["query"], a=b, b=a))
        n += 1
    print(f"prepared {n} pairs x 2 orders")


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    by_cat: dict[str, list] = {}
    for m in manifest:
        vs = []
        for order, v2_is_a in (("o1", True), ("o2", False)):
            vp = OUT / f"verdict_{m['id']}_{order}.txt"
            if not vp.exists():
                vs.append(None)
                continue
            match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text())
            vs.append(None if not match or match.group(1) == "TIE"
                      else (match.group(1) == "A") == v2_is_a)
        both = [v for v in vs if v is not None]
        if len(both) == 2 and all(both):
            res = "V2"
        elif len(both) == 2 and not any(both):
            res = "CHUNKS"
        else:
            res = "TIE"
        by_cat.setdefault(m["category"], []).append(res)
        print(f"  {m['id']:<22} [{m['category']:<7}] {res}")
    summary = {}
    for cat, rs in sorted(by_cat.items()):
        summary[cat] = {"v2": rs.count("V2"), "chunks": rs.count("CHUNKS"),
                        "tie": rs.count("TIE")}
    total = {"v2": sum(s["v2"] for s in summary.values()),
             "chunks": sum(s["chunks"] for s in summary.values()),
             "tie": sum(s["tie"] for s in summary.values())}
    print(json.dumps({"by_category": summary, "total": total}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
