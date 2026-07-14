#!/usr/bin/env python3
"""Blind head-to-head: classic semantic-chunk RAG over raw articles vs the
tuned fact pipeline (asm@50), at MATCHED context budget (same char count).

  build       chunk data/news_corpus bodies (~1200 chars, 200 overlap), embed
              (same MiniLM), index into Qdrant 'market_color_chunks'.
  prep        for each eval/blind_questions.jsonl question: tuned context =
              asm@50 rendered facts; chunk context = top chunks truncated to
              the SAME char budget. Writes gen prompts + manifest.
  judge-prep  blind pairwise judge prompts, randomized A/B per question id.
  score       verdicts -> facts wins / chunk wins / ties.

Generation/judging run via eval/run_ablation.sh with OUT=blind_out.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import index_corpus as ic  # noqa: E402

OUT = HERE / "blind_out"
CHUNK_COLLECTION = "market_color_chunks"
K_FACTS = 50

GEN_TMPL = """You are a market analyst. Answer the question using ONLY the numbered context items below. Cite every claim with the item number in brackets, e.g. [3]. If the context does not fully answer the question, answer as far as it allows and say what is missing. 2-6 sentences.

QUESTION: {query}

CONTEXT:
{items}
"""

JUDGE_TMPL = """You are judging two answers to a market question, each written from retrieved context. Judge which answer better explains the CAUSAL CHAIN — the specific root events and mechanism, not vague gestures at "the conflict" or "market conditions". Groundedness matters: prefer specific, attributed causes. Ignore style and length.

QUESTION: {query}

ANSWER A:
{a}

ANSWER B:
{b}

Reply with exactly one line: VERDICT: A or VERDICT: B or VERDICT: TIE
Then one sentence of justification."""


def _chunks_of(text: str, size: int = 1200, overlap: int = 200):
    text = re.sub(r"\s+", " ", text or "").strip()
    step = size - overlap
    for s in range(0, max(len(text) - overlap, 1), step):
        c = text[s:s + size]
        if len(c) > 200:
            yield c


def cmd_build(args):
    import uuid
    from qdrant_client.http import models
    client = ic._client("http://localhost:6333", None)
    rows = ic.load_rows(ROOT / "data" / "news_corpus", None, None, desk=None,
                        source=None, only_full_body=True, limit=None)
    chunks, meta = [], []
    for r in rows:
        for i, c in enumerate(_chunks_of(r.get("body_text") or "")):
            chunks.append(c)
            meta.append({"doc_id": r["doc_id"], "chunk_index": i, "text": c,
                         "title": r.get("title"), "source_name": r.get("source_name"),
                         "published_date": str(r.get("published_date") or ""),
                         "url": r.get("url")})
    print(f"[chunks] {len(rows)} docs -> {len(chunks)} chunks", file=sys.stderr)
    if client.collection_exists(CHUNK_COLLECTION):
        client.delete_collection(CHUNK_COLLECTION)
    vecs0 = ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL, chunks[:1],
                     ic.DEFAULT_OLLAMA_URL, None)
    client.create_collection(
        collection_name=CHUNK_COLLECTION, on_disk_payload=True,
        vectors_config=models.VectorParams(size=len(vecs0[0]),
                                           distance=models.Distance.COSINE))
    done = 0
    for s in range(0, len(chunks), 256):
        vecs = ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL, chunks[s:s + 256],
                        ic.DEFAULT_OLLAMA_URL, None)
        pts = [models.PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{m['doc_id']}|{m['chunk_index']}")),
            vector=v, payload=m)
            for m, v in zip(meta[s:s + 256], vecs, strict=True)]
        client.upsert(collection_name=CHUNK_COLLECTION, points=pts, wait=True)
        done += len(pts)
        if done % 2560 < 256:
            print(f"[chunks] indexed {done}/{len(chunks)}", file=sys.stderr)
    print(json.dumps({"chunks_indexed": len(chunks), "collection": CHUNK_COLLECTION}))


def _fact_line(i, f):
    cause = f" DRIVER: {f['cause']}" if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else ""
    return f"[{i}] {f.get('claim')} (source: {f.get('source_name')}, {f.get('published_date')}){cause}"


def cmd_prep(args):
    import ppr_experiment as px
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
        facts = [f for _, f in rt.run("asm", q["query"], K_FACTS)]
        fact_items = "\n".join(_fact_line(i + 1, f) for i, f in enumerate(facts))
        budget = len(fact_items)

        qv = ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL, [q["query"]],
                      ic.DEFAULT_OLLAMA_URL, None)[0]
        pts = client.query_points(collection_name=CHUNK_COLLECTION, query=qv,
                                  limit=200 if args.restrict else 40,
                                  with_payload=True).points
        if args.restrict:  # only docs the fact layer also saw (coverage control)
            extracted = {f["doc_id"] for f in rt.graph.facts}
            pts = [p for p in pts if (p.payload or {}).get("doc_id") in extracted]
        lines, used = [], 0
        for i, p in enumerate(pts):
            pl = p.payload or {}
            line = (f"[{len(lines) + 1}] {pl['text']} "
                    f"(source: {pl.get('source_name')}, {pl.get('published_date')})")
            if used + len(line) > budget and lines:
                break
            lines.append(line)
            used += len(line)
        chunk_items = "\n".join(lines)

        (OUT / f"gen_{q['id']}_facts.txt").write_text(
            GEN_TMPL.format(query=q["query"], items=fact_items))
        (OUT / f"gen_{q['id']}_chunks.txt").write_text(
            GEN_TMPL.format(query=q["query"], items=chunk_items))
        manifest.append({"id": q["id"], "query": q["query"],
                         "budget_chars": budget, "n_facts": len(facts),
                         "n_chunks": len(lines)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for m in manifest:
        print(f"  {m['id']:<24} budget={m['budget_chars']:>6} chars  "
              f"facts={m['n_facts']}  chunks={m['n_chunks']}")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        ans = {}
        for arm in ("facts", "chunks"):
            p = OUT / f"ans_{m['id']}_{arm}.txt"
            if not p.exists() or not p.read_text().strip():
                print(f"missing answer: {p}", file=sys.stderr)
                break
            ans[arm] = p.read_text().strip()
        else:
            facts_is_a = int(hashlib.sha256(m["id"].encode()).hexdigest(), 16) % 2 == 0
            a, b = (ans["facts"], ans["chunks"]) if facts_is_a else (ans["chunks"], ans["facts"])
            (OUT / f"judge_{m['id']}.txt").write_text(
                JUDGE_TMPL.format(query=m["query"], a=a, b=b))
            m["facts_is_a"] = facts_is_a
            n += 1
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {n} judge prompts")


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    wins = losses = ties = 0
    for m in manifest:
        if "facts_is_a" not in m:
            continue
        vp = OUT / f"verdict_{m['id']}.txt"
        if not vp.exists():
            continue
        match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text())
        if not match:
            print(f"  {m['id']:<24} UNPARSED")
            continue
        v = match.group(1)
        facts_won = (v == "A") == m["facts_is_a"] if v != "TIE" else None
        label = {True: "FACTS_WIN", False: "CHUNKS_WIN", None: "TIE"}[facts_won]
        print(f"  {m['id']:<24} {label}")
        if facts_won is True:
            wins += 1
        elif facts_won is False:
            losses += 1
        else:
            ties += 1
    n = wins + losses + ties
    print(json.dumps({"judged": n, "facts_wins": wins, "chunk_wins": losses,
                      "ties": ties}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "prep", "judge-prep", "score"])
    p.add_argument("--restrict", action="store_true",
                   help="chunks only from docs the fact layer extracted")
    args = p.parse_args()
    {"build": cmd_build, "prep": cmd_prep,
     "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
