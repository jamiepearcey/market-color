#!/usr/bin/env python3
"""Hybrid v2: dual-granularity passage nomination + contextual fact embeddings.

Fixes the two diagnosed causes of the satellite-fact weak spot:
  1. facts are embedded WITH their doc title ("{title} — {claim}") — restores
     the document-level signal atomization destroyed (measured: ¥11.73tn fact
     rank #194 -> #11 for the USD/JPY query);
  2. passages are nominated at BOTH granularities — by contextual fact
     retrieval (fact -> parent passage) AND by direct chunk retrieval — and
     fused in passage space (score = max of the two routes), so a passage
     whose facts all rank poorly can still be nominated by its own text.

Chunk nomination is RESTRICTED to fact-extracted docs (mechanism test, not a
coverage test). Render: top passages with KEY FACT/DRIVER annotations, then
atomic contextual facts fill the same char budget as facts@50.

Stages: prep -> (run_ablation.sh gen, OUT=blind_out_hybrid2) -> judge-prep ->
(judge vs chunks + vs hybrid v1) -> score.
"""
from __future__ import annotations

import argparse
import hashlib
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

OUT = HERE / "blind_out_hybrid2"
BASE = HERE / "blind_out"           # ans_*_chunks.txt (restricted run)
V1 = HERE / "blind_out_hybrid"      # ans_*_hybrid.txt
N_PASSAGES = 5
FACT_K = 50


def _embed_all(texts):
    out = []
    for chunk in ic._chunked(texts, 256):
        out.extend(ic.embed("fastembed", ic.DEFAULT_FASTEMBED_MODEL, chunk,
                            ic.DEFAULT_OLLAMA_URL, None))
    m = np.asarray(out, dtype="float32")
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def cmd_prep(args):
    import pyarrow.parquet as pq
    from qdrant_client.http import models
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    extracted_docs = {f["doc_id"] for f in facts}
    print(f"[v2] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions.jsonl") if l.strip()]
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in questions:
        qv = _embed_all([q["query"]])[0]
        fact_order = np.argsort(-(fmat @ qv))[:FACT_K]
        fact_hits = [(float((fmat @ qv)[i]), facts[int(i)]) for i in fact_order]
        budget = len("\n".join(_fact_line(i + 1, f) for i, (_, f) in enumerate(fact_hits)))

        # --- passage nomination, route A: top facts -> parent passage
        passages: dict[str, dict] = {}  # key doc_id|chunk_index
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

        # --- route B: direct chunk retrieval (restricted to extracted docs)
        pts = client.query_points(collection_name=CHUNK_COLLECTION, query=qv,
                                  limit=100, with_payload=True).points
        for p in pts:
            pl = p.payload or {}
            if pl.get("doc_id") not in extracted_docs:
                continue
            key = f"{pl['doc_id']}|{pl['chunk_index']}"
            entry = passages.setdefault(key, {"pl": pl, "score": 0.0, "facts": []})
            entry["score"] = max(entry["score"], float(p.score))
            if len([1 for e in passages.values() if e["score"] > 0]) >= N_PASSAGES * 3:
                break

        ranked = sorted(passages.values(), key=lambda e: -e["score"])
        # annotate chunk-nominated passages with their doc's best facts
        by_doc: dict[str, list] = {}
        for f in facts:
            by_doc.setdefault(f["doc_id"], []).append(f)
        lines, used, n_pass = [], 0, 0
        used_fids = set()
        for e in ranked:
            if n_pass >= N_PASSAGES:
                break
            pl = e["pl"]
            key_facts = e["facts"]
            if not key_facts:  # chunk-nominated: attach best same-doc fact
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
        for _, f in fact_hits:  # breadth fill
            if f["fact_id"] in used_fids:
                continue
            line = _fact_line(len(lines) + 1, f)
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
            used_fids.add(f["fact_id"])
        (OUT / f"gen_{q['id']}_hybrid2.txt").write_text(
            GEN_TMPL.format(query=q["query"], items="\n".join(lines)))
        manifest.append({"id": q["id"], "query": q["query"], "budget_chars": budget,
                         "used_chars": used, "passages": n_pass,
                         "atomic_facts": len(lines) - n_pass})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    for m in manifest:
        print(f"  {m['id']:<22} passages={m['passages']} atomic={m['atomic_facts']} "
              f"used={m['used_chars']}/{m['budget_chars']}")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    for opp, src in (("chunks", BASE), ("hybrid1", V1)):
        d = HERE / f"blind_h2_vs_{opp}"
        d.mkdir(exist_ok=True)
        n = 0
        suffix = "chunks" if opp == "chunks" else "hybrid"
        for m in manifest:
            mine = OUT / f"ans_{m['id']}_hybrid2.txt"
            theirs = src / f"ans_{m['id']}_{suffix}.txt"
            if not mine.exists() or not theirs.exists():
                print(f"missing answers for {m['id']} vs {opp}", file=sys.stderr)
                continue
            mine_is_a = int(hashlib.sha256(f"{m['id']}|h2|{opp}".encode()).hexdigest(), 16) % 2 == 0
            a, b = ((mine.read_text().strip(), theirs.read_text().strip()) if mine_is_a
                    else (theirs.read_text().strip(), mine.read_text().strip()))
            (d / f"judge_{m['id']}.txt").write_text(
                JUDGE_TMPL.format(query=m["query"], a=a, b=b))
            m[f"h2_is_a_vs_{opp}"] = mine_is_a
            n += 1
        print(f"prepared {n} judge prompts vs {opp}")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


def cmd_score(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    for opp in ("chunks", "hybrid1"):
        d = HERE / f"blind_h2_vs_{opp}"
        wins = losses = ties = 0
        print(f"== hybrid2 vs {opp}")
        for m in manifest:
            key = f"h2_is_a_vs_{opp}"
            vp = d / f"verdict_{m['id']}.txt"
            if key not in m or not vp.exists():
                continue
            match = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text())
            if not match:
                continue
            v = match.group(1)
            won = (v == "A") == m[key] if v != "TIE" else None
            print(f"  {m['id']:<22} " + {True: "H2_WIN", False: f"{opp.upper()}_WIN",
                                          None: "TIE"}[won])
            if won is True:
                wins += 1
            elif won is False:
                losses += 1
            else:
                ties += 1
        print(json.dumps({"vs": opp, "h2_wins": wins, f"{opp}_wins": losses, "ties": ties}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
