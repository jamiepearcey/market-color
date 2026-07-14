#!/usr/bin/env python3
"""End-to-end judge validation of the fusion champion.

Arm `fusion` = hybrid v2 rendering (passages + atomic fill) with candidate
generation replaced by the EQUAL-WEIGHT 4-route fusion that won held-out
chain coverage (dense + BM25 lexical + driver-walk + entity-graph, RRF).
Judged double-order vs the existing v2 answers on the 26 VERIFIED questions
(`eval/blind_questions_ww.jsonl`, v2 answers reused from eval/ww_out/).

Stages: prep -> (OUT=fe_out run_ablation.sh gen) -> judge-prep ->
(OUT=fe_out run_ablation.sh judge) -> score.
"""
from __future__ import annotations

import argparse
import collections
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
from chunk_baseline import CHUNK_COLLECTION, GEN_TMPL, JUDGE_TMPL, _fact_line  # noqa: E402
from strat_blind import _embed_all  # noqa: E402
from fusion_train import _bm25, _tok, _hay  # noqa: E402

OUT = HERE / "fe_out"
WW = HERE / "ww_out"
RRF_K = 60
N_PASSAGES = 5


def fused_ranking(query, facts, fmat, docs_tf, docs_len, idf, avg, ent_index, excluded):
    valid = np.array([f["doc_id"] not in excluded for f in facts])
    qv = _embed_all([query])[0]
    dense = fmat @ qv
    dense[~valid] = -9
    r_dense = list(np.argsort(-dense)[:150])

    qtf = collections.Counter(_tok(query))
    lex = np.zeros(len(facts))
    for i in range(len(facts)):
        if not valid[i]:
            continue
        tf = docs_tf[i]
        s = 0.0
        for t in qtf:
            cnt = tf.get(t)
            if cnt:
                s += idf.get(t, 0) * cnt * 2.2 / (cnt + 1.2 * (0.25 + 0.75 * docs_len[i] / avg))
        lex[i] = s
    r_lex = list(np.argsort(-lex)[:150])

    walk_scores = np.zeros(len(facts))
    drivers = [facts[int(i)] for i in r_dense[:8]
               if facts[int(i)].get("cause")
               and str(facts[int(i)]["cause"]).lower() not in ("none", "null")]
    if drivers:
        for dv in _embed_all([str(d["cause"]) for d in drivers]):
            s = fmat @ dv
            s[~valid] = -9
            walk_scores = np.maximum(walk_scores, s)
    r_walk = list(np.argsort(-walk_scores)[:150])

    top_ents = set()
    for i in r_dense[:5]:
        top_ents |= set(facts[int(i)].get("entities") or [])
    graph_scores = np.zeros(len(facts))
    for e in top_ents:
        for i in ent_index.get(e, ()):
            if valid[i]:
                graph_scores[i] += 1
    r_graph = list(np.argsort(-graph_scores)[:150])

    scores: dict[int, float] = collections.defaultdict(float)
    for lst in (r_dense, r_lex, r_walk, r_graph):
        for rank, i in enumerate(lst):
            scores[int(i)] += 1.0 / (RRF_K + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda t: -t[1])], qv


def cmd_prep(args):
    import pyarrow.parquet as pq
    from qdrant_client.http import models
    facts = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    by_doc: dict[str, list] = {}
    for f in facts:
        by_doc.setdefault(f["doc_id"], []).append(f)
    docs_tokens = [_tok(_hay(f)) for f in facts]
    idf, avg = _bm25(docs_tokens)
    docs_tf = [collections.Counter(t) for t in docs_tokens]
    docs_len = [len(t) for t in docs_tokens]
    ent_index = collections.defaultdict(list)
    for i, f in enumerate(facts):
        for e in set(f.get("entities") or []):
            ent_index[e].append(i)
    print(f"[e2e] contextual-embedding {len(facts)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in facts])
    client = ic._client("http://localhost:6333", None)
    questions = [json.loads(l) for l in open(HERE / "blind_questions_ww.jsonl") if l.strip()]
    OUT.mkdir(exist_ok=True)
    manifest = []
    for q in questions:
        excluded = set(q["exclude_doc_ids"])
        order, qv = fused_ranking(q["query"], facts, fmat, docs_tf, docs_len,
                                  idf, avg, ent_index, excluded)
        fact_hits = [facts[i] for i in order[:50]]
        budget = len("\n".join(_fact_line(i + 1, f) for i, f in enumerate(fact_hits)))

        # v2-style rendering over the FUSED order
        lines, used, used_fids, n_pass, seen_docs = [], 0, set(), 0, set()
        for f in fact_hits:
            if n_pass >= N_PASSAGES or f["doc_id"] in seen_docs:
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
            cause = (f" DRIVER: {f['cause']}"
                     if f.get("cause") and str(f["cause"]).lower() not in ("none", "null") else "")
            line = (f"[{len(lines) + 1}] PASSAGE: {pl['text']} "
                    f"(source: {pl.get('source_name')}, {pl.get('published_date')}) "
                    f"KEY FACT: {f.get('claim')}{cause}")
            if used + len(line) > budget * 0.6 and lines:
                break
            lines.append(line)
            used += len(line)
            used_fids.add(f["fact_id"])
            n_pass += 1
        for f in fact_hits:
            if f["fact_id"] in used_fids:
                continue
            line = _fact_line(len(lines) + 1, f)
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
            used_fids.add(f["fact_id"])
        (OUT / f"gen_{q['id']}_fusion.txt").write_text(
            GEN_TMPL.format(query=q["query"], items="\n".join(lines)))
        # carry over the v2 answer
        src = WW / f"ans_{q['id']}_v2.txt"
        if src.exists():
            shutil.copy(src, OUT / f"ans_{q['id']}_v2.txt")
        manifest.append({"id": q["id"], "query": q["query"], "budget": budget,
                         "passages": n_pass, "atomic": len(lines) - n_pass})
        print(f"  {q['id']:<24} passages={n_pass} atomic={len(lines) - n_pass}",
              file=sys.stderr)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"prepared {len(manifest)} questions")


def cmd_judge_prep(args):
    manifest = json.loads((OUT / "manifest.json").read_text())
    n = 0
    for m in manifest:
        a_p = OUT / f"ans_{m['id']}_fusion.txt"
        b_p = OUT / f"ans_{m['id']}_v2.txt"
        if not a_p.exists() or not b_p.exists():
            continue
        a, b = a_p.read_text().strip(), b_p.read_text().strip()
        (OUT / f"judge_{m['id']}_o1.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=a, b=b))
        (OUT / f"judge_{m['id']}_o2.txt").write_text(JUDGE_TMPL.format(query=m["query"], a=b, b=a))
        n += 1
    print(f"prepared {n} pairs x 2 orders")


def cmd_score(args):
    import math
    manifest = json.loads((OUT / "manifest.json").read_text())
    W = L = T = 0
    for m in manifest:
        vs = []
        for order, mine_is_a in (("o1", True), ("o2", False)):
            vp = OUT / f"verdict_{m['id']}_{order}.txt"
            mt = re.search(r"VERDICT:\s*(A|B|TIE)", vp.read_text()) if vp.exists() else None
            vs.append(None if not mt or mt.group(1) == "TIE"
                      else (mt.group(1) == "A") == mine_is_a)
        both = [v for v in vs if v is not None]
        res = ("W" if len(both) == 2 and all(both) else
               "L" if len(both) == 2 and not any(both) else "T")
        print(f"  {m['id']:<24} {res}")
        W += res == "W"
        L += res == "L"
        T += res == "T"
    n = W + L
    p = sum(math.comb(n, k) for k in range(W, n + 1)) / 2**n if n else 1.0
    print(json.dumps({"fusion_wins": W, "v2_wins": L, "ties": T,
                      "one_sided_p": round(p, 4)}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["prep", "judge-prep", "score"])
    args = p.parse_args()
    {"prep": cmd_prep, "judge-prep": cmd_judge_prep, "score": cmd_score}[args.stage](args)


if __name__ == "__main__":
    main()
