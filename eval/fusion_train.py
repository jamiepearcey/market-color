#!/usr/bin/env python3
"""Learned fusion: tune multi-route retrieval weights on synthetic causal
chains with MECHANICAL labels; judge-free training and generalization.

Chain label (from canonical-exclusion construction): for excluded canonical
fact F (claim=endpoint, cause=root), targets are
  endpoint_support = facts in OTHER docs mentioning F.subject
  root_support     = facts in OTHER docs mentioning any F.cause_entity
Metric: chain-coverage@K — top-K fused facts contain >=1 of each set.

Routes (candidate generation, fused by weighted reciprocal rank):
  r_dense   dense(query) over title-contextualized fact embeddings
  r_lex     BM25-lite over claim+cause tokens (the never-tested sparse leg)
  r_walk    dense(top facts' DRIVER texts), best per driver
  r_graph   facts sharing entities with top-5 dense hits (co-occurrence hop)
Priors multiplying each candidate's fused score:
  p_conf    0.5 + confidence/2, weight w_conf in [0,1] blends toward 1
  p_time    event-time validity vs the dense-top fact (1 / soft / penalty)

Tuned params (8): w_dense, w_lex, w_walk, w_graph, w_conf, w_time,
rrf_k, K fixed at 30. Random search on TRAIN, report TEST. 70/30 split.

Stages: mine-train, rank (precompute per-chain route rankings), tune.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from strat_blind import _embed_all  # noqa: E402
from weighted_walk_test import norm_event_time  # noqa: E402

OUT = HERE / "fusion_out"
K = 30
MAX_PER_SUBJECT = 3
MIN_CAUSE_WORDS = 4


def _hay(f):
    return (str(f.get("claim", "")) + " " + str(f.get("cause", ""))).lower()


def _load_judge_canonicals():
    used = set()
    for qf in ("blind_questions_ww.jsonl", "blind_questions_gate.jsonl",
               "blind_questions_chains.jsonl"):
        p = HERE / qf
        if p.exists():
            for l in open(p):
                q = json.loads(l)
                used.update(q.get("exclude_doc_ids", []))
    return used


def cmd_mine_train(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    judge_docs = _load_judge_canonicals()
    subj_count = collections.Counter()
    chains = []
    for fi, f in enumerate(rows):
        cause = str(f.get("cause") or "")
        if not cause or cause.lower() in ("none", "null") or len(cause.split()) < MIN_CAUSE_WORDS:
            continue
        if f["doc_id"] in judge_docs:
            continue
        subj = str(f.get("subject") or "").lower().strip()
        ce = [e for e in (f.get("cause_entities") or []) if len(e) > 3]
        if not subj or len(subj) < 4 or not ce or subj_count[subj] >= MAX_PER_SUBJECT:
            continue
        endpoint = [i for i, g in enumerate(rows)
                    if g["doc_id"] != f["doc_id"] and subj in _hay(g)]
        root = [i for i, g in enumerate(rows)
                if g["doc_id"] != f["doc_id"] and any(e in _hay(g) for e in ce)]
        if len(endpoint) >= 2 and len(root) >= 2:
            subj_count[subj] += 1
            chains.append({"id": f"T{len(chains):03d}", "canon_idx": fi,
                           "query": f"Explain why this happened: "
                                    f"{str(f.get('claim')).rstrip('.')}.",
                           "exclude_doc": f["doc_id"],
                           "endpoint_ids": endpoint[:60], "root_ids": root[:60]})
    OUT.mkdir(exist_ok=True)
    (OUT / "train_chains.jsonl").write_text(
        "\n".join(json.dumps(c) for c in chains) + "\n")
    print(f"mined {len(chains)} labeled training chains "
          f"(judge-set canonicals excluded)")


def _bm25(tokens_list):
    """Minimal BM25 index over token lists."""
    df = collections.Counter()
    for toks in tokens_list:
        df.update(set(toks))
    N = len(tokens_list)
    idf = {t: math.log(1 + (N - c + 0.5) / (c + 0.5)) for t, c in df.items()}
    avg = sum(len(t) for t in tokens_list) / max(N, 1)
    return idf, avg


def _tok(s):
    return re.findall(r"[a-z0-9][a-z0-9.%$/-]{1,}", str(s).lower())


def cmd_rank(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    for f in rows:
        f["_etime"] = norm_event_time(f.get("time"), f.get("published_date"))
    print(f"[fusion] embedding {len(rows)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in rows])
    docs_tokens = [_tok(_hay(f)) for f in rows]
    idf, avg = _bm25(docs_tokens)
    docs_tf = [collections.Counter(t) for t in docs_tokens]
    docs_len = [len(t) for t in docs_tokens]
    ent_index = collections.defaultdict(set)
    for i, f in enumerate(rows):
        for e in set(f.get("entities") or []):
            ent_index[e].add(i)

    chains = [json.loads(l) for l in open(OUT / "train_chains.jsonl") if l.strip()]
    ranked = []
    for ci, c in enumerate(chains):
        valid = np.array([f["doc_id"] != c["exclude_doc"] for f in rows])
        qv = _embed_all([c["query"]])[0]
        dense = fmat @ qv
        dense[~valid] = -9
        r_dense = list(np.argsort(-dense)[:150])

        qt = _tok(c["query"])
        qtf = collections.Counter(qt)
        lex = np.zeros(len(rows))
        for i in range(len(rows)):
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

        walk_scores = np.zeros(len(rows))
        drivers = [rows[int(i)] for i in r_dense[:8]
                   if rows[int(i)].get("cause")
                   and str(rows[int(i)]["cause"]).lower() not in ("none", "null")]
        if drivers:
            dvecs = _embed_all([str(d["cause"]) for d in drivers])
            for dv in dvecs:
                s = fmat @ dv
                s[~valid] = -9
                walk_scores = np.maximum(walk_scores, s)
        r_walk = list(np.argsort(-walk_scores)[:150])

        top_ents = set()
        for i in r_dense[:5]:
            top_ents |= set(rows[int(i)].get("entities") or [])
        graph_cand = set()
        for e in top_ents:
            graph_cand |= ent_index[e]
        graph_scores = np.zeros(len(rows))
        for i in graph_cand:
            if valid[i]:
                graph_scores[i] = len(set(rows[i].get("entities") or []) & top_ents)
        r_graph = list(np.argsort(-graph_scores)[:150])

        anchor_et = rows[int(r_dense[0])]["_etime"] if len(r_dense) else None
        cand = set(map(int, r_dense[:80])) | set(map(int, r_lex[:80])) \
            | set(map(int, r_walk[:80])) | set(map(int, r_graph[:80]))
        feats = {}
        for i in cand:
            f = rows[i]
            et = f["_etime"]
            if anchor_et and et:
                p_time = 1.0 if et <= anchor_et + dt.timedelta(days=1) else 0.4
            else:
                p_time = 0.75
            feats[i] = {"conf": float(f.get("confidence") or 0.5), "time": p_time}
        ranked.append({
            "id": c["id"],
            "routes": {"dense": [int(x) for x in r_dense],
                       "lex": [int(x) for x in r_lex],
                       "walk": [int(x) for x in r_walk],
                       "graph": [int(x) for x in r_graph]},
            "feats": {str(k): v for k, v in feats.items()},
            "endpoint_ids": c["endpoint_ids"], "root_ids": c["root_ids"]})
        if ci % 25 == 0:
            print(f"[fusion] ranked {ci}/{len(chains)}", file=sys.stderr)
    (OUT / "rankings.json").write_text(json.dumps(ranked))
    print(f"precomputed route rankings for {len(ranked)} chains")


def _coverage(r, w, rrf_k):
    scores = collections.defaultdict(float)
    for route, lst in r["routes"].items():
        wr = w[route]
        if wr <= 0:
            continue
        for rank, i in enumerate(lst):
            scores[i] += wr / (rrf_k + rank)
    out = []
    for i, s in scores.items():
        ft = r["feats"].get(str(i))
        if ft:
            s *= (1 - w["conf"] + w["conf"] * (0.5 + ft["conf"] / 2))
            s *= (1 - w["time"] + w["time"] * ft["time"])
        out.append((s, i))
    out.sort(key=lambda t: -t[0])
    top = {i for _, i in out[:K]}
    ep, rt = set(r["endpoint_ids"]), set(r["root_ids"])
    return bool(top & ep and top & rt)


def cmd_tune(args):
    ranked = json.loads((OUT / "rankings.json").read_text())
    rng = random.Random(7)
    rng.shuffle(ranked)
    cut = int(len(ranked) * 0.7)
    train, test = ranked[:cut], ranked[cut:]

    def evl(chains, w, rrf_k):
        return sum(_coverage(r, w, rrf_k) for r in chains) / len(chains)

    baselines = {
        "dense_only": ({"dense": 1, "lex": 0, "walk": 0, "graph": 0,
                        "conf": 0, "time": 0}, 60),
        "dense+lex_equal": ({"dense": 1, "lex": 1, "walk": 0, "graph": 0,
                             "conf": 0, "time": 0}, 60),
        "all_equal": ({"dense": 1, "lex": 1, "walk": 1, "graph": 1,
                       "conf": 0, "time": 0}, 60),
    }
    print("baselines (train / test):")
    for name, (w, rk) in baselines.items():
        print(f"  {name:<18} {evl(train, w, rk):.3f} / {evl(test, w, rk):.3f}")

    best = None
    for _ in range(600):
        w = {"dense": rng.uniform(0.3, 1.5), "lex": rng.uniform(0, 1.5),
             "walk": rng.uniform(0, 1.0), "graph": rng.uniform(0, 1.0),
             "conf": rng.uniform(0, 1), "time": rng.uniform(0, 1)}
        rk = rng.choice([20, 40, 60, 90])
        s = evl(train, w, rk)
        if best is None or s > best[0]:
            best = (s, w, rk)
    s_train, w, rk = best
    print(f"\ntuned weights: { {k: round(v, 2) for k, v in w.items()} } rrf_k={rk}")
    print(json.dumps({"train_coverage": round(s_train, 3),
                      "TEST_coverage": round(evl(test, w, rk), 3),
                      "test_dense_only": round(evl(test, *baselines['dense_only']), 3),
                      "n_train": len(train), "n_test": len(test)}, indent=2))
    (OUT / "tuned_weights.json").write_text(json.dumps({"w": w, "rrf_k": rk}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["mine-train", "rank", "tune"])
    args = p.parse_args()
    {"mine-train": cmd_mine_train, "rank": cmd_rank, "tune": cmd_tune}[args.stage](args)


if __name__ == "__main__":
    main()
