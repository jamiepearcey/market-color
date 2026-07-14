#!/usr/bin/env python3
"""Typed representations + LLM reprojection, evaluated head-weighted.

Hypotheses under test (user's):
  H1 identities live in sparse space, meaning in dense space — split the
     lexical route into ENTITY-FIELD-scoped BM25 (subject+entities+object)
     vs full-text BM25, weight separately.
  H2 a lightweight output->query REPROJECTION model (haiku) produces more
     directed traversal than embedding raw driver spans.

Metrics (head-weighted, per Postscript 6):
  hw    mean of reciprocal ranks of the FIRST endpoint-support and FIRST
        root-support fact in the fused list (rank 1-based)
  cov10 both support sets represented in the top-10

Stages:
  reproject  haiku call per chain (resume-safe cache in fusion_out/reproj/)
  rank2      recompute all route rankings incl. entity-lex + reproj routes
  compare    fixed configs + small tuned search, 70/30 split, head-weighted
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from strat_blind import _embed_all  # noqa: E402
from fusion_train import _bm25, _tok, _hay  # noqa: E402

OUT = HERE / "fusion_out"
REPROJ = OUT / "reproj"
CLAUDE = "/opt/homebrew/bin/claude"

REPROJ_TMPL = """You are a retrieval query planner. A research question's context mentions a driver whose own cause is not yet explained.

QUESTION: {query}
STATED DRIVER (unexplained): {driver}

Write ONE short search query (max 12 words) naming the specific concrete upstream event, actor or decision most likely to explain that driver. Output only the query text."""


def cmd_reproject(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    chains = [json.loads(l) for l in open(OUT / "train_chains.jsonl") if l.strip()]
    REPROJ.mkdir(parents=True, exist_ok=True)
    print(f"[reproj] embedding for driver selection", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in rows])
    # embed queries up front: fastembed is not thread-safe, calling it from
    # pool workers can deadlock
    qvs = _embed_all([c["query"] for c in chains])

    def job(ci_c):
        ci, c = ci_c
        fp = REPROJ / f"{c['id']}.txt"
        if fp.exists() and fp.read_text().strip():
            return False
        valid = np.array([f["doc_id"] != c["exclude_doc"] for f in rows])
        qv = qvs[ci]
        dense = fmat @ qv
        dense[~valid] = -9
        driver = None
        for i in np.argsort(-dense)[:8]:
            f = rows[int(i)]
            cz = f.get("cause")
            if cz and str(cz).lower() not in ("none", "null"):
                driver = str(cz)
                break
        if not driver:
            fp.write_text("NONE")
            return True
        prompt = REPROJ_TMPL.format(query=c["query"], driver=driver)
        r = subprocess.run([CLAUDE, "-p", "--model", "claude-haiku-4-5"],
                           input=prompt, capture_output=True, text=True, timeout=120)
        fp.write_text((r.stdout or "NONE").strip().split("\n")[0][:200])
        return True

    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for ran in ex.map(job, enumerate(chains)):
            done += bool(ran)
    have = len([p for p in REPROJ.glob("*.txt") if p.read_text().strip()])
    print(f"reprojections cached: {have}/{len(chains)} (ran {done} this pass)")


def cmd_rank2(args):
    import pyarrow.parquet as pq
    rows = pq.read_table(ROOT / "facts_work" / "facts.parquet").to_pylist()
    print(f"[rank2] embedding {len(rows)} facts", file=sys.stderr)
    fmat = _embed_all([f"{f.get('doc_title') or ''} — {f.get('claim') or ''}" for f in rows])

    full_tokens = [_tok(_hay(f)) for f in rows]
    idf_full, avg_full = _bm25(full_tokens)
    tf_full = [collections.Counter(t) for t in full_tokens]
    len_full = [len(t) for t in full_tokens]

    ent_text = [" ".join([str(f.get("subject") or ""), str(f.get("object") or "")]
                         + list(f.get("entities") or [])) for f in rows]
    ent_tokens = [_tok(t) for t in ent_text]
    idf_ent, avg_ent = _bm25(ent_tokens)
    tf_ent = [collections.Counter(t) for t in ent_tokens]
    len_ent = [len(t) for t in ent_tokens]

    def bm25_route(query, tfs, lens, idf, avg, valid):
        qtf = collections.Counter(_tok(query))
        sc = np.zeros(len(rows))
        for i in range(len(rows)):
            if not valid[i]:
                continue
            tf = tfs[i]
            s = 0.0
            for t in qtf:
                cnt = tf.get(t)
                if cnt:
                    s += idf.get(t, 0) * cnt * 2.2 / (cnt + 1.2 * (0.25 + 0.75 * (lens[i] or 1) / avg))
            sc[i] = s
        return list(np.argsort(-sc)[:150])

    chains = [json.loads(l) for l in open(OUT / "train_chains.jsonl") if l.strip()]
    ranked = []
    for ci, c in enumerate(chains):
        valid = np.array([f["doc_id"] != c["exclude_doc"] for f in rows])
        qv = _embed_all([c["query"]])[0]
        dense = fmat @ qv
        dense[~valid] = -9
        r_dense = list(np.argsort(-dense)[:150])
        r_lex_full = bm25_route(c["query"], tf_full, len_full, idf_full, avg_full, valid)
        r_lex_ent = bm25_route(c["query"], tf_ent, len_ent, idf_ent, avg_ent, valid)

        walk_scores = np.zeros(len(rows))
        drivers = [rows[int(i)] for i in r_dense[:8]
                   if rows[int(i)].get("cause")
                   and str(rows[int(i)]["cause"]).lower() not in ("none", "null")]
        if drivers:
            for dv in _embed_all([str(d["cause"]) for d in drivers]):
                s = fmat @ dv
                s[~valid] = -9
                walk_scores = np.maximum(walk_scores, s)
        r_walk = list(np.argsort(-walk_scores)[:150])

        rp = REPROJ / f"{c['id']}.txt"
        rq = rp.read_text().strip() if rp.exists() else "NONE"
        if rq and rq != "NONE":
            rv = _embed_all([rq])[0]
            rs = fmat @ rv
            rs[~valid] = -9
            r_reproj = list(np.argsort(-rs)[:150])
        else:
            r_reproj = []

        ranked.append({"id": c["id"],
                       "routes": {"dense": [int(x) for x in r_dense],
                                  "lex_full": [int(x) for x in r_lex_full],
                                  "lex_ent": [int(x) for x in r_lex_ent],
                                  "walk": [int(x) for x in r_walk],
                                  "reproj": [int(x) for x in r_reproj]},
                       "endpoint_ids": c["endpoint_ids"], "root_ids": c["root_ids"]})
        if ci % 50 == 0:
            print(f"[rank2] {ci}/{len(chains)}", file=sys.stderr)
    (OUT / "rankings2.json").write_text(json.dumps(ranked))
    print(f"rank2 complete: {len(ranked)} chains, 5 routes")


def _fuse(r, w, rrf_k=60):
    scores = collections.defaultdict(float)
    for route, lst in r["routes"].items():
        wr = w.get(route, 0)
        if wr <= 0:
            continue
        for rank, i in enumerate(lst):
            scores[i] += wr / (rrf_k + rank)
    return [i for i, _ in sorted(scores.items(), key=lambda t: -t[0])]


def _metrics(r, order):
    ep, rt = set(r["endpoint_ids"]), set(r["root_ids"])
    fe = fr = None
    for rank, i in enumerate(order[:100], 1):
        if fe is None and i in ep:
            fe = rank
        if fr is None and i in rt:
            fr = rank
        if fe and fr:
            break
    hw = ((1 / fe if fe else 0) + (1 / fr if fr else 0)) / 2
    top10 = set(order[:10])
    return hw, bool(top10 & ep and top10 & rt)


def cmd_compare(args):
    ranked = json.loads((OUT / "rankings2.json").read_text())
    rng = random.Random(7)
    rng.shuffle(ranked)
    cut = int(len(ranked) * 0.7)
    train, test = ranked[:cut], ranked[cut:]

    def evl(chains, w):
        hws, covs = [], []
        for r in chains:
            hw, c10 = _metrics(r, _fuse(r, w))
            hws.append(hw)
            covs.append(c10)
        return sum(hws) / len(hws), sum(covs) / len(covs)

    configs = {
        "dense_only":        {"dense": 1},
        "dense+lex_full":    {"dense": 1, "lex_full": 1},
        "dense+lex_ent":     {"dense": 1, "lex_ent": 1},
        "typed(d+full+ent)": {"dense": 1, "lex_full": 0.7, "lex_ent": 0.7},
        "dense+walk":        {"dense": 1, "walk": 0.5},
        "dense+reproj":      {"dense": 1, "reproj": 0.5},
        "typed+reproj":      {"dense": 1, "lex_full": 0.7, "lex_ent": 0.7, "reproj": 0.5},
        "all_equal":         {"dense": 1, "lex_full": 1, "lex_ent": 1, "walk": 1, "reproj": 1},
    }
    print(f"{'config':<20} {'hw_train':>9} {'hw_TEST':>9} {'cov10_TEST':>11}")
    for name, w in configs.items():
        hw_tr, _ = evl(train, w)
        hw_te, c10 = evl(test, w)
        print(f"{name:<20} {hw_tr:>9.3f} {hw_te:>9.3f} {c10:>11.3f}")

    best = None
    for _ in range(800):
        w = {k: rng.uniform(0, 1.5) for k in
             ("dense", "lex_full", "lex_ent", "walk", "reproj")}
        w["dense"] = max(w["dense"], 0.4)
        s, _ = evl(train, w)
        if best is None or s > best[0]:
            best = (s, w)
    s_tr, w = best
    hw_te, c10 = evl(test, w)
    print(f"\ntuned: { {k: round(v, 2) for k, v in w.items()} }")
    print(json.dumps({"tuned_hw_train": round(s_tr, 3),
                      "tuned_hw_TEST": round(hw_te, 3),
                      "tuned_cov10_TEST": round(c10, 3),
                      "n_train": len(train), "n_test": len(test)}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["reproject", "rank2", "compare"])
    args = p.parse_args()
    {"reproject": cmd_reproject, "rank2": cmd_rank2,
     "compare": cmd_compare}[args.stage](args)


if __name__ == "__main__":
    main()
