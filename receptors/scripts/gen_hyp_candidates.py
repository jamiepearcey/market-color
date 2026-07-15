# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "fastembed>=0.3"]
# ///
"""Hypothesis-driven cosine union -> candidate drivers per query, per arm.

This is the discovery engine RETRIEVAL_FINDINGS.md credits: for each hypothesis
query, cosine-search the corpus; union the per-hypothesis rankings (RRF); the result
is a candidate driver set that reaches upstream docs the bare effect phrase misses.
The arms differ ONLY in their hypotheses (gen_hypotheses.py) — so any gap between
arms is attributable to the hypothesis source, nothing downstream.

Optionally layer the validated cos+ndir reranker (direction_head.py) on the union
pool (`--rerank ndir`) — measuring feature extraction the RIGHT way (a novelty-aware
reranker fused with cosine), not as a standalone first-stage retriever.

Also emits a BASELINE arm (cosine on the effect phrase alone, no hypotheses): the
reference every novelty judgement is made against. Reusing the effect as a single
"hypothesis" keeps baseline and arms on identical machinery.

Output (existing cand_ format so build_novelty_packets / build_judge_packets consume it):
  data/eval/cand_<arm>.json = {qid: [{doc_id, rank, title, snippet, via}]}
  data/eval/prov_<arm>.json = {qid: {doc_id: [hypotheses that surfaced it]}}  (provenance)

Usage:
  uv run scripts/gen_hyp_candidates.py --arm baseline --hyps effect  --k 10
  uv run scripts/gen_hyp_candidates.py --arm C        --hyps hyps_C  --k 10 --rerank ndir
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, "scripts")

D = Path(os.environ.get("RECEPTORS_DATA_DIR", "data"))
DEVAL = Path("data/eval")
PER_HYP = 50   # depth pulled per hypothesis before union
RRF_K = 60.0


def _load_docspace():
    docs = [json.loads(l) for l in (D / "docs.jsonl").read_text().splitlines() if l.strip()]
    E = np.load(D / "embeddings.npy").astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ids = [d["doc_id"] for d in docs]
    epoch = np.array([d.get("published_epoch", 0) for d in docs])
    try:
        dmeta = json.loads((D / "doc_meta.json").read_text())
    except FileNotFoundError:
        dmeta = {d["doc_id"]: {"title": d.get("title", "?")} for d in docs}
    return docs, E, ids, epoch, dmeta


def _snippets():
    """doc_id -> best-chunk text, keyed for whichever effect. Returns a resolver."""
    ch = np.load(D / "chunks.npy").astype(np.float32)
    ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
    texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
    by_doc = {}
    for ci, m in enumerate(meta):
        by_doc.setdefault(m["doc_id"], []).append(ci)

    def resolve(doc_id, qv):
        cis = by_doc.get(doc_id)
        if not cis:
            return ""
        best = max(cis, key=lambda ci: float(ch[ci] @ qv))
        return texts[best][:600]
    return resolve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)          # output label
    ap.add_argument("--hyps", required=True)          # "effect" | "hyps_<ARM>" file stem
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--rerank", choices=["none", "ndir"], default="none")
    ap.add_argument("--rerank-w", type=float, default=1.0)
    ap.add_argument("--asof", action="store_true", help="honour per-query asof epoch")
    a = ap.parse_args()

    import hyp_sources as hs  # for _embed (shares the configured substrate model)

    queries = json.load(open(DEVAL / "queries.json"))
    docs, E, ids, epoch, dmeta = _load_docspace()
    resolve = _snippets()

    # hypotheses per query
    if a.hyps == "effect":
        hyp_map = {q["id"]: {"hypotheses": [q["effect"]]} for q in queries}
    else:
        hyp_map = json.load(open(DEVAL / f"{a.hyps}.json"))

    dh = None
    if a.rerank == "ndir":
        import direction_head as dhmod
        try:
            dh = dhmod.load_or_train()
        except dhmod.InsufficientPairs as e:
            print(f"WARN: {e}\nWARN: falling back to cosine-union order (rerank disabled).")
            a.rerank = "none"

    out, prov = {}, {}
    for q in queries:
        qid, effect = q["id"], q["effect"]
        hyps = hyp_map.get(qid, {}).get("hypotheses") or [effect]
        asof = (int(q.get("asof") or q.get("asof_epoch") or 0)) if a.asof else 0
        live = (epoch < asof) if asof else np.ones(len(ids), dtype=bool)

        # cosine each hypothesis; RRF-union the per-hypothesis rankings
        rrf = {}
        surfaced_by = {}
        effect_vec = hs._embed([effect])[0]
        for h in hyps:
            hv = hs._embed([h])[0]
            sc = E @ hv
            order = np.argsort(-sc)
            rank = 0
            for di in order:
                if not live[di]:
                    continue
                did = ids[di]
                rrf[did] = rrf.get(did, 0.0) + 1.0 / (RRF_K + rank)
                surfaced_by.setdefault(did, []).append(h)
                rank += 1
                if rank >= PER_HYP:
                    break

        ranked = sorted(rrf.items(), key=lambda kv: -kv[1])

        # optional cos+ndir rerank of the union pool (novelty-aware precision layer)
        if dh is not None and ranked:
            cand_ids = [d for d, _ in ranked]
            idx = {ids[i]: i for i in range(len(ids))}
            rows = np.asarray([E[idx[d]] for d in cand_ids], np.float32)
            cos = rows @ effect_vec
            nd = dh.ndir(rows, effect_vec)
            fused = dh.fuse(cos, nd, w=a.rerank_w)
            ranked = [(cand_ids[i], float(fused[i]))
                      for i in np.argsort(-fused)]

        res, pmap = [], {}
        for did, _ in ranked[:a.k]:
            res.append({"doc_id": did, "rank": len(res) + 1,
                        "title": dmeta.get(did, {}).get("title", "?"),
                        "snippet": resolve(did, effect_vec),
                        "via": "hyp-union" if a.hyps != "effect" else "baseline-cosine"})
            pmap[did] = list(dict.fromkeys(surfaced_by.get(did, [])))
        out[qid] = res
        prov[qid] = pmap

    DEVAL.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(DEVAL / f"cand_{a.arm}.json", "w"), indent=1)
    json.dump(prov, open(DEVAL / f"prov_{a.arm}.json", "w"), indent=1)
    tot = sum(len(v) for v in out.values())
    print(f"wrote cand_{a.arm}.json + prov_{a.arm}.json  "
          f"({len(out)} q, {tot} candidates, k={a.k}, rerank={a.rerank})")


if __name__ == "__main__":
    main()
