# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Receptor tools — the README's embedding-space learnings as callable tools.

NO graph anywhere at inference: pure matrix algebra over embeddings (the
receptors thesis). The cause_entities labels were used once, as training
supervision for W; these tools only do BLAS.

  causal_find      W-only first-stage CAUSE retrieval (the README's biggest
                   finding: in-dist W beats cosine as a *finder*, R@10 +53%
                   random / +34% temporal). Query = an effect/phenomenon;
                   results = candidate upstream causes, ranked by cos(chunk·W, q).
                   Per README: W-only for finding (cosine injects co-topical
                   non-causes); results are CANDIDATE causes -> feed the
                   validation gate, don't trust the algebra to have proven cause.
  causal_direction Which of two texts is the cause? sign of s(a->b) - s(b->a)
                   (0.80-0.83 accuracy vs 0.50 cosine floor).

CLI:  uv run scripts/receptors_tools.py find "effect phenomenon" [-k 8] [--before D] [--json]
      uv run scripts/receptors_tools.py direction "text a" "text b"
"""
import json, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data"
_model = None


def _embed(texts):
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    V = np.array(list(_model.embed(texts)), dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V


def _load_index():
    T = np.load(D / "chunks_transported.npy")          # normalize(chunk·W)
    ch = np.load(D / "chunks.npy").astype(np.float32)
    ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
    texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
    dmeta = json.loads((D / "doc_meta.json").read_text())
    return T, ch, meta, texts, dmeta


def causal_find(query, k=8, before=None, exclude=None):
    """W-only cause finder: rank all chunks by cos(chunk·W, query), fold to docs."""
    T, ch, meta, texts, dmeta = _load_index()
    q = _embed([query])[0]
    s = T @ q            # causal score (candidate cause -> transported -> effect?)
    cos = ch @ q         # shown for transparency: LOW cosine + HIGH causal = the
                         # cross-vocabulary reach cosine cannot make
    cutoff = None
    if before:
        import datetime
        y, m, dd = map(int, before.split("-"))
        cutoff = int(datetime.datetime(y, m, dd, tzinfo=datetime.timezone.utc).timestamp())
    excl = set((exclude or "").split(",")) - {""}
    best = {}
    for ci in np.argsort(-s)[: k * 60]:
        mm = meta[ci]
        did = mm["doc_id"]
        if did in excl or (cutoff is not None and mm.get("epoch", 0) >= cutoff):
            continue
        if did not in best or s[ci] > best[did][0]:
            best[did] = (float(s[ci]), int(ci))
        if len(best) >= k * 6:
            break
    out = []
    for did, (sc, ci) in sorted(best.items(), key=lambda kv: -kv[1][0])[:k]:
        dm = dmeta.get(did, {})
        out.append({"doc_id": did, "causal_score": round(sc, 3),
                    "cosine": round(float(cos[ci]), 3),
                    "src": dm.get("src", "?"), "tier": dm.get("tier", 9),
                    "date": dm.get("date", "?"), "title": dm.get("title", "?"),
                    "snippet": texts[ci]})
    return out


_graph = None


def _load_graph():
    """The semantic causal graph, precomputed at index time (build_semantic_graph.py).
    Loading is O(file); inference is a sparse power iteration — no graph construction
    at query time (that cost is paid once, offline)."""
    global _graph
    if _graph is None:
        g = np.load(D / "semantic_graph.npz", allow_pickle=True)
        _graph = {k: g[k] for k in g.files}
    return _graph


def _ppr(seed, edges_idx, edges_wt, alpha=0.9, iters=20):
    """PPR power iteration s = alpha*seed + (1-alpha)*P*s, P moves mass from an
    effect to its causes (edge i->j = i causes j). Multi-hop = causes-of-causes."""
    n = seed.shape[0]
    s = seed.copy()
    for _ in range(iters):
        y = np.zeros(n, dtype=np.float32)
        # y[i] += wt * s[j] for each cause i of effect j
        for kk in range(edges_idx.shape[1]):
            idx = edges_idx[:, kk]
            m = idx >= 0
            np.add.at(y, idx[m], edges_wt[m, kk] * s[m])
        s = alpha * seed + (1.0 - alpha) * y
    return s


def _rrf_rank(rankings, take=100):
    """Reciprocal-rank fusion of several node rankings -> fused node order (best first)."""
    score = {}
    for lst in rankings:
        for r, ni in enumerate(lst[:take]):
            score[ni] = score.get(ni, 0.0) + 1.0 / (60.0 + r)
    return [ni for ni, _ in sorted(score.items(), key=lambda kv: -kv[1])]


def causal_chain(query, k=8, before=None, exclude=None, alpha=0.9, iters=20):
    """MULTI-HOP cause finder over the precomputed SEMANTIC GRAPH (abductive fusion —
    the crowned method in the Rust scoreboard). Pipeline, all BLAS, graph precomputed:
      1-hop:   seed = relu(doc_w . q)                    (direct causes)
      chain:   PPR back through the graph                (causes-of-causes, 2-hop)
      support: cosine of a hypothesis centroid (top chain docs) over doc_emb
      cosine:  effect-anchored topical
    then 4-way RRF. On held-out 2-hop chain gold this beats 1-hop W R@10 0.044 vs
    0.034 (+29%) / R@30 0.099 vs 0.083, hub-rate 0.33 (raw PPR alone = 0.85, unusable).
    Zero inference-time graph construction: the graph is built once, offline.
    Returns CANDIDATE causes -> feed the gate; the algebra proposes, it does not prove."""
    g = _load_graph()
    edges_idx, edges_wt = g["edges_idx"], g["edges_wt"]
    doc_w = g["doc_w"].astype(np.float32)
    doc_emb = g["doc_emb"].astype(np.float32)
    gepoch, gids = g["epoch"], list(g["doc_ids"])
    q = _embed([query])[0]

    cutoff = None
    if before:
        import datetime
        y, m, dd = map(int, before.split("-"))
        cutoff = int(datetime.datetime(y, m, dd, tzinfo=datetime.timezone.utc).timestamp())
    live = (gepoch < cutoff) if cutoff is not None else np.ones(len(gids), dtype=bool)

    def _rank(score):
        idx = np.nonzero(live)[0]
        return idx[np.argsort(-score[idx])].tolist()

    seed = np.maximum(doc_w @ q, 0.0).astype(np.float32)
    seed = np.where(live, seed, 0.0)
    ssum = float(seed.sum())
    if ssum > 0:
        seed = seed / ssum
    r_1hop = _rank(seed)

    hyp = _ppr(seed, edges_idx, edges_wt, alpha=alpha, iters=iters)
    hyp = np.where(live, hyp, 0.0)
    r_chain = _rank(hyp)

    r_cos = _rank(doc_emb @ q)

    # abductive support: centroid of top-15 chain docs -> cosine RAG for corroboration
    top = np.argsort(-hyp)[:15]
    qhyp = (doc_emb[top] * hyp[top, None]).sum(axis=0)
    nn = float(np.linalg.norm(qhyp))
    r_support = _rank(doc_emb @ (qhyp / nn)) if nn > 0 else r_cos

    fused = _rrf_rank([r_cos, r_1hop, r_chain, r_support], take=100)

    # snippet = the doc's best chunk by cosine to the query (doc-level result)
    ch = np.load(D / "chunks.npy").astype(np.float32)
    ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
    texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
    dmeta = json.loads((D / "doc_meta.json").read_text())
    cos_ch = ch @ q
    by_doc = {}
    for ci, mm in enumerate(meta):
        did = mm["doc_id"]
        if did not in by_doc or cos_ch[ci] > cos_ch[by_doc[did]]:
            by_doc[did] = ci
    chain_rankpos = {ni: r for r, ni in enumerate(r_chain)}
    onehop_rankpos = {ni: r for r, ni in enumerate(r_1hop)}

    excl = set((exclude or "").split(",")) - {""}
    out = []
    for ni in fused:
        did = gids[ni]
        if did in excl or did not in by_doc:
            continue
        ci = by_doc[did]
        dm = dmeta.get(did, {})
        # "reach" = surfaced by the multi-hop chain but NOT near the top of 1-hop
        reach = chain_rankpos.get(ni, 10**9) < 30 and onehop_rankpos.get(ni, 10**9) >= 30
        out.append({"doc_id": did, "rank": len(out) + 1,
                    "via": "2-hop-chain" if reach else "direct/topical",
                    "src": dm.get("src", "?"), "tier": dm.get("tier", 9),
                    "date": dm.get("date", "?"), "title": dm.get("title", "?"),
                    "snippet": texts[ci]})
        if len(out) >= k:
            break
    return out


def causal_direction(a, b):
    """Directed arrow between two texts via the asymmetric operator."""
    W = np.load(D / "transport_w_indist.npy")
    va, vb = _embed([a, b])
    ta = va @ W; ta /= (np.linalg.norm(ta) + 1e-9)
    tb = vb @ W; tb /= (np.linalg.norm(tb) + 1e-9)
    s_ab, s_ba = float(ta @ vb), float(tb @ va)
    return {"a_causes_b": s_ab, "b_causes_a": s_ba,
            "arrow": "a->b" if s_ab > s_ba else "b->a",
            "margin": round(abs(s_ab - s_ba), 4),
            "note": "candidate direction (0.80-0.83 acc on held-out pairs); verify with the gate"}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find")
    f.add_argument("query"); f.add_argument("-k", type=int, default=8)
    f.add_argument("--before"); f.add_argument("--exclude"); f.add_argument("--json", action="store_true")
    c = sub.add_parser("chain")
    c.add_argument("query"); c.add_argument("-k", type=int, default=8)
    c.add_argument("--before"); c.add_argument("--exclude"); c.add_argument("--json", action="store_true")
    d = sub.add_parser("direction")
    d.add_argument("a"); d.add_argument("b")
    a = ap.parse_args()

    if a.cmd == "find":
        res = causal_find(a.query, k=a.k, before=a.before, exclude=a.exclude)
        if a.json:
            print(json.dumps(res, indent=2)); return
        print(f"CAUSAL-FIND (W-only, embedding traversal) — {a.query!r}")
        print("candidate UPSTREAM CAUSES (low cosine + high causal = reach cosine lacks):")
        for i, r in enumerate(res, 1):
            print(f"\n[{i}] causal={r['causal_score']:.3f} cos={r['cosine']:.3f}  "
                  f"[{r['src']}] t{r['tier']}  {r['date']}  ({r['doc_id'][:8]})")
            print(f"    {r['title'][:90]}")
            print(f"    {' '.join(r['snippet'].split()[:45])}")
    elif a.cmd == "chain":
        res = causal_chain(a.query, k=a.k, before=a.before, exclude=a.exclude)
        if a.json:
            print(json.dumps(res, indent=2)); return
        print(f"CAUSAL-CHAIN (abductive fusion over the precomputed semantic graph) — {a.query!r}")
        print("candidate CAUSES incl. causes-of-causes; via=2-hop-chain marks multi-hop reach:")
        for i, r in enumerate(res, 1):
            print(f"\n[{i}] via={r['via']:<14} "
                  f"[{r['src']}] t{r['tier']}  {r['date']}  ({r['doc_id'][:8]})")
            print(f"    {r['title'][:90]}")
            print(f"    {' '.join(r['snippet'].split()[:45])}")
    else:
        print(json.dumps(causal_direction(a.a, a.b), indent=2))


if __name__ == "__main__":
    main()
