# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Build the SEMANTIC CAUSAL GRAPH once, at index time, from the transport
operator W — a faithful Python port of the Rust `build_causal_graph` (src/rerank.rs).

The whole point: the graph is PRECOMPUTED here (no LLM, no symbolic extraction,
no per-query graph construction). At inference `causal_chain` just runs a sparse
power iteration over this file — so we get multi-hop causes-of-causes reach with
ZERO inference-time graph-building cost. This is the "semantic graph": nodes are
docs, edges i->j = "i is a cause of j" derived purely from doc·W scores.

Method (matches Rust, content-hub-suppressed variant, K=10, gamma=1.0):
  doc_emb = mean-pooled normalised chunk embeddings per doc
  doc_w   = normalize(doc_emb @ W)
  S[i,j]  = doc_w[i] . doc_emb[j]        # "i causes j"
  per effect j: keep top-K earlier causes i, softmax((S-mx)*4) with a content
  hub penalty exp(-gamma * z(doc_w[i].mean_emb)) so "cause-of-everything" macro
  hubs stop dominating; renormalise -> column-stochastic edge weights.

Outputs data/semantic_graph.npz:
  edges_idx (N,K) int32  cause node ids per effect (-1 padded)
  edges_wt  (N,K) f32    column-stochastic weights
  epoch     (N,)  int64
  doc_w     (N,D) f32    transported doc rows (query seed = relu(doc_w @ q))
  doc_ids   (N,)  <U..   doc id strings
"""
import json, numpy as np
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "data"
K = 10
GAMMA = 1.0

docs = [json.loads(l) for l in (D / "docs.jsonl").read_text().splitlines() if l.strip()]
doc_ids = [d["doc_id"] for d in docs]
id2idx = {did: i for i, did in enumerate(doc_ids)}
epoch = np.array([d.get("published_epoch", 0) for d in docs], dtype=np.int64)
N = len(docs)

# ---- doc embeddings = mean-pooled normalised chunk embeddings (matches Rust) ----
ch = np.load(D / "chunks.npy").astype(np.float32)
ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
cmeta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
Dd = ch.shape[1]
doc_emb = np.zeros((N, Dd), dtype=np.float32)
cnt = np.zeros(N, dtype=np.int64)
for ci, m in enumerate(cmeta):
    a = id2idx.get(m["doc_id"])
    if a is not None:
        doc_emb[a] += ch[ci]
        cnt[a] += 1
valid = cnt > 0
doc_emb /= (np.linalg.norm(doc_emb, axis=1, keepdims=True) + 1e-9)

W = np.load(D / "transport_w_indist.npy").astype(np.float32)
doc_w = doc_emb @ W
doc_w /= (np.linalg.norm(doc_w, axis=1, keepdims=True) + 1e-9)

# S[i,j] = "i causes j"  (N x N; 3433^2 f32 ~ 47MB — fine)
S = doc_w @ doc_emb.T

# ---- content genericness: z-score of doc_w[i] . mean_emb (hub = points at centroid) ----
mean_emb = doc_emb[valid].mean(axis=0)
gen = doc_w @ mean_emb
gmean = gen[valid].mean()
gstd = max(float(gen[valid].std()), 1e-6)
zgen = (gen - gmean) / gstd
pen = np.exp(-GAMMA * zgen)
pen = np.clip(pen, 1e-3, 1e3)  # generic hubs down-weighted

edges_idx = np.full((N, K), -1, dtype=np.int32)
edges_wt = np.zeros((N, K), dtype=np.float32)
for j in range(N):
    if not valid[j]:
        continue
    # candidate causes: earlier in time, not self, valid
    mask = valid & (epoch < epoch[j])
    mask[j] = False
    cand = np.nonzero(mask)[0]
    if cand.size == 0:
        continue
    sc = S[cand, j]
    top = cand[np.argsort(-sc)[:K]]
    ssc = S[top, j]
    mx = float(ssc.max())
    e = np.exp((ssc - mx) * 4.0) * pen[top]
    z = float(e.sum())
    if z <= 0:
        continue
    e = e / z
    edges_idx[j, : top.size] = top.astype(np.int32)
    edges_wt[j, : top.size] = e.astype(np.float32)

np.savez(
    D / "semantic_graph.npz",
    edges_idx=edges_idx,
    edges_wt=edges_wt,
    epoch=epoch,
    doc_w=doc_w.astype(np.float32),
    doc_emb=doc_emb.astype(np.float32),
    doc_ids=np.array(doc_ids),
)
deg = np.zeros(N, dtype=np.int64)
for j in range(N):
    for i in edges_idx[j]:
        if i >= 0:
            deg[i] += 1
print(f"semantic_graph: N={N} valid={int(valid.sum())} K={K} gamma={GAMMA} "
      f"edges={(edges_idx >= 0).sum()} max_indeg={int(deg.max())} "
      f"median_indeg={int(np.median(deg))}")
