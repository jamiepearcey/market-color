# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Fit the IN-DISTRIBUTION transport operator W — the README's strongest finder
config (standalone first-stage cause retrieval: R@10 +53% random / +34% temporal
over cosine).

Pairs are (cause_chunk_embedding -> effect_claim_embedding): the exact objects W
ranks at inference. Gold cause docs for a claim = earlier docs ABOUT a specific
cause_entity of that claim (labels already paid for; used ONLY as training
supervision — no graph at inference, pure embedding traversal).

Outputs:
  data/transport_w_indist.npy   (384x384)
  data/chunks_transported.npy   (21422x384, L2-normalised chunk·W — the live
                                 cause-finder index; one matmul, cached)
"""
import json, numpy as np
from collections import Counter

D = "data"
docs = [json.loads(l) for l in open(f"{D}/docs.jsonl")]
epoch_d = {d["doc_id"]: d["published_epoch"] for d in docs}
ents = {d["doc_id"]: set(d.get("entities") or []) for d in docs}
N = len(docs)

cents_all = [set(d.get("cause_entities") or []) for d in docs]
df = Counter(e for s in cents_all for e in s)
idf = {e: np.log(N / c) for e, c in df.items()}
# "reasonably specific": drop only genuinely generic tokens (appear in >2% of docs),
# matching the Rust miner's much broader pair yield.
specific = {e for e, c in df.items() if c <= max(5, int(0.02 * N))}

about = {}
for did, s in ents.items():
    for e in s:
        about.setdefault(e, []).append(did)

queries = [json.loads(l) for l in open(f"{D}/queries.jsonl")]
Q = np.load(f"{D}/queries.npy").astype(np.float32)
Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)

ch = np.load(f"{D}/chunks.npy").astype(np.float32)
ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
cmeta = [json.loads(l) for l in open(f"{D}/chunks.jsonl")]
chunks_by_doc = {}
for i, m in enumerate(cmeta):
    chunks_by_doc.setdefault(m["doc_id"], []).append(i)

CAP = 12  # chunks per query (sorted gold for determinism, per README)
Xs, Ys = [], []
n_q = 0
for q, qv in zip(queries, Q):
    ces = set(q["cause_entities"]) & specific
    if not ces:
        continue
    gold = sorted({d for e in ces for d in about.get(e, ())
                   if epoch_d.get(d, 0) < q["epoch"] and d != q["effect_doc_id"]})
    if not gold:
        continue
    n_q += 1
    cnt = 0
    for d in gold:
        for ci in chunks_by_doc.get(d, ()):
            Xs.append(ci); Ys.append(qv)
            cnt += 1
            if cnt >= CAP:
                break
        if cnt >= CAP:
            break
X = ch[Xs]
Y = np.array(Ys, dtype=np.float32)
print(f"queries with gold: {n_q}  in-dist pairs (cause_chunk -> effect_claim): {len(Xs)}")

lam = 1.0
W = np.linalg.solve(X.T @ X + lam * np.eye(384, dtype=np.float32), X.T @ Y)
np.save(f"{D}/transport_w_indist.npy", W.astype(np.float32))

T = ch @ W
T /= (np.linalg.norm(T, axis=1, keepdims=True) + 1e-9)
np.save(f"{D}/chunks_transported.npy", T.astype(np.float32))
print(f"saved transport_w_indist.npy {W.shape} + chunks_transported.npy {T.shape}")
