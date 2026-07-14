# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Fit and persist the deployment transport operator W for live retrieval.

Same recipe as feature_ablation.py (ridge cause->effect over pairs mined from
cause_entities), but fit on ALL mined pairs — the labels are training supervision
only; at inference W operates on raw embeddings (any doc, any phrasing), which is
the receptors thesis: causal traversal in embedding space, no symbolic layer.
"""
import json, numpy as np
from collections import Counter

D = "data"
docs = [json.loads(l) for l in open(f"{D}/docs.jsonl")]
E = np.load(f"{D}/embeddings.npy").astype(np.float32)
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
N = len(docs)
epoch = np.array([d["published_epoch"] for d in docs])
ents = [set(d.get("entities") or []) for d in docs]
cents = [set(d.get("cause_entities") or []) for d in docs]

df = Counter(e for s in cents for e in s)
idf = {e: np.log(N / c) for e, c in df.items()}
med = np.median(list(idf.values()))
specific = {e for e, v in idf.items() if v >= med}

about = {}
for i, s in enumerate(ents):
    for e in s:
        about.setdefault(e, []).append(i)

best = {}
for b in range(N):
    for e in (cents[b] & specific):
        for a in about.get(e, ()):
            if epoch[a] < epoch[b] and a != b:
                if (a, b) not in best or idf[e] > idf[best[(a, b)]]:
                    best[(a, b)] = e
pairs = list(best)
print(f"mined pairs (all, no split): {len(pairs)}")

A = E[[a for a, _ in pairs]]
B = E[[b for _, b in pairs]]
lam = 1.0
W = np.linalg.solve(A.T @ A + lam * np.eye(E.shape[1], dtype=np.float32), A.T @ B)
np.save(f"{D}/transport_w.npy", W.astype(np.float32))
print(f"saved data/transport_w.npy  {W.shape}")
