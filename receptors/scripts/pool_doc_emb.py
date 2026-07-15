# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Build a dimension-matched embeddings.npy for a re-embedded variant dir by
mean-pooling its chunk vectors per doc, in docs.jsonl row order. This replaces the
symlinked MiniLM embeddings.npy so the Rust Dataset dim (ds.d()) matches chunks.npy.
(rerank with INDIST=1 never uses embeddings.npy values — only its column count.)

Usage: uv run scripts/pool_doc_emb.py --dir data_bge_base
"""
import argparse, json, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
a = ap.parse_args()
d = Path(a.dir).resolve()

chunks = np.load(d / "chunks.npy").astype(np.float32)
dim = chunks.shape[1]
cmeta = [json.loads(l) for l in open(d / "chunks.jsonl") if l.strip()]
docs = [json.loads(l) for l in open(d / "docs.jsonl") if l.strip()]
id2row = {doc["doc_id"]: i for i, doc in enumerate(docs)}

acc = np.zeros((len(docs), dim), dtype=np.float32)
cnt = np.zeros(len(docs), dtype=np.float32)
for i, m in enumerate(cmeta):
    r = id2row.get(m["doc_id"])
    if r is not None:
        acc[r] += chunks[i]; cnt[r] += 1
nz = cnt > 0
acc[nz] /= cnt[nz, None]
norm = np.linalg.norm(acc, axis=1, keepdims=True)
acc = acc / (norm + 1e-9)

# replace the symlink/file
p = d / "embeddings.npy"
if p.is_symlink() or p.exists():
    p.unlink()
np.save(p, acc)
print(f"wrote {p} {acc.shape} (docs={len(docs)}, with-chunks={int(nz.sum())})")
