# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Re-embed the EXISTING chunk + query texts with an alternative embedding model,
into an isolated data dir, so the chain-retrieval eval can be run on a richer
substrate without disturbing the MiniLM baseline.

Only chunks.npy + queries.npy change (that's all the multihop/retrieve eval reads
for embeddings — doc_emb is mean-pooled from chunks at eval time). Everything else
(chunks.jsonl, queries.jsonl, docs.jsonl) is symlinked from the base data dir.

Usage:
  uv run scripts/reembed_variant.py --model BAAI/bge-base-en-v1.5 --out data_bge_base \
        --doc-prefix "" --query-prefix ""
Prefix schemes (model-dependent, matters a lot for retrieval-tuned models):
  bge:   query "Represent this sentence for searching relevant passages: "  doc ""
  e5:    query "query: "  doc "passage: "
  nomic: query "search_query: "  doc "search_document: "
"""
import argparse, json, os
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True, help="output data dir (created)")
ap.add_argument("--base", default="data")
ap.add_argument("--doc-prefix", default="")
ap.add_argument("--query-prefix", default="")
ap.add_argument("--batch", type=int, default=256)
a = ap.parse_args()

base = Path(a.base).resolve()
out = Path(a.out).resolve()
out.mkdir(parents=True, exist_ok=True)

chunk_texts = [json.loads(l)["t"] for l in open(base / "chunk_texts.jsonl") if l.strip()]
qtexts = [json.loads(l)["claim"][:2000] for l in open(base / "queries.jsonl") if l.strip()]
print(f"{len(chunk_texts)} chunks, {len(qtexts)} queries; model={a.model}")

from fastembed import TextEmbedding
model = TextEmbedding(model_name=a.model)

def embed(texts, prefix):
    texts = [prefix + t for t in texts] if prefix else texts
    v = np.array(list(model.embed(texts, batch_size=a.batch)), dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v

print("embedding chunks (docs) ...")
cvec = embed(chunk_texts, a.doc_prefix)
print("embedding queries ...")
qvec = embed(qtexts, a.query_prefix)
np.save(out / "chunks.npy", cvec)
np.save(out / "queries.npy", qvec)

# symlink the unchanged sidecar files so RECEPTORS_DATA=<out> is self-contained
for fn in ["chunks.jsonl", "queries.jsonl", "docs.jsonl", "chunk_texts.jsonl",
           "doc_meta.json", "embeddings.npy"]:
    src = base / fn
    dst = out / fn
    if src.exists() and not dst.exists():
        os.symlink(src, dst)
print(f"wrote {out}/chunks.npy {cvec.shape}, queries.npy {qvec.shape} (dim={cvec.shape[1]})")
