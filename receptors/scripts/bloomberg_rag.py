# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "numpy", "fastembed>=0.3"]
# ///
"""
Build the RAG-mode artifacts for the Bloomberg batch so `receptors rerank`
(and the in-distribution / W-as-finder / multihop levers) can run:

  chunks.npy    (M,384) float32 L2-normed  -- embed chunk_texts.jsonl, aligned to chunks.jsonl
  queries.jsonl Q lines {query_id, effect_doc_id, epoch, claim, cause_entities}
  queries.npy   (Q,384) float32 L2-normed  -- embed the query claim text

Query = an EFFECT doc (one per doc carrying cause_entities); the query TEXT is the
Bloomberg HEADLINE (a faithful effect-claim proxy, since we deliberately dropped
free-text claims). Gold (built in Rust) = earlier docs sharing a cause entity.

Usage:  uv run receptors/scripts/bloomberg_rag.py
"""
import json, hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
BBG  = Path("/Users/jamiepearcey/Downloads/bloomberg_financial_data.parquet.gzip")
OUTD = ROOT / "receptors" / "data_bloomberg"

def l2(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---- chunks.npy : embed chunk_texts.jsonl (aligned to chunks.jsonl) ----
ctexts = [json.loads(l)["t"] for l in open(OUTD / "chunk_texts.jsonl")]
cmeta  = [json.loads(l) for l in open(OUTD / "chunks.jsonl")]
assert len(ctexts) == len(cmeta), "chunk text/meta mismatch"
print(f"embedding {len(ctexts)} chunks ...")
cvec = l2(list(model.embed(ctexts)))
np.save(OUTD / "chunks.npy", cvec)
print(f"  wrote chunks.npy {cvec.shape}")

# ---- headlines by doc_id (query claim text) ----
docs = [json.loads(l) for l in open(OUTD / "docs.jsonl")]
qdocs = [d for d in docs if d.get("cause_entities")]
want = {d["doc_id"] for d in qdocs}
t = pq.ParquetFile(BBG).read()
H = t.column("Headline").to_pylist(); L = t.column("Link").to_pylist()
head_by_id = {}
for h, l in zip(H, L):
    did = "bbg_" + hashlib.sha1((l or h).encode()).hexdigest()[:16]
    if did in want:
        head_by_id[did] = h or ""

# ---- queries.jsonl + queries.npy ----
queries, qtexts = [], []
for d in qdocs:
    claim = head_by_id.get(d["doc_id"], "")
    if not claim.strip():
        continue
    queries.append({"query_id": d["doc_id"], "effect_doc_id": d["doc_id"],
                    "epoch": d["published_epoch"], "claim": claim,
                    "cause_entities": d["cause_entities"]})
    qtexts.append(claim)
print(f"embedding {len(queries)} effect-claim queries (headlines) ...")
qvec = l2(list(model.embed(qtexts)))
with open(OUTD / "queries.jsonl", "w") as f:
    for q in queries: f.write(json.dumps(q) + "\n")
np.save(OUTD / "queries.npy", qvec)
print(f"  wrote queries.jsonl ({len(queries)}) and queries.npy {qvec.shape}")
