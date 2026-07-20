# /// script
# requires-python = ">=3.10"
# dependencies = ["pyarrow", "numpy", "fastembed>=0.3"]
# ///
"""
Embed the Bloomberg batch (receptors/data_bloomberg/docs.jsonl) with the SAME
model the receptor experiment uses (all-MiniLM-L6-v2, 384-d, L2-normalized),
so data_bloomberg/ is a drop-in sibling of data/ for the Rust receptor.

Row i of embeddings.npy <-> line i of docs.jsonl (already sorted by
(published_epoch, doc_id) by the extractor).

Usage:
  uv run receptors/scripts/bloomberg_embed.py
"""
import json, re, hashlib
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq, pyarrow.compute as pc

ROOT = Path(__file__).resolve().parents[2]
BBG  = Path("/Users/jamiepearcey/Downloads/bloomberg_financial_data.parquet.gzip")
OUTD = ROOT / "receptors" / "data_bloomberg"
MAXC = 8000

docs = [json.loads(l) for l in open(OUTD / "docs.jsonl")]
want = {d["doc_id"] for d in docs}
print(f"{len(docs)} docs to embed")

# rebuild doc_id -> text by recomputing the same hash the extractor used.
t = pq.ParquetFile(BBG).read()
H = t.column("Headline").to_pylist(); A = t.column("Article").to_pylist(); L = t.column("Link").to_pylist()
text_by_id = {}
for h, a, l in zip(H, A, L):
    did = "bbg_" + hashlib.sha1((l or h).encode()).hexdigest()[:16]
    if did in want:
        text_by_id[did] = ((h or "") + "\n\n" + (a or ""))[:MAXC]
texts = [text_by_id.get(d["doc_id"], "") for d in docs]
missing = sum(1 for x in texts if not x.strip())
print(f"matched text for {len(docs)-missing}/{len(docs)} docs")

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
vecs = np.array(list(model.embed(texts)), dtype=np.float32)
vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
np.save(OUTD / "embeddings.npy", vecs)
print(f"wrote {OUTD/'embeddings.npy'} {vecs.shape}")
