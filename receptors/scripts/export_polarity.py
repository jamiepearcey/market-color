# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "numpy", "fastembed>=0.3"]
# ///
"""Export labeled data for the SIGNED-IMPACT (polarity) receptor: fact claims
with an up/down market direction. Polarity is orthogonal to topic, so this tests
whether a linear receptor recovers a signal plain cosine similarity cannot."""
import json
from pathlib import Path
import duckdb, numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "data"
FACTS = (ROOT / "facts_work" / "facts.parquet").as_posix()

con = duckdb.connect()
rows = con.execute(f"""
    SELECT claim, subject, direction
    FROM '{FACTS}'
    WHERE direction IN ('up','down') AND claim IS NOT NULL AND length(claim) > 8
      AND subject IS NOT NULL
""").fetchall()
texts = [((r[1] or "") + ": " + r[0])[:400] for r in rows]
labels = [1 if r[2] == 'up' else -1 for r in rows]
subs = [(r[1] or "").strip().lower() for r in rows]
print(f"{len(rows)} labeled claims ({sum(l==1 for l in labels)} up / {sum(l==-1 for l in labels)} down)")

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
v = np.array(list(model.embed(texts)), dtype=np.float32)
v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
np.save(OUT / "claims.npy", v)
(OUT / "claims_labels.jsonl").write_text(
    "\n".join(json.dumps({"y": y, "subj": s}) for y, s in zip(labels, subs)) + "\n")
print(f"wrote claims.npy {v.shape}, claims_labels.jsonl")
