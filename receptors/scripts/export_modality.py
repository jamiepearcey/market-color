# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "numpy", "fastembed>=0.3"]
# ///
"""Export labeled data for the MODALITY (epistemic-status) receptor: fact claims
with their `predicate` class (event/forecast/statement/policy_action/...).
Modality is orthogonal to topic AND to direction — tests whether a linear probe
separates fact-vs-forecast-vs-opinion that plain cosine cannot."""
import json
from pathlib import Path
import duckdb, numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "data"
FACTS = (ROOT / "facts_work" / "facts.parquet").as_posix()

# meaningful modality classes (drop tiny/catch-all)
KEEP = ["price_move","forecast","statement","policy_action","corporate_action",
        "event","deal_or_contract","supply_change","demand_change","production_change","sanction"]
cls2idx = {c: i for i, c in enumerate(KEEP)}

con = duckdb.connect()
rows = con.execute(f"""
    SELECT claim, subject, lower(predicate) p
    FROM '{FACTS}'
    WHERE lower(predicate) IN ({','.join("'"+c+"'" for c in KEEP)})
      AND claim IS NOT NULL AND length(claim) > 8 AND subject IS NOT NULL
""").fetchall()
texts = [((r[1] or "") + ": " + r[0])[:400] for r in rows]
ys = [cls2idx[r[2]] for r in rows]
subs = [(r[1] or "").strip().lower() for r in rows]
import collections
print(f"{len(rows)} claims across {len(KEEP)} modalities")
for c, n in collections.Counter(rows and [r[2] for r in rows]).most_common():
    print(f"  {c:20} {n}")

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
v = np.array(list(model.embed(texts)), dtype=np.float32)
v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
np.save(OUT / "modality.npy", v)
(OUT / "modality_labels.jsonl").write_text(
    "\n".join(json.dumps({"y": y, "subj": s}) for y, s in zip(ys, subs)) + "\n")
(OUT / "modality_classes.json").write_text(json.dumps(KEEP))
print(f"wrote modality.npy {v.shape}, modality_labels.jsonl, modality_classes.json")
