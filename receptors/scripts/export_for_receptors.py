# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "pyarrow", "numpy", "fastembed>=0.3"]
# ///
"""
Export a clean float32 embedding matrix + aligned per-doc metadata for the
receptor / transport-operator experiment.

We deliberately re-embed (rather than pull quantised vectors out of Qdrant) so
the Rust side gets clean, deterministic float32 with the *same* model the
project already uses: sentence-transformers/all-MiniLM-L6-v2 (384-d, cosine).

Outputs (into receptors/data/):
  embeddings.npy   (N, 384) float32, L2-normalised, row i <-> docs.jsonl line i
  docs.jsonl       N lines: {doc_id, published_epoch, date, entities[], cause_entities[]}

Only fact-bearing docs are exported (those are the ones with mineable
cause_entities labels). Row order is deterministic: (published_epoch, doc_id).
"""
import json, datetime as dt
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]  # research/market-color
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)

FACTS = ROOT / "facts_work" / "facts.parquet"
CORPUS = ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet"

# Small alias map mirroring the project's entity normalisation, so cause_entities
# in one doc line up with entities in another.
ALIASES = {
    "us": "united states", "u.s.": "united states", "usa": "united states",
    "fed": "federal reserve", "the fed": "federal reserve",
    "uk": "united kingdom", "ecb": "european central bank",
    "boj": "bank of japan", "rbnz": "reserve bank of new zealand",
}

def norm(e: str) -> str:
    e = (e or "").strip().lower()
    return ALIASES.get(e, e)

def to_epoch(ts: str) -> int:
    if not ts:
        return 0
    try:
        return int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except Exception:
        try:
            return int(dt.datetime.fromisoformat(ts + "T00:00:00+00:00").timestamp())
        except Exception:
            return 0

con = duckdb.connect()

# Aggregate facts per doc: union of entities and cause_entities.
print("aggregating facts per doc ...")
rows = con.execute(f"""
    SELECT doc_id,
           any_value(published_utc)  AS published_utc,
           any_value(published_date) AS published_date,
           flatten(list(entities))        AS ents,
           flatten(list(cause_entities))  AS causes
    FROM '{FACTS.as_posix()}'
    GROUP BY doc_id
""").fetchall()

docs = []
for doc_id, put, pdate, ents, causes in rows:
    ents = sorted({norm(e) for e in (ents or []) if e and e.strip()})
    causes = sorted({norm(e) for e in (causes or []) if e and e.strip()})
    if not ents:          # need something to bind on
        continue
    docs.append({
        "doc_id": doc_id,
        "published_epoch": to_epoch(put or (pdate + "T00:00:00+00:00" if pdate else "")),
        "date": (pdate or (put or "")[:10]),
        "entities": ents,
        "cause_entities": causes,
    })

docs.sort(key=lambda d: (d["published_epoch"], d["doc_id"]))
ids = [d["doc_id"] for d in docs]
print(f"  {len(docs)} fact-bearing docs")

# Pull title+body for those docs from the corpus.
print("loading corpus text ...")
con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
text_rows = con.execute(f"""
    SELECT c.doc_id, c.title, c.body_text
    FROM '{CORPUS.as_posix()}' c
    JOIN want w USING (doc_id)
""").fetchall()
text_by_id = {r[0]: ((r[1] or "") + "\n\n" + (r[2] or ""))[:8000] for r in text_rows}
texts = [text_by_id.get(i, "") for i in ids]
missing = sum(1 for t in texts if not t.strip())
print(f"  {len(text_rows)} matched, {missing} missing text")

# Embed with the same model the project uses.
print("embedding (all-MiniLM-L6-v2) ...")
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
vecs = np.array(list(model.embed(texts)), dtype=np.float32)
# L2 normalise (cosine geometry).
vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
print(f"  embeddings {vecs.shape}")

np.save(OUT / "embeddings.npy", vecs)
with open(OUT / "docs.jsonl", "w") as f:
    for d in docs:
        f.write(json.dumps(d) + "\n")
print(f"wrote {OUT/'embeddings.npy'} and {OUT/'docs.jsonl'}")
