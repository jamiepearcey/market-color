# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "pyarrow", "numpy", "fastembed>=0.3"]
# ///
"""
Export the RAG chunk pool + causal queries for the reranking experiment.

Outputs (into receptors/data/):
  chunks.npy    (M, 384) f32 L2-normalised passage embeddings
  chunks.jsonl  M lines: {chunk_id, doc_id, epoch}
  queries.npy   (Q, 384) f32 L2-normalised claim embeddings
  queries.jsonl Q lines: {query_id, effect_doc_id, epoch, claim, cause_entities[]}

Naive chunking: title + fixed ~120-word body windows, max 6 per doc.
Query = an atomic fact's `claim` (an effect); its gold causes are computed in
Rust from cause_entities. Only facts with cause_entities are emitted.
"""
import json, datetime as dt
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1] / "data"
OUT.mkdir(parents=True, exist_ok=True)

FACTS = ROOT / "facts_work" / "facts.parquet"
CORPUS = ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet"

ALIASES = {
    "us": "united states", "u.s.": "united states", "usa": "united states",
    "fed": "federal reserve", "the fed": "federal reserve",
    "uk": "united kingdom", "ecb": "european central bank",
    "boj": "bank of japan", "rbnz": "reserve bank of new zealand",
}
def norm(e):
    e = (e or "").strip().lower(); return ALIASES.get(e, e)
def to_epoch(ts):
    if not ts: return 0
    try: return int(dt.datetime.fromisoformat(ts.replace("Z","+00:00")).timestamp())
    except Exception:
        try: return int(dt.datetime.fromisoformat(ts+"T00:00:00+00:00").timestamp())
        except Exception: return 0

WORDS, MAXC = 120, 6
def chunk_text(title, body):
    body = body or ""
    ws = body.split()
    out = []
    for i in range(0, min(len(ws), WORDS*MAXC), WORDS):
        seg = " ".join(ws[i:i+WORDS])
        out.append(((title or "") + " — " + seg).strip()[:2000])
    if not out:
        out = [((title or "").strip() or "untitled")[:2000]]
    return out

con = duckdb.connect()

# ---- chunk pool: ALL corpus articles ----
print("loading corpus ...")
corpus = con.execute(f"""
    SELECT doc_id, any_value(title) title, any_value(body_text) body,
           any_value(published_utc) put, any_value(published_date) pdate
    FROM '{CORPUS.as_posix()}' GROUP BY doc_id
""").fetchall()
chunk_meta, chunk_texts = [], []
for doc_id, title, body, put, pdate in corpus:
    ep = to_epoch(put or (pdate+"T00:00:00+00:00" if pdate else ""))
    for ci, txt in enumerate(chunk_text(title, body)):
        chunk_meta.append({"chunk_id": f"{doc_id}:{ci}", "doc_id": doc_id, "epoch": ep})
        chunk_texts.append(txt)
print(f"  {len(corpus)} docs -> {len(chunk_texts)} chunks")

# ---- queries: facts with cause_entities ----
print("loading query facts ...")
qrows = con.execute(f"""
    SELECT fact_id, doc_id, claim, cause_entities, published_utc, published_date
    FROM '{FACTS.as_posix()}'
    WHERE cause_entities IS NOT NULL AND len(cause_entities) > 0
      AND claim IS NOT NULL AND length(claim) > 8
""").fetchall()
queries, qtexts = [], []
for fid, doc_id, claim, causes, put, pdate in qrows:
    ce = sorted({norm(e) for e in (causes or []) if e and e.strip()})
    if not ce: continue
    queries.append({
        "query_id": fid, "effect_doc_id": doc_id,
        "epoch": to_epoch(put or (pdate+"T00:00:00+00:00" if pdate else "")),
        "claim": claim, "cause_entities": ce,
    })
    qtexts.append(claim[:2000])
print(f"  {len(queries)} queries")

# ---- embed ----
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
def embed(texts):
    v = np.array(list(model.embed(texts)), dtype=np.float32)
    v /= (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
    return v

print("embedding chunks ...")
cvec = embed(chunk_texts)
print("embedding queries ...")
qvec = embed(qtexts)

np.save(OUT / "chunks.npy", cvec)
np.save(OUT / "queries.npy", qvec)
with open(OUT / "chunks.jsonl", "w") as f:
    for m in chunk_meta: f.write(json.dumps(m) + "\n")
with open(OUT / "queries.jsonl", "w") as f:
    for q in queries: f.write(json.dumps(q) + "\n")
print(f"wrote chunks {cvec.shape}, queries {qvec.shape}")
