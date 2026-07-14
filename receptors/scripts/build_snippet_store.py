# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0"]
# ///
"""Precompute the fast evidence store the live retrieval tool reads.

Reconstructs chunk texts with the SAME chunker as export_rag.py (title + 120-word
body windows, max 6) so chunk_texts.jsonl aligns row-for-row with chunks.npy, and
emits doc_meta.json (source tier / domain / date / epoch) for point-in-time and
corroboration. Idempotent; run once after any corpus/chunk rebuild."""
import json
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()

WORDS, MAXC = 120, 6
def chunk_text(title, body):
    out, ws = [], (body or "").split()
    for i in range(0, min(len(ws), WORDS * MAXC), WORDS):
        out.append(f"{title} {' '.join(ws[i:i+WORDS])}".strip())
    return out or [title or ""]

# chunk order in chunks.npy is exactly the iteration order of chunks.jsonl
cmeta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
con = duckdb.connect()
rows = con.execute(f"""
    SELECT doc_id, any_value(title) title, any_value(body_text) body,
           any_value(source_name) src, any_value(source_domain) dom,
           any_value(source_tier) tier, any_value(published_date) pdate,
           any_value(url) url
    FROM '{CORPUS}' GROUP BY doc_id""").fetchall()
docs = {r[0]: r for r in rows}

# rebuild per-doc chunk text lists, then map chunk_id -> text
by_doc = {}
for did, (_, title, body, *_ ) in docs.items():
    by_doc[did] = chunk_text(title or "", body or "")

texts, miss = [], 0
for m in cmeta:
    did, cid = m["doc_id"], m["chunk_id"]
    ci = int(cid.split(":")[1])
    lst = by_doc.get(did, [])
    texts.append(lst[ci] if ci < len(lst) else (lst[0] if lst else ""))
    if did not in docs:
        miss += 1

with open(D / "chunk_texts.jsonl", "w") as f:
    for t in texts:
        f.write(json.dumps({"t": t}) + "\n")

doc_meta = {}
for did, (_, title, body, src, dom, tier, pdate, url) in docs.items():
    doc_meta[did] = {
        "title": title or "?", "src": src or "?", "dom": dom or "?",
        "tier": tier if tier is not None else 9,
        "date": str(pdate) if pdate is not None else "?",
        "url": url or "",
    }
(D / "doc_meta.json").write_text(json.dumps(doc_meta))
print(f"chunk_texts.jsonl: {len(texts)} rows (aligned={len(texts)==len(cmeta)}, missing_docs={miss})")
print(f"doc_meta.json: {len(doc_meta)} docs; tiers={sorted({m['tier'] for m in doc_meta.values()})}")
