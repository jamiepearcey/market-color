# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Emit blind judge packets for the (qid,doc) pairs the stacking experiment surfaced
that no prior judge covered, so the rerankers can be scored at full coverage."""
import json
from pathlib import Path
import numpy as np
DATA = Path("data"); DEVAL = DATA / "eval"
unc = json.load(open(DEVAL / "stack_uncovered.json"))
by_q = {}
for qid, did in unc: by_q.setdefault(qid, []).append(did)
queries = {q["id"]: q["effect"] for q in json.load(open(DEVAL / "queries.json"))}
dmeta = json.loads((DATA / "doc_meta.json").read_text())
ch = np.load(DATA / "chunks.npy").astype(np.float32); ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
meta = [json.loads(l) for l in (DATA / "chunks.jsonl").read_text().splitlines() if l.strip()]
texts = [json.loads(l)["t"] for l in (DATA / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
by_doc = {}
for ci, m in enumerate(meta): by_doc.setdefault(m["doc_id"], []).append(ci)
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
key = {}
for qid, dids in by_q.items():
    qv = np.array(list(model.embed([queries[qid]])), np.float32)[0]; qv /= (np.linalg.norm(qv)+1e-9)
    docs, kmap = [], {}
    for i, did in enumerate(sorted(set(dids))):
        cis = by_doc.get(did)
        if not cis: continue
        sn = texts[max(cis, key=lambda ci: float(ch[ci] @ qv))][:600]
        lid = f"D{i+1:02d}"
        docs.append({"lid": lid, "title": dmeta.get(did, {}).get("title", "?"), "snippet": sn})
        kmap[lid] = {"doc_id": did}
    json.dump({"effect": queries[qid], "docs": docs}, open(DEVAL / f"packet_unc_{qid}.json", "w"), indent=1)
    key[qid] = kmap
json.dump(key, open(DEVAL / "key_unc.json", "w"), indent=1)
print("uncovered packets:", {q: len(k) for q, k in key.items()})
