# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "numpy", "fastembed>=0.3"]
# ///
"""HyDE-causal retrieval: embed the LLM-distilled hypotheses, retrieve
conventionally, and measure whether they surface RELEVANT docs the naive first
pass missed. Guards against drift by scoring new docs' relevance to the ORIGINAL
question (not the hypothesis)."""
import json
from pathlib import Path
import numpy as np, duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()

# doc embeddings (mean-pool chunks)
chunks = np.load(D / "chunks.npy"); chunks /= (np.linalg.norm(chunks,axis=1,keepdims=True)+1e-9)
cmeta = [json.loads(l) for l in (D/"chunks.jsonl").read_text().splitlines() if l.strip()]
docs = {}
for i,m in enumerate(cmeta):
    docs.setdefault(m["doc_id"], []).append(i)
doc_ids = list(docs)
did2row = {d:i for i,d in enumerate(doc_ids)}
demb = np.array([chunks[docs[d]].mean(0) for d in doc_ids]); demb /= (np.linalg.norm(demb,axis=1,keepdims=True)+1e-9)

res = json.loads((D/"ask_results.json").read_text())
qemb = np.load(D/"questions_custom.npy"); qemb /= (np.linalg.norm(qemb,axis=1,keepdims=True)+1e-9)
hyps = {json.loads(l)["i"]: json.loads(l)["hyps"] for l in (D/"hypotheses.jsonl").read_text().splitlines() if l.strip()}

from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
all_h = [(i,h) for i in hyps for h in hyps[i]]
hvecs = np.array(list(model.embed([h for _,h in all_h]))); hvecs /= (np.linalg.norm(hvecs,axis=1,keepdims=True)+1e-9)
hyp_vec = {}
for (i,h),v in zip(all_h, hvecs):
    hyp_vec.setdefault(i, []).append((h,v))

TOPN=8; HK=6; REL=0.45
new_counts=[]; naive_rel=[]; new_rel=[]
examples=[]
for i,r in enumerate(res):
    q = qemb[i]
    naive = [did2row[d] for d in r["cosine"][:TOPN] if d in did2row]
    naive_set=set(naive)
    hyde=set()
    for h,v in hyp_vec.get(i,[]):
        top = np.argsort(-(demb @ v))[:HK]
        hyde.update(int(x) for x in top)
    new = [d for d in hyde if d not in naive_set]
    q_rel_new = [float(demb[d] @ q) for d in new]
    new_counts.append(len(new))
    naive_rel.extend(float(demb[d] @ q) for d in naive)
    new_rel.extend(q_rel_new)
    n_relevant_new = sum(1 for c in q_rel_new if c>REL)
    if i<3:
        examples.append((i, r["question"], [(doc_ids[d], round(c,3)) for d,c in
                        sorted(zip(new,q_rel_new), key=lambda x:-x[1])[:5]]))

import statistics as st
print(f"HyDE-causal over {len(res)} questions (naive top-{TOPN}, {HK}/hyp)")
print(f"  new docs surfaced by hypotheses (not in naive top-{TOPN}): mean {st.mean(new_counts):.1f}/q")
print(f"  relevance-to-QUESTION of naive top-{TOPN} docs : mean {st.mean(naive_rel):.3f}")
print(f"  relevance-to-QUESTION of NEW hypothesis docs   : mean {st.mean(new_rel):.3f}")
print(f"  new docs that are ALSO highly relevant to the question (cos>{REL}): "
      f"{sum(1 for c in new_rel if c>REL)}/{len(new_rel)} ({100*sum(1 for c in new_rel if c>REL)/max(len(new_rel),1):.0f}%)")

ids=sorted({d for _,_,ex in examples for d,_ in ex})
con=duckdb.connect(); con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)",[(x,) for x in ids])
title={r[0]:(r[1] or '?') for r in con.execute(f"SELECT doc_id, any_value(title) FROM '{CORPUS}' JOIN want USING(doc_id) GROUP BY doc_id").fetchall()}
print("\n=== NEW relevant docs the hypotheses surfaced (not in naive top-8) ===")
for i,qt,ex in examples:
    print(f"\nQ{i}: {qt[:72]}")
    for d,c in ex:
        print(f"   qrel={c:.2f} | {title.get(d,'?')[:66]}")
