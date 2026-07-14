# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "numpy", "fastembed>=0.3"]
# ///
"""Validation loop: for each LLM-distilled hypothesis, retrieve corpus evidence
so a validator can rule SUPPORTED / PARTIAL / UNSUPPORTED. Measures whether the
loop catches speculative (unsupported) causal claims before they enter a report."""
import json
from pathlib import Path
import numpy as np, duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()

chunks = np.load(D/"chunks.npy"); chunks /= (np.linalg.norm(chunks,axis=1,keepdims=True)+1e-9)
cmeta = [json.loads(l) for l in (D/"chunks.jsonl").read_text().splitlines() if l.strip()]
dmap={}
for i,m in enumerate(cmeta): dmap.setdefault(m["doc_id"], []).append(i)
doc_ids=list(dmap)
demb=np.array([chunks[dmap[d]].mean(0) for d in doc_ids]); demb/=(np.linalg.norm(demb,axis=1,keepdims=True)+1e-9)
res = json.loads((D/"ask_results.json").read_text())
hyps=[(json.loads(l)["i"], k, h) for l in (D/"hypotheses.jsonl").read_text().splitlines() if l.strip()
      for k,h in enumerate(json.loads(l)["hyps"])]

from fastembed import TextEmbedding
model=TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
hv=np.array(list(model.embed([h for _,_,h in hyps]))); hv/=(np.linalg.norm(hv,axis=1,keepdims=True)+1e-9)

# retrieve top-5 evidence docs per hypothesis
retr={}
for (qi,k,h),v in zip(hyps,hv):
    top=list(np.argsort(-(demb@v))[:5])
    retr[(qi,k)]=[doc_ids[d] for d in top]
ids=sorted({d for v in retr.values() for d in v})
con=duckdb.connect(); con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)",[(x,) for x in ids])
meta={r[0]:(r[1] or '?', r[2] or '') for r in con.execute(f"SELECT doc_id, any_value(title), any_value(body_text) FROM '{CORPUS}' JOIN want USING(doc_id) GROUP BY doc_id").fetchall()}

blocks=[[],[]]
for idx,(qi,k,h) in enumerate(hyps):
    ev="\n".join(f"  - {meta[d][0][:70]} :: {' '.join(meta[d][1].split()[:55])}" for d in retr[(qi,k)])
    blk=f"{'='*80}\nHYPOTHESIS {qi}.{k}: {h}\n\nRETRIEVED EVIDENCE:\n{ev}\n"
    blocks[0 if idx<20 else 1].append(blk)
for bi,b in enumerate(blocks,1):
    (D/f"hyde_valid_{bi}.txt").write_text("\n".join(b))
print(f"wrote hyde_valid_1.txt ({len(blocks[0])} hyps), hyde_valid_2.txt ({len(blocks[1])} hyps)")
