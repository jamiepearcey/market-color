# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb>=1.0", "numpy", "fastembed>=0.3"]
# ///
"""Build blind A/B contexts: NAIVE (cosine top-6) vs HYDE (naive ∪ hypothesis-
retrieved, RRF-fused, novelty-diversified to 6). Same 6-doc budget for fairness.
A/B assignment alternates by question index (recorded) to keep the judge blind."""
import json
from pathlib import Path
import numpy as np, duckdb

ROOT = Path(__file__).resolve().parents[2]
D = Path(__file__).resolve().parents[1] / "data"
CORPUS = (ROOT / "data" / "news_corpus" / "dt=*" / "*.parquet").as_posix()

chunks = np.load(D/"chunks.npy"); chunks /= (np.linalg.norm(chunks,axis=1,keepdims=True)+1e-9)
cmeta = [json.loads(l) for l in (D/"chunks.jsonl").read_text().splitlines() if l.strip()]
dmap = {}
for i,m in enumerate(cmeta): dmap.setdefault(m["doc_id"], []).append(i)
doc_ids = list(dmap); did2row = {d:i for i,d in enumerate(doc_ids)}
demb = np.array([chunks[dmap[d]].mean(0) for d in doc_ids]); demb /= (np.linalg.norm(demb,axis=1,keepdims=True)+1e-9)

res = json.loads((D/"ask_results.json").read_text())
hyps = {json.loads(l)["i"]: json.loads(l)["hyps"] for l in (D/"hypotheses.jsonl").read_text().splitlines() if l.strip()}
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
allh = [(i,h) for i in hyps for h in hyps[i]]
hv = np.array(list(model.embed([h for _,h in allh]))); hv /= (np.linalg.norm(hv,axis=1,keepdims=True)+1e-9)
hbyq = {}
for (i,_),v in zip(allh, hv): hbyq.setdefault(i, []).append(v)

def mmr(pool, relscore, k=6, lam=0.7):
    sel=[]; cand=list(pool)
    while cand and len(sel)<k:
        best,bi=-1e9,0
        for ii,d in enumerate(cand):
            ms=max([float(demb[d]@demb[s]) for s in sel], default=0.0)
            sc=lam*relscore[d]-(1-lam)*ms
            if sc>best: best,bi=sc,ii
        sel.append(cand.pop(bi))
    return sel

K=6
mapping={}
out=[]
for i,r in enumerate(res):
    naive=[did2row[d] for d in r["cosine"][:15] if d in did2row]
    naive_top=naive[:K]
    # hypothesis retrievals
    hyp_ranklist=[]
    for v in hbyq.get(i,[]):
        hyp_ranklist.append(list(np.argsort(-(demb@v))[:8]))
    # RRF fuse naive(rank) + best hypothesis rank
    pool=set(naive) | {int(d) for hl in hyp_ranklist for d in hl}
    def rrf(d):
        s=0.0
        if d in naive: s+=1/(60+naive.index(d))
        for hl in hyp_ranklist:
            if d in hl: s=max(s, s+1/(60+list(hl).index(d)))
        return s
    rel={d:rrf(d) for d in pool}
    hyde_top=mmr(sorted(pool,key=lambda d:-rel[d]), rel, K)
    mapping[i]={"naive":[doc_ids[d] for d in naive_top], "hyde":[doc_ids[d] for d in hyde_top]}
    out.append((i, r["question"], naive_top, hyde_top))

ids=sorted({d for m in mapping.values() for k in ("naive","hyde") for d in m[k]})
con=duckdb.connect(); con.execute("CREATE TEMP TABLE want(doc_id VARCHAR)")
con.executemany("INSERT INTO want VALUES (?)",[(x,) for x in ids])
meta={r[0]:(r[1] or '?', r[2] or '') for r in con.execute(f"SELECT doc_id, any_value(title), any_value(body_text) FROM '{CORPUS}' JOIN want USING(doc_id) GROUP BY doc_id").fetchall()}
def ctx(rows):
    return "\n".join(f"  - {meta[doc_ids[d]][0][:70]} :: {' '.join(meta[doc_ids[d]][1].split()[:60])}" for d in rows)

# alternate A/B; record which label is naive/hyde
ab={}
batches=[[],[]]
for i,q,nt,ht in out:
    if i%2==0: A,B,aid,bid=("naive",nt,"naive","hyde"); a_rows,b_rows=nt,ht
    else:      a_rows,b_rows=ht,nt
    a_is = "naive" if i%2==0 else "hyde"
    ab[i]={"A":a_is, "B":("hyde" if a_is=="naive" else "naive")}
    a_rows = nt if a_is=="naive" else ht
    b_rows = ht if a_is=="naive" else nt
    block=f"{'='*80}\nQUESTION {i}: {q}\n\n### CONTEXT A ###\n{ctx(a_rows)}\n\n### CONTEXT B ###\n{ctx(b_rows)}\n"
    batches[0 if i<10 else 1].append(block)

for bi,b in enumerate(batches,1):
    (D/f"hyde_eval_{bi}.txt").write_text("\n".join(b))
(D/"hyde_ab_map.json").write_text(json.dumps(ab))
print("wrote hyde_eval_1.txt, hyde_eval_2.txt, hyde_ab_map.json")
# overlap sanity
for i,m in mapping.items():
    ov=len(set(m["naive"])&set(m["hyde"]))
    if i<5: print(f"  Q{i}: naive∩hyde = {ov}/{K} (higher overlap = smaller test)")
