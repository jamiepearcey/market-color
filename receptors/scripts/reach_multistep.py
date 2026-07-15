# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Isolate the PURE MULTI-STEP method's hypothesis quality at depth. Buckets:
  costop      cosine top-8                              (control)
  cosdeep     cosine rank 11..40                        (control: read deeper)
  onehopreach 1-hop W top-40, cosine rank >40           (cross-vocab DIRECT causes, no 2-hop)
  chainreach  PPR-chain top-40, cosine>40 AND 1-hop>40  (GENUINE 2-hop-only: causes-of-causes
                                                          neither cosine nor the 1-hop operator finds)
Judge blind 0/1/2. Answers: does the MULTI-STEP hop specifically find good drivers,
separate from the 1-hop operator? Requires the union graph (GRAPH_GAMMA=0 GRAPH_K=25).
"""
import sys, json
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np, hashlib
import receptors_tools as rt

D = Path("data"); DEVAL = Path("data/eval")
queries = json.load(open(DEVAL / "queries.json"))
g = rt._load_graph()
doc_emb = g["doc_emb"].astype(np.float32); doc_w = g["doc_w"].astype(np.float32)
gids = list(g["doc_ids"]); edges_idx, edges_wt = g["edges_idx"], g["edges_wt"]
dmeta = json.loads((D / "doc_meta.json").read_text())
ch = np.load(D / "chunks.npy").astype(np.float32); ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
by_doc = {}
for ci, mm in enumerate(meta):
    by_doc.setdefault(mm["doc_id"], []).append(ci)

def snip(did, qv):
    cis = by_doc.get(did)
    if not cis: return None
    return texts[max(cis, key=lambda ci: float(ch[ci] @ qv))][:600]

key = {}
for q in queries:
    qid, eff = q["id"], q["effect"]
    qv = rt._embed([eff])[0]
    cos = doc_emb @ qv
    cos_order = [gids[i] for i in np.argsort(-cos)]
    cos_rank = {d: r for r, d in enumerate(cos_order)}
    onehop = doc_w @ qv
    onehop_order = [gids[i] for i in np.argsort(-onehop)]
    onehop_rank = {d: r for r, d in enumerate(onehop_order)}
    # concentrated seed (deployed union recipe) -> PPR
    seed = np.maximum(doc_w @ qv, 0.0).astype(np.float32)
    keep = np.argsort(-seed)[:10]; m = np.zeros_like(seed, bool); m[keep] = True
    seed = np.where(m, seed, 0.0); seed = seed / (seed.sum() + 1e-9)
    hyp = rt._ppr(seed, edges_idx, edges_wt, alpha=0.9, iters=20)
    chain_order = [gids[i] for i in np.argsort(-hyp)]
    chain_rank = {d: r for r, d in enumerate(chain_order)}

    buckets = {}
    for d in cos_order[:8]: buckets[d] = "costop"
    for d in cos_order[10:40]: buckets.setdefault(d, "cosdeep")
    # 1-hop cross-vocab reach (direct causes cosine buries)
    oh = [d for d in onehop_order[:40] if cos_rank[d] > 40][:12]
    for d in oh: buckets[d] = "onehopreach"
    # pure 2-hop reach: chain surfaces, BOTH cosine and 1-hop bury it
    cr = [d for d in chain_order[:40] if cos_rank[d] > 40 and onehop_rank[d] > 40][:12]
    for d in cr: buckets[d] = "chainreach"   # overwrites if also onehopreach (kept as chain only if 1-hop>40)

    cosdeep = [d for d, b in buckets.items() if b == "cosdeep"][:10]
    keep_docs = [d for d, b in buckets.items() if b in ("costop", "onehopreach", "chainreach")] + cosdeep
    kmap, docs = {}, []
    for i, d in enumerate(sorted(set(keep_docs), key=lambda x: hashlib.md5(f"{qid}|{x}".encode()).hexdigest())):
        s = snip(d, qv)
        if s is None: continue
        lid = f"D{i+1:02d}"
        docs.append({"lid": lid, "title": dmeta.get(d, {}).get("title", "?"), "snippet": s})
        kmap[lid] = {"doc_id": d, "bucket": buckets[d], "cos_rank": cos_rank[d],
                     "onehop_rank": onehop_rank[d], "chain_rank": chain_rank[d]}
    json.dump({"effect": eff, "docs": docs}, open(DEVAL / f"packet_ms_{qid}.json", "w"), indent=1)
    key[qid] = kmap

json.dump(key, open(DEVAL / "key_ms.json", "w"), indent=1)
nb = {}
for kmap in key.values():
    for e in kmap.values(): nb[e["bucket"]] = nb.get(e["bucket"], 0) + 1
print(f"wrote 12 multistep packets; bucket totals: {nb}")
