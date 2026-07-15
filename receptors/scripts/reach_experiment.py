# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""THE hypothesis-generation-at-depth test. For each effect query, bucket docs:
  costop  = cosine top-8            (control: what plain RAG already shows — should be relevant)
  cosdeep = cosine rank 11..40      (control: does just reading DEEPER in cosine find drivers?)
  reach   = union-chain top-40 docs that cosine ranks > 40
            (the method's genuine cross-vocab reach — the hypothesis candidates cosine can't get)
Emit a BLIND judge packet (anonymised) + a key tagging each doc's bucket + ranks.
Then judges grade relevance 0/1/2; scoring compares reach-precision vs the controls.
Requires the UNION graph built (GRAPH_GAMMA=0 GRAPH_K=25).
"""
import sys, json, os
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np, hashlib
import receptors_tools as rt

D = Path("data"); DEVAL = Path("data/eval")
queries = json.load(open(DEVAL / "queries.json"))
g = rt._load_graph()
doc_emb = g["doc_emb"].astype(np.float32); gids = list(g["doc_ids"]); gepoch = g["epoch"]
dmeta = json.loads((D / "doc_meta.json").read_text())
ch = np.load(D / "chunks.npy").astype(np.float32); ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
by_doc = {}
for ci, mm in enumerate(meta):
    by_doc.setdefault(mm["doc_id"], []).append(ci)

def snippet(did, qv):
    cis = by_doc.get(did)
    if not cis: return None
    best = max(cis, key=lambda ci: float(ch[ci] @ qv))
    return texts[best][:600]

packets, key = {}, {}
for q in queries:
    qid, eff = q["id"], q["effect"]
    qv = rt._embed([eff])[0]
    cos = doc_emb @ qv
    cos_order = [gids[i] for i in np.argsort(-cos)]
    cos_rank = {did: r for r, did in enumerate(cos_order)}
    # union reach ranking (deployed reach config, mode=union seed_top=10)
    union = rt.causal_chain(eff, k=40, mode="union", seed_top=10)
    union_rank = {r["doc_id"]: i for i, r in enumerate(union)}

    buckets = {}
    for did in cos_order[:8]:
        buckets[did] = "costop"
    for did in cos_order[10:40]:
        buckets.setdefault(did, "cosdeep")
    for r in union:
        did = r["doc_id"]
        if cos_rank.get(did, 10**9) > 40:
            buckets[did] = "reach"          # method surfaces, cosine buries — the test set
    # keep what an analyst would actually see: costop (8), the TOP reach docs by
    # union rank (12), and a cosdeep control (10). Cap so the packet stays judgeable.
    reach = sorted([d for d, b in buckets.items() if b == "reach"],
                   key=lambda d: union_rank.get(d, 10**9))[:12]
    costop = [d for d, b in buckets.items() if b == "costop"]
    cosdeep = [d for d, b in buckets.items() if b == "cosdeep"][:10]
    keep = costop + reach + cosdeep

    docs, kmap = [], {}
    items = sorted(keep, key=lambda d: hashlib.md5(f"{qid}|{d}".encode()).hexdigest())
    for i, did in enumerate(items):
        sn = snippet(did, qv)
        if sn is None: continue
        lid = f"D{i+1:02d}"
        docs.append({"lid": lid, "title": dmeta.get(did, {}).get("title", "?"), "snippet": sn})
        kmap[lid] = {"doc_id": did, "bucket": buckets[did],
                     "cos_rank": cos_rank.get(did), "union_rank": union_rank.get(did)}
    packets[qid] = {"effect": eff, "docs": docs}
    key[qid] = kmap
    json.dump(packets[qid], open(DEVAL / f"packet_reach_{qid}.json", "w"), indent=1)

json.dump(key, open(DEVAL / "key_reach.json", "w"), indent=1)
nb = {}
for kmap in key.values():
    for e in kmap.values(): nb[e["bucket"]] = nb.get(e["bucket"], 0) + 1
print(f"wrote {len(packets)} reach packets; bucket totals: {nb}")
