# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Emit top-K candidate docs per eval query for ONE retrieval arm, using whatever
semantic_graph.npz is currently built. Arms:
  cosine     : plain topical retrieval (doc_emb . q)      — no graph
  chain      : causal_chain(mode, seed_top)               — uses current graph
Writes data/eval/cand_<arm>.json = {qid: [{doc_id,title,snippet,rank}, ...]}.
Run AFTER building the graph you want for this arm.
"""
import sys, json, argparse, os
sys.path.insert(0, "scripts")
from pathlib import Path
import numpy as np
import receptors_tools as rt

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)               # label for the output file
ap.add_argument("--method", choices=["cosine", "chain"], required=True)
ap.add_argument("--mode", default="union")            # chain fusion mode
ap.add_argument("--seed-top", type=int, default=10)
ap.add_argument("--k", type=int, default=10)
a = ap.parse_args()

D = Path(os.environ.get("RECEPTORS_DATA_DIR", "data"))
# eval queries + outputs always live under the canonical data/eval (symlinked dirs share it)
DEVAL = Path("data/eval")
queries = json.load(open(DEVAL / "queries.json"))
out = {}

if a.method == "cosine":
    g = rt._load_graph()
    doc_emb = g["doc_emb"].astype(np.float32); gids = list(g["doc_ids"])
    dmeta = json.loads((D / "doc_meta.json").read_text())
    ch = np.load(D / "chunks.npy").astype(np.float32)
    ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
    texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
    by_doc = {}
    for ci, mm in enumerate(meta):
        by_doc.setdefault(mm["doc_id"], []).append(ci)
    for q in queries:
        qv = rt._embed([q["effect"]])[0]
        cos = doc_emb @ qv
        order = np.argsort(-cos)
        res = []
        for ni in order:
            did = gids[ni]
            if did not in by_doc:
                continue
            best = max(by_doc[did], key=lambda ci: float(ch[ci] @ qv))
            dm = dmeta.get(did, {})
            res.append({"doc_id": did, "rank": len(res) + 1,
                        "title": dm.get("title", "?"), "snippet": texts[best][:600]})
            if len(res) >= a.k:
                break
        out[q["id"]] = res
else:
    for q in queries:
        res = rt.causal_chain(q["effect"], k=a.k, mode=a.mode, seed_top=a.seed_top)
        out[q["id"]] = [{"doc_id": r["doc_id"], "rank": r["rank"],
                         "title": r["title"], "snippet": r["snippet"][:600]} for r in res]

DEVAL.mkdir(exist_ok=True)
json.dump(out, open(DEVAL / f"cand_{a.arm}.json", "w"), indent=1)
print(f"wrote data/eval/cand_{a.arm}.json  ({len(out)} queries, k={a.k})")
