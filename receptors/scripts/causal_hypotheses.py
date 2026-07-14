# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Causal-graph hypothesis + confound generation (Stage 2 / confound analysis).

Given an effect question, don't just retrieve topically — traverse the causal graph
to surface the UPSTREAM DRIVERS that could explain it, and separate the primary
thesis driver from RIVAL drivers (confounds). Uses the free cause_entities graph
(a doc is ABOUT entities, CAUSED-BY cause_entities), not chunk cosine — so it can
name a driver whose vocabulary doesn't overlap the effect.

Method:
  1. embed the question; take the top effect docs by cosine.
  2. pool their cause_entities, weighted by doc relevance x IDF -> candidate drivers.
  3. cluster drivers into the PRIMARY (highest mass) and RIVALS (other distinct
     high-mass drivers = confounds to rule out).
  4. for each driver, list upstream evidence docs (docs ABOUT that entity, earlier
     in time than the effect) — the leads to dig.

Usage: uv run scripts/causal_hypotheses.py "Russia fuel crisis spillover to Central Asia"
"""
import sys, json, numpy as np
from collections import defaultdict

docs = [json.loads(l) for l in open("data/docs.jsonl")]
E = np.load("data/embeddings.npy").astype(np.float32)
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
epoch = np.array([d["published_epoch"] for d in docs])
ents = [set(d.get("entities") or []) for d in docs]
cents = [set(d.get("cause_entities") or []) for d in docs]
N = len(docs)

from collections import Counter
df = Counter()
for s in cents:
    for e in s: df[e] += 1
idf = {e: np.log(N / c) for e, c in df.items()}

q = sys.argv[1] if len(sys.argv) > 1 else "Russia fuel crisis spillover to Central Asia"
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
qv = np.array(list(model.embed([q]))[0], dtype=np.float32); qv /= (np.linalg.norm(qv) + 1e-9)
rel = E @ qv
top = np.argsort(-rel)[:25]                     # effect docs
eff_epoch = np.median(epoch[top])

# pool cause_entities weighted by relevance x IDF
mass = defaultdict(float)
for i in top:
    r = max(float(rel[i]), 0.0)
    for e in cents[i]:
        mass[e] += r * idf.get(e, 0.0)
drivers = sorted(mass.items(), key=lambda kv: -kv[1])
# drop entities too generic (low idf) or trivially the effect itself
drivers = [(e, m) for e, m in drivers if idf.get(e, 0) > 0.5][:10]

about = defaultdict(list)
for i, s in enumerate(ents):
    for e in s:
        about[e].append(i)

def upstream(e, k=3):
    cand = [i for i in about.get(e, []) if epoch[i] < eff_epoch]
    cand.sort(key=lambda i: -float(rel[i]))
    return [docs[i]["doc_id"][:8] for i in cand[:k]]

print(f"QUESTION: {q}\neffect docs (top cosine): {len(top)}\n")
print("CANDIDATE CAUSAL DRIVERS (upstream, from cause_entities graph):")
tot = sum(m for _, m in drivers) or 1.0
for rank, (e, m) in enumerate(drivers, 1):
    tag = "PRIMARY" if rank == 1 else "rival/confound"
    up = upstream(e)
    print(f"  {rank:>2}. [{tag:<14}] {e:<22} mass={m/tot:.2f}  upstream_docs={up}")

print("\nCONFOUND CHECK — distinct high-mass drivers competing to explain the effect:")
prim = drivers[0][0] if drivers else None
rivals = [e for e, m in drivers[1:] if m >= 0.4 * (drivers[0][1] if drivers else 1)]
print(f"  primary thesis driver : {prim}")
print(f"  rival drivers to rule out (confounds): {rivals if rivals else '(none above 40% of primary mass)'}")
