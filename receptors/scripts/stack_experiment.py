# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "fastembed>=0.3"]
# ///
"""STACKING the nonlinear direction signal onto cosine for causal-driver ranking.

Reframe from earlier failures: the operator is a poor first-stage RETRIEVER but may be
a strong RERANKER of cosine's shortlist — promoting genuine causal antecedents over
co-topical non-causes. Uses the parallel chain's finding (direction_deep.py): a
nonlinear HistGB head on [E_a,E_b,E_a-E_b,E_a*E_b] beats the linear operator (+8.6 AUC).

For each effect query: take cosine's top-N candidates, score each cand with the head's
P(cand -> effect), and compare rankings:
  cosine          baseline
  dhead           rerank cosine-N purely by direction score
  stack           z(cos)+z(dhead)
  stack_abstain   use dhead to reorder only where the head is CONFIDENT, else cosine
  deviation       causal-lift: dhead score minus cosine rank (cross-vocab causal picks)
Scored by driver relevance, reusing all prior judgments; uncovered docs are reported.
"""
import json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; DEVAL = DATA / "eval"
RNG = np.random.default_rng(0)
def norml(m): return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

# ---- master grade map from ALL prior judging (qid,doc_id)->mean grade ----------
def master_grades():
    keymap = {"": "key.json", "sub": "key_sub.json", "reach": "key_reach.json",
              "ms": "key_ms.json", "unc": "key_unc.json"}
    g = defaultdict(list)
    for suf, kf in keymap.items():
        kp = DEVAL / kf
        if not kp.exists(): continue
        key = json.load(open(kp))
        for qid, kmap in key.items():
            rf = DEVAL / (f"rating_{qid}.json" if suf == "" else f"rating_{suf}_{qid}.json")
            if not rf.exists(): continue
            rt = json.load(open(rf))
            for lid, e in kmap.items():
                if lid in rt:
                    g[(qid, e["doc_id"])].append(rt[lid])
    return {k: float(np.mean(v)) for k, v in g.items()}

# ---- load embeddings + docs (MiniLM article space, direction_deep's space) -----
emb = norml(np.load(DATA / "embeddings.npy").astype(np.float32))
docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
id2idx = {d["doc_id"]: i for i, d in enumerate(docs)}
dmeta = json.loads((DATA / "doc_meta.json").read_text())
df = defaultdict(int)
for d in docs:
    for e in set(d["entities"]): df[e] += 1
cap = math.ceil(0.03 * len(docs))
post = defaultdict(list)
for i, d in enumerate(docs):
    for e in d["entities"]: post[e].append(i)

def mine_causal():
    pos = []
    for b, d in enumerate(docs):
        tb = d["published_epoch"]
        if tb == 0: continue
        for c in d.get("cause_entities", []):
            if 0 < df.get(c, 0) <= cap:
                for a in post.get(c, []):
                    if a != b and docs[a]["published_epoch"] > 0 and tb - docs[a]["published_epoch"] > 0:
                        pos.append((a, b))
    pos = list({p for p in pos}); RNG.shuffle(pos); return pos[:9000]

def feats(ea, eb): return np.concatenate([ea, eb, ea - eb, ea * eb])
def matrix(pairs): return np.asarray([feats(emb[a], emb[b]) for a, b in pairs], np.float32)

pairs = mine_causal()
ep = np.array([docs[b]["published_epoch"] for _, b in pairs])
cut = np.quantile(ep, 0.7); te = ep >= cut
Xp, Xr = matrix(pairs), matrix([(b, a) for a, b in pairs])
X = np.vstack([Xp, Xr]); y = np.r_[np.ones(len(pairs)), np.zeros(len(pairs))]
mask = np.r_[te, te]
clf = HistGradientBoostingClassifier(max_iter=300, random_state=0)
clf.fit(X[~mask], y[~mask])
auc = roc_auc_score(y[mask], clf.predict_proba(X[mask])[:, 1])
print(f"nonlinear direction head — temporal test AUC {auc:.3f}  (train {(~mask).sum()} / test {mask.sum()})")

# ---- embed queries (same MiniLM space) -----------------------------------------
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
queries = json.load(open(DEVAL / "queries.json"))
def embed(t):
    v = np.array(list(model.embed([t])), np.float32)[0]; return v / (np.linalg.norm(v) + 1e-9)

N = 40; K = 10
GR = master_grades()
def zc(x):
    x = np.asarray(x, float); s = x.std(); return (x - x.mean()) / (s + 1e-9)

rankings = {m: {} for m in ["cosine", "dhead", "stack", "stack_abstain", "deviation"]}
for q in queries:
    qid, qv = q["id"], embed(q["effect"])
    cos = emb @ qv
    cand = np.argsort(-cos)[:N]                         # cosine shortlist
    ex, ey = emb[cand], np.tile(qv, (len(cand), 1))
    F = np.hstack([ex, ey, ex - ey, ex * ey]).astype(np.float32)
    d = clf.predict_proba(F)[:, 1]                      # P(cand -> effect)
    coss = cos[cand]
    conf = np.abs(d - 0.5)
    order = {}
    order["cosine"] = list(range(len(cand)))            # already cosine order
    order["dhead"] = list(np.argsort(-d))
    order["stack"] = list(np.argsort(-(zc(coss) + zc(d))))
    # abstain: reorder by dhead only among confident; keep cosine rank as tiebreak
    key_ab = [(-(d[i] if conf[i] >= 0.15 else 0.0), i) for i in range(len(cand))]
    order["stack_abstain"] = [i for _, i in sorted(key_ab)]
    # deviation / causal-lift: how much dhead rank beats cosine rank
    drank = {i: r for r, i in enumerate(np.argsort(-d))}
    order["deviation"] = sorted(range(len(cand)), key=lambda i: drank[i] - i)  # dhead>>cosine first
    for m, idxs in order.items():
        rankings[m][qid] = [docs[cand[i]]["doc_id"] for i in idxs[:K]]

json.dump(rankings, open(DEVAL / "stack_rankings.json", "w"), indent=1)

# ---- score by reused judgments; report coverage + uncovered --------------------
print(f"\n  {'method':<15} {'cov%':>5} {'mean':>6} {'%rel':>6} {'%strong':>8}")
uncovered = defaultdict(list)
perq = {m: {} for m in rankings}
for m in rankings:
    gr, cov, tot = [], 0, 0
    for qid, dids in rankings[m].items():
        qg = [GR[(qid, did)] for did in dids if (qid, did) in GR]
        perq[m][qid] = np.mean(qg) if qg else None
        for did in dids:
            tot += 1
            if (qid, did) in GR: gr.append(GR[(qid, did)]); cov += 1
            else: uncovered[m].append((qid, did))
    n = len(gr) or 1
    print(f"  {m:<15} {100*cov/tot:>4.0f}% {sum(gr)/n:>6.2f} "
          f"{100*sum(1 for x in gr if x>=1)/n:>5.0f}% {100*sum(1 for x in gr if x>=1.5)/n:>7.0f}%")
allunc = sorted({p for v in uncovered.values() for p in v})
json.dump([list(p) for p in allunc], open(DEVAL / "stack_uncovered.json", "w"), indent=1)
print(f"\n  uncovered (qid,doc) needing judgment: {len(allunc)}")

# ---- paired per-query: each method vs cosine (mean grade of top-K) --------------
print(f"\n  paired vs cosine (per-query mean grade, {len(queries)} queries):")
for m in ["dhead", "stack", "stack_abstain", "deviation"]:
    diffs = [perq[m][q["id"]] - perq["cosine"][q["id"]]
             for q in queries if perq[m][q["id"]] is not None and perq["cosine"][q["id"]] is not None]
    w = sum(1 for d in diffs if d > 1e-9); l = sum(1 for d in diffs if d < -1e-9)
    print(f"    {m:<15} mean Δ {np.mean(diffs):+.3f}   win {w} / tie {len(diffs)-w-l} / loss {l}")
