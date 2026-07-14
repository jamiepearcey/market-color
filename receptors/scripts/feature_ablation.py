# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Situational-load-bearing test for the CAUSAL TRANSPORT feature.

Question: on a task designed to favour the feature (directed cause->effect
retrieval), does transport add marginal recall OVER plain cosine, and — the sharp
version — is that lift CONCENTRATED where cosine is weakest (low cause<->effect
surface overlap, transport's designed niche)?

Gold is mined from `cause_entities` (A ABOUT entity e, B CAUSED-BY e, t_A<t_B) — it
is NOT cosine-derived, so unlike the Q3 report ablation there is no survivorship
bias toward cosine-findable docs.

Design:
  - doc-level MiniLM embeddings (embeddings.npy aligned to docs.jsonl).
  - mine directed pairs A->B on a SPECIFIC (rarer, IDF-weighted) shared entity.
  - temporal split: fit W on pairs whose effect B is early; TEST on late effects.
  - W = ridge (A^T A + lam I)^-1 A^T B  (cause embedding transported toward effect).
  - per test effect B (query): rank all strictly-earlier docs by
        cosine        = cos(A, B)
        transport     = cos(A·W, B)
        fusion        = z(cosine) + beta * z(transport)
    gold = earlier docs sharing the specific cause entity.
  - metric: recall@10 (macro over queries) AND, stratified, per-gold-pair hit@10
    bucketed by entity-Jaccard(A,B) — LOW bucket = cross-vocabulary (cosine-hard).
"""
import json, numpy as np

D = "data"
docs = [json.loads(l) for l in open(f"{D}/docs.jsonl")]
E = np.load(f"{D}/embeddings.npy").astype(np.float32)
E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
N = len(docs)
epoch = np.array([d["published_epoch"] for d in docs])
ents = [set(d.get("entities") or []) for d in docs]
cents = [set(d.get("cause_entities") or []) for d in docs]

# ---- IDF over cause entities; keep the SPECIFIC half (rarer entities carry signal)
from collections import Counter
df = Counter()
for s in cents:
    for e in s: df[e] += 1
idf = {e: np.log(N / c) for e, c in df.items()}
med = np.median(list(idf.values()))
specific = {e for e, v in idf.items() if v >= med}

# ---- mine directed pairs A(cause, about e) -> B(effect, caused-by e), t_A < t_B
# index docs by the entities they are ABOUT
about = {}
for i, s in enumerate(ents):
    for e in s:
        about.setdefault(e, []).append(i)
pairs = []  # (a, b, shared_entity)
for b in range(N):
    for e in (cents[b] & specific):
        for a in about.get(e, ()):
            if epoch[a] < epoch[b] and a != b:
                pairs.append((a, b, e))
# dedup by (a,b) keeping the most specific entity
best = {}
for a, b, e in pairs:
    if (a, b) not in best or idf[e] > idf[best[(a, b)]]:
        best[(a, b)] = e
pairs = [(a, b, e) for (a, b), e in best.items()]
print(f"docs={N}  mined directed pairs={len(pairs)}  specific-entities={len(specific)}")

# ---- temporal split on effect epoch
cut = np.quantile([epoch[b] for _, b, _ in pairs], 0.6)
train = [(a, b) for a, b, _ in pairs if epoch[b] < cut]
test_pairs = [(a, b, e) for a, b, e in pairs if epoch[b] >= cut]
print(f"train pairs={len(train)}  test pairs={len(test_pairs)}")

# ---- fit transport W on train pairs: A·W ≈ B
A = E[[a for a, _ in train]]
B = E[[b for _, b in train]]
lam = 1.0
W = np.linalg.solve(A.T @ A + lam * np.eye(E.shape[1], dtype=np.float32), A.T @ B)
EW = E @ W
EW /= (np.linalg.norm(EW, axis=1, keepdims=True) + 1e-9)

def z(x):
    return (x - x.mean()) / (x.std() + 1e-9)

# ---- group test gold by effect (query)
gold_by_b = {}
for a, b, e in test_pairs:
    gold_by_b.setdefault(b, {})[a] = e

def jac(i, j):
    u = ents[i] | ents[j]
    return len(ents[i] & ents[j]) / len(u) if u else 0.0

K = 10
cos_recall, fus_recall, tr_recall, adp_recall = [], [], [], []
# per-pair hit stratified by overlap
buckets = {"low (J<0.05)": [[], [], []], "mid (0.05-0.15)": [[], [], []], "high (>=0.15)": [[], [], []]}
BETA = 1.0
for b, golds in gold_by_b.items():
    earlier = np.where(epoch < epoch[b])[0]
    earlier = earlier[earlier != b]
    if len(earlier) < K or not golds:
        continue
    q = E[b]
    cos = E[earlier] @ q
    tr = EW[earlier] @ q
    fus = z(cos) + BETA * z(tr)
    # ADAPTIVE (overlap-gated): transport is trusted only where cosine is UNSURE.
    # Per-candidate gate = how far below the query's cosine ceiling this candidate
    # sits (weak-cosine candidates get the transport signal; strong ones keep pure
    # cosine). Computable at query time — no gold needed.
    gate = np.clip((cos.max() - cos) / (cos.max() - cos.min() + 1e-9), 0, 1)
    adp = z(cos) + BETA * gate * z(tr)
    def topset(scores):
        idx = earlier[np.argsort(-scores)[:K]]
        return set(int(x) for x in idx)
    tc, tt, tf, ta = topset(cos), topset(tr), topset(fus), topset(adp)
    g = set(golds)
    cos_recall.append(len(g & tc) / len(g))
    tr_recall.append(len(g & tt) / len(g))
    fus_recall.append(len(g & tf) / len(g))
    adp_recall.append(len(g & ta) / len(g))
    for a in g:
        J = jac(a, b)
        key = "low (J<0.05)" if J < 0.05 else "mid (0.05-0.15)" if J < 0.15 else "high (>=0.15)"
        buckets[key][0].append(1.0 if a in tc else 0.0)
        buckets[key][1].append(1.0 if a in tt else 0.0)
        buckets[key][2].append(1.0 if a in tf else 0.0)

print(f"\nqueries scored={len(cos_recall)}   (macro recall@{K})")
print(f"  cosine     : {np.mean(cos_recall):.3f}")
print(f"  transport  : {np.mean(tr_recall):.3f}")
print(f"  fusion     : {np.mean(fus_recall):.3f}   (lift over cosine: {np.mean(fus_recall)-np.mean(cos_recall):+.3f})  [naive global]")
print(f"  adaptive   : {np.mean(adp_recall):.3f}   (lift over cosine: {np.mean(adp_recall)-np.mean(cos_recall):+.3f})  [overlap-gated]")

print(f"\nSTRATIFIED per-gold-pair hit@{K} by cause<->effect entity overlap:")
print(f"  {'bucket':<18}{'n':>6}{'cosine':>9}{'transport':>11}{'fusion':>9}{'fus-cos':>9}")
for key, (c, t, f) in buckets.items():
    if not c:
        continue
    print(f"  {key:<18}{len(c):>6}{np.mean(c):>9.3f}{np.mean(t):>11.3f}{np.mean(f):>9.3f}{np.mean(f)-np.mean(c):>+9.3f}")
