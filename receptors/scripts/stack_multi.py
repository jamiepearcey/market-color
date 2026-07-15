# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "fastembed>=0.3"]
# ///
"""Multi-signal LEARNED stack for causal-driver ranking, LEAVE-ONE-QUERY-OUT CV.

Uses the 828 judged (query,doc) driver-relevance labels accumulated across all prior
experiments as ground truth. Per (query,doc) builds features from the VALIDATED signals:
  cos        cosine(doc, effect)                             (topical baseline)
  ndir       nonlinear direction head P(doc -> effect)       (Exp E, AUC 0.782)
  tmax,tmean typed mechanism tensor: doc transported under each per-predicate W_r,
             aligned to effect; max & mean over predicates   (A1: 0.742 typed > 0.649)
  conf       doc self-reported confidence                    (weak prior)
Then LEAVE-ONE-QUERY-OUT: train a stacker on 11 queries' labels, rank the held-out
query's judged candidates, score mean-grade@10 + nDCG@10 vs ranking by cosine alone.
The operators are trained on the global pair corpus (never see judged labels) -> no leak.
Bootstrap CI over the 12 queries; linear-stack coefficients = which signal carries weight.
"""
import json, math, os
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"; DEVAL = DATA / "eval"
RNG = np.random.default_rng(0)
def norml(m): return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

# ---- judged pool ---------------------------------------------------------------
km = {"": "key.json", "sub": "key_sub.json", "reach": "key_reach.json", "ms": "key_ms.json", "unc": "key_unc.json"}
G = defaultdict(list)
for suf, kf in km.items():
    if not (DEVAL / kf).exists(): continue
    key = json.load(open(DEVAL / kf))
    for qid, kmap in key.items():
        rf = DEVAL / (f"rating_{qid}.json" if suf == "" else f"rating_{suf}_{qid}.json")
        if not rf.exists(): continue
        rt = json.load(open(rf))
        for lid, e in kmap.items():
            if lid in rt: G[(qid, e["doc_id"])].append(rt[lid])
GR = {k: float(np.mean(v)) for k, v in G.items()}

# ---- embeddings + docs ---------------------------------------------------------
emb = norml(np.load(DATA / "embeddings.npy").astype(np.float32))
docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
id2idx = {d["doc_id"]: i for i, d in enumerate(docs)}
df = defaultdict(int)
for d in docs:
    for e in set(d["entities"]): df[e] += 1
cap = math.ceil(0.03 * len(docs)); post = defaultdict(list)
for i, d in enumerate(docs):
    for e in d["entities"]: post[e].append(i)

def mine():
    pos = []
    for b, d in enumerate(docs):
        tb = d["published_epoch"]
        if tb == 0: continue
        for c in d.get("cause_entities", []):
            if 0 < df.get(c, 0) <= cap:
                for a in post.get(c, []):
                    if a != b and docs[a]["published_epoch"] > 0 and tb - docs[a]["published_epoch"] > 0:
                        pos.append((a, b))
    pos = list({p for p in pos}); RNG.shuffle(pos); return pos
pairs = mine()

# ---- nonlinear direction head (Exp E) ------------------------------------------
def feats_dir(ea, eb): return np.concatenate([ea, eb, ea - eb, ea * eb])
Xp = np.asarray([feats_dir(emb[a], emb[b]) for a, b in pairs[:9000]], np.float32)
Xr = np.asarray([feats_dir(emb[b], emb[a]) for a, b in pairs[:9000]], np.float32)
Xd = np.vstack([Xp, Xr]); yd = np.r_[np.ones(len(Xp)), np.zeros(len(Xr))]
head = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(Xd, yd)

# ---- mechanism tensor: per-effect-predicate ridge W_r (A1) ---------------------
def ridge_W(A, B, lam=1.0):
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32), A.T @ B).astype(np.float32)
by_pred = defaultdict(list)
for a, b in pairs:
    by_pred[docs[b].get("predicate", "other")].append((a, b))
Wr = {}
for r, ps in by_pred.items():
    if len(ps) < 50: continue
    A = np.asarray([emb[a] for a, b in ps], np.float32); B = np.asarray([emb[b] for a, b in ps], np.float32)
    Wr[r] = ridge_W(A, B)
print(f"mechanism tensor: {len(Wr)} predicate operators; direction head trained")

# ---- query embeddings ----------------------------------------------------------
from fastembed import TextEmbedding
model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
queries = json.load(open(DEVAL / "queries.json"))
qemb = {}
for q in queries:
    v = np.array(list(model.embed([q["effect"]])), np.float32)[0]; qemb[q["id"]] = v / (np.linalg.norm(v) + 1e-9)

# ---- build feature rows for every judged (query,doc) ---------------------------
FEATS = ["cos", "ndir", "tmax", "tmean", "conf"]
rows = []
for (qid, did), g in GR.items():
    if did not in id2idx or qid not in qemb: continue
    e = emb[id2idx[did]]; qv = qemb[qid]
    cos = float(e @ qv)
    ndir = float(head.predict_proba(feats_dir(e, qv)[None])[0, 1])
    ts = []
    for r, W in Wr.items():
        t = e @ W; t = t / (np.linalg.norm(t) + 1e-9); ts.append(float(t @ qv))
    tmax, tmean = (max(ts), float(np.mean(ts))) if ts else (0.0, 0.0)
    conf = float(docs[id2idx[did]].get("confidence", 0.5) or 0.5)
    rows.append((qid, did, g, [cos, ndir, tmax, tmean, conf]))

X = np.asarray([r[3] for r in rows], np.float32)
y = np.asarray([r[2] for r in rows], np.float32)
qids = np.asarray([r[0] for r in rows])
mu, sd = X.mean(0), X.std(0) + 1e-9
Xz = (X - mu) / sd

def dcg(gs, k=10): return sum(g / math.log2(i + 2) for i, g in enumerate(gs[:k]))
def ndcg(order_grades, all_grades, k=10):
    ideal = dcg(sorted(all_grades, reverse=True), k) or 1.0
    return dcg(order_grades, k) / ideal

def eval_ranker(score_fn):
    mg, nd = [], []
    for q in queries:
        m = qids == q["id"]
        if m.sum() == 0: continue
        gg = y[m]; sc = score_fn(Xz[m], q["id"])
        order = np.argsort(-sc)
        og = gg[order]
        mg.append(float(np.mean(og[:10]))); nd.append(ndcg(list(og), list(gg)))
    return np.array(mg), np.array(nd)

# baseline: cosine feature alone
cos_mg, cos_nd = eval_ranker(lambda Xq, qid: Xq[:, 0])

# LOQO learned stacks (Ridge = interpretable linear; feature subsets = ablation)
def loqo_scores(feat_idx):
    out = {}
    for q in queries:
        tr = qids != q["id"]; teq = qids == q["id"]
        if teq.sum() == 0: continue
        reg = Ridge(alpha=1.0).fit(Xz[tr][:, feat_idx], y[tr])
        out[q["id"]] = reg.predict(Xz[teq][:, feat_idx])
    return out
def make_fn(feat_idx):
    sc = loqo_scores(feat_idx)
    return lambda Xq, qid: sc[qid]

subsets = {
    "cosine (baseline)": None,
    "cos+ndir": [0, 1],
    "cos+typed(tmax,tmean)": [0, 2, 3],
    "cos+ndir+typed": [0, 1, 2, 3],
    "cos+ndir+typed+conf (all)": [0, 1, 2, 3, 4],
}
print(f"\n  {'ranker':<28} {'mean-grade@10':>14} {'nDCG@10':>10}   (LOQO, {len(rows)} pairs, 12 q)")
base_mg = cos_mg
for name, idx in subsets.items():
    if idx is None:
        mg, nd = cos_mg, cos_nd
    else:
        mg, nd = eval_ranker(make_fn(idx))
    # bootstrap CI over 12 queries on the delta vs cosine
    d = mg - base_mg
    bs = [np.mean(d[RNG.integers(0, len(d), len(d))]) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    w = int((d > 1e-9).sum()); l = int((d < -1e-9).sum())
    tag = "" if idx is None else f"  Δ{np.mean(d):+.3f} [{lo:+.3f},{hi:+.3f}]  W{w}/L{l}"
    print(f"  {name:<28} {np.mean(mg):>14.3f} {np.mean(nd):>10.3f}{tag}")

# full-data linear coefficients (which signal carries weight)
reg = Ridge(alpha=1.0).fit(Xz, y)
print("\n  linear-stack standardized coefficients (signal weight):")
for f, c in sorted(zip(FEATS, reg.coef_), key=lambda x: -abs(x[1])):
    print(f"    {f:<8} {c:+.3f}")
# AUC of each single feature vs strong-driver label
print("\n  single-feature AUC vs strong-driver(grade==2):")
ybin = (y >= 2).astype(int)
for i, f in enumerate(FEATS):
    try: print(f"    {f:<8} {roc_auc_score(ybin, X[:, i]):.3f}")
    except Exception: pass
