# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","torch","scikit-learn","matplotlib"]
# ///
"""
news_market_clip.py

Question: does a LEARNED joint news<->market space recover impact/spillover
better than the ridge map that failed temporally?

- Docs with >=1 price move -> target = 29-dim signed-zscore basket (0 where no move).
- Text head 384->64, market head 29->64 (small MLPs). InfoNCE aligns each doc's
  text embedding with ITS move-basket using in-batch negatives.
- TEMPORAL split by earliest move_date per doc (train earliest 70%, test latest 30%).

Baselines:
  (i)  ridge doc_emb(384)->basket(29), score candidate baskets by cosine.
  (ii) naive cosine(doc_emb, symbol_name_emb) for per-symbol ranking.

Eval on held-out test docs:
  (1) doc->basket retrieval among all test baskets: MRR + Recall@1 (CLIP vs ridge).
  (2) per-symbol recall@3 of actually-moved symbols (CLIP text->market vs naive).
  (3) impact magnitude: does ||pred basket|| rank docs by realized max|z| TEMPORALLY (Spearman)?

CPU-only, deterministic, small.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge


def spearmanr(a, b):
    """Spearman rho + two-sided p (normal approx), numpy-only."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    n = len(a)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    rho = float((ra * rb).sum() / denom) if denom > 0 else 0.0
    if n > 3 and abs(rho) < 1.0:
        t = rho * np.sqrt((n - 2) / (1 - rho * rho))
        # normal approx to two-sided t p-value
        from math import erf, sqrt
        p = 2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2))))
    else:
        p = 0.0
    return rho, float(p)

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
DEVICE = "cpu"

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MET = DATA / "metrics"
FIG = DATA / "figures"
MET.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)


def load_jsonl(p):
    return [json.loads(l) for l in open(p)]


# ---------------------------------------------------------------- load
docs = load_jsonl(DATA / "docs.jsonl")
doc_emb = np.load(DATA / "embeddings.npy").astype(np.float32)  # (3433,384) aligned to docs
symbols = load_jsonl(DATA / "symbols.jsonl")
sym_emb = np.load(DATA / "symbols.npy").astype(np.float32)      # (29,384)
pairs = load_jsonl(DATA / "price_pairs.jsonl")

assert doc_emb.shape[0] == len(docs)
sym_order = [s["symbol"] for s in symbols]
sym_idx = {s: i for i, s in enumerate(sym_order)}
NSYM = len(sym_order)
doc_row = {d["doc_id"]: i for i, d in enumerate(docs)}

# ---------------------------------------------------------------- build baskets
# per doc: 29-dim signed zscore basket + earliest move_date
from collections import defaultdict

basket_z = defaultdict(lambda: np.zeros(NSYM, dtype=np.float32))
earliest = {}
moved_syms = defaultdict(set)
for p in pairs:
    did = p["doc_id"]
    if did not in doc_row:
        continue
    s = p["symbol"]
    if s not in sym_idx:
        continue
    j = sym_idx[s]
    # keep the largest-magnitude signed zscore if multiple to same symbol
    z = float(p["zscore"])
    if abs(z) > abs(basket_z[did][j]):
        basket_z[did][j] = z
    moved_syms[did].add(j)
    md = p["move_date"]
    if did not in earliest or md < earliest[did]:
        earliest[did] = md

doc_ids = [d for d in earliest if moved_syms[d]]  # docs with >=1 move
# TEMPORAL split by earliest move_date, train earliest 70%
doc_ids.sort(key=lambda d: (earliest[d], d))
n = len(doc_ids)
cut = int(round(n * 0.70))
train_ids = doc_ids[:cut]
test_ids = doc_ids[cut:]

# ensure the temporal boundary is a real date boundary (no leakage of same-day docs
# straddling; here we simply report the boundary dates)
train_last = earliest[train_ids[-1]]
test_first = earliest[test_ids[0]]


def stack(ids):
    X = np.stack([doc_emb[doc_row[d]] for d in ids]).astype(np.float32)
    B = np.stack([basket_z[d] for d in ids]).astype(np.float32)
    return X, B


Xtr, Btr = stack(train_ids)
Xte, Bte = stack(test_ids)


def l2n(a, axis=-1, eps=1e-8):
    return a / (np.linalg.norm(a, axis=axis, keepdims=True) + eps)


# ---------------------------------------------------------------- CLIP model
class Head(nn.Module):
    def __init__(self, din, dh=128, dout=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, dh), nn.GELU(), nn.Linear(dh, dout))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


torch.manual_seed(SEED)
text_head = Head(384).to(DEVICE)
mkt_head = Head(NSYM).to(DEVICE)
logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32))
# learned scalar magnitude head on text embedding (for eval 3)
mag_head = nn.Sequential(nn.Linear(384, 64), nn.GELU(), nn.Linear(64, 1)).to(DEVICE)

params = list(text_head.parameters()) + list(mkt_head.parameters()) + \
    list(mag_head.parameters()) + [logit_scale]
opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)

Xtr_t = torch.tensor(Xtr, device=DEVICE)
Btr_t = torch.tensor(Btr, device=DEVICE)
tr_max_z = torch.tensor(np.abs(Btr).max(axis=1), device=DEVICE)  # realized impact target

g = torch.Generator().manual_seed(SEED)
BS = 128
EPOCHS = 200
ntr = Xtr.shape[0]
for ep in range(EPOCHS):
    perm = torch.randperm(ntr, generator=g)
    for i in range(0, ntr, BS):
        idx = perm[i:i + BS]
        if idx.numel() < 4:
            continue
        t = text_head(Xtr_t[idx])
        m = mkt_head(Btr_t[idx])
        scale = logit_scale.exp().clamp(max=100.0)
        logits = scale * t @ m.t()
        labels = torch.arange(idx.numel(), device=DEVICE)
        loss = 0.5 * (F.cross_entropy(logits, labels) +
                      F.cross_entropy(logits.t(), labels))
        # magnitude regression (log-space, monotone-friendly)
        pred_mag = mag_head(Xtr_t[idx]).squeeze(-1)
        loss = loss + 0.3 * F.mse_loss(pred_mag, tr_max_z[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

text_head.eval(); mkt_head.eval(); mag_head.eval()
with torch.no_grad():
    Tte = text_head(torch.tensor(Xte, device=DEVICE)).cpu().numpy()   # (nte,64)
    Mte = mkt_head(torch.tensor(Bte, device=DEVICE)).cpu().numpy()    # (nte,64) candidate baskets
    pred_mag = mag_head(torch.tensor(Xte, device=DEVICE)).squeeze(-1).cpu().numpy()
    # symbol prototypes in market space: one-hot signed baskets -> market head
    eye = np.eye(NSYM, dtype=np.float32)
    Msym = mkt_head(torch.tensor(eye, device=DEVICE)).cpu().numpy()   # (29,64)

# ---------------------------------------------------------------- ridge baseline
ridge = Ridge(alpha=10.0)
ridge.fit(Xtr, Btr)
Rpred = ridge.predict(Xte).astype(np.float32)  # (nte,29) predicted basket

# ---------------------------------------------------------------- Eval 1: retrieval
# candidate set = all test baskets. rank the true one for each test doc.
nte = Xte.shape[0]


def retrieval_metrics(query, cand):
    # query (nte,d), cand (nte,d) ; true index == row index
    q = l2n(query); c = l2n(cand)
    sim = q @ c.T  # (nte,nte)
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(nte, dtype=np.int64)
    for i in range(nte):
        ranks[i] = int(np.where(order[i] == i)[0][0])
    mrr = float(np.mean(1.0 / (ranks + 1)))
    r1 = float(np.mean(ranks == 0))
    return mrr, r1


clip_mrr, clip_r1 = retrieval_metrics(Tte, Mte)
# ridge: text-side query is ridge's predicted basket; candidates are true test baskets
ridge_mrr, ridge_r1 = retrieval_metrics(Rpred, Bte)

# ---------------------------------------------------------------- Eval 2: per-symbol recall@3
def per_symbol_recall_at_k(sym_scores, k=3):
    # sym_scores (nte,29): higher = more likely moved. recall@k of actually-moved syms
    order = np.argsort(-sym_scores, axis=1)[:, :k]
    recs = []
    for i, did in enumerate(test_ids):
        truth = moved_syms[did]
        if not truth:
            continue
        topk = set(order[i].tolist())
        recs.append(len(topk & truth) / len(truth))
    return float(np.mean(recs))


clip_sym_scores = Tte @ Msym.T             # (nte,29)
naive_sym_scores = l2n(Xte) @ l2n(sym_emb).T
ridge_sym_scores = np.abs(Rpred)           # ridge magnitude ranking (bonus context)

clip_r3 = per_symbol_recall_at_k(clip_sym_scores, 3)
naive_r3 = per_symbol_recall_at_k(naive_sym_scores, 3)
ridge_r3 = per_symbol_recall_at_k(ridge_sym_scores, 3)

# ---------------------------------------------------------------- Eval 3: impact magnitude (temporal)
realized_maxz = np.abs(Bte).max(axis=1)
clip_basket_norm = np.linalg.norm(Rpred, axis=1)  # ridge basket norm (reference)
clip_pred_norm = np.linalg.norm(mkt_head_out := Mte, axis=1)  # ~1 (normed) - not useful, keep learned head

rho_learned, p_learned = spearmanr(pred_mag, realized_maxz)
rho_ridgenorm, p_ridgenorm = spearmanr(clip_basket_norm, realized_maxz)
rho_learned = float(rho_learned); rho_ridgenorm = float(rho_ridgenorm)

# ---------------------------------------------------------------- verdict
# SIGNAL if CLIP clearly beats ridge on retrieval AND spillover beats prior 0.12
# baseline meaningfully; NON-STATIONARY if retrieval/spillover ok but impact
# magnitude does NOT generalize temporally; WEAK if marginal; NO-SIGNAL otherwise.
retr_win = clip_mrr > ridge_mrr + 0.02
spill_beats_prior = clip_r3 > 0.12 + 0.05
spill_beats_naive = clip_r3 > naive_r3
impact_generalizes = (rho_learned > 0.15) and (p_learned < 0.05)
# a SIGNIFICANT temporal sign-flip (or vanish) = the impact signal is non-stationary
impact_nonstationary = (p_learned < 0.05) and (rho_learned <= 0.0)

if retr_win and spill_beats_naive and impact_generalizes:
    verdict = "SIGNAL"
elif impact_nonstationary:
    # the joint space did NOT rescue impact: it inverts out-of-sample
    verdict = "NON-STATIONARY"
elif (retr_win and spill_beats_prior) or spill_beats_naive:
    verdict = "WEAK"
else:
    verdict = "NO-SIGNAL"

metrics = {
    "n_docs_with_move": int(n),
    "n_train": int(len(train_ids)),
    "n_test": int(len(test_ids)),
    "temporal_boundary": {"train_last_date": train_last, "test_first_date": test_first},
    "n_symbols": int(NSYM),
    "retrieval": {
        "clip_mrr": round(clip_mrr, 4), "clip_recall@1": round(clip_r1, 4),
        "ridge_mrr": round(ridge_mrr, 4), "ridge_recall@1": round(ridge_r1, 4),
    },
    "per_symbol_recall@3": {
        "clip": round(clip_r3, 4), "naive_cosine": round(naive_r3, 4),
        "ridge_absnorm": round(ridge_r3, 4),
        "prior_spillover_baseline": 0.12, "prior_target": 0.60,
    },
    "impact_magnitude_temporal_spearman": {
        "learned_mag_head": {"rho": round(rho_learned, 4), "p": float(f"{p_learned:.3g}")},
        "ridge_basket_norm": {"rho": round(rho_ridgenorm, 4), "p": float(f"{p_ridgenorm:.3g}")},
    },
    "verdict": verdict,
    "seed": SEED,
}
with open(MET / "news_market_clip.json", "w") as f:
    json.dump(metrics, f, indent=2)

# ---------------------------------------------------------------- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# retrieval
x = np.arange(2); w = 0.35
ax[0].bar(x - w / 2, [clip_mrr, clip_r1], w, label="CLIP", color="#2b6cb0")
ax[0].bar(x + w / 2, [ridge_mrr, ridge_r1], w, label="ridge", color="#a0aec0")
ax[0].set_xticks(x); ax[0].set_xticklabels(["MRR", "Recall@1"])
ax[0].set_title("doc->basket retrieval (test baskets)")
ax[0].set_ylim(0, 1); ax[0].legend(); ax[0].grid(axis="y", alpha=0.3)
# per-symbol recall@3
labels = ["CLIP", "naive cos", "ridge |z|"]
vals = [clip_r3, naive_r3, ridge_r3]
ax[1].bar(labels, vals, color=["#2b6cb0", "#a0aec0", "#cbd5e0"])
ax[1].axhline(0.12, ls="--", c="red", label="prior 0.12")
ax[1].axhline(0.60, ls="--", c="green", label="target 0.60")
ax[1].set_ylim(0, 1); ax[1].set_title("per-symbol recall@3 (moved syms)")
ax[1].legend(); ax[1].grid(axis="y", alpha=0.3)
fig.suptitle(f"news<->market CLIP  |  impact Spearman(learned)={rho_learned:.2f}  VERDICT={verdict}")
fig.tight_layout()
fig.savefig(FIG / "news_market_clip.png", dpi=110)

# ---------------------------------------------------------------- summary (<=4 lines)
print(f"Retrieval: CLIP MRR={clip_mrr:.3f} R@1={clip_r1:.3f} vs ridge MRR={ridge_mrr:.3f} R@1={ridge_r1:.3f} (n_test={nte})")
print(f"Spillover recall@3: CLIP={clip_r3:.3f} naive={naive_r3:.3f} (prior 0.12 / target 0.60)")
print(f"Impact magnitude temporal Spearman: learned={rho_learned:.3f} (p={p_learned:.3g}), ridge-norm={rho_ridgenorm:.3f}")
print(f"VERDICT: {verdict}")
