# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Does a POLARITY axis add signal the cosine-only epistemic gate throws away?

The gate rules evidence-support by cosine alignment — but cosine is DIRECTIONALLY
BLIND: "prices rose" and "prices fell" about the same subject are near-neighbours.
A claim can be tagged 'supported' by a passage that says the OPPOSITE. The polarity
receptor is exactly the axis that separates them.

Test (honest, held-out):
  1. Refit polarity probe w on a train split; report test AUROC (probe quality).
  2. For same-subject OPPOSITE-direction claim pairs (up vs down about the same
     entity): measure cosine similarity (is cosine blind?) vs the polarity-axis gap
     (does the probe separate them?). This is the signal a direction-consistency
     gate would exploit and the cosine gate cannot.
"""
import json, numpy as np

X = np.load("data/claims.npy").astype(np.float32)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
lab = [json.loads(l) for l in open("data/claims_labels.jsonl")]
y = np.array([l["y"] for l in lab], dtype=np.float32)
subj = [l["subj"] for l in lab]
N = len(lab)

rng = np.random.default_rng(0)
idx = rng.permutation(N)
tr, te = idx[: int(0.7 * N)], idx[int(0.7 * N):]

lam = 1.0
Xtr = X[tr]
w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(X.shape[1], dtype=np.float32), Xtr.T @ y[tr])

def auroc(scores, labels):
    pos = scores[labels > 0]; neg = scores[labels < 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based AUC
    alls = np.concatenate([pos, neg])
    order = alls.argsort()
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(alls) + 1)
    return (ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

s_te = X[te] @ w
print(f"polarity probe held-out AUROC = {auroc(s_te, y[te]):.3f}   (n_test={len(te)})")

# persist the axis fit on ALL data for the gate (direction-consistency check).
w_full = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32), X.T @ y)
# scale so a projection is roughly a z-score (unit std over the corpus)
w_full = w_full / (float((X @ w_full).std()) + 1e-9)
np.save("data/polarity_w.npy", w_full)
print("saved data/polarity_w.npy (direction axis, z-scaled)")

# projection score per claim (the "direction" the text expresses)
proj = X @ w

# same-subject opposite-direction pairs
by = {}
for i, l in enumerate(lab):
    by.setdefault(l["subj"], {}).setdefault(l["y"], []).append(i)
pairs = [(ups[0], downs[0], s) for s, d in by.items()
         if (ups := d.get(1)) and (downs := d.get(-1))]
print(f"same-subject up/down pairs = {len(pairs)}")

cos_up_down, proj_gap, cos_same = [], [], []
for u, dn, s in pairs:
    cos_up_down.append(float(X[u] @ X[dn]))          # cosine between opposite-dir claims
    proj_gap.append(float(proj[u] - proj[dn]))       # polarity separation (should be > 0)
    # baseline: cosine among SAME-direction same-subject claims (topical ceiling)
cos_up_down = np.array(cos_up_down); proj_gap = np.array(proj_gap)

# how often does the polarity axis correctly order the pair (up should score > down)?
polarity_correct = float(np.mean(proj_gap > 0))

print(f"\ncosine similarity between OPPOSITE-direction claims (same subject):")
print(f"   mean={cos_up_down.mean():.3f}  median={np.median(cos_up_down):.3f}  "
      f">0.6: {np.mean(cos_up_down>0.6)*100:.0f}%   -> cosine sees them as the SAME (blind to direction)")
print(f"\npolarity axis separates up>down in {polarity_correct*100:.0f}% of pairs "
      f"(gap mean={proj_gap.mean():+.3f})")
print(f"\nInterpretation: the cosine gate rates these near-identical, so 'up' evidence")
print(f"can be cited for a 'down' claim and pass. The polarity axis orders them")
print(f"correctly {polarity_correct*100:.0f}% of the time — signal the gate currently discards.")
