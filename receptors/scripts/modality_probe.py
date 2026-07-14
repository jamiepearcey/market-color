# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""Modality (evidence-type) probe for the epistemic gate.

"Corroborated by 2 forecasts" is NOT the same as "corroborated by 2 event reports."
The modality receptor types a passage's predicate (price_move / event / policy_action /
... vs forecast / statement). A FACTUAL claim supported only by SOFT-modality passages
(forecast/opinion) should not earn the same status as one backed by hard events.

Refit one-vs-rest ridge W (d x k), report held-out accuracy, persist W + classes.
"""
import json, numpy as np

X = np.load("data/modality.npy").astype(np.float32)
X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
lab = [json.loads(l) for l in open("data/modality_labels.jsonl")]
classes = json.load(open("data/modality_classes.json"))
y = np.array([l["y"] for l in lab])
K = len(classes)
N = len(y)

# SOFT = projection/opinion; HARD = something that happened / a measured move.
SOFT = {"forecast", "statement"}
soft_idx = {i for i, c in enumerate(classes) if c in SOFT}

rng = np.random.default_rng(0)
idx = rng.permutation(N)
tr, te = idx[: int(0.7 * N)], idx[int(0.7 * N):]
Y = np.full((len(tr), K), -1.0, dtype=np.float32)
Y[np.arange(len(tr)), y[tr]] = 1.0
lam = 1.0
Xtr = X[tr]
W = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(X.shape[1], dtype=np.float32), Xtr.T @ Y)

pred = (X[te] @ W).argmax(1)
acc = float((pred == y[te]).mean())
# hard-vs-soft binary accuracy (the distinction the gate actually uses)
soft_true = np.array([1 if t in soft_idx else 0 for t in y[te]])
soft_pred = np.array([1 if p in soft_idx else 0 for p in pred])
bin_acc = float((soft_true == soft_pred).mean())

print(f"modality probe: {K}-way held-out accuracy = {acc:.3f}  (chance={1/K:.3f}, majority={max(np.bincount(y))/N:.3f})")
print(f"  HARD-vs-SOFT (forecast/statement) binary accuracy = {bin_acc:.3f}")
print(f"  soft classes = {sorted(SOFT)}  ({len(soft_idx)} of {K})")

# persist W fit on ALL data
Yf = np.full((N, K), -1.0, dtype=np.float32); Yf[np.arange(N), y] = 1.0
Wf = np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1], dtype=np.float32), X.T @ Yf)
np.save("data/modality_w.npy", Wf)
print("saved data/modality_w.npy")
