# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""Nonlinear direction head as a reusable RERANKER — the one measured retrieval-level
win (Exp E / SIGNALS.md: temporal AUC 0.696->0.782).

Extracted from stack_multi.py so the discovery-eval arms can layer the SAME validated
signal on their candidate pools. Two lessons from the prior experiments are baked in:

  1. The signal is relational ASYMMETRY: the difference vector e_a-e_b carries it,
     the Hadamard product is noise (0.500). So features = [e_a, e_b, e_a-e_b] by
     default (product droppable via RECEPTORS_DIR_PRODUCT=1 to reproduce the old set).
  2. It only helps FUSED with cosine (standalone AUC ~= chance; it works by correcting
     cosine's errors — orthogonality). So `fuse()` returns z(cos)+w*z(ndir), never
     ndir alone.

Training supervision = mined (cause, effect) pairs from docs.jsonl cause_entities,
using rare (<=3% doc-freq) causes and strict time-order (cause precedes effect) —
identical to stack_multi.mine(). The head NEVER sees the judged eval labels -> no leak.
"""
import os, math
from pathlib import Path
from collections import defaultdict
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
D = _ROOT / os.environ.get("RECEPTORS_DATA_DIR", "data")
_USE_PRODUCT = os.environ.get("RECEPTORS_DIR_PRODUCT", "0") == "1"
RNG = np.random.default_rng(0)


class InsufficientPairs(RuntimeError):
    """Raised when too few causal pairs exist to train the direction head."""


def _feats(ea, eb):
    parts = [ea, eb, ea - eb]
    if _USE_PRODUCT:
        parts.append(ea * eb)
    return np.concatenate(parts)


def _mine(docs, emb):
    df = defaultdict(int)
    for d in docs:
        for e in set(d.get("entities", [])):
            df[e] += 1
    cap = math.ceil(0.03 * len(docs))
    post = defaultdict(list)
    for i, d in enumerate(docs):
        for e in d.get("entities", []):
            post[e].append(i)
    pos = set()
    for b, d in enumerate(docs):
        tb = d.get("published_epoch", 0)
        if tb == 0:
            continue
        for c in d.get("cause_entities", []):
            if 0 < df.get(c, 0) <= cap:
                for a in post.get(c, []):
                    ta = docs[a].get("published_epoch", 0)
                    if a != b and ta > 0 and tb - ta > 0:
                        pos.add((a, b))
    pos = list(pos)
    RNG.shuffle(pos)
    return pos


class DirectionHead:
    """Fitted head + z-score stats for cosine fusion."""

    def __init__(self, head, cos_mu, cos_sd):
        self.head = head
        self.cos_mu, self.cos_sd = cos_mu, cos_sd
        self._nd_mu = self._nd_sd = None

    def ndir(self, doc_emb_rows, effect_vec):
        """P(doc -> effect) for each row of doc_emb_rows (unit-normalised)."""
        X = np.asarray([_feats(e, effect_vec) for e in doc_emb_rows], np.float32)
        return self.head.predict_proba(X)[:, 1]

    def fuse(self, cos, ndir, w=1.0):
        """z(cos) + w*z(ndir): the validated fused reranker score. z-stats for ndir
        are estimated per-call over the candidate pool (relative reranking)."""
        cz = (np.asarray(cos) - self.cos_mu) / (self.cos_sd + 1e-9)
        nd = np.asarray(ndir)
        nz = (nd - nd.mean()) / (nd.std() + 1e-9)
        return cz + w * nz


def load_or_train():
    import json
    from sklearn.ensemble import HistGradientBoostingClassifier
    docs = [json.loads(l) for l in (D / "docs.jsonl").read_text().splitlines() if l.strip()]
    emb = np.load(D / "embeddings.npy").astype(np.float32)
    emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    pairs = _mine(docs, emb)[:9000]
    if len(pairs) < 50:
        raise InsufficientPairs(
            f"only {len(pairs)} causal training pairs mined (<50) — cannot train the "
            f"direction head. Check docs.jsonl entities/cause_entities/published_epoch.")
    Xp = np.asarray([_feats(emb[a], emb[b]) for a, b in pairs], np.float32)
    Xr = np.asarray([_feats(emb[b], emb[a]) for a, b in pairs], np.float32)
    X = np.vstack([Xp, Xr])
    y = np.r_[np.ones(len(Xp)), np.zeros(len(Xr))]
    head = HistGradientBoostingClassifier(max_iter=300, random_state=0).fit(X, y)
    # cosine z-stats over the mined positive pairs (a stable reference scale)
    cos = np.array([float(emb[a] @ emb[b]) for a, b in pairs], np.float32)
    return DirectionHead(head, float(cos.mean()), float(cos.std()))


if __name__ == "__main__":
    dh = load_or_train()
    print("direction head trained; features =",
          "[e_a, e_b, e_a-e_b" + (", e_a*e_b]" if _USE_PRODUCT else "]"))
