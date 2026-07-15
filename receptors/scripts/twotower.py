# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "torch", "scikit-learn", "matplotlib"]
# ///
"""
TWO-TOWER cause->effect receptor vs a single linear transport operator W.

Hypothesis: the cause->effect relation is ASYMMETRIC, so two separate learned
towers f_c (cause) and f_e (effect) on FROZEN embeddings should beat one linear
operator W (and raw cosine) at (1) getting pair DIRECTION right and (2) lifting
first-stage RETRIEVAL of the correct later effect doc -- and, crucially, should
survive a TEMPORAL split (train early pairs, test late pairs).

Score(a->b) = <f_c(e_a), f_e(e_b)>.
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SEED = 0

# ------------------------------------------------------------------ data / pairs
def norml(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)

def load():
    emb = norml(np.load(DATA / "embeddings.npy").astype(np.float32))
    docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
    return emb, docs

def mine_pairs(docs, rng):
    """Same miner as probe_temporal.py: rare cause_entity that appears earlier as
    an entity of doc a, later stated as cause of effect doc b (strict time)."""
    n = len(docs)
    df = defaultdict(int)
    for d in docs:
        for e in set(d["entities"]):
            df[e] += 1
    cap = math.ceil(0.03 * n)
    post = defaultdict(list)
    for i, d in enumerate(docs):
        for e in d["entities"]:
            post[e].append(i)
    pos = []
    for b, d in enumerate(docs):
        tb = d["published_epoch"]
        if tb == 0:
            continue
        for c in d.get("cause_entities", []):
            if 0 < df.get(c, 0) <= cap:
                for a in post.get(c, []):
                    ta = docs[a]["published_epoch"]
                    if a != b and ta > 0 and 0 < tb - ta:
                        pos.append((a, b))
    pos = list({p for p in pos})
    rng.shuffle(pos)
    return pos[:9000]

# ------------------------------------------------------------------ baselines
def ridge_W(A, B, lam=10.0):
    """W = (A^T A + lam I)^-1 A^T B  s.t.  A @ W ~= B  (transport cause->effect)."""
    d = A.shape[1]
    return np.linalg.solve(A.T @ A + lam * np.eye(d, dtype=np.float64), A.T @ B)

# ------------------------------------------------------------------ two-tower
def build_towers():
    import torch, torch.nn as nn
    def head():
        return nn.Sequential(
            nn.Linear(384, 256), nn.ReLU(),
            nn.Linear(256, 128),
        )
    class TwoTower(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = head()
            self.fe = head()
        def cause(self, x):
            z = self.fc(x); return z / (z.norm(dim=-1, keepdim=True) + 1e-9)
        def effect(self, x):
            z = self.fe(x); return z / (z.norm(dim=-1, keepdim=True) + 1e-9)
    return TwoTower()

def train_towers(emb, pairs_tr, epochs=40, bs=256, tau=0.07):
    import torch
    torch.manual_seed(SEED)
    dev = "cpu"
    model = build_towers().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    A = torch.from_numpy(emb[[a for a, _ in pairs_tr]]).to(dev)
    B = torch.from_numpy(emb[[b for _, b in pairs_tr]]).to(dev)
    n = A.shape[0]
    g = torch.Generator().manual_seed(SEED)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if idx.numel() < 2:
                continue
            ca = model.cause(A[idx])          # (m,128)
            ce = model.effect(B[idx])         # (m,128)
            logits = (ca @ ce.t()) / tau      # in-batch negatives
            labels = torch.arange(idx.numel(), device=dev)
            # symmetric InfoNCE (cause->effect and effect->cause retrieval)
            loss = 0.5 * (torch.nn.functional.cross_entropy(logits, labels) +
                          torch.nn.functional.cross_entropy(logits.t(), labels))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        Fc = model.cause(torch.from_numpy(emb)).numpy()
        Fe = model.effect(torch.from_numpy(emb)).numpy()
    return Fc, Fe

# ------------------------------------------------------------------ eval
def direction_acc(idx_a, idx_b, score_fn):
    """score_fn(list_a, list_b) -> vector of scores for a->b. Accuracy that
    score(a->b) > score(b->a) on held-out pairs. Ties (e.g. symmetric cosine,
    which scores a->b == b->a) count as 0.5 -> the correct chance baseline."""
    s_ab = score_fn(idx_a, idx_b)
    s_ba = score_fn(idx_b, idx_a)
    win = (s_ab > s_ba).astype(np.float64)
    tie = np.isclose(s_ab, s_ba)
    win[tie] = 0.5
    return float(np.mean(win))

def retrieval(cause_ids, gold_ids, epochs_arr, cause_vecs, cand_vecs, train_targets):
    """For each test cause, rank all STRICTLY-LATER docs; measure recall@k / MRR
    of the true effect. Exclude train targets from candidate set."""
    ks = [10, 30, 100]
    hits = {k: 0 for k in ks}
    rr = 0.0
    n_eval = 0
    train_mask = np.zeros(len(epochs_arr), dtype=bool)
    train_mask[list(train_targets)] = True
    for a, gold in zip(cause_ids, gold_ids):
        ta = epochs_arr[a]
        cand = (epochs_arr > ta) & (~train_mask)
        cand[gold] = True                 # keep the gold effect even if it was a train target
        cand_idx = np.nonzero(cand)[0]
        if cand_idx.size < 2:
            continue
        scores = cand_vecs[cand_idx] @ cause_vecs[a]
        order = cand_idx[np.argsort(-scores)]
        pos = np.nonzero(order == gold)[0]
        if pos.size == 0:
            continue
        rank = int(pos[0])
        rr += 1.0 / (rank + 1)
        for k in ks:
            if rank < k:
                hits[k] += 1
        n_eval += 1
    if n_eval == 0:
        return {"recall@10": 0.0, "recall@30": 0.0, "recall@100": 0.0, "mrr": 0.0, "n": 0}
    return {f"recall@{k}": round(hits[k] / n_eval, 4) for k in ks} | {
        "mrr": round(rr / n_eval, 4), "n": n_eval}

# ------------------------------------------------------------------ main
def main():
    (DATA / "metrics").mkdir(exist_ok=True)
    (DATA / "figures").mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    emb, docs = load()
    pairs = mine_pairs(docs, rng)
    eff_ep = np.array([docs[b]["published_epoch"] for _, b in pairs])
    cut = np.quantile(eff_ep, 0.7)
    is_te = eff_ep >= cut
    pairs_tr = [p for p, t in zip(pairs, is_te) if not t]
    pairs_te = [p for p, t in zip(pairs, is_te) if t]
    epochs_arr = np.array([d["published_epoch"] for d in docs], dtype=np.int64)
    train_targets = {b for _, b in pairs_tr}

    # ---- baseline transport W (fit on train pairs)
    A = emb[[a for a, _ in pairs_tr]].astype(np.float64)
    B = emb[[b for _, b in pairs_tr]].astype(np.float64)
    W = ridge_W(A, B)
    embW = norml((emb.astype(np.float64) @ W)).astype(np.float32)  # transported cause reps

    # ---- two-tower
    Fc, Fe = train_towers(emb, pairs_tr)

    te_a = [a for a, _ in pairs_te]
    te_b = [b for _, b in pairs_te]

    # score fns for DIRECTION
    def s_cos(ia, ib):   return np.sum(emb[ia] * emb[ib], axis=1)
    def s_W(ia, ib):     return np.sum(embW[ia] * emb[ib], axis=1)   # <W e_a, e_b>
    def s_tt(ia, ib):    return np.sum(Fc[ia] * Fe[ib], axis=1)

    direction = {
        "cosine": round(direction_acc(te_a, te_b, s_cos), 4),
        "W":      round(direction_acc(te_a, te_b, s_W), 4),
        "two_tower": round(direction_acc(te_a, te_b, s_tt), 4),
        "n_test_pairs": len(pairs_te),
    }

    # ---- retrieval (first stage): cause vecs vs candidate (effect) vecs
    ret_cos = retrieval(te_a, te_b, epochs_arr, emb, emb, train_targets)
    ret_W   = retrieval(te_a, te_b, epochs_arr, embW, emb, train_targets)
    ret_tt  = retrieval(te_a, te_b, epochs_arr, Fc, Fe, train_targets)

    metrics = {
        "seed": SEED,
        "n_pairs_total": len(pairs),
        "n_train_pairs": len(pairs_tr),
        "n_test_pairs": len(pairs_te),
        "temporal_cut_epoch": int(cut),
        "direction_accuracy": direction,
        "retrieval": {"cosine": ret_cos, "W": ret_W, "two_tower": ret_tt},
    }
    (DATA / "metrics" / "twotower.json").write_text(json.dumps(metrics, indent=2))

    # ---- verdict
    d_tt, d_W, d_cos = direction["two_tower"], direction["W"], direction["cosine"]
    r_tt, r_cos, r_W = ret_tt["recall@10"], ret_cos["recall@10"], ret_W["recall@10"]
    beats_dir = d_tt > d_W + 0.01
    beats_ret = r_tt > max(r_cos, r_W) + 0.005
    # non-stationary check: two-tower direction below chance despite training
    if d_tt < 0.5:
        verdict = "NON-STATIONARY"
    elif beats_dir and beats_ret:
        verdict = "SIGNAL"
    elif beats_dir or beats_ret:
        verdict = "WEAK"
    else:
        verdict = "NO-SIGNAL"

    # ---- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    names = ["cosine", "W", "two-tower"]
    vals = [d_cos, d_W, d_tt]
    colors = ["#888", "#3b7", "#e55"]
    ax1.bar(names, vals, color=colors)
    ax1.axhline(0.5, ls="--", c="k", lw=1, label="chance")
    for i, v in enumerate(vals):
        ax1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax1.set_ylim(0, 1); ax1.set_ylabel("direction accuracy")
    ax1.set_title(f"Direction: score(a->b)>score(b->a)  (n={len(pairs_te)})")
    ax1.legend()
    ks = [10, 30, 100]
    for ret, nm, c in [(ret_cos, "cosine", "#888"), (ret_W, "W", "#3b7"), (ret_tt, "two-tower", "#e55")]:
        ax2.plot(ks, [ret[f"recall@{k}"] for k in ks], "-o", color=c, label=nm)
    ax2.set_xlabel("k"); ax2.set_ylabel("recall@k"); ax2.set_xticks(ks)
    ax2.set_title(f"First-stage retrieval (temporal test, n={ret_tt['n']})")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle(f"Two-tower vs linear transport (temporal split)  ->  {verdict}")
    fig.tight_layout()
    fig.savefig(DATA / "figures" / "twotower.png", dpi=110)

    print(f"DIRECTION acc: two-tower {d_tt:.3f} vs W {d_W:.3f} vs cosine {d_cos:.3f}")
    print(f"RETRIEVAL R@10: two-tower {r_tt:.3f} vs cosine {r_cos:.3f} vs W {r_W:.3f} (MRR tt {ret_tt['mrr']:.3f})")
    print(f"Learned towers {'beat' if (beats_dir or beats_ret) else 'do NOT beat'} the linear operator temporally.")
    print(f"VERDICT: {verdict}")

if __name__ == "__main__":
    main()
