# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
GEOMETRY: is the embedding's anisotropy hurting the linear receptors, and does
the causal-direction signal live on a low-dimensional manifold?

1. ANISOTROPY: PCA variance spectrum of facts.npy (top-1/5/20 dims).
2. WHITENING TEST: linear logistic probes for polarity (direction sign) and
   predicate (top classes) on raw / ZCA-whitened / all-but-top-k (k=1,5)
   representations. Does decorrelating / removing dominant topical directions
   HELP or HURT? (fact-level de-gist question.)
3. INTRINSIC DIMENSION: TwoNN MLE estimator on a sample of facts.npy.
4. LOW-D SUBSPACE for direction: mined direction ordering task (interaction
   features), sweep #PCA components -> linear probe accuracy. Where does it
   saturate?
"""
import json, math
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RNG = np.random.default_rng(0)


def norml(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# 1. ANISOTROPY
# ---------------------------------------------------------------------------
def anisotropy(F):
    pca = PCA(n_components=min(384, F.shape[0], F.shape[1])).fit(F)
    ev = pca.explained_variance_ratio_
    cum = np.cumsum(ev)
    return {
        "top1_pct": round(float(cum[0]) * 100, 2),
        "top5_pct": round(float(cum[4]) * 100, 2),
        "top20_pct": round(float(cum[19]) * 100, 2),
        "spectrum": ev.tolist(),
        "cum": cum.tolist(),
    }


# ---------------------------------------------------------------------------
# whitening / top-k removal transforms (FIT ON TRAIN ONLY)
# ---------------------------------------------------------------------------
def fit_transforms(Xtr):
    mu = Xtr.mean(0)
    Xc = Xtr - mu
    # SVD for a stable eigen-decomposition of the covariance
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    w, V = np.linalg.eigh(cov)  # ascending
    w = w[::-1]
    V = V[:, ::-1]
    eps = 1e-5
    W_zca = V @ np.diag(1.0 / np.sqrt(w + eps)) @ V.T  # ZCA whitening

    def make(kind):
        if kind == "raw":
            return lambda X: X - mu
        if kind == "zca":
            return lambda X: (X - mu) @ W_zca
        if kind.startswith("top"):
            k = int(kind[3:])
            Vk = V[:, :k]
            P = np.eye(V.shape[0]) - Vk @ Vk.T  # remove top-k directions
            return lambda X: (X - mu) @ P
        raise ValueError(kind)

    return make


def probe_on(Xtr, Xte, ytr, yte, make, kind):
    tf = make(kind)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(tf(Xtr), ytr)
    return round(accuracy_score(yte, clf.predict(tf(Xte))), 4)


def whitening_test(F, facts):
    kinds = ["raw", "zca", "top1", "top5"]
    n = F.shape[0]
    te = RNG.random(n) < 0.30

    # -- polarity: direction sign, drop neutral (0) --
    ysign = np.array([int(np.sign(round(f["direction"]))) for f in facts])
    pmask = ysign != 0
    Xp, yp = F[pmask], (ysign[pmask] > 0).astype(int)
    tep = te[pmask]
    pol = {}
    make_p = fit_transforms(Xp[~tep])
    for k in kinds:
        pol[k] = probe_on(Xp[~tep], Xp[tep], yp[~tep], yp[tep], make_p, k)

    # -- predicate: top-6 classes --
    preds = [f["predicate"] for f in facts]
    top = [p for p, _ in Counter(preds).most_common(6)]
    lut = {p: i for i, p in enumerate(top)}
    pmask2 = np.array([p in lut for p in preds])
    Xq = F[pmask2]
    yq = np.array([lut[preds[i]] for i in range(len(facts)) if pmask2[i]])
    teq = te[pmask2]
    predc = {}
    make_q = fit_transforms(Xq[~teq])
    for k in kinds:
        predc[k] = probe_on(Xq[~teq], Xq[teq], yq[~teq], yq[teq], make_q, k)

    return {
        "kinds": kinds,
        "polarity": pol,
        "polarity_n_test": int(tep.sum()),
        "predicate": predc,
        "predicate_classes": top,
        "predicate_n_test": int(teq.sum()),
    }


# ---------------------------------------------------------------------------
# 3. INTRINSIC DIMENSION (TwoNN, Facco et al. 2017)
# ---------------------------------------------------------------------------
def twonn(F, sample=4000):
    idx = RNG.choice(F.shape[0], min(sample, F.shape[0]), replace=False)
    X = F[idx]
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=3).fit(X)
    d, _ = nn.kneighbors(X)  # col0 self=0
    r1 = d[:, 1]
    r2 = d[:, 2]
    good = r1 > 1e-12
    mu = r2[good] / r1[good]
    mu = mu[mu > 1.0]
    # MLE with linear-regression on empirical CDF (drop top 10% outliers)
    mu_sorted = np.sort(mu)
    N = len(mu_sorted)
    Femp = np.arange(1, N + 1) / N
    keep = Femp < 0.9
    x = np.log(mu_sorted[keep])
    y = -np.log(1.0 - Femp[keep])
    d_mle = float(np.sum(x * y) / np.sum(x * x))  # slope through origin
    return {"id_twonn": round(d_mle, 2), "ambient": int(F.shape[1]), "n_sample": int(len(idx))}


# ---------------------------------------------------------------------------
# 4. LOW-D SUBSPACE for direction (mined ordering task)
# ---------------------------------------------------------------------------
def mine_direction_pairs():
    emb = norml(np.load(DATA / "embeddings.npy").astype(np.float32))
    docs = [json.loads(l) for l in (DATA / "docs.jsonl").read_text().splitlines() if l.strip()]
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
                    if a != b and docs[a]["published_epoch"] > 0 and 0 < tb - docs[a]["published_epoch"]:
                        pos.append((a, b))
    pos = list({p for p in pos})
    RNG.shuffle(pos)
    pos = pos[:9000]

    def feat(a, b):
        ea, eb = emb[a], emb[b]
        return np.concatenate([ea, eb, ea - eb, ea * eb])

    Xp = np.array([feat(a, b) for a, b in pos], np.float32)
    Xr = np.array([feat(b, a) for a, b in pos], np.float32)
    X = np.vstack([Xp, Xr])
    y = np.r_[np.ones(len(pos)), np.zeros(len(pos))].astype(int)
    return X, y


def dim_sweep(X, y, dims=(8, 16, 32, 64, 128, 384)):
    te = RNG.random(X.shape[0]) < 0.30
    Xtr, Xte = X[~te], X[te]
    ytr, yte = y[~te], y[te]
    mu = Xtr.mean(0)
    pca = PCA(n_components=max(dims), random_state=0).fit(Xtr - mu)
    Ztr = pca.transform(Xtr - mu)
    Zte = pca.transform(Xte - mu)
    out = {}
    for k in dims:
        clf = LogisticRegression(max_iter=2000).fit(Ztr[:, :k], ytr)
        out[k] = round(accuracy_score(yte, clf.predict(Zte[:, :k])), 4)
    return {"dims": list(dims), "acc": out, "n_test": int(te.sum())}


# ---------------------------------------------------------------------------
# FIGURE
# ---------------------------------------------------------------------------
def figure(ani, whit, sweep, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # variance spectrum
    ev = np.array(ani["spectrum"])
    ax[0].plot(np.arange(1, len(ev) + 1), np.cumsum(ev), lw=2)
    ax[0].axhline(0, color="k", lw=0.3)
    for d in (1, 5, 20):
        ax[0].axvline(d, ls="--", color="grey", lw=0.7)
    ax[0].set_title(f"Anisotropy: top1={ani['top1_pct']}% "
                    f"top5={ani['top5_pct']}% top20={ani['top20_pct']}%")
    ax[0].set_xlabel("PCA dim")
    ax[0].set_ylabel("cumulative variance")
    ax[0].set_xscale("log")

    # whitening bars
    kinds = whit["kinds"]
    xpos = np.arange(len(kinds))
    pol = [whit["polarity"][k] for k in kinds]
    prd = [whit["predicate"][k] for k in kinds]
    w = 0.38
    ax[1].bar(xpos - w / 2, pol, w, label="polarity")
    ax[1].bar(xpos + w / 2, prd, w, label="predicate")
    ax[1].set_xticks(xpos)
    ax[1].set_xticklabels(kinds)
    ax[1].set_ylim(0.4, 1.0)
    ax[1].set_title("Linear probe acc: raw vs ZCA vs top-k removed")
    ax[1].set_ylabel("held-out accuracy")
    ax[1].legend()

    # dim sweep
    dims = sweep["dims"]
    acc = [sweep["acc"][d] for d in dims]
    ax[2].plot(dims, acc, "o-", lw=2)
    ax[2].set_xscale("log")
    ax[2].set_xticks(dims)
    ax[2].set_xticklabels(dims)
    ax[2].set_title("Direction acc vs #PCA components")
    ax[2].set_xlabel("#components (interaction feats)")
    ax[2].set_ylabel("held-out accuracy")
    ax[2].axhline(max(acc), ls="--", color="grey", lw=0.7)

    fig.tight_layout()
    fig.savefig(path, dpi=110)


# ---------------------------------------------------------------------------
def main():
    F = norml(np.load(DATA / "facts.npy").astype(np.float32))
    facts = [json.loads(l) for l in (DATA / "facts.jsonl").read_text().splitlines() if l.strip()]

    ani = anisotropy(F)
    whit = whitening_test(F, facts)
    idn = twonn(F)
    Xd, yd = mine_direction_pairs()
    sweep = dim_sweep(Xd, yd)

    metrics = {
        "anisotropy": {k: ani[k] for k in ("top1_pct", "top5_pct", "top20_pct")},
        "whitening": whit,
        "intrinsic_dim": idn,
        "direction_dim_sweep": sweep,
    }
    (DATA / "metrics" / "geometry.json").write_text(json.dumps(metrics, indent=2))
    figure(ani, whit, sweep, DATA / "figures" / "geometry.png")

    # ---- verdict logic ----
    pol = whit["polarity"]
    prd = whit["predicate"]
    best_alt_pol = max(pol["zca"], pol["top1"], pol["top5"])
    best_alt_prd = max(prd["zca"], prd["top1"], prd["top5"])
    pol_gain = best_alt_pol - pol["raw"]
    prd_gain = best_alt_prd - prd["raw"]
    whiten_helps = (pol_gain >= 0.01) or (prd_gain >= 0.01)

    acc = sweep["acc"]
    full = acc[max(sweep["dims"])]
    lowd = None
    for d in sweep["dims"]:
        if acc[d] >= full - 0.01:
            lowd = d
            break
    lowd_helps = lowd is not None and lowd <= 64

    signal = whiten_helps or lowd_helps
    verdict = "SIGNAL" if signal else ("WEAK" if (pol_gain > 0 or prd_gain > 0) else "NO-SIGNAL")

    print(f"WHITENING: polarity raw={pol['raw']} best-alt={best_alt_pol} "
          f"({pol_gain:+.3f}); predicate raw={prd['raw']} best-alt={best_alt_prd} "
          f"({prd_gain:+.3f}) -> {'HELPS' if whiten_helps else 'HURTS/neutral'}")
    print(f"INTRINSIC DIM: TwoNN ~{idn['id_twonn']} vs ambient {idn['ambient']}")
    print(f"DIRECTION SIGNAL: saturates at ~{lowd} PCA dims "
          f"(acc {acc[lowd] if lowd else full} vs full-384 {full})")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
