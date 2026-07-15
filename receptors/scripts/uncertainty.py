# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "matplotlib"]
# ///
"""
Can calibrated abstention convert the noisy direction receptor into a
high-precision-on-a-subset tool, and does active labeling beat random?

Task: cause->effect direction ordering. For a mined (a,b) pair (a causes b,
a published before b), build features [e_a, e_b, e_a-e_b, e_a*e_b] with label 1;
the flipped (b,a) gets label 0. TEMPORAL split (train early, test late).

1. SELECTIVE PREDICTION: confidence = |p - 0.5|. Risk-coverage curve.
2. CALIBRATION: reliability + Brier, raw vs isotonic (fit on a val slice).
3. ACTIVE LEARNING: uncertainty (margin) sampling vs random, learning curves.
"""
import json, math
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]; DATA = ROOT / "data"
RNG = np.random.default_rng(0)


def norml(m):
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-9)


def build_direction_dataset():
    """Mine cause->effect pairs (mirror of probe_temporal.py) and build the
    direction-ordering task with a temporal split on the effect timestamp."""
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
    eff_ep = np.array([docs[b]["published_epoch"] for _, b in pos])

    def feat(a, b):
        ea, eb = emb[a], emb[b]
        return np.concatenate([ea, eb, ea - eb, ea * eb])

    Xp = np.array([feat(a, b) for a, b in pos], np.float32)
    Xr = np.array([feat(b, a) for a, b in pos], np.float32)
    X = np.vstack([Xp, Xr])
    y = np.r_[np.ones(len(pos)), np.zeros(len(pos))]
    # temporal split shared across the (a,b)/(b,a) copies of each pair
    cut = np.quantile(eff_ep, 0.7)
    te_pair = eff_ep >= cut
    te = np.r_[te_pair, te_pair]
    return X, y, te, len(pos)


def risk_coverage(conf, correct):
    """Accuracy on the most-confident x% of test items, for x in 10..100."""
    order = np.argsort(-conf)  # high confidence first
    correct_sorted = correct[order]
    pts = []
    n = len(conf)
    for cov in range(10, 101, 10):
        k = max(1, int(round(cov / 100.0 * n)))
        acc = float(correct_sorted[:k].mean())
        pts.append({"coverage": cov / 100.0, "accuracy": round(acc, 4), "n": k})
    return pts


def acc_at_coverage(conf, correct, cov_frac):
    order = np.argsort(-conf)
    k = max(1, int(round(cov_frac * len(conf))))
    return float(correct[order][:k].mean())


def threshold_for_target(conf, correct, target=0.90):
    """Smallest confidence threshold whose retained accuracy >= target;
    returns (threshold, coverage, accuracy)."""
    order = np.argsort(-conf)
    cs = correct[order]
    confs = conf[order]
    n = len(conf)
    best = None
    for k in range(n, 0, -1):
        acc = float(cs[:k].mean())
        if acc >= target:
            best = (float(confs[k - 1]), k / n, acc)
            break
    if best is None:
        # even the single most-confident item doesn't hit target
        best = (float(confs[0]), 1.0 / n, float(cs[:1].mean()))
    return best


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        out.append({"conf": round(float(p[m].mean()), 4),
                    "acc": round(float(y[m].mean()), 4),
                    "n": int(m.sum())})
    return out


def active_learning(Xtr, ytr, Xte, yte, seed=200, step=200, rounds=14, reps=3):
    """Simulate pool-based AL: uncertainty (margin) sampling vs random.
    Averaged over `reps` seeds. Returns per-strategy learning curves."""
    ntr = len(Xtr)
    curves = {"uncertainty": defaultdict(list), "random": defaultdict(list)}
    label_sizes = None
    for rep in range(reps):
        rng = np.random.default_rng(100 + rep)
        for strat in ("uncertainty", "random"):
            perm = rng.permutation(ntr)
            labeled = list(perm[:seed])
            pool = list(perm[seed:])
            sizes = []
            for r in range(rounds):
                clf = LogisticRegression(max_iter=1000, C=1.0)
                clf.fit(Xtr[labeled], ytr[labeled])
                acc = accuracy_score(yte, clf.predict(Xte))
                curves[strat][len(labeled)].append(acc)
                sizes.append(len(labeled))
                if not pool:
                    break
                if strat == "random":
                    take = pool[:step]
                    pool = pool[step:]
                else:
                    p = clf.predict_proba(Xtr[pool])[:, 1]
                    unc = -np.abs(p - 0.5)  # most uncertain first
                    order = np.argsort(-unc)
                    take_idx = order[:step]
                    take = [pool[i] for i in take_idx]
                    keep = set(range(len(pool))) - set(take_idx.tolist())
                    pool = [pool[i] for i in sorted(keep)]
                labeled = labeled + take
            if label_sizes is None:
                label_sizes = sizes
    def summarize(d):
        xs = sorted(d.keys())
        return [{"labels": int(x), "acc": round(float(np.mean(d[x])), 4)} for x in xs]
    return {"uncertainty": summarize(curves["uncertainty"]),
            "random": summarize(curves["random"])}


def main():
    X, y, te, n_pairs = build_direction_dataset()
    Xtr_all, ytr_all = X[~te], y[~te]
    Xte, yte = X[te], y[te]

    # carve a validation slice out of train for isotonic calibration
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(Xtr_all))
    n_val = max(500, int(0.15 * len(Xtr_all)))
    val_idx, fit_idx = perm[:n_val], perm[n_val:]
    Xfit, yfit = Xtr_all[fit_idx], ytr_all[fit_idx]
    Xval, yval = Xtr_all[val_idx], ytr_all[val_idx]

    clf = LogisticRegression(max_iter=2000).fit(Xfit, yfit)
    p_raw = clf.predict_proba(Xte)[:, 1]
    pred = (p_raw >= 0.5).astype(float)
    correct = (pred == yte).astype(float)
    conf = np.abs(p_raw - 0.5)

    # ---- 1. selective prediction ----
    rc = risk_coverage(conf, correct)
    acc_100 = acc_at_coverage(conf, correct, 1.0)
    acc_50 = acc_at_coverage(conf, correct, 0.5)
    acc_20 = acc_at_coverage(conf, correct, 0.2)
    thr, cov_at_90, acc_at_90 = threshold_for_target(conf, correct, target=0.90)

    # ---- 2. calibration (raw vs isotonic on val slice) ----
    p_val = clf.predict_proba(Xval)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_val, yval)
    p_cal = iso.predict(p_raw)
    brier_raw = float(brier_score_loss(yte, p_raw))
    brier_cal = float(brier_score_loss(yte, p_cal))
    rel_raw = reliability(p_raw, yte)
    rel_cal = reliability(p_cal, yte)

    # ---- 3. active learning ----
    al = active_learning(Xtr_all, ytr_all, Xte, yte)

    # target accuracy = 98% of full-train accuracy; labels each strategy needs
    full_acc = acc_100
    target = 0.98 * full_acc
    def labels_to_reach(curve, tgt):
        for pt in curve:
            if pt["acc"] >= tgt:
                return pt["labels"]
        return None
    lab_unc = labels_to_reach(al["uncertainty"], target)
    lab_rnd = labels_to_reach(al["random"], target)

    metrics = {
        "task": "cause->effect direction ordering (temporal split)",
        "n_pairs": n_pairs, "n_train": int((~te).sum()), "n_test": int(te.sum()),
        "selective_prediction": {
            "risk_coverage": rc,
            "acc_at_100pct": round(acc_100, 4),
            "acc_at_50pct": round(acc_50, 4),
            "acc_at_20pct": round(acc_20, 4),
            "target_selective_acc": 0.90,
            "threshold_conf": round(thr, 4),
            "coverage_at_target": round(cov_at_90, 4),
            "acc_at_target": round(acc_at_90, 4),
        },
        "calibration": {
            "brier_raw": round(brier_raw, 4),
            "brier_calibrated": round(brier_cal, 4),
            "brier_improvement": round(brier_raw - brier_cal, 4),
            "reliability_raw": rel_raw,
            "reliability_calibrated": rel_cal,
        },
        "active_learning": {
            "seed": 200, "step": 200,
            "full_train_acc": round(full_acc, 4),
            "target_acc": round(target, 4),
            "labels_uncertainty_to_target": lab_unc,
            "labels_random_to_target": lab_rnd,
            "curve_uncertainty": al["uncertainty"],
            "curve_random": al["random"],
        },
    }

    (DATA / "metrics").mkdir(exist_ok=True)
    (DATA / "figures").mkdir(exist_ok=True)
    (DATA / "metrics" / "uncertainty.json").write_text(json.dumps(metrics, indent=2))

    # ---- figure ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    covs = [p["coverage"] for p in rc]
    accs = [p["accuracy"] for p in rc]
    ax[0].plot(covs, accs, "o-", color="#1f77b4", lw=2)
    ax[0].axhline(acc_100, ls="--", color="gray", label=f"full-coverage acc {acc_100:.3f}")
    ax[0].scatter([cov_at_90], [acc_at_90], color="red", zorder=5,
                  label=f"90% sel-acc @ cov {cov_at_90:.2f}")
    ax[0].set_xlabel("coverage (most-confident fraction)")
    ax[0].set_ylabel("accuracy on retained")
    ax[0].set_title("Risk-Coverage (direction receptor)")
    ax[0].invert_xaxis()
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ux = [p["labels"] for p in al["uncertainty"]]
    uy = [p["acc"] for p in al["uncertainty"]]
    rx = [p["labels"] for p in al["random"]]
    ry = [p["acc"] for p in al["random"]]
    ax[1].plot(ux, uy, "o-", color="#d62728", lw=2, label="uncertainty sampling")
    ax[1].plot(rx, ry, "s-", color="#2ca02c", lw=2, label="random sampling")
    ax[1].axhline(target, ls="--", color="gray", label=f"target {target:.3f}")
    ax[1].set_xlabel("# labeled pairs")
    ax[1].set_ylabel("test accuracy")
    ax[1].set_title("Active learning: uncertainty vs random")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(DATA / "figures" / "uncertainty.png", dpi=110)

    # ---- verdict ----
    abst_gain = acc_20 - acc_100
    al_beats = (lab_unc is not None and lab_rnd is not None and lab_unc < lab_rnd) or \
               (lab_unc is not None and lab_rnd is None)
    abstention_helps = abst_gain >= 0.03 or cov_at_90 >= 0.15
    verdict = "SIGNAL" if (abstention_helps or al_beats) else \
              ("WEAK" if (abst_gain > 0.01 or (lab_unc and lab_rnd and lab_unc <= lab_rnd)) else "NO-SIGNAL")

    print(f"Selective: acc {acc_100:.3f}@100% -> {acc_50:.3f}@50% -> {acc_20:.3f}@20%; "
          f"90% sel-acc buys {cov_at_90*100:.0f}% coverage.")
    print(f"Calibration: Brier {brier_raw:.4f} raw -> {brier_cal:.4f} isotonic "
          f"(improvement {brier_raw-brier_cal:+.4f}).")
    al_msg = (f"reaches {target:.3f} at {lab_unc} vs {lab_rnd} labels" if lab_unc and lab_rnd
              else f"unc={lab_unc} rnd={lab_rnd} labels to target")
    print(f"Active learning ({'beats' if al_beats else 'ties/loses vs'} random): {al_msg}.")
    print(f"VERDICT: {verdict}")


if __name__ == "__main__":
    main()
