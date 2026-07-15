# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","scikit-learn","matplotlib"]
# ///
"""
Koopman / DMD on entity-state of the news system.

Question: does the news system have PREDICTABLE linear dynamics (a Koopman /
transfer operator) on entity activity, and what are the coherent modes?

Method:
  - Build X (entities x days), cell = summed signed direction (also raw counts)
    for that entity on that day. Sort days ascending.
  - DMD: X1=X[:,:-1], X2=X[:,1:]; low-rank SVD of X1 -> reduced operator A~;
    eigen-decompose -> DMD eigenvalues (|lambda|, angle) and modes.
  - Predictive test: fit A on earliest ~70% of transitions, one-step-ahead
    predict on held-out latest days; MSE vs persistence & mean baselines.
    Koopman "has signal" only if it beats BOTH out of sample.
"""
import json, os
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FACTS = os.path.join(ROOT, "data", "facts.jsonl")
MET_DIR = os.path.join(ROOT, "data", "metrics")
FIG_DIR = os.path.join(ROOT, "data", "figures")
os.makedirs(MET_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

N_ENT = 60
RANK = 12  # SVD truncation for DMD


def load():
    """Return (days sorted, signed matrix, count matrix, entity list)."""
    freq = Counter()
    rows = []
    for line in open(FACTS):
        d = json.loads(line)
        if not d.get("date"):
            continue
        ents = d.get("entities") or []
        direction = float(d.get("direction") or 0.0)
        rows.append((d["date"], direction, ents))
        for e in ents:
            freq[e] += 1
    top = [e for e, _ in freq.most_common(N_ENT)]
    eidx = {e: i for i, e in enumerate(top)}
    days = sorted({r[0] for r in rows})
    didx = {dd: j for j, dd in enumerate(days)}
    Xs = np.zeros((len(top), len(days)))  # signed activity
    Xc = np.zeros((len(top), len(days)))  # raw counts
    for date, direction, ents in rows:
        j = didx[date]
        for e in ents:
            i = eidx.get(e)
            if i is None:
                continue
            Xs[i, j] += direction
            Xc[i, j] += 1.0
    return days, Xs, Xc, top


def dmd(X, rank):
    """Exact DMD. Returns eigenvalues, modes (entity loadings), A_reduced info."""
    X1, X2 = X[:, :-1], X[:, 1:]
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    r = min(rank, np.sum(S > 1e-10), X1.shape[1])
    r = max(r, 1)
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r, :].conj().T
    Atil = Ur.conj().T @ X2 @ Vr @ np.diag(1.0 / Sr)
    evals, W = np.linalg.eig(Atil)
    # DMD modes (projected)
    Phi = X2 @ Vr @ np.diag(1.0 / Sr) @ W
    return evals, Phi, r


def one_step_eval(X, rank, train_frac=0.7):
    """Fit reduced linear operator on early transitions, predict held-out days.
    Returns dict of MSEs and the fraction of pairs held out."""
    npairs = X.shape[1] - 1
    ntrain = max(2, int(round(npairs * train_frac)))
    ntrain = min(ntrain, npairs - 1)  # keep >=1 held out
    # training transitions: columns 0..ntrain (pairs 0..ntrain-1)
    Xtr1 = X[:, :ntrain]
    Xtr2 = X[:, 1:ntrain + 1]
    # Reduced-space operator learned on training data only (POD basis on Xtr1)
    U, S, Vt = np.linalg.svd(Xtr1, full_matrices=False)
    r = min(rank, int(np.sum(S > 1e-10)), Xtr1.shape[1])
    r = max(r, 1)
    Ur = U[:, :r]
    # Least-squares reduced operator: minimize ||Ur^T Xtr2 - A (Ur^T Xtr1)||
    Ztr1 = Ur.conj().T @ Xtr1
    Ztr2 = Ur.conj().T @ Xtr2
    A, *_ = np.linalg.lstsq(Ztr1.T, Ztr2.T, rcond=None)
    A = A.T  # so that Ztr2 ~ A @ Ztr1
    train_mean = X[:, :ntrain + 1].mean(axis=1, keepdims=True)

    se_k, se_p, se_m = [], [], []
    for t in range(ntrain, npairs):  # held-out target columns t+1
        xt = X[:, t]
        xt1 = X[:, t + 1]
        pred_k = Ur @ (A @ (Ur.conj().T @ xt))
        pred_k = np.real(pred_k)
        se_k.append(np.mean((xt1 - pred_k) ** 2))
        se_p.append(np.mean((xt1 - xt) ** 2))          # persistence
        se_m.append(np.mean((xt1 - train_mean[:, 0]) ** 2))  # mean baseline
    return {
        "koopman_mse": float(np.mean(se_k)),
        "persistence_mse": float(np.mean(se_p)),
        "mean_mse": float(np.mean(se_m)),
        "n_train_pairs": ntrain,
        "n_test_pairs": npairs - ntrain,
        "reduced_rank": int(r),
    }


def top_modes(evals, Phi, entities, k=6):
    order = np.argsort(-np.abs(evals))
    out = []
    for idx in order[:k]:
        lam = evals[idx]
        mode = Phi[:, idx]
        mag = np.abs(mode)
        dom = np.argsort(-mag)[:5]
        ang = float(np.angle(lam))
        period = float(2 * np.pi / abs(ang)) if abs(ang) > 1e-8 else None
        out.append({
            "abs_lambda": float(np.abs(lam)),
            "angle_rad": ang,
            "period_days": period,
            "lambda_re": float(lam.real),
            "lambda_im": float(lam.imag),
            "dominant_entities": [
                {"entity": entities[i], "loading": float(mag[i])} for i in dom
            ],
        })
    return out


def main():
    days, Xs, Xc, entities = load()
    # Center per-entity (remove entity mean) for dynamics on fluctuations.
    Xs_c = Xs - Xs.mean(axis=1, keepdims=True)

    # DMD on signed activity (primary) and raw counts (secondary).
    evals_s, Phi_s, r_s = dmd(Xs_c, RANK)
    modes_s = top_modes(evals_s, Phi_s, entities, k=6)

    pred_signed = one_step_eval(Xs_c, RANK)
    pred_counts = one_step_eval(Xc - Xc.mean(axis=1, keepdims=True), RANK)

    beats = (pred_signed["koopman_mse"] < pred_signed["persistence_mse"] and
             pred_signed["koopman_mse"] < pred_signed["mean_mse"])
    beats_counts = (pred_counts["koopman_mse"] < pred_counts["persistence_mse"] and
                    pred_counts["koopman_mse"] < pred_counts["mean_mse"])
    # Margin vs the better baseline on signed representation.
    best_base = min(pred_signed["persistence_mse"], pred_signed["mean_mse"])
    margin = (best_base - pred_signed["koopman_mse"]) / best_base if best_base > 0 else 0.0

    if beats and margin > 0.10:
        verdict = "SIGNAL"
    elif beats or beats_counts:
        verdict = "WEAK"
    else:
        verdict = "NO-SIGNAL"

    metrics = {
        "n_days": len(days),
        "day_range": [days[0], days[-1]],
        "n_entities": len(entities),
        "dmd_eigenvalues_signed": [
            {"re": float(e.real), "im": float(e.imag), "abs": float(np.abs(e))}
            for e in evals_s
        ],
        "top_modes_signed": modes_s,
        "prediction_signed": pred_signed,
        "prediction_counts": pred_counts,
        "koopman_beats_both_baselines_signed": bool(beats),
        "koopman_beats_both_baselines_counts": bool(beats_counts),
        "margin_vs_best_baseline_signed": float(margin),
        "verdict": verdict,
        "caveat": ("Only %d usable days -> ~%d one-step transitions; DMD spectra "
                   "and out-of-sample MSE are high-variance and indicative only."
                   % (len(days), len(days) - 1)),
    }
    with open(os.path.join(MET_DIR, "koopman.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # ---- Figure ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    th = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(th), np.sin(th), "k--", lw=1, alpha=0.6)
    ax1.scatter(evals_s.real, evals_s.imag, c=np.abs(evals_s),
                cmap="viridis", s=80, edgecolor="k", zorder=3)
    ax1.axhline(0, color="gray", lw=0.5)
    ax1.axvline(0, color="gray", lw=0.5)
    ax1.set_title("DMD eigenvalues (signed activity) in unit disk")
    ax1.set_xlabel("Re(lambda)")
    ax1.set_ylabel("Im(lambda)")
    ax1.set_aspect("equal")

    labels = ["koopman", "persistence", "mean"]
    vals = [pred_signed["koopman_mse"], pred_signed["persistence_mse"],
            pred_signed["mean_mse"]]
    colors = ["#2a9d8f", "#e76f51", "#8888aa"]
    ax2.bar(labels, vals, color=colors)
    ax2.set_title("One-step-ahead MSE (held-out days, signed)")
    ax2.set_ylabel("MSE")
    for i, v in enumerate(vals):
        ax2.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"Koopman/DMD on news entity-state  |  VERDICT: {verdict}",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(os.path.join(FIG_DIR, "koopman.png"), dpi=120)

    # ---- Summary (<=4 lines) ----
    m0 = modes_s[0]
    ents0 = ", ".join(e["entity"] for e in m0["dominant_entities"][:3])
    print("Dominant mode |lambda|=%.3f (period=%s) driven by: %s" % (
        m0["abs_lambda"],
        ("%.1fd" % m0["period_days"]) if m0["period_days"] else "non-osc",
        ents0))
    print("Predictive (signed, held-out %d days): koopman MSE=%.4f vs "
          "persistence=%.4f vs mean=%.4f" % (
              pred_signed["n_test_pairs"], pred_signed["koopman_mse"],
              pred_signed["persistence_mse"], pred_signed["mean_mse"]))
    print("Koopman beats both baselines: %s (margin %.1f%% vs best); n=%d days "
          "-> results indicative only." % (
              beats, 100 * margin, len(days)))
    print("VERDICT: %s" % verdict)


if __name__ == "__main__":
    main()
