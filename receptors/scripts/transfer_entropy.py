# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy","scikit-learn","matplotlib"]
# ///
"""
Transfer-entropy / linear-Granger experiment on daily signed-activity series.

Independent, time-series evidence of DIRECTED causality between entities, and a
corroboration test against the embedding-mined (cause_entities) direction.

Reads  ./data/facts.jsonl, ./data/docs.jsonl
Writes ./data/metrics/transfer_entropy.json, ./data/figures/transfer_entropy.png
"""
import json
import os
from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # receptors/
DATA = os.path.join(ROOT, "data")
METRICS = os.path.join(DATA, "metrics")
FIGURES = os.path.join(DATA, "figures")
os.makedirs(METRICS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

TOP_N = 40          # top entities by frequency
TOP_EDGES = 10      # directed edges to report
MIN_ACTIVE_DAYS = 3 # entity must be active on at least this many days


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ----------------------------------------------------------------------------
# 1. Load facts, pick top entities, build daily signed-activity series
# ----------------------------------------------------------------------------
facts = [d for d in load_jsonl(os.path.join(DATA, "facts.jsonl")) if d.get("date")]

ent_freq = Counter()
for f in facts:
    for e in f.get("entities") or []:
        ent_freq[e] += 1

top_entities = [e for e, _ in ent_freq.most_common(TOP_N)]
ent_idx = {e: i for i, e in enumerate(top_entities)}

days = sorted({f["date"] for f in facts})
day_idx = {d: i for i, d in enumerate(days)}
T = len(days)
E = len(top_entities)

# series[e, t] = sum of direction over facts on day t mentioning entity e
series = np.zeros((E, T), dtype=float)
counts = np.zeros((E, T), dtype=float)
for f in facts:
    t = day_idx[f["date"]]
    dr = float(f.get("direction") or 0.0)
    for e in f.get("entities") or []:
        j = ent_idx.get(e)
        if j is not None:
            series[j, t] += dr
            counts[j, t] += 1.0

active_days = (counts > 0).sum(axis=1)
keep = active_days >= MIN_ACTIVE_DAYS
kept_entities = [top_entities[i] for i in range(E) if keep[i]]
series = series[keep]
E = len(kept_entities)
ent_idx = {e: i for i, e in enumerate(kept_entities)}


# ----------------------------------------------------------------------------
# 2. Directed predictability: linear Granger, lag 1
#    Compare residual var of  Y_t ~ Y_{t-1}  vs  Y_t ~ Y_{t-1} + X_{t-1}
#    TE-like score = log(var_reduced / var_full)  (>=0, larger = more directed info)
# ----------------------------------------------------------------------------
def ols_resid_var(y, X):
    """Residual variance of OLS y ~ [1, X]. X is (n, k)."""
    A = np.hstack([np.ones((X.shape[0], 1)), X])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(np.mean(resid**2))


def zscore(v):
    s = v.std()
    return (v - v.mean()) / s if s > 1e-9 else v - v.mean()


# standardize each series so scores are comparable across entities
Z = np.vstack([zscore(series[i]) for i in range(E)])

Y_t = Z[:, 1:]      # (E, T-1)  target at time t
Y_lag = Z[:, :-1]   # (E, T-1)  lag-1

edges = []           # (X, Y, te_score, F_stat)
n = T - 1            # number of usable observations per regression
for yi in range(E):
    y = Y_t[yi]
    ylag = Y_lag[yi].reshape(-1, 1)
    var_reduced = ols_resid_var(y, ylag)
    for xi in range(E):
        if xi == yi:
            continue
        xlag = Y_lag[xi].reshape(-1, 1)
        X_full = np.hstack([ylag, xlag])
        var_full = ols_resid_var(y, X_full)
        var_full = max(var_full, 1e-12)
        te = float(np.log(var_reduced / var_full))  # >=0 when X helps
        # F-stat for adding 1 regressor: df1=1, df2=n-3
        df2 = n - 3
        if df2 > 0 and var_full > 1e-12:
            fstat = ((var_reduced - var_full) / 1.0) / (var_full / df2)
        else:
            fstat = 0.0
        edges.append((kept_entities[xi], kept_entities[yi], max(te, 0.0), max(fstat, 0.0)))

edges.sort(key=lambda z: z[2], reverse=True)
top_edges = edges[:TOP_EDGES]

# Symmetric lag-0 correlation baseline (undirected control)
corr = np.corrcoef(Z)
sym_pairs = []
for i in range(E):
    for j in range(i + 1, E):
        sym_pairs.append((kept_entities[i], kept_entities[j], float(corr[i, j])))
sym_pairs.sort(key=lambda z: abs(z[2]), reverse=True)

# TE score lookup for every ordered pair (directed) and correlation lookup
te_map = {(x, y): te for (x, y, te, f) in edges}
corr_map = {}
for i in range(E):
    for j in range(E):
        if i != j:
            corr_map[(kept_entities[i], kept_entities[j])] = float(corr[i, j])


# ----------------------------------------------------------------------------
# 3. Mine directed pairs from cause_entities (the "embedding-mined" system proxy)
#    cause_entity -> effect_entity: an entity that appears in one doc's `entities`
#    and later shows up as a `cause_entity` of another doc is treated as a driver.
#    Concretely: for each doc, every cause_entity C -> every entity E in that same
#    doc is a mined directed pair (C drives E), consistent with the causal-graph
#    construction used by the embedding pipeline.
# ----------------------------------------------------------------------------
docs = list(load_jsonl(os.path.join(DATA, "docs.jsonl")))
mined = Counter()  # (cause, effect) -> weight
for d in docs:
    ces = d.get("cause_entities") or []
    ents = d.get("entities") or []
    for c in ces:
        for e in ents:
            if c == e:
                continue
            mined[(c, e)] += 1

# restrict mined pairs to those whose BOTH endpoints are in our time-series universe
uni = set(kept_entities)
mined_uni = {(c, e): w for (c, e), w in mined.items() if c in uni and e in uni}


# ----------------------------------------------------------------------------
# 4. Corroboration test
#    For each mined directed pair (C->E) present in both systems, does Granger
#    prefer the mined direction?  Compare TE(C->E) vs TE(E->C).
#    Control: does the (symmetric) correlation "agree"? Correlation has no
#    direction, so as a control we ask whether |corr| alone would predict the
#    mined direction — which is chance by construction (50%). We instead build a
#    directional control by randomizing direction (bootstrap 50%).
# ----------------------------------------------------------------------------
overlap = []
granger_agree = 0
tie = 0
for (c, e), w in mined_uni.items():
    if (e, c) not in mined_uni or mined_uni[(e, c)] < w:
        # treat as a net directed pair only if forward weight dominates
        te_fwd = te_map.get((c, e), 0.0)
        te_rev = te_map.get((e, c), 0.0)
        if te_fwd == te_rev:
            tie += 1
            direction_ok = None
        else:
            direction_ok = te_fwd > te_rev
            if direction_ok:
                granger_agree += 1
        overlap.append(
            {
                "cause": c,
                "effect": e,
                "mined_weight": int(w),
                "te_forward": round(te_fwd, 4),
                "te_reverse": round(te_rev, 4),
                "corr": round(corr_map.get((c, e), 0.0), 4),
                "granger_agrees": direction_ok,
            }
        )

decided = [o for o in overlap if o["granger_agrees"] is not None]
n_overlap = len(decided)
agreement = (granger_agree / n_overlap) if n_overlap else float("nan")

# Symmetric-correlation control: correlation is undirected, so the best a
# direction-blind predictor can do is chance. We estimate it via a large random
# assignment to give an empirical control baseline.
rng = np.random.default_rng(0)
ctrl_trials = 2000
ctrl_scores = []
for _ in range(ctrl_trials):
    hits = rng.integers(0, 2, size=max(n_overlap, 1)).sum()
    ctrl_scores.append(hits / max(n_overlap, 1))
ctrl_mean = float(np.mean(ctrl_scores))
ctrl_p975 = float(np.quantile(ctrl_scores, 0.975)) if n_overlap else float("nan")

# Binomial one-sided p-value that agreement > 0.5 (normal approx)
if n_overlap > 0:
    se = np.sqrt(0.25 / n_overlap)
    z = (agreement - 0.5) / se if se > 0 else 0.0
    from math import erf, sqrt

    p_value = 0.5 * (1 - erf(z / sqrt(2)))  # one-sided P(agree>0.5)
else:
    z, p_value = float("nan"), float("nan")


# ----------------------------------------------------------------------------
# 5. Verdict
# ----------------------------------------------------------------------------
# NOTE: with only ~15 days, the single top F-stat over 40*39 ordered pairs is
# expected to be large under the null (multiple comparisons). A stricter marker of
# genuine directed structure is whether MANY edges clear a Bonferroni-ish bar.
n_tests = E * (E - 1)
# Conservative per-test bar to counter multiple comparisons: F(1, df2) grows
# roughly with -2*log(alpha/n_tests). For df2~11 and n_tests~1560 a Bonferroni
# 5% F-crit is very large (~30+); use a fixed conservative bar.
bonf_bar = 25.0
strong_edges = sum(1 for (_x, _y, _te, f) in edges if f > bonf_bar)
directed_signal = bool(strong_edges >= max(3, 0.01 * n_tests))

# The corroboration test is the PRIMARY question: does time-series direction match
# the embedding-mined direction beyond chance AND beyond the symmetric control?
corroborates = bool(
    n_overlap >= 5 and agreement > 0.5 and agreement > ctrl_p975 and p_value < 0.05
)

if corroborates:
    verdict = "SIGNAL"
elif n_overlap >= 5 and agreement > 0.5 and directed_signal:
    verdict = "WEAK"
else:
    # Primary corroboration test is null (agreement ~ chance) -> no directed signal
    # that survives the embedding cross-check, regardless of raw per-pair F-stats.
    verdict = "NO-SIGNAL"


# ----------------------------------------------------------------------------
# 6. Emit JSON
# ----------------------------------------------------------------------------
metrics = {
    "n_days": T,
    "date_range": [days[0], days[-1]] if days else [],
    "n_entities_universe": E,
    "top_entities": kept_entities,
    "top_directed_edges": [
        {"x": x, "y": y, "te_score": round(te, 4), "f_stat": round(f, 3)}
        for (x, y, te, f) in top_edges
    ],
    "top_symmetric_pairs": [
        {"a": a, "b": b, "corr": round(c, 4)} for (a, b, c) in sym_pairs[:TOP_EDGES]
    ],
    "corroboration": {
        "n_mined_pairs_universe": len(mined_uni),
        "n_overlapping_decided": n_overlap,
        "n_ties": tie,
        "granger_agreement_pct": round(agreement * 100, 1) if n_overlap else None,
        "chance_pct": 50.0,
        "control_symmetric_mean_pct": round(ctrl_mean * 100, 1),
        "control_symmetric_p975_pct": round(ctrl_p975 * 100, 1) if n_overlap else None,
        "z": round(z, 3) if n_overlap else None,
        "p_value_one_sided": round(p_value, 4) if n_overlap else None,
        "overlap_detail": sorted(
            overlap, key=lambda o: o["mined_weight"], reverse=True
        )[:20],
    },
    "directed_signal": directed_signal,
    "corroborates_mined_direction": corroborates,
    "verdict": verdict,
    "caveat": f"Only {T} days of daily data; Granger estimates are noisy and directional conclusions are indicative, not definitive.",
}

with open(os.path.join(METRICS, "transfer_entropy.json"), "w") as f:
    json.dump(metrics, f, indent=2)


# ----------------------------------------------------------------------------
# 7. Figure
# ----------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

labels = [f"{x} → {y}" for (x, y, te, f) in top_edges][::-1]
scores = [te for (x, y, te, f) in top_edges][::-1]
ax1.barh(labels, scores, color="#2b8cbe")
ax1.set_xlabel("TE-like score  log(var_reduced / var_full)")
ax1.set_title(f"Top {len(top_edges)} directed edges (linear Granger, lag 1)")
ax1.grid(axis="x", alpha=0.3)

if n_overlap:
    bars = ["Granger\nagreement", "Chance", "Sym-corr\ncontrol p97.5"]
    vals = [agreement * 100, 50.0, ctrl_p975 * 100]
    colors = ["#238b45", "#999999", "#cccccc"]
    ax2.bar(bars, vals, color=colors)
    ax2.axhline(50, color="red", ls="--", lw=1, label="50% chance")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% agreement with mined direction")
    ax2.set_title(f"Corroboration vs embedding-mined direction (n={n_overlap})")
    for i, v in enumerate(vals):
        ax2.text(i, v + 1.5, f"{v:.0f}%", ha="center")
    ax2.legend()
else:
    ax2.text(0.5, 0.5, "No overlapping pairs", ha="center", va="center")
    ax2.set_axis_off()

fig.suptitle(
    f"Transfer entropy / directed causality  —  {T} days "
    f"({days[0]}…{days[-1]})  —  VERDICT: {verdict}",
    fontsize=12,
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(os.path.join(FIGURES, "transfer_entropy.png"), dpi=110)


# ----------------------------------------------------------------------------
# 8. Summary (<=4 lines, ends with VERDICT)
# ----------------------------------------------------------------------------
top = top_edges[0] if top_edges else ("-", "-", 0, 0)
print(
    f"Directed time-series signal: top edge {top[0]}→{top[1]} "
    f"TE={top[2]:.3f} F={top[3]:.1f} over {T} days, {E} entities."
)
if n_overlap:
    print(
        f"Corroboration: {n_overlap} overlapping mined pairs, Granger agrees with "
        f"mined direction {agreement*100:.0f}% (chance 50%, ctrl p97.5={ctrl_p975*100:.0f}%, p={p_value:.3f})."
    )
else:
    print("Corroboration: no overlapping mined pairs in the time-series universe.")
print(f"Caveat: only {T} days — directional estimates are noisy/indicative.")
print(f"VERDICT: {verdict}")
