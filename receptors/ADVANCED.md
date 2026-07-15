# Advanced directions — grounded in a probing-ladder diagnostic

**Date:** 2026-07-15 · Follow-on to [RECEPTOR_RESULTS.md](RECEPTOR_RESULTS.md).
Question the user asked: *far more advanced ways to explore this better.* Rather
than guess, we first measured **where** advanced methods can actually pay off,
then built the agenda around the answer.

## The diagnostic (do this before building anything)

`scripts/probe_ladder.py` + `scripts/probe_temporal.py`: for each task, compare a
**linear probe** vs **MLP / gradient-boosting** vs **model-free mutual
information**, held-out — and for the two live candidates, under a **temporal**
split. This decides, per signal, whether to (a) stop, (b) add model capacity, or
(c) fix the representation. Figure: `data/figures/diagnostic.png`.

| task | linear | best nonlinear | temporal? | verdict |
|---|---|---|---|---|
| direction (pair ordering) | AUC 0.874 | **GB 0.958** | **+0.084 AUC survives** (0.698→0.782) | **NONLINEAR HEADROOM, transferable** |
| impact (\|zscore\|) | ρ 0.36 (random) | GB 0.37 | **collapses temporally** (ρ −0.10) | **NON-STATIONARY, not signal-free** |
| polarity | AUC 0.945 | 0.951 | — | linear-saturated |
| predicate (11-way) | 0.66 | 0.68 | — | linear-saturated |
| corroboration | AUC 0.79 | MLP 0.71 (worse) | — | linear-saturated (accuracy hidden by 87% imbalance) |

**Two findings that reset the priorities:**

1. **The flagship direction task is under-powered by its linear operator.** A
   nonlinear model beats ridge transport by **+8 AUC points even on a temporal
   split** — this is real, forward-generalizing headroom, not memorization. The
   single highest-ROI advanced move is a better *direction* model/representation.
2. **The impact "negative" was non-stationarity, not absence.** Text carries
   contemporaneous impact (ρ≈0.36 within-period) that **does not transfer forward**
   — what moves markets rotates week to week. More capacity doesn't help (GB no
   better temporally); this needs *online / regime-aware* modeling or *exogenous*
   (price-context) features, not a bigger static probe.

Everything else (polarity, predicate, corroboration) is **linear-saturated** —
the receptors already capture it; don't spend model budget there.

---

## Four frontiers (prioritized by the diagnostic)

### Frontier 1 — Representation learning  ★ diagnostic-confirmed, highest ROI
The wins so far are all *linear maps over frozen MiniLM*. Direction proves the
representation/operator is the bottleneck, not the signal.

- **1a. Nonlinear direction operator (ship now).** Replace ridge `W` with a shallow
  MLP head on the interaction features `[e_a, e_b, e_a−e_b, e_a⊙e_b]`. Two matmuls
  + ReLU → **exportable to the existing Rust BLAS path** (train the head in Python,
  ship weights; still no LLM at inference). Measured payoff: temporal AUC
  0.698→0.782. This is the concrete next build.
- **1b. Two-tower contrastive fine-tune.** Train cause-tower `f_c` and effect-tower
  `f_e` (LoRA/adapter on MiniLM, or MLP projection heads) with InfoNCE/triplet on
  the mined cause→effect pairs. Asymmetry then comes from *two towers*, not one `W`
  — a principled, higher-capacity transport. Directional embedding space also lifts
  first-stage recall (the standing bottleneck) where the linear metric (B5) failed.
- **1c. Sparse autoencoder over the embeddings.** The atlas but *learned*: an
  overcomplete SAE yields a dictionary of monosemantic concept directions; read
  which concepts the corpus actually uses instead of 6 hand-picked axes. The
  advanced way to *explore the embedding geometry itself*.

### Frontier 2 — Dynamics / operator-theoretic
Treat the corpus as a dynamical system, not a bag of pairs.

- **2a. Koopman / DMD on daily entity-state vectors.** Eigenmodes = coherent causal
  modes with a decay rate (persistence) and phase (lead-lag). Unifies the spectral
  reading + latency (B1) into one object with generalization guarantees.
- **2b. Hawkes / neural point process on the fact event stream.** A cause raises the
  intensity of later effects; the learned kernel jointly gives **transmission
  latency (B1) + magnitude + which-effects** as one generative model — the natural
  home for the temporal structure impact needs.

### Frontier 3 — Causal inference proper (beyond associational "direction")
Everything so far is associational. Real causal tooling:

- **3a. PCMCI / transfer entropy** on entity time series (built from the fact stream
  + prices): nonlinear, directional, **lagged**, FDR-controlled causal graph.
  State-of-the-art for "which entity drives which, at what lag" — upgrades
  direction + latency + confounder with statistical rigor and causal (not
  associational) semantics. (`tigramite` for PCMCI.)
- **3b. KG embeddings — RotatE / ComplEx** on `(cause, predicate, effect)` triples.
  Relations as rotations in complex space: asymmetric *and composable*, so multi-hop
  chains fall out of composition. A principled upgrade of the mechanism tensor `{W_r}`.
- **3c. Confounder, done right.** Backdoor adjustment / do-calculus on the entity
  graph; instrumental variables using exogenous shocks (a geopolitical event as an
  instrument for supply→price) — turns B3 from a probe into an identification strategy.

### Frontier 4 — Geometry, information theory, uncertainty (how to *explore better*)
- **4a. The probing ladder as a standing gate.** Run `probe_ladder.py` on every new
  label *before* building a receptor — linear vs nonlinear vs MI decides the
  investment. (This diagnostic is the reusable meta-tool.)
- **4b. Manifold / anisotropy analysis.** Intrinsic dimension (TwoNN), whitening/ISO
  to fix the measured 18% top-direction anisotropy before any operator; persistent
  homology to test whether causal signal lives on a low-d submanifold.
- **4c. Multimodal contrastive (news ↔ market).** CLIP-style alignment of a news
  embedding with the realized cross-asset move vector — *learns* the impact/spillover
  space instead of ridge-mapping into it. Given impact is partly a representation
  mismatch, this is the right tool for B2/A3 (paired with online adaptation for 2).
- **4d. Uncertainty + active learning.** Conformal prediction sets for direction
  (calibrated abstention); ensemble/Bayesian operators for per-edge epistemic
  uncertainty; an active loop that picks which pairs to LLM-verify to most improve
  the operator — turns noisy free labels into a principled budget.

---

## Recommended sequence

1. **Build 1a now** — nonlinear direction head, Python-trained / Rust-served. Cheap,
   diagnostic-confirmed +8 temporal AUC, preserves the no-LLM-at-inference contract.
2. **Then 1b (two-tower)** — the productionizable representation; also attacks the
   first-stage-recall ceiling that killed the linear metric.
3. **Then 3a (PCMCI/transfer entropy)** — the rigorous causal-graph upgrade; reuses
   the entity time series 2a/2b also need.
4. **Fold impact into an online/regime frame (2b or 4c)** rather than a static probe.
5. Keep polarity/predicate/corroboration as the linear receptors they already are.

Everything above is measured-motivated: the diagnostic said *where* the signal is
linear, nonlinear-but-real, non-stationary, or absent — so effort goes where the
numbers say it will pay, not where the method is fashionable.
