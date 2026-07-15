# SIGNALS.md — the signal hunt

**Date:** 2026-07-15 · Follow-on to [ADVANCED.md](ADVANCED.md). We worked through
every frontier in the advanced agenda (plus extra controls) as **9 held-out
experiments**, each with an explicit baseline, a **temporal split** wherever the
label carries time, a metrics JSON, a figure, and an honest verdict. Master figure:
`data/figures/signal_map.png`; per-experiment figures in `data/figures/`.

## Signal catalog

| verdict | experiment | headline (held-out) | baseline it had to beat |
|---|---|---|---|
| ✅ SIGNAL | **direction, nonlinear** | temporal AUC 0.696→**0.782** (GB); signal is the *difference vector* e_a−e_b (0.716), product is noise (0.500); recency-control 0.561 | linear transport 0.696 / recency artifact |
| ✅ SIGNAL | **direction, abstention** | selective accuracy **0.636→0.814 @20% coverage** (0.845 @10%) | full-coverage 0.636 |
| 🟠 WEAK | two-tower contrastive | asymmetry beats W on direction (0.572 vs 0.526) **but degrades retrieval** (R@10 0.070 vs cosine 0.127) | ridge W / cosine |
| 🟠 WEAK | sparse autoencoder | recon R²=0.68; **monosemantic concept-neurons emerge** (metals/gold 0.79, markets-falling 0.92) but desk decodability −14 pt | hand-picked atlas / raw dims |
| 🟠 WEAK | Koopman / DMD | entity **attention-volume** has predictable linear dynamics (beats persistence on counts); **signed direction does not** | persistence + mean |
| 🔴 NON-STAT | news↔market CLIP / impact | impact-magnitude Spearman **−0.36 (inverts out-of-sample)**; spillover 0.158 < naive 0.499 | ridge map / naive cosine |
| ⚫ NO-SIGNAL | transfer-entropy / Granger | Granger direction agrees with embedding direction **48%** (≤ chance) | 50% chance / correlation |
| ✅ SIGNAL | **KGE directionality** | asymmetric **RotatE Hits@10 0.348 vs symmetric DistMult 0.266** on 49k causal triples — the entity graph is genuinely directional | symmetric DistMult |
| ⚫ NO-SIGNAL | geometry (whiten/low-d) | whitening neutral-to-harmful; de-gist hurts; direction spread across dims (no low-d saturation) | raw linear probe |

![signal map](data/figures/signal_map.png)

---

## 1. What carries real, forward-generalizing signal

### Direction is the one robust, text-intrinsic signal — and it's genuinely causal
`direction_deep.py` is the decisive result. Three things, all on a **temporal** split:
- **Nonlinear headroom is real and transfers forward:** gradient boosting lifts
  direction AUC 0.696→**0.782** (+0.086) — the linear transport operator is
  leaving ~8 points on the table that survive train-early / test-late.
- **The signal is relational asymmetry, not a topic prior.** Feature ablation: the
  *difference* vector e_a−e_b carries almost all of it (GB 0.716); the Hadamard
  product is pure noise (0.500); the marginal e_a / e_b ("which topics tend to be
  causes") only matches, doesn't exceed. So the model is reading *how A relates to
  B*, not *what A is about*.
- **It is not a recency artifact.** Co-topical but non-causal pairs labelled by
  publication time-order reach only 0.561 — subtracting that leaves a **+0.22
  genuine causal component**. This is the control that separates signal from
  "detecting which article came first."

### Confidence abstention converts it into a high-precision tool
`uncertainty.py`: gating on prediction margin lifts direction accuracy from 0.636
(all) to **0.814 on the most-confident 20%** (0.845 @10%). Probabilities are already
near-calibrated (isotonic doesn't help). Active-learning by uncertainty **did not**
beat random sampling (margin sampling hoards ambiguous pairs on a noisy boundary).
So the deployable win is *selective prediction*, not active labelling.

### Partial wins worth keeping
- **Sparse autoencoder** (`sae.py`): a learned overcomplete dictionary surfaces
  **monosemantic concept-neurons** — a "markets-falling" neuron at 0.92 purity, a
  "gold/metals" neuron, an "asia/China-growth" neuron — the atlas but *discovered*,
  not hand-picked. Recon R²=0.68 at ~34 active features/fact. It doesn't strictly
  dominate raw decodability (desk −14 pt), so it's a complementary interpretability
  layer, not a replacement.
- **Two-tower** (`twotower.py`): the two-tower asymmetry beats the single linear W
  on direction (0.572 vs 0.526) — confirming asymmetry is worth modelling — but the
  trained space *degrades* first-stage retrieval below raw cosine. Capture the
  asymmetry (see §4), don't rewrite the retrieval representation.
- **Koopman/DMD** (`koopman.py`): entity **attention/volume** has a predictable
  linear transfer operator (beats persistence on counts); the *signed direction* of
  news does not. A real but narrow "attention dynamics" signal.

### The entity causal graph is genuinely directional
`kge.py`: on 49k mined `(cause, predicate, effect)` triples, the **asymmetric**
RotatE beats the **symmetric** DistMult on link prediction (Hits@10 0.348 vs 0.266;
MRR 0.134 vs 0.124). Absolute link-prediction is hard (MRR ~0.13, a 4,280-entity
graph), but the RotatE > DistMult gap is exactly the test that matters: it confirms
the causal structure is directional, not co-occurrence — a second, independent
corroboration of the direction finding, this time at the entity (not document) level.
RotatE relations also compose, giving multi-hop chains for free — the principled
upgrade path for the mechanism tensor `{W_r}`.

---

## 2. The one thing that kills every market-coupled signal: corpus length

The most important cross-cutting finding. **Every** experiment that couples text to
market or time collapses the same way, and they all hit the same wall — **~15 days /
9 move-dates** of corpus:
- Impact magnitude **inverts** out-of-sample (ρ=−0.36, p=5e-10) — a learned joint
  space makes it *worse*, not better (`news_market_clip.py`).
- Spillover is not rescued (0.158 < naive 0.499).
- Transfer-entropy / Granger finds **no** directed time-series structure that
  corroborates the embedding direction (48% agreement, 15 days).
- Koopman's signed-direction dynamics are unpredictable (15 days).
- Regime (from the earlier suite) had only 9 usable overlapping days.

This is **non-stationarity from a tiny time window**, not a modeling-capacity
problem — gradient boosting, contrastive learning, and Koopman all fail it
identically. *What moves markets rotates faster than 15 days of data can pin down.*
The binding constraint on the whole market-signal program is **corpus duration**,
and no method fixes that.

## 3. What does NOT help (stop spending here)
- **Geometry fixes** — whitening is neutral-to-harmful, de-gisting destroys fact-level
  signal, and although the facts live on a ~9.5-dim manifold the direction signal is
  spread across many dimensions, so low-d projection only loses accuracy.
- **Active learning** by uncertainty (worse than random here).
- **Full retrieval-tower rewrite** (degrades the cosine structure it's built on).
- **Polarity / predicate / corroboration capacity** — linear-saturated (from the
  diagnostic); more model is wasted there.

---

## 4. Recommendations (what to build, what to shelve)

**Build now (text-intrinsic, deployable, no-LLM-at-inference):**
1. **Nonlinear direction head** — features `[e_a, e_b, e_a−e_b]` (drop the noise
   product) into a shallow MLP or small GBM; train in Python, **serve in Rust** as a
   few matmuls. +8.6 temporal AUC over the linear operator, verified causal & not a
   recency artifact. This upgrades the flagship receptor and the mechanism tensor.
2. **Confidence abstention** on top of it — ship the 0.81@20%-coverage high-precision
   mode for anywhere direction gates a decision.
3. **SAE concept dictionary** as a learned, monosemantic companion to the atlas.

**Shelve until the corpus is longer (months–quarters), then revisit:**
- Impact, spillover, regime, Koopman signed-dynamics, transfer-entropy/PCMCI,
  news↔market CLIP. All are *non-stationarity-blocked*, not method-blocked. The
  single highest-leverage action for the entire market-signal program is **let the
  cron crawler accumulate a multi-month corpus**, then re-run this exact battery.

**Meta-tool:** keep the probing-ladder + temporal-split as a standing gate — run it
on any new label *before* building a receptor, to classify it linear-saturated /
nonlinear-headroom / non-stationary / no-signal and spend effort accordingly.

## Honest caveats
15-day corpus (2026-06-27→07-11); free LLM-extracted labels (associational, not
price-validated); direction is a mined-pair proxy. The direction and abstention wins
are robust to temporal split and a recency control; the market negatives are
sample-starved, so they are "no signal *at this corpus size*," not proofs of absence.
