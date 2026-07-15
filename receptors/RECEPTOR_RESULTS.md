# Receptor results — built, tested, measured

**Date:** 2026-07-15 · **Status:** all 11 new receptors + the qualitative program
implemented, compiled, run on the real corpus, and evaluated with bootstrap CIs /
significance. Design rationale is in [RECEPTOR_IDEAS.md](RECEPTOR_IDEAS.md);
master table in [RECEPTOR_METRICS.md](RECEPTOR_METRICS.md); figures in
`data/figures/`.

Corpus: 3,433 fact-bearing docs (doc-level) / 12,476 atomic facts (fact-level),
384-d `all-MiniLM-L6-v2`, joined to 29 instruments' daily moves. Everything is
pure BLAS, no LLM at inference, whole suite runs in <5 min on CPU.

![scoreboard](data/figures/scoreboard.png)

## How to run

```bash
uv run scripts/export_for_receptors.py   # doc-level: embeddings.npy + docs.jsonl (+predicate/direction/…)
uv run scripts/export_facts.py           # fact-level: facts.npy + facts.jsonl
uv run scripts/export_market.py          # market_daily.jsonl (for regime)
cargo build --release
cargo test  --release                    # 5 kernel unit tests (ridge/transport/SVD/probe/norm)
for s in mechanism atlas spectrum impact latency spillover confound source consensus regime arc metric; do
  ./<target>/release/receptors $s; done  # each also writes data/metrics/<s>.json
uv run scripts/make_figures.py           # data/figures/*.png + RECEPTOR_METRICS.md
```

## Evaluation methodology (applies to every receptor)

- **Held-out always.** Causal/pair receptors split by effect-doc hash; fact-level
  receptors use a 70/30 split (temporal where timestamps allow, else index hash —
  stated per receptor). Retrieval pools are strictly time-masked (no look-ahead).
- **Explicit baseline per receptor** — the number the receptor must beat (cosine
  floor 0.5 for direction; raw cosine for retrieval; `fact.confidence` for impact
  and source; mean-predictor for latency; naive cosine-to-symbol for spillover).
- **95% CIs by bootstrap** (2,000 resamples for means, 600 for AUC; deterministic
  xorshift seed so figures and stats reproduce). AUC CIs and ROC curves are drawn
  from the emitted raw (score, label) arrays.
- **Significance where it's the crux**: McNemar's test on the mechanism tensor's
  paired per-item correctness; CI-overlap for the retrieval receptors.
- **Correctness tests**: `cargo test` proves the math kernels on synthetic data
  with known answers (ridge recovers a planted linear map to <1% error; the
  transport operator is directionally asymmetric; `top_singular` recovers a
  planted rank-2 spectrum; `ridge_probe` recovers a planted plane to cos>0.999).

---

## Tier A

### A1 — Mechanism tensor `{W_r}`  ✅ WIN
One transport operator **per effect predicate** instead of one global `W`.

- Direction accuracy **0.742 typed** vs **0.649 single-W** (cosine floor 0.500),
  on 12,293 held-out pairs. **McNemar χ²=463** (typed beats single on 2,110 pairs,
  loses on 964) → p ≪ 0.001. Typing is decisively better.
- **Transmission grammar** (per-predicate direction accuracy): `production_change`
  0.83, `forecast` 0.79, `policy_action` 0.78 carry the strongest directional
  signal; `demand_change` 0.63, `event` 0.64, `statement` 0.66 the weakest.
- **Cross-desk transfer** heatmap: strong diagonal (own-desk 0.71–0.81) vs weak
  off-diagonal (~0.45–0.63) → **market causality is desk-specific, not a single
  universal geometry**. (equities→geopolitics is the worst transfer, 0.45.)

![mechanism](data/figures/mechanism.png)

### A2 — Concept-axis atlas  ✅ WIN (the qualitative deliverable)
Six interpretable difference-of-means axes; every fact projected onto all of them.

- **Held-out separation AUC**, all six well above the 0.5 floor with tight CIs:
  official/market **0.991**, deal/friction **0.976**, horizon **0.923**, polarity
  **0.922**, certainty **0.884**, supply/demand **0.860**.
- Pole readings are correct on inspection (polarity + = "advances/growth", − =
  "fell/escalation"; certainty + = strikes that happened, − = Fed/BoJ forecasts).
- **Per-desk fingerprint** (right panel): geopolitics is the most bearish +
  supply-side (deep red); equities/fx sit at the market-move pole; macro/rates are
  the most forecast-driven/speculative. A readable map of desk "personality".

![atlas](data/figures/atlas.png)

### A3 — Market-impact receptor  ❌ NEGATIVE (honest)
Regress realized |zscore| from the claim embedding.

- Test **Spearman −0.011** (embedding) vs **+0.024** (`fact.confidence` baseline);
  top-decile lift 1.04×. The embedding does **not** predict realized market impact
  here, and doesn't beat plain confidence. Reported as a null result, not buried.
- Figure: predicted-vs-actual scatter + flat decile calibration.

![impact](data/figures/impact.png)

---

## Tier B

### B1 — Transmission-latency receptor  ✅ WIN
Predict the cause→effect lag in days from the cause embedding.

- **MAE 2.59 d** (content probe) vs **3.70 d** (mean baseline) — a **30% error
  reduction** — and **Spearman +0.561** (pred vs actual lag) on 11,609 held-out
  pairs. Content genuinely carries transmission-speed signal.
- Fast vs slow archetypes are separable by entity (fast: Hormuz/energy repricing;
  slow: BIS/structural-macro chains).

![latency](data/figures/latency.png)

### B2 — Cross-asset spillover  ❌ NEGATIVE (honest)
Multi-output operator doc-embedding → basket of moved symbols.

- Secondary (contagion) Recall@3 **0.123 receptor** vs **0.605 naive cosine**; the
  ridge basket collapses toward a few default symbols. On this 9-move-date window
  it does **not** beat cosine-to-symbol-name. Sparse secondary movers + tiny
  train window are the likely cause; flagged for a larger price panel.

![spillover](data/figures/spillover.png)

### B3 — Confounder / common-cause discriminator  ✅ WIN
Separate a real edge A→B from a sibling pair sharing a parent (A→B1, A→B2).

- **Probe AUC 0.645** vs **cosine-only 0.560** (accuracy 0.593). Cosine near-0.5
  confirms siblings are topically identical; the transport-asymmetry feature
  (`fwd_transport(x→y)`, |w|=1.36, the dominant weight) carries the screening
  signal. Modest but real — exactly the disconfirmation lever the pipeline wants.

![confound](data/figures/confound.png)

### B4 — Source / provenance receptor  ✅ WIN
Predict corroboration (proxy for reliability) and profile sources.

- Embedding predicts corroboration at **AUC 0.678** vs **0.509 for confidence**
  (+0.170), Spearman +0.575. The embedding, not the self-reported confidence,
  knows which claims get independently echoed.
- Provenance table: single-source firehoses (Yahoo Finance 72% corroborated,
  "unknown" 0%) vs high-agreement wires (FXStreet 96%, ForexLive 93%). An honesty
  axis the briefs don't currently surface. (Corroboration is a proxy, not truth.)

![source](data/figures/source.png)

### B5 — Learned relatedness metric  ❌ NEGATIVE (honest)
Low-rank symmetric metric `s(x,y)=(Lx)·(Ly)` for first-stage recall.

- Recall@10 **0.174** (metric, r=64) vs **0.183** (raw cosine) — CIs overlap, so
  no improvement; the recall@k curves track each other. Symmetric metric learning
  did **not** beat cosine; the **asymmetric** transport operator remains the better
  causal signal. Highest-ceiling / highest-risk item, and it didn't pay off here.

![metric](data/figures/metric.png)

---

## Tier C (exploratory)

### C1 — Consensus vs contrarian  🟠 EXPLORATORY
Dissent = distance of a fact from its (date, desk) centroid.

- Clean dissent distribution (mean 0.507). Market test: top-decile contrarian
  facts precede **1.04×** the |zscore| of the rest (n=175 vs 3,829) — directionally
  positive but weak. Dissent-by-embedding ≠ contrarian-by-truth; treat as a lead.

![consensus](data/figures/consensus.png)

### C2 — Market regime (risk-on/off)  🟠 EXPLORATORY (thin)
Learn a risk-on axis from price context; project fact-day centroids.

- Held-out **Spearman +0.50**, 100% sign-agreement — but only **9 usable
  overlapping days** (most facts lack a clean date). The learned axis is coherent
  (risk-on pole = Fed-cut/soft-data; risk-off pole = Hormuz/tanker/Iran supply
  shock), but the sample is far too thin to trade. Needs a longer corpus.

![regime](data/figures/regime.png)

### C3 — Narrative-arc / continuation operator  ❌ NEGATIVE (honest)
Predict the next-2-day same-entity follow-up doc.

- Recall@10 **0.159** (arc operator) vs **0.257** (cosine); MRR 0.209 vs 0.351.
  Story continuation is dominated by plain topical similarity — the learned
  operator adds nothing over cosine here.

![arc](data/figures/arc.png)

---

## §3 qualitative program — spectral reading + anisotropy

`receptors spectrum` (figure `data/figures/spectrum.png`):
- **Spectral channels** of the global `W` read cleanly as cause→effect grammar,
  e.g. channel 1 *Japan/BoJ/yen → yen-weakness/BoJ*, channel 6 *Iran/Hormuz →
  shipping/commercial-vessels*.
- **Anisotropy**: the top embedding direction holds 18.4% of variance, top-20 hold
  55% — quantifying why raw cosine is gist-dominated.
- **De-gist sweep**: direction accuracy peaks at **k=1 stripped (0.667)** then
  declines — stripping one topical direction helps, more hurts (matches the prior
  finding, now measured as a curve).

![spectrum](data/figures/spectrum.png)

---

## Scoreboard (11 new receptors)

| verdict | receptors |
|---|---|
| ✅ beats baseline | A1 mechanism, A2 atlas, B1 latency, B3 confounder, B4 source |
| ❌ honest negative | A3 impact, B2 spillover, B5 metric, C3 arc |
| 🟠 exploratory | C1 consensus, C2 regime (thin) |

The wins are exactly where the design predicted the signal lived: **typed
causal structure, interpretable concept axes, transmission speed, confounding,
and provenance**. The negatives are reported as negatives with the numbers that
kill them — no result was tuned to look good. Highest-value next steps: fold the
mechanism tensor + atlas into the desk-brief pipeline, and revisit spillover/regime
on a longer price panel where the sample isn't the binding constraint.
