# Receptor Ideas — new operators for extracting qualitative meaning from the embeddings

**Status:** research memo · 2026-07-15
**Scope:** what *other* receptors could prove useful, and how to turn the 384-d
embedding cloud into readable, qualitative structure. Grounded in the current
`facts.parquet` / `prices.parquet` / `docs.jsonl` schemas and the existing Rust core.

---

## 0. What a receptor actually is (and the two jobs it can do)

A receptor is a **learned bilinear form** `s(x, y) = ⟨x, M y⟩` (or a projection `P x`)
that isolates *one axis of meaning* the raw geometry buries. Cosine similarity has two
structural blind spots that every receptor here exploits:

1. **It is symmetric.** `cos(a,b) = cos(b,a)`, so it can never encode *direction*
   (cause vs effect, lead vs lag, escalation vs de-escalation). An **asymmetric**
   operator `W` breaks the symmetry and the asymmetry *is* the arrow. This is the
   whole transport-operator result (0.80 vs 0.50).
2. **It is topic-dominated.** ~1–3 leading singular directions carry generic "what
   this is about" gist; the interesting axes (polarity, certainty, mechanism) live in
   the residual and are drowned out. **De-gisting** / projection receptors recover them.

Every receptor below is one of a small **operator zoo** applied to a **free label**
(a column we already extracted, or a price join we already compute). Two distinct jobs:

- **Job 1 — retrieval lift.** Beat cosine on a task (find causes, rerank, classify).
  Measured by Recall@k / MRR / accuracy. (transport, price, polarity receptors.)
- **Job 2 — qualitative insight.** Turn 384 opaque dimensions into a *handful of
  readable dials* and *interpretable channels* so a human can read the corpus's
  structure directly. This is the "derive more meaning" ask and is under-served today.

The best new work sits in **Job 2** and in **typed extensions of Job 1**.

---

## 1. The operator zoo (the toolbox we can draw from)

| Form | Shape | Symmetric? | Learns | Best for |
|------|-------|-----------|--------|----------|
| Linear probe `w` | D | — (scalar out) | one interpretable score | polarity, confidence, impact |
| Difference-of-means axis | D | — | a single concept direction, ~free | concept-axis atlas (Job 2) |
| LDA / CCA subspace `P_k` | D×k | — | the k dims that separate groups | mechanism archetypes (Job 2) |
| Ridge transport `W` | D×D | **no** | asymmetric relation (direction) | causal find/rerank *(shipped)* |
| Low-rank `W = U Vᵀ` | D×r each | no | r interpretable cause→effect *channels* | spectral reading (Job 2) |
| PSD metric `M` (Mahalanobis) | D×D | **yes** | a task-tuned *replacement for cosine* | stage-1 recall (the real bottleneck) |
| Relational tensor `{W_r}` | R × D×D | no | one arrow *per predicate* | mechanism-typed causality |
| Projection / removal `I − P_k` | D×D | yes | strips gist / desk / topic | de-gisting, residual analysis (Job 2) |

Everything is BLAS-sized (384×384 dense, or 384×r), fits the existing `linalg.rs`
ridge + eigendecomposition machinery, and runs sub-second on CPU. No new inference-time
LLM cost — labels are mined once, offline.

**One required plumbing change:** `export_for_receptors.py` currently exports only
`entities` + `cause_entities`. Add `predicate`, `direction`, `confidence`, `magnitude`,
`source_access`, and the event-vs-published time delta to `docs.jsonl` (they already
exist in `facts.parquet`). That single export change unlocks Tiers A–B below.

---

## 2. New receptor candidates, ranked

Ranking = (product/insight value) × (cheapness) × (fit to existing infra) × novelty.

### TIER A — build these next

#### A1. Mechanism tensor `{W_r}` — one causal arrow *per predicate* ★ top pick
The current `W` collapses all causality into a single operator. But the fact schema
already types every fact by **`predicate`**: `policy_action | supply_change |
demand_change | production_change | sanction | event | price_move | deal_or_contract |
corporate_action | forecast | statement | other`. Fit a **separate transport operator
`W_r` per predicate** (RESCAL-style relational factorization) so the arrow "a
sanction *causes* a supply_change" is a different geometric map than "a forecast
*causes* a price_move."

- **Free label:** `(cause_predicate, effect_predicate)` on the 24.7k mined pairs.
- **Operator:** ridge per relation `W_r = (AᵣᵀAᵣ + λI)⁻¹ AᵣᵀBᵣ`; share a low-rank
  core across relations if a predicate is data-thin (`W_r = U diag(d_r) Vᵀ`).
- **Eval:** does typed direction accuracy beat the single-W 0.80? Does routing a query
  to the right `W_r` improve causal Recall@k?
- **Qualitative payoff (big):** you can now *read the transmission grammar of the
  market* — which mechanism types cause which. Rank the `‖W_r‖` / off-diagonal mass to
  see the dominant causal channels (e.g. sanction→supply strong, statement→price weak).
  This is the single most information-rich extension and reuses the exact ridge path.

#### A2. Concept-axis atlas — interpretable 1-D dials over the whole corpus ★ the "meaning" deliverable
The generalization of the polarity probe. Learn a **battery of single directions**,
each a difference-of-means (or LDA) axis between two free-labelled groups, then
**project every fact onto all of them** to get a low-dim, *human-readable coordinate
system*. Candidate axes, all from columns we already have:

| Axis | Poles | Free label source |
|------|-------|-------------------|
| Polarity | bullish ↔ bearish | `direction` *(have the probe)* |
| Certainty | hard fact ↔ speculation | `predicate ∈ {event,statement}` vs `{forecast}`; `confidence` |
| Novelty | first-mention ↔ restated | cross-corpus entity/claim match *(have novelty.rs)* |
| Time horizon | realized ↔ anticipated | event-time vs published-time delta; `forecast` predicate |
| Supply/demand | supply-side ↔ demand-side | `predicate ∈ {supply_change,production_change}` vs `{demand_change}` |
| Escalation | escalation ↔ de-escalation | `direction` within `geopolitics` desk facts |
| Official/market | policy action ↔ market reaction | `predicate ∈ {policy_action,sanction}` vs `{price_move}` |
| Magnitude | large ↔ small | `|zscore_20d|` on price-linked facts; `magnitude` field |

- **Operator:** each axis `u = (μ⁺ − μ⁻)/‖·‖` after de-gisting; project `p = x·u`.
- **Qualitative payoff:** turns the 384-d black box into ~8 dials. Now *any* fact,
  article, or desk brief carries a readable signature — "bearish, speculative, novel,
  supply-side, escalating." You can sort a brief's facts by any axis, colour the
  causal graph by mechanism, or diff two days along "escalation." Read the *extremes*
  of each axis (nearest facts at ±3σ) to name and sanity-check it. This directly
  answers "derive more qualitative insight from the embeddings."

#### A3. Market-impact (surprise) receptor — "how market-moving is this fact?"
The price receptor learns *which symbol* moved. A complementary probe learns
**how much** — regress `|zscore_20d|` (magnitude of the post-fact move) from the fact
embedding. Output = a scalar "expected market-moving-ness" for *any* fact, even ones
with no realized price join yet.

- **Free label:** join fact → symbol move within its desk window, target `|zscore|` (already computed in `prices.parquet`).
- **Operator:** ridge/Huber linear probe (robust to the fat tail); temporal split on move date.
- **Payoff:** rank each brief's facts by expected impact instead of by cosine — a
  materially better "what matters today" ordering, and an interpretable saliency score.

### TIER B — strong, slightly more work

#### B1. Transmission-latency (lead-lag) receptor
Predict the **lag in days** between cause and effect on the mined pairs (regression on
`effect.epoch − cause.epoch`). Splits causes into *fast-transmission* (same-day repricing)
vs *slow-transmission* (weeks). Qualitatively labels every causal edge with a speed;
product-wise flags "this driver hasn't fully propagated yet."

#### B2. Cross-asset spillover operator
Learn `W_spill : fact_embedding → basket weights` over the 29 instruments, trained on
which *other* assets moved after a fact (not just the primary one). Recovers contagion
structure (oil shock → FX of exporters → EM rates). A genuinely new signal — the
relation "spills into asset X" exists in *no single document*, only in the price panel.

#### B3. Confounder / common-cause discriminator
Serves the pitch-deck's "confound-check" narrative directly. Given `A→B`, decide
whether it's a real edge or an artefact of a shared upstream cause `C` (`A←C→B`). Learn
a probe on triangle features in the derived causal graph (does a common predecessor
explain both?). Turns the graph from "co-mentions" into "screened edges" and is exactly
the disconfirmation machinery the process narrative sells.

#### B4. Source-reliability / provenance receptor
`source_access` (open / headline / paywall) and source tier are free labels. Learn how
much to **discount** a claim by provenance and recalibrate `confidence`. Qualitatively:
separates "wire-confirmed" from "headline-only / single-source" facts — an honesty axis
the briefs currently don't surface.

#### B5. Mahalanobis relatedness metric `M` — attack the real bottleneck
Every eval says the ceiling is **stage-1 recall** (only 26% of gold causes reach the
cosine top-100). A symmetric PSD metric `M` learned so that causally-linked pairs are
*near* under `⟨x, M y⟩` (metric learning, not direction) could raise first-stage recall
without abandoning symmetric ANN indexing (factor `M = LᵀL`, index `Lx`). Higher-risk,
highest-ceiling: it improves *finding*, not just reordering.

### TIER C — research-y, do if Tiers A/B land

- **C1. Consensus vs contrarian receptor** — use corroboration counts; a fact far in
  embedding space from the day's consensus centroid but high-confidence = contrarian signal.
- **C2. Regime receptor** — project the day's briefs onto a risk-on/off axis learned
  from price context (VIX, DXY, yields) → a macro-state dial over the whole corpus.
- **C3. Narrative-arc operator** — asymmetric operator predicting *next-day follow-up*
  facts (story continuation), a temporal cousin of the causal transport.

---

## 3. The qualitative-insight program (deep dive on Job 2)

Beyond individual receptors, four analyses turn the embedding space itself into readable
structure — cheap, and the most direct answer to "derive more meaning."

1. **Spectral reading of the operators.** `W` is already computed. Take its SVD
   `W = Σ σ_k u_k v_kᵀ`. Each `(v_k → u_k)` pair is a **dominant cause→effect channel**:
   facts loading on input direction `v_k` map to effect direction `u_k`. Name the top ~10
   channels by their nearest facts. This reads the *latent causal grammar* straight out of
   the operator you already fit — near-zero cost, high insight. Do the same per `W_r`.

2. **Anisotropy / gist accounting.** Formalize de-gisting as variance accounting:
   report what fraction of embedding variance is topical gist vs the residual where
   causal/polarity structure lives. Quantifies *why* cosine underperforms and *how much*
   signal the residual holds. Sweep `RECEPTORS_K` and plot direction-accuracy vs
   variance-removed.

3. **Residual-space archetypes.** After removing gist + desk directions, cluster the
   residual. Clusters are **mechanism archetypes** ("supply-shock repricing",
   "central-bank-guidance", "geopolitical-risk-premium") independent of surface topic —
   a qualitative taxonomy the corpus reveals on its own.

4. **Cross-desk transfer geometry.** Fit `W` on energy pairs, test on rates. High
   transfer ⇒ market causality is a *universal* geometry; low ⇒ desk-specific mechanisms.
   Either answer is a publishable qualitative finding and tells you whether to ship one
   `W` or a mixture-of-desks.

---

## 4. How to validate (don't only chase AUROC)

- **Interpretability first for Job 2:** for every learned axis/channel, print the facts
  at both extremes and *name* it. An axis you can't name is overfit.
- **Temporal splits, always** (`RECEPTORS_SPLIT=temporal`) — the honest generalization
  number is the one that counts.
- **Calibration** for scalar receptors (impact, confidence) — reliability curves, not
  just accuracy.
- **Ablate against cosine-kNN and against the single-`W` baseline** — a typed/relational
  receptor must beat *both* its symmetric and its untyped parent to earn its complexity.
- **Reuse the free eval suite** (`eval/cases.jsonl`, 30 frozen cases) as a regression gate.

---

## 5. Recommended first build

Do **A1 (mechanism tensor `{W_r}`)** and **A2 (concept-axis atlas)** together, because
they share the same one-line export change and the same ridge/difference-of-means
machinery, and because they cover both jobs:

- A1 upgrades the *retrieval/causal* story (typed arrows, spectral channels).
- A2 delivers the *qualitative* story (readable dials over the whole corpus) the user
  is asking for — and it's the cheapest high-value thing in the whole list.

Then layer **A3 (impact)** to re-rank brief facts by expected market move, and run the
**spectral reading (§3.1)** on the operators you've now fit — that last step is almost
free and is where the "more meaning from the embeddings" really lands.

**Immediate next step if approved:** extend `export_for_receptors.py` to carry
`predicate/direction/confidence/source_access/time-delta`, add a `receptors mechanism`
and `receptors atlas` subcommand mirroring the existing `polarity`/`price` pattern, and
report typed-direction accuracy + a named axis atlas on the temporal split.
