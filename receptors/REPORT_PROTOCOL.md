# Compositional Report-Generation Protocol

A quality market-intelligence report is a *disciplined process*, not a sum of
retrievals. This spec formalises that process over the components we built
(topical + causal retrieval, direction/modality/novelty axes, HyDE hypotheses,
retrieval-validation gate) and defines how to assess whether the *process* — not
the components — is robust.

The claim of a report produced this way is **rigour and faithfulness**, not price
prediction. It should tell you what is known, how well it is known, and what is
not known — with citations and honest epistemic status.

---

## Architecture: an iterative research engine (not GraphRAG)

This is not retrieve-then-generate. It is a staged research loop where each stage
has an explicit job and can send control *backwards*:

```
      Question
         ↓
   Planning  ────────────────── objectives, sub-questions, success criteria   (Stage 1)
         ↓
   Hypothesis generation ─────── HyDE / causal traversal → candidate claims    (Stage 2)
         ↓
   Evidence search  ──────────── retrieve.py (topical | causal | --diverse)    (Stage 3)
         ↓
   Disconfirmation search ────── query the OPPOSITE; fill counter_cite         (Stage 3/4)
         ↓
   Confound analysis  ────────── is the effect explained by a rival driver?    (Stage 3/4)
         ↓                        (name it; don't credit the thesis for it)
   Coverage assessment ───────── did we span the facets & meet every objective?(Stage 3)
         ↓                        gaps re-enter Evidence search (the loop)
   Structured report ─────────── two-axis ledger + framing                     (Stage 4/5)
```

Two stages the prototype did only *informally*, now first-class:
- **Confound analysis** — before crediting a driver, search for the rival
  explanation. (Q3's Central-Asia fuel-price rise had an Iran-war supply confound;
  the report must attribute, not assume.) A claim whose effect is equally explained
  by a confound is capped in status until the confound is ruled out.
- **Coverage assessment** — a completeness critic: every objective mapped to
  evidence, facets spanned (`--diverse` redundancy low), and any unmet objective
  re-enters Evidence search rather than being silently dropped.

The stages below detail each box.

---

## Stage 1 — Objective framing (how to ask)
Reproject the query into an explicit research plan *before* retrieving:
- **Decision context**: what decision/brief does this inform?
- **Objectives**: 2–5 concrete things a complete answer must establish.
- **Sub-questions**: each objective → searchable sub-questions.
- **Success criteria**: what would make the report complete and defensible?
- **Scope/time-box**: as-of date (point-in-time), breadth vs depth.

Failure mode: answering a subtly *different* question than was asked.

## Stage 2 — Retrieval strategy (what to find)
Per sub-question, choose the mode and axes deliberately:
- **Mode**: topical (cosine) | causal (operator, upstream drivers) | hypothesis
  (HyDE) | metadata filter (source tier, desk, time).
- **Axes to impose**: direction (polarity), epistemic status (modality: prefer
  events/data over forecasts/opinion where facts are needed), distinctness
  (novelty — de-dup restatements).
- **Coverage (facet spread)** — a retrieval objective distinct from relevance,
  causality, novelty and time. Ten individually-relevant, individually-novel docs
  can all cover the *same* facet (e.g. all "Saudi price cuts") and leave the report
  blind to Hormuz shipping / Fed expectations / OPEC output / China demand. Impose
  coverage with `retrieve.py --diverse` (MMR): once a facet is represented its
  near-duplicates are penalised and a NEW facet surfaces. Measure it: intra-list
  redundancy (mean pairwise cosine) should drop. Relevance-only is the *wrong*
  default for a survey sub-question.
- Hybrid: fuse modes (cosine ∪ causal), then diversify by novelty + facet-coverage.

## Stage 3 — Adaptive investigation (where to dig)
An explicit loop, not a fixed k:
`retrieve → assess coverage & confidence → identify the gap → decide {dig deeper | follow a lead | stop}`
- Dig when a load-bearing claim is thinly evidenced or a sub-question is unmet.
- Stop when marginal evidence stops changing the conclusion.
- Record the stop decision and why (auditability).

## Stage 4 — Epistemic ledger (how well is it known)
Every claim carries a **status** (what the evidence does) and a **confidence**
qualifier (how strongly) — never collapsed into one label.

**Status** — four distinct states, each meaning something different to a decision:
- **corroborated** — ≥2 *independent* sources agree (and any specific figures verify).
- **supported** — one good source.
- **contradicted** — a *disconfirming* search surfaced counter-evidence (recorded in
  `counter_cite`; running that search is mandatory, not optional).
- **no evidence found** — search neither supported nor refuted. Qualify it:
  confidence **high** = *searched thoroughly, genuinely absent* (a real, often
  decision-useful negative); **low** = *coverage gap, not searched hard enough* (an
  admission of ignorance). Collapsing these two — the old flat "unsupported" — is the
  single most misleading thing a ledger can do.
- (**speculative** — model-inferred hypothesis, never asserted as evidenced.)

**Numeric verification (gate on promotion).** Specific figures — refinery-capacity
%, import tonnages, barrel counts — are where a report is most precisely wrong. A
claim's figures must actually appear in its cited passages or the claim CANNOT be
promoted to *corroborated*; it is capped at *supported* and the unverified figure is
flagged. "≈30% of capacity" earns corroboration only if "30%" is in the source text.

The gate (`verify_citations.py`) re-derives status + confidence from the actual cited
passages, runs the numeric check, and RECLASSIFIES when the tag disagrees:
corroborated-but-<2-aligned-sources → supported; any-affirmed-with-no-aligned-cite →
no evidence found; contradicted-with-no-counter-evidence → no evidence found;
affirmed-but-counter-evidence-present → contradicted; corroborated-with-unverified-
figure → supported. Status is earned by evidence, not assigned by the writer. Ledger
entries carry `cite`, `counter_cite`, and `searched`.

## Stage 5 — Framing & relay (how to say it)
Shape output to objective and audience:
- **View**: executive brief | analyst deep-dive | watch-list.
- **Frame**: lead with what matters for the decision; separate *established* from
  *emerging* from *speculative*.
- **Citations**: every non-trivial claim cites its source(s).
- **Foreground the gaps**: what's unknown / unsupported / disproven is stated, not
  hidden. A good report is legible about its own limits.

---

## Robustness rubric (assess the PROCESS, score 0–3 each)

| # | dimension | question | 0 (broken) | 3 (robust) |
|---|-----------|----------|-----------|-----------|
| 1 | **Objective fidelity** | do stated objectives match what was asked? | drifts to a different question | objectives fully capture the ask + decision context |
| 2 | **Coverage** | did investigation find what an expert would? | misses load-bearing evidence | complete vs an expert audit |
| 3 | **Label calibration** | are epistemic tags correct & not over/under-confident? | asserts unsupported as fact | tags match the evidence, tails trimmed |
| 4 | **Stop-decision quality** | dig too shallow or too deep? | stops before a key gap / pads & drifts | digs exactly the load-bearing gaps |
| 5 | **Reproducibility** | same query → consistent objectives & conclusions? | conclusions swing across runs | stable core conclusions |
| 6 | **Adversarial robustness** | captured by confident-wrong / misleading consensus? | repeats consensus, buries the counter-view | surfaces disconfirming evidence |

A report can be fluent and *still* fail 1, 3, 4, 6 — those are the failures that
turn a data dump into misinformation, and the ones the process exists to prevent.

---

## Implementation — the process is tool-enforced, not discretionary

The prototype proved the *stages* work but relied on a disciplined analyst overriding
weak tooling. The hardened build makes the load-bearing guarantees mechanical:

- **`scripts/retrieve.py`** — the live substrate for Stage 3. Passage-level cosine
  over the full corpus, fresh per query, with `--before` (point-in-time, no
  look-ahead), `--exclude` (force NEW evidence on a dig iteration), `--tier`, and
  `--diverse` (MMR facet-coverage; reports intra-list redundancy). This is what
  makes the adaptive loop *active* rather than a fixed top-k. An analyst issues
  10–20 targeted retrievals, using `--diverse` to survey a question's facets and
  plain relevance to dig a specific one.

- **Claim-specific corroboration** — `retrieve.py --corroborate` counts a source
  only if its passage is BOTH cosine-near AND covers the claim's salient terms
  (`--min-cover`). Topically-adjacent neighbours are reported separately as
  `topical-only`, never counted as confirmation. (Lexical coverage is a proxy for
  entailment; claims that share vocabulary with adjacent docs still need a reader —
  flagged, not hidden.)

- **Feature-routed checks** — the extracted axes are wired into the roles where they
  measure strongly, NOT as chunk-retrieval rerankers (their weakest role, ~5% lift):
  - **polarity → direction-consistency** (`verify_citations.py`): probe held-out AUROC
    0.947; flags an OK cite whose passage direction opposes the claim — a class cosine
    is blind to ("rose" vs "fell" cosine ~0.6). HONEST LIMIT: the probe conflates crisis
    *sentiment* with price *direction* on real passages (it false-flagged a doc that
    literally said "prices climb"), so it is **ADVISORY by default** (guarded to
    move-modality claims); hard demotion only under `--enforce-direction`, pending a
    sentiment-robust re-fit.
  - **modality → evidence-type cap**: probe hard-vs-soft acc 0.848; a factual claim
    "corroborated" only by forecast/opinion passages is capped to *supported* (needs
    ≥1 event/data source). Rarely fires — the corpus is event-dense — which is itself
    reassuring.
  - **novelty → coverage** (`retrieve.py --novel`): surfaces first-story developments
    over restatements (redundancy 0.71→0.46; pulled a load-bearing lead cosine buried).
  - **causal graph → hypothesis + confound** (`causal_hypotheses.py`): traverses
    cause_entities to name upstream drivers and RIVAL drivers. On Q3 it auto-surfaced
    the Iran/Hormuz supply confound the analyst had found by hand.
- **Two-axis self-correcting ledger** — `scripts/verify_citations.py` re-fetches
  every cited doc, scores claim↔passage alignment (OK / WEAK < 0.40 / MISSING), and
  emits **(status, confidence)** for each claim, re-deriving both from the evidence
  and RECLASSIFYING when the analyst's tag disagrees. `supported/high` therefore
  *means* ≥2 distinct well-aligned sources; `insufficient/high` (searched-not-found)
  is held distinct from `insufficient/low` (coverage gap). A WEAK/MISSING cite can
  never earn confidence.

Pipeline: analyst dig-loop (with `--diverse` for coverage) → `ledger.json`
(`cite`, `counter_cite`, `searched`) → `verify_citations.py` → (status × confidence)
per claim, reclassified where evidence disagrees. Every assertion is evidence-earned
or explicitly flagged.
