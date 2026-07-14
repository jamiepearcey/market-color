# Pitch deck structure — v4 (reframed build spec)

Successor to v3. **Every number, caveat, negative finding, visualisation, and motion
direction from v3 is preserved.** What changes is the *hierarchy*: the deck no longer
presents four innovations at equal weight. The new spine:

> **One research engine with two reach extensions.**
> Core = an auditable research process + a mechanical evidence gate (most of the lift).
> Extension 1 = a learned causal receptor (closes the vocabulary gap on causal questions).
> Extension 2 = a semantic causal graph *derived from the receptor* (closes second-order
> reach; activates only where multi-hop reasoning is needed).

One-sentence understanding the audience must leave with:
**"An auditable research engine that uses LLMs for exploration, but deterministic
mechanisms for evidence, verification, and trust."**

Emotional arc (replaces "RAG broken → GraphRAG expensive → we invented mechanisms"):
*AI research fails because it writes convincing stories without trying to disprove them
→ we built an auditable research process → this fixes most failures, measurably → but
researchers also discover things they didn't know to search for, and that reach problem
remains → we add learned causal reach — an operator, and a graph derived from it → the
system combines reliability and discovery.*

Structural moves vs v3 (not for the deck):
- The engine (was beat 04) and the gate (were beats 09–10) move UP to beats 02–04: the
  core system now owns the first third of the deck.
- The measured process-only lift (was inside beat 05's stack) becomes its own beat 05 —
  "most of the lift is the process" is now a slide-level claim with its own bars.
- GraphRAG (was beat 02, upfront) moves to beat 07, inside the reach section — it is now
  the *known expensive answer to the remaining gap*, not an opening rival. This is the
  one framing change to the user's earlier "GraphRAG upfront" ordering, required by the
  new hierarchy: retrieval must not dominate the opening.
- The receptor and graph (were "Innovations 1, 2, 2b") become "Extension 1" and
  "Extension 2 — unlocked by Extension 1." The graph is explicitly *derived*, optional,
  and scoped to second-order questions.
- Language audit: "institutional-quality research" is qualified; preferred vocabulary
  throughout: **auditable research workflow · measured improvements in research quality ·
  research-grade evidence discipline · quality gains from process rather than model
  scale.**
- All v3 corrections stand (recall-ceiling stat not "63% of corpus"; no (Q6,Q19) naming;
  raw-PPR failure disclosed; gate-vs-judges attribution; ~82% WEAK-cite noise honesty).

All numbers verified against `data/ab/RESULTS.md` and `README.md` on 2026-07-12.

---

## Narrative arc — 15 numbered beats + title (counter `/ 15`)

**00 — Title.** "Plausible → Auditable." Subtitle: **"An auditable research engine —
LLMs for exploration, deterministic mechanisms for evidence, verification, and trust."**
Support line: *"Measured improvements in research quality from process and verification —
not bigger models, larger datasets, or costly inference."* Ribbon: **"A research result —
ready to bring into a project."** Animate: the word "Plausible" resolves (glitch/split)
into "Auditable."

**01 — The problem: convincing stories, no attempt to disprove them.** AI research fails
in a specific way: it can write a fluent, plausible report **without ever trying to prove
itself wrong.** Hero stat: across **20 questions / 256 baseline claims, plain RAG
surfaced counter-evidence exactly ZERO times.** It admits gaps — what it never does is go
looking for what would disprove it. One understated receipt: a plain-RAG report stated
**gold at $4,525/oz** when its own cited corpus said **~$4,050–4,100** — a forecast
quoted as spot, the kind of error that survives *because nobody re-reads the source.*
Animate: the "0" holds; the $4,525 receipt stamps in.
*Sell line:* "The failure isn't hallucination. It's a system with no mechanism for
doubting itself."

**02 — The core system: a research process that tries to disprove itself.** Not a
retrieval trick — **an auditable research workflow.** Animate the pipeline flowing and
looping back:
`Question → Plan → Hypothesise → Search → Disconfirm → Confound-check → Coverage → Report`
with the backward arrow Coverage → Search (coverage gaps re-enter search). Caption:
**raw source text is the only thing searched at inference — decomposition and
graph-building never sit in the query path.** Every stage exists to catch a specific
failure: Disconfirm catches the tidy narrative; Confound-check catches the spurious
driver; Coverage catches the loud facet crowding out the rest.
*Sell line:* "Force the system to behave like a researcher: hypothesise, disconfirm,
check confounds — then verify mechanically."

**03 — The core system: the epistemic gate.** The process explores; **a deterministic
gate decides what survives.** Every claim gets a **status × confidence** re-derived
**mechanically** from the cited passages — and the gate **overrules the writer.**
Statuses: corroborated / supported / contradicted / speculative / no-evidence-found /
under-searched — splitting flat "unsupported" into **"searched, genuinely absent" (high
confidence)** vs **"didn't look hard enough" (low)** — different facts for a decision.
Claim-specific corroboration (distinct sources confirming the *exact* proposition) and
**numeric verification** (every figure — percent, quantity, price, bps — must appear in a
cited passage to be promoted). Honest scope, on the slide: **the gate catches unsourced
numbers; a wrong-context number that does sit in a cited source (the $4,525 gold) slips
the mechanical check — the blind judges caught that one. Two independent nets; we name
which catches what.** Animate: claims drop onto a status grid; a couple auto-demote with
a stamp.
*Sell line:* "Honesty as a mechanism, not a personality trait."

**04 — The core system: verification that can't be charmed by fluent prose.** The gate
checks the **proposition, not the vibe** — corroboration counts sources confirming the
exact claim, not topical neighbours; **facet coverage & first-story novelty** force the
report to span sub-topics instead of restating the loudest one. Cost honesty lands here:
**~1.8× tokens, ~3–4× retrieval calls** — and the output deliberately makes **fewer,
harder claims.** The asymmetry favours it: checking costs cents; a wrong number
propagates into models, positions, and rework.
*Sell line:* "Fluency is free. Research-grade evidence discipline is what you're paying
for — 2.10 vs 1.26 OK cites per claim."

**05 — The measured result: most of the lift IS the process.** Quality gains from
process rather than model scale — same model, same corpus, process only. Animate three
racing bars (baseline → +process), amber fill:
- corroborated claims: **14.5% → 45%**
- counter-evidence surfaced: **0 → 27**
- evidence density: **1.26 → 2.09** OK cites/claim
The structured process + the mechanical gate (~400 lines) deliver most of the measured
lift — a spec plus prompts. Cost × impact 2×2 as a small inset: process + gate sit
high-impact / low-cost; bigger model, more data, knowledge-graph-in-the-loop sit
high-cost / lower-leverage.
*Sell line:* "Optimise the process and the gate first. The model and the data are not
where the leverage is — we measured it."

**06 — The remaining gap: reach.** A reliable process can only verify **what it manages
to find** — and researchers discover things they didn't know to search for. Two failures
survive the process: **cosine returns the abundant, not the relevant** (it gravitates to
whatever the corpus repeats most), and **the upstream cause rarely shares vocabulary with
its effect** (a heatwave doesn't mention copper margins). Animate: latent-space scatter;
cosine pulls toward a dense "abundant ≠ relevant" blob while the true upstream driver
sits dark, unreached. Landing line: *"The hard part was never verification once you have
the evidence. It's knowing where to look."*
*Sell line:* "Reliability is solved by process. Discovery is a reach problem — and reach
is what's left."

**07 — The known answer is expensive: GraphRAG.** GraphRAG earns its reputation on
exactly this gap: it answers what **no single passage states** — global search composes
facts across documents and infers **causal chains** link by link. Animate: a knowledge
graph where a causal chain lights up across nodes no passage connects. Then the bill:
**expensive to build and maintain** (LLM extraction over the whole corpus, again on every
refresh), **node explosion** (entity resolution, dedupe, drift; edges grow ~n²), and
**opaque** (why an edge exists is itself hard to audit — you traded unverifiable prose
for unverifiable edges). Keep the costs qualitative — no invented numbers. Closing
question, big type: **"Can we get graph traversal's reach — without paying for graph
construction?"**
*Sell line:* "The reach is real. The question is whether you must pay for it this way."

**08 — The move: learn the relation, discard the graph.** The core idea behind both
extensions: the LLM-extracted cause→effect pairs are **expensive to build but ideal
training data.** Learn the relation once into a lightweight **operator W over
embeddings — then throw the extracted graph away.** The causal structure now lives in
latent space, **free at inference.** And (set up beat 10): once the relation is an
operator, **a graph comes back for free** — derived, not extracted. Animate: extraction
graph → "train" arrow → latent scatter with a **W** glyph; the extracted graph fades out;
a faint new lattice shimmers behind the scatter.
*Sell line:* "We throw the expensive graph away — and get one back for free."

**09 — Extension 1: the causal receptor.** A targeted fix for the vocabulary gap — not a
new architecture. A single learned **asymmetric matrix `W`** maps a *cause* embedding
into its *effect* region; because `W` isn't symmetric, **`score(a→b) ≠ score(b→a)` — the
asymmetry IS the causal arrow. No LLM at inference.** Animate: a vector rotating through
the matrix into a highlighted effect region. Stats (count up):
- **0.80** direction accuracy vs the **0.50** coin-flip floor
- **+53%** Recall@10 over cosine as a standalone cause-finder; **+34%** on the honest
  temporal split (train early, test later) — it learns transferable structure, not the
  corpus
- **cosine's recall ceiling: 63% even at rank 300. W lifts it to 75%** — causes cosine
  can *never* rank, at any depth
- **< 1s on CPU · zero LLM calls · free, already-extracted labels**
Honesty callout (keep prominent): the receptor won **all 9** causal/driver/second-order
questions vs the process alone — and added **~nothing on descriptive questions.** A
scalpel for the questions the process can't reach, not a general boost.
*Sell line:* "One 384×384 matrix. It finds what similarity search structurally cannot —
and only claims credit where it should."

**10 — Extension 2: the semantic causal graph (unlocked by the operator).** Some drivers
sit **two hops upstream** — the cause of the cause. Instead of extracting a knowledge
graph, **derive** the one W already implies: `S[i,j] = (dᵢ·W)·dⱼᵀ` = "i causes j"; keep
each effect's **top-K (=10) softmax-weighted causes** → a sparse column-stochastic
operator **P** (~31k edges over 3.4k docs), built **once, offline, in seconds of sparse
BLAS — no LLM, no entity pipeline, zero inference-time graph cost.** It activates only
where multi-hop reasoning is needed. Then the design finding, stated plainly: **the naive
walk fails.** Raw PageRank over the graph is an **85% hub magnet** — macro
"cause-of-everything" boilerplate. The usable signal needs **hub-suppression at build
time + abductive fusion at query time** (direct-hop ∪ chain-walk ∪ topical ∪
hypothesis-centroid, rank-fused; <0.5s on CPU). Measured payoff:
- **+29%** recall on held-out **2-hop chain** targets vs the single hop (R@10 0.044 vs 0.034)
- head-to-head vs the single-hop method (blind, 20 Qs): **12/5/3**, and **7/9 on
  causal/second-order** questions — near-even on descriptive, exactly the designed shape
- counter-evidence per run nearly doubles (**27 → 44**): more rival upstream drivers
  proposed means more things to disconfirm
- discipline: only **1–4 two-hop docs actually cited per report** — a scalpel, not a
  firehose
Animate: dense score grid collapses to top-K columns; a seed pulse propagates seed →
causes → causes-of-causes; a "hub" node flares and is dimmed by the suppression term.
*Sell line:* "Graph traversal's reach, from a matrix multiply — and we're showing you the
failure mode we had to engineer out, because we measured it."

**11 — The proof (the money slide).** Blind A/B, **n=20** (full benchmark set), same
raw-text search surface, same model. Baseline = plain cosine RAG; Method = the full
engine (process + gate + both extensions). Two independent instruments: mechanical gate
(arm-blind) + blind pairwise judges (labels randomised, content-only, corpus
spot-checks).
**Panel 1 — judges:** the method wins **20 / 20** — avg **10.75 vs 8.10** / 12; **13 of
20 decisive** (≥3 pts); only 2 within a point; on no question did the baseline score
higher. **Causal/second-order subset: 9/9.**
**Panel 2 — the racing bars (mechanical, arm-blind):**

| metric | baseline | method |
|---|---|---|
| corroborated (≥2 aligned sources) | 14.5% | **42.4%** |
| **counter-evidence surfaced** | **0** | **44** |
| evidence density (OK cites / claim) | 1.26 | **2.10** |
| figures asserted (same ~96% sourcing discipline) | 116 | **181** |
| **cross-vocabulary causal evidence** (causal-lift cites) | 1.5% | **4.4%** — 70% load-bearing |
| hallucinated citations | 0 | 0 |
| mis-aligned cites (honest cost, amber) | 4.2% | **5.5%** |

**Panel 3 — the dose-response stack (small inset, the credibility panel):** each layer
wins exactly where it's designed to. The **process** produces most of the general lift
(corroboration, counter-evidence, density). Adding the **receptor** won the **causal
questions 9/9** (≈nothing on descriptive). Adding the **graph** beat the single hop
**7/9 on causal/second-order** (near-even on descriptive). The edge concentrates
precisely where each mechanism should act — that shape is hard to fake.
Concrete saves, attributed honestly: baseline asserted **$4,525 gold**, a **−4.1% China
FAI** presented as fact, and an **IEA supply figure mislabelled as demand** — all caught
by blind judges against the corpus; the method made none of them. Amber footnote, exact:
the higher mis-aligned rate is the price of reach — and honestly, **most low-cosine cites
are noise (~82%), not hidden value**; the value shows up in the separate causal-lift
measure (4.4%, 70% on corroborated/contradicted claims). Hallucinated stays 0.
*Sell line:* "Across 256 baseline claims: zero counter-evidence. The engine: 44. That is
the difference between a summary and a research process."

**12 — Shown, not told (Q19, the second-order question).** Real question: *second-order
effects of the European heatwave on emerging-market manufacturers.* Left panel (cosine
only → "plausible"): the **one loud facet** — "Chinese AC makers sold out across Europe"
— and a dimmed list of what it never surfaced: the EU trade-policy second-order effect,
the "temperature gap" counter-narrative, the cross-country span, any disconfirming check.
Right panel (full engine → "auditable"): the **citation ledger** — every claim carries a
status × confidence chip and a **found-by** tag (cosine / corroborate / receptor / chain /
disconfirm), exactly as the run recorded them. Star the chain find: the **upstream copper
margin-squeeze** — the derived graph connected the AC export boom back to copper input
costs (a driver two hops from the question's vocabulary), with a **direction check** on
the chain (heatwave → AC export surge, margin 0.143 ✓) — and show the discipline: **the
other 2-hop candidates (BIS AI-debt, Iran) were checked and discarded as off-topic.**
Caption: *"Same corpus, same model — every line traces to a source passage. The gap is
what focused retrieval never goes looking for."*
*Sell line:* "The judge's own words: the upstream copper mechanism 'with verified
figures' — the single-hop method missed it entirely."

**13 — The playbook + cost.** Animated checklist — in adoption order, core first:
1. Write the **process** (hypothesise + disconfirm + confound + coverage) — biggest
   lift, least code.
2. Build the **mechanical gate** — honesty as a guarantee, not analyst luck.
3. **Decompose to learn / retrieve the original** — structure as signal, source as
   evidence.
4. When causal questions matter, add the **learned causal operator** for hypothesis
   discovery; when second-order questions matter, derive the **semantic causal graph**
   from it offline — query cost stays flat.
5. Keep LLM decomposition **training-time only** — never in the inference loop.
Cost tags animate in: **~3–4× retrieval calls** *(cheap)*, **~1.8× tokens** *(cheap vs.
being wrong)*, graph build *(seconds, offline, no LLM)*.
*Sell line:* "Steps 1–3 are a spec and ~400 lines — that's the engine. The extensions
bolt on when your questions need the reach."

**14 — The honest limits.** Eyebrow: **"The honest limits."** (ramps whiter, one per beat)
- **n = 20** curated questions on a ~2-week single-domain corpus — a real result, not an
  independent benchmark; the mechanical-gate aggregates (above all **44 vs 0**
  counter-evidence across 256 baseline claims) are the robust part.
- **2 of 20 wins were within a single point** — on saturated-coverage topics a strong
  plain analyst nearly matches; **the guarantee is the floor, not the ceiling** *(punch
  phrase)*.
- **5.5% vs 4.2%** mis-aligned cites — and most low-cosine cites are noise, not secret
  value; the reach benefit is a measured, targeted 4.4%.
- The **receptor adds ~nothing on descriptive questions** — its 9/9 is on causal
  questions only; we scope the claim to where it was measured.
- The **graph's lift is narrow by design**: 2-hop chain targets, after hub-suppression +
  fusion (raw PageRank alone is 85% hub noise). Its head-to-head vs the single hop was
  measured across runs, so protocol variance isn't fully isolated — the causal-vs-
  descriptive *shape* (7/9 vs near-even) is the honest signal, not the margin.
- The **direction oracle** is a *lead*, not evidence — recorded, never counted as proof.
- One judgment (Q19) was re-run once: the first judge's rationale credited the baseline
  with content a file-diff proved only the method contained. Documented in RESULTS.md;
  nothing else was touched. (Include this — it *demonstrates* the audit culture.)
*Sell line:* "This slide is the pitch. A system that reports its own confounds is the
product working on itself."

**15 — Close.** Big line: **"Plain RAG gives you a plausible report. This gives you an
auditable one — and across 20 questions the baseline surfaced counter-evidence exactly
zero times, while asserting figures its own sources contradict."** The one-sentence
takeaway, set alone: *"An auditable research engine — LLMs for exploration, deterministic
mechanisms for evidence, verification, and trust."* Cost-asymmetry line: *wrong
information doesn't stay wrong quietly — it **propagates** into models, positions and
rework; an auditable report is **cheap insurance**, and the causal reach also finds the
questions **you didn't know to ask.*** Footer: `n=20 · blind-judged · measured 2026-07 ·
a research result, ready to bring into a project`.

---

## Hard rules on the data
- Use **ONLY** the numbers above; do not invent, re-round, or extrapolate. Keep GraphRAG
  cost claims (beat 07) qualitative. If a visual needs an unlabeled magnitude, keep the
  label qualitative.
- Never state or imply the mechanical gate caught the $4,525 / FAI / IEA errors — the
  judges did; the gate's scope is unsourced numbers.
- Do not name which two questions were within a point (not recorded for this run).
- Do not reintroduce the polarity/sentiment probe or the "63% of the corpus reachable
  only this way" phrasing (replaced by the recall-ceiling stat: 63%→75% @300).
- Avoid unqualified superlatives ("institutional-quality"); prefer *auditable research
  workflow*, *measured improvements in research quality*, *research-grade evidence
  discipline*, *quality gains from process rather than model scale*.
- Present the receptor and graph as **extensions of one engine**, never as standalone
  innovations; the graph is **derived from the operator**, not a separate architecture.
- Keep every caveat on beat 14 — the honesty is the pitch.

## Design & motion (unchanged from v3 — the sleek look stays)
- Near-black `#0B0D13`; ONE amber accent `#E0B062`; red = wrong/contradicted; green =
  verified; blue `#5B8FFF` reserved for the receptor/graph extensions (visually marking
  them as the *extension* layer against the amber core). Monospace for numbers, clean
  grotesque for prose.
- Motion signature: stats count up; bars race; the pipeline flows and loops; the receptor
  is a vector rotating through a matrix; claims drop and stamp; curves draw in. Kept from
  v3: the hub-node flare-and-dim on beat 10; the extracted-graph fade → latent lattice
  shimmer on beat 08. New in v4: the core/extension colour split (amber engine, blue
  extensions) as a persistent visual grammar across beats.
- Timeline scrubber + right-rail beat picker; counter `/ 15`; credibility ribbon
  `n=20 · blind-judged · measured 2026-07`.
