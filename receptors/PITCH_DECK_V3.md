# Pitch deck structure — v3 (optimised build spec)

Successor to the v2 "as built" doc. **Preserves the v2 framing exactly** — problem →
GraphRAG alternative → the question → the engine-as-solution → 90/10 → pivot to latent
space → four innovations → proof → shown-not-told → playbook → limits → close. What v3
changes is the *quality* of each beat: sharper sell lines, corrected stats, resolved
internal contradictions, and previously-unused measured results folded in.

All numbers verified against `data/ab/RESULTS.md` and `README.md` on 2026-07-12.

---

## Corrections vs v2 (why these edits exist — not for the deck)

1. **"63% of the corpus reachable ONLY this way" is not a measured stat — removed.**
   The real, stronger measurement (README, retrieval eval): even scanning its top **300**
   candidates, cosine finds only **63%** of true causes; the operator lifts that ceiling
   to **75%**. That is a *recall-ceiling* claim, not a "corpus share" claim. Beat 07 now
   states it correctly.
2. **Limits beat no longer names (Q6, Q19)** as the within-a-point pair — that pairing is
   from the *earlier single-hop* run. In the latest (latent-graph) run Q19 was an 11–9
   (2-point) win and the two within-a-point questions were not individually recorded.
   "Only 2 of 20 within a single point" stays; the question IDs go.
3. **Beat 08 no longer sells raw PageRank as the mechanism.** Measured: raw PPR alone is
   an **85% hub magnet** (macro "cause-of-everything" boilerplate) and *loses* to the
   single hop. What ships is hub-suppressed edges + **abductive fusion** (direct-hop +
   chain-walk + topical + hypothesis-centroid, rank-fused). This is now the beat's
   design insight — "the naive walk fails; we measured it and fixed it" — which is more
   credible AND more impressive than pretending PPR just works.
4. **The graph contradiction is resolved with one consistent line.** v2 said "we don't
   have to pre-build the edges" (beat 03) and "no knowledge graph in the loop" (beat 04)
   while beat 08 pre-builds a graph. The consistent truth, used verbatim everywhere:
   **"No LLM-built knowledge graph — the graph we do use is derived from W in one sparse
   BLAS pass at index time (seconds, ~31k edges over 3.4k docs), and costs zero at
   inference."** Beat 06's arc becomes the sell: *we throw the expensive graph away — and
   get one back for free, in latent space.*
5. **Error-catch attribution fixed.** Numeric sourcing is a measured WASH (~96% verify
   rate all arms — the hardened checker covers $, prices, bps). The $4,525 / −4.1% FAI /
   IEA errors are *wrong-context* numbers that DO appear in a cited source; the
   mechanical gate cannot catch those — **blind judges did**. Beats 09 and 11 now say
   "two independent nets" and never credit the gate with those catches.
6. **The ablation stack is restored** (v2 dropped it). The dose-response shape —
   process-only < +receptor (won causal Qs **9/9**) < +graph (won causal Qs **7/9** vs
   the single hop, 12/5/3 overall) — is the strongest evidence the *mechanism* works,
   not just the ensemble. Folded into beat 11 as a second panel.
7. **The mis-aligned-cite honesty is made exact.** Measured: **~82% of the method's WEAK
   (low-cosine) cites are genuine noise; only ~18% are high-causal reach.** The value
   story lives in the separate causal-lift metric (4.4% of all cites, **70%
   load-bearing** on corroborated/contradicted claims vs 20% for baseline). Beat 11's
   amber row now says the honest version — which inoculates against the one audit that
   could kill the deck.
8. **The 90/10 gets a measured backbone.** Process-only *alone* moves corroboration
   14.5%→45%, counter-evidence 0→27, density 1.26→2.09. Beat 05 now shows the measured
   stack instead of an asserted percentage.
9. **New concrete stats put to work**: figures asserted **181 vs 116 (+55%) at the same
   ~96% sourcing discipline** (beat 11); **1–4 two-hop docs actually cited per report**
   — a scalpel, not a firehose (beat 08); graph arm surfaces **44 vs 27** counter-evidence
   versus the single hop because it proposes more rival upstream drivers to disconfirm
   (beat 08); **+29% recall on held-out 2-hop chain targets** (beat 08); **+34% R@10 on
   the honest temporal split** kept on beat 07.

---

## Narrative arc — 15 numbered beats + title (unchanged order; `/ 15` counter)

**00 — Title.** "Plausible → Auditable." Subtitle: **"Institutional-quality research at a
fraction of the cost."** Support: *"Quality comes from a rigorous process and mechanical
verification — not bigger models, larger datasets, or costly inference."* Ribbon:
**"A research result — ready to bring into a project."** Animate: "Plausible" resolves
(glitch/split) into "Auditable."

**01 — The problem.** Plain RAG produces reports that are **plausible — and confidently
wrong.** Hero stat: across **20 questions / 256 baseline claims, plain RAG surfaced
counter-evidence exactly ZERO times.** It admits gaps — what it never does is go looking
for what would prove it wrong. One understated receipt: a plain-RAG report stated **gold
at $4,525/oz** when its own cited corpus said **~$4,050–4,100** — a forecast quoted as
spot, the kind of error that survives *because nobody re-reads the source.* Animate: the
"0" holds; the $4,525 receipt stamps in.
*Sell line:* "The failure isn't hallucination. It's a system with no mechanism for
doubting itself."

**02 — The alternative (GraphRAG).** GraphRAG earns its reputation: it answers what **no
single passage states** — global search composes facts across documents and infers
**causal chains** link by link. Animate: a knowledge graph where a causal chain lights up
across nodes no passage connects. Then the bill arrives: **expensive to build and
maintain** (LLM extraction over the whole corpus, again on every refresh), **node
explosion** (entity resolution, dedupe, drift; edges grow ~n²), and **opaque** (why an
edge exists is itself hard to audit — you traded unverifiable prose for unverifiable
edges). Keep the costs qualitative — no invented numbers.
*Sell line:* "The reach is real. The question is whether you must pay for it this way."

**03 — The question.** **"How do we get graph traversal's reach — without paying for
graph construction? Build it in semantic space."** Reframe: traversal is just
**multi-step reasoning that follows the relevant result** — and the two things that
actually break plain retrieval are: **distillation loses context** (a graph node keeps
the label, drops the passage) and **cosine returns the abundant, not the relevant** (it
gravitates to whatever the corpus repeats most; the upstream cause rarely *looks like*
its effect). Animate: latent-space scatter; cosine pulls toward a dense "abundant ≠
relevant" blob while a guided path hops to the true target. Landing line: *"The hard part
was never the edges. It's knowing where to look — so we think like a researcher."*
(Language discipline: this beat promises reach without *LLM-built* graphs; the semantic
graph revealed in beat 08 is derived, free, and offline — do not say "no pre-built edges.")

**04 — The solution (the engine).** Not "GraphRAG." **A loop that tries to disprove
itself.** Pipeline flows and loops back:
`Question → Plan → Hypothesise → Search → Disconfirm → Confound-check → Coverage → Report`
with the backward arrow Coverage → Search. Caption: **raw source text is the only thing
searched at inference — decomposition and graph-building never sit in the query path.**
*Sell line:* "Every stage exists to catch a specific failure: Disconfirm catches the tidy
narrative; Confound-check catches the spurious driver; Coverage catches the loud facet
crowding out the rest."

**05 — The insight (the 90/10).** **The biggest quality driver is also the cheapest
artifact.** Replace the asserted percentage with the measured stack — animate three bars
growing left to right (baseline → +process → +operators):
- corroborated claims: **14.5% → 45% → 42%** (the process does this)
- counter-evidence: **0 → 27 → 44** (the process starts it; the graph doubles it)
- evidence density: **1.26 → 2.09 → 2.10** OK cites/claim (the process does this)
Message: **the structured process + the mechanical gate (~400 lines) deliver most of the
lift — a spec plus prompts. The learned operators then add their edge exactly where the
process can't reach: the causal questions.** Cost × impact 2×2 stays as a small inset:
process + gate high-impact/low-cost; bigger model, more data, knowledge-graph-in-the-loop
high-cost/lower-leverage.
*Sell line:* "Optimise the process and the gate first. The model and the data are not
where the leverage is — we measured it."

**06 — The pivot.** **"Reasoning only gets us so far."** The agent explores only what it
can already name; two gaps remain: **Diversity** (an embedding tuned for "most relevant"
returns near-duplicates) and **Reach** (the upstream cause rarely shares vocabulary with
its effect). The move — and this is the core idea of the whole deck: the LLM-extracted
cause→effect pairs are **expensive to build but ideal training data.** Learn the relation
once into a lightweight **operator W over embeddings — then throw the extracted graph
away.** The causal structure now lives in latent space, **free at inference.** And (set
up beat 08): once the relation is an operator, **a graph comes back for free** — derived,
not extracted. Animate: extraction graph → "train" arrow → latent scatter with a **W**
glyph; the extracted graph fades out; a faint new lattice shimmers behind the scatter.
*Sell line:* "We throw the expensive graph away — and get one back for free."

**07 — Innovation 1: the causal receptor.** A single learned **asymmetric matrix `W`**
maps a *cause* embedding into its *effect* region; because `W` isn't symmetric,
**`score(a→b) ≠ score(b→a)` — the asymmetry IS the causal arrow. No LLM at inference.**
Animate: a vector rotating through the matrix into a highlighted effect region. Stats
(count up):
- **0.80** direction accuracy vs the **0.50** coin-flip floor
- **+53%** Recall@10 over cosine as a standalone cause-finder; **+34%** on the honest
  temporal split (train early, test later) — it learns transferable structure, not the
  corpus
- **cosine's recall ceiling: 63% even at rank 300. W lifts it to 75%** — causes cosine
  can *never* rank, at any depth
- **< 1s on CPU · zero LLM calls · free, already-extracted labels**
Honesty callout: the receptor won **all 9** causal/driver/second-order questions vs the
process alone — and added ~nothing on descriptive ones. A scalpel, exactly where
inference matters.
*Sell line:* "One 384×384 matrix. It finds what similarity search structurally cannot."

**08 — Innovation 2: from one hop to a causal graph.** The single-hop operator finds
direct causes; some drivers sit **two hops upstream**. So derive the graph W implies:
`S[i,j] = (dᵢ·W)·dⱼᵀ` = "i causes j"; keep each effect's **top-K (=10) softmax-weighted
causes** → a sparse column-stochastic operator **P** (~31k edges over 3.4k docs), built
**once, offline, in seconds of sparse BLAS — no LLM, no entity pipeline, zero
inference-time graph cost.** Then the design insight, stated as a finding: **the naive
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
*Sell line:* "GraphRAG's reach, from a matrix multiply. And we're showing you the failure
mode we had to engineer out — because we measured it."

**09 — Innovation 3: the epistemic gate.** Every claim gets a **status × confidence**
re-derived **mechanically** from the cited passages — and the gate **overrules the
writer.** Statuses: corroborated / supported / contradicted / speculative /
no-evidence-found / under-searched — splitting flat "unsupported" into **"searched,
genuinely absent" (high confidence)** vs **"didn't look hard enough" (low)** — different
facts for a decision. Claim-specific corroboration (distinct sources confirming the
*exact* proposition) and **numeric verification** (every figure — percent, quantity,
price, bps — must appear in a cited passage to be promoted). Honest scope, on the slide:
**the gate catches unsourced numbers; a wrong-context number that does sit in a cited
source (the $4,525 gold) slips the mechanical check — the blind judges caught that one.
Two independent nets; we name which catches what.** Animate: claims drop onto a status
grid; a couple auto-demote with a stamp.
*Sell line:* "Honesty as a mechanism, not a personality trait."

**10 — Innovation 4: verification that can't be charmed by fluent prose.** The gate
checks the **proposition, not the vibe** — corroboration counts sources confirming the
exact claim, not topical neighbours; **facet coverage & first-story novelty** force the
report to span sub-topics instead of restating the loudest one. Cost honesty lands here:
**~1.8× tokens, ~3–4× retrieval calls** — and the output deliberately makes **fewer,
harder claims.** The asymmetry favours it: checking costs cents; a wrong number
propagates into models, positions, and rework.
*Sell line:* "Fluency is free. Evidence density is what you're paying for — 2.10 vs 1.26
OK cites per claim."

**11 — The proof (the money slide).** Blind A/B, **n=20** (full benchmark set), same
raw-text search surface, same model. Baseline = plain cosine RAG; Method = the whole
process incl. the latent-space causal graph. Two independent instruments: mechanical
gate (arm-blind) + blind pairwise judges (labels randomised, content-only, corpus
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
wins exactly where it's designed to — process alone lifts corroboration and counter-
evidence; adding the receptor won the **causal questions 9/9** (≈nothing on descriptive);
adding the graph beat the single hop **7/9 on causal** (near-even on descriptive). The
edge concentrates precisely where each mechanism should act — that shape is hard to fake.
Concrete saves, attributed honestly: baseline asserted **$4,525 gold**, a **−4.1% China
FAI** presented as fact, and an **IEA supply figure mislabelled as demand** — all caught
by blind judges against the corpus; the method made none of them. Amber footnote, exact:
the higher mis-aligned rate is the price of reach — and honestly, **most low-cosine cites
are noise (~82%), not hidden value**; the value shows up in the separate causal-lift
measure (4.4%, 70% on corroborated/contradicted claims). Hallucinated stays 0.
*Sell line:* "Across 256 baseline claims: zero counter-evidence. The method: 44. That is
the difference between a summary and a research process."

**12 — Shown, not told (Q19, the second-order question).** Real question: *second-order
effects of the European heatwave on emerging-market manufacturers.* Left panel (cosine
only → "plausible"): the **one loud facet** — "Chinese AC makers sold out across Europe"
— and a dimmed list of what it never surfaced: the EU trade-policy second-order effect,
the "temperature gap" counter-narrative, the cross-country span, any disconfirming check.
Right panel (full method → "auditable"): the **citation ledger** — every claim carries a
status × confidence chip and a **found-by** tag (cosine / corroborate / receptor / chain /
disconfirm), exactly as the run recorded them. Star the chain find: the **upstream copper
margin-squeeze** — the graph connected the AC export boom back to copper input costs
(a driver two hops from the question's vocabulary), with a **direction check** on the
chain (heatwave → AC export surge, margin 0.143 ✓) — and show the discipline: **the
other 2-hop candidates (BIS AI-debt, Iran) were checked and discarded as off-topic.**
Caption: *"Same corpus, same model — every line traces to a source passage. The gap is
what focused retrieval never goes looking for."*
*Sell line:* "The judge's own words: the upstream copper mechanism 'with verified
figures' — the single-hop method missed it entirely."

**13 — The playbook + cost.** Animated checklist:
1. Write the **process** (disconfirm + confound + coverage) — biggest lift, least code.
2. Build the **mechanical gate** — honesty as a guarantee, not analyst luck.
3. **Decompose to learn / retrieve the original** — structure as signal, source as
   evidence.
4. Add the **learned causal operator** for hypothesis discovery; derive the **semantic
   causal graph** from it offline for second-order reach — query cost stays flat.
5. Keep LLM decomposition **training-time only** — never in the inference loop.
Cost tags animate in: **~3–4× retrieval calls** *(cheap)*, **~1.8× tokens** *(cheap vs.
being wrong)*, graph build *(seconds, offline, no LLM)*.
*Sell line:* "Steps 1–3 are a spec and ~400 lines. Start there; the operators bolt on."

**14 — The honest limits.** Eyebrow: **"The honest limits."** (ramps whiter, one per beat)
- **n = 20** curated questions on a ~2-week single-domain corpus — a real result, not an
  independent benchmark; the mechanical-gate aggregates (above all **44 vs 0**
  counter-evidence across 256 baseline claims) are the robust part.
- **2 of 20 wins were within a single point** — on saturated-coverage topics a strong
  plain analyst nearly matches; **the guarantee is the floor, not the ceiling** *(punch
  phrase)*.
- **5.5% vs 4.2%** mis-aligned cites — and most low-cosine cites are noise, not secret
  value; the reach benefit is a measured, targeted 4.4%.
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
zero times, while asserting figures its own sources contradict."** Cost-asymmetry line:
*wrong information doesn't stay wrong quietly — it **propagates** into models, positions
and rework; an auditable report is **cheap insurance**, and a one-matrix causal receptor
also finds the questions **you didn't know to ask.*** Footer: `n=20 · blind-judged ·
measured 2026-07 · a research result, ready to bring into a project`.

---

## Hard rules on the data
- Use **ONLY** the numbers above; do not invent, re-round, or extrapolate. Keep GraphRAG
  cost claims (beat 02) qualitative. If a visual needs an unlabeled magnitude, keep the
  label qualitative.
- Never state or imply the mechanical gate caught the $4,525 / FAI / IEA errors — the
  judges did; the gate's scope is unsourced numbers.
- Do not name which two questions were within a point (not recorded for this run).
- Do not reintroduce the polarity/sentiment probe or the "63% of the corpus reachable
  only this way" phrasing (replaced by the recall-ceiling stat: 63%→75% @300).
- Keep every caveat on beat 14 — the honesty is the pitch.

## Design & motion (unchanged from v2)
- Near-black `#0B0D13`; ONE amber accent `#E0B062`; red = wrong/contradicted; green =
  verified; blue `#5B8FFF` reserved for the receptor/graph. Monospace numbers, clean
  grotesque prose.
- Stats count up; bars race; pipeline flows and loops; the receptor is a vector rotating
  through a matrix; claims drop and stamp; curves draw in. New in v3: the hub node
  flare-and-dim on beat 08; the extracted-graph fade → latent lattice shimmer on beat 06.
- Timeline scrubber + right-rail beat picker; counter `/ 15`; credibility ribbon
  `n=20 · blind-judged · measured 2026-07`.
