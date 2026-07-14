# Prompt: Animated infographic pitch deck — "Auditable market intelligence, cheaply"

Paste everything below the line into Claude (extended thinking on, a Claude 4.x model).
It asks for a **single self-contained HTML file** — an animated, scroll-driven
infographic pitch deck. Every number is real and measured; the prompt forbids
inventing any others.

> NOTE TO SELF (not for the deck): all A/B numbers below are FINAL from the full
> 20-question run (`data/ab/RESULTS.md`). The unproven polarity probe is REMOVED —
> do not reintroduce it.

---

You are a senior motion-design engineer + data-storyteller. Build me a **single
self-contained `.html` file** (no build step; only GSAP + one Google Font via CDN are
allowed) that is an **animated, scroll-driven infographic pitch deck**. It must run by
double-clicking. Target 16:9 laptop; degrade gracefully on mobile and with
`prefers-reduced-motion`.

## What I'm pitching

A research methodology for **market-intelligence report generation** built for the
places markets are hardest to read: **sparse-information, emerging/frontier, and
foreign-language coverage.** Two theses:

1. **Most of the quality is cheap structure.** ~90% of the lift comes from a research
   *process* and a *mechanical verification gate* — a spec plus a few hundred lines of
   code. The clever 10% (a learned causal operator) adds only **~1.8× inference cost**.
2. **Verification is asymmetric insurance.** Proving a claim costs one retrieval call.
   *Carrying a wrong one* costs the meeting it triggers, the model it feeds, the
   position it sizes, and the analyst-hours burned before someone catches it. This
   method makes the cheap thing automatic.

The tone is **credible-because-honest**, not hype. No buzzwords. Persuasion comes from
real numbers, honest caveats, and the cost argument. Show what *didn't* work too.

## Narrative arc (one full-viewport section per beat; animate on scroll-into-view)

**0 — Title.** "Plausible → Auditable." Subtitle: "Most of the quality of an expensive
research system, from cheap structure — and it works best exactly where information is
thin." Tag: "20-question head-to-head, blind-judged. A research result." Animate: the
word "Plausible" resolves into "Auditable."

**1 — The problem, as a cost asymmetry (NOT a scare line).** Plain RAG produces
*plausible reports that are confidently wrong* — and wrong information is expensive in
a way a wrong trade alone doesn't capture: it propagates. Animate a single unverified
claim fanning out into "→ briefing → model input → position size → hours of rework."
Caption: "The cost of being wrong — time, rework, risk, opportunity — dwarfs the cost
of checking. Proof is cheap insurance." One real receipt, understated: a plain-RAG
report stated **gold at $4,525/oz** when its own cited corpus said **~$4,050–4,100** —
the kind of error that survives *because nobody re-reads the source.*

**2 — The 90/10.** The biggest quality driver (a structured process + a mechanical
gate) is also the *cheapest* artifact — a markdown spec plus a few hundred lines.
Animate a stacked bar: **~90% of the lift = cheap structure, assembled fast**; the
remaining **~10% = a learned operator at only ~1.8× inference cost.** Message:
**optimise the process and the gate first. The model and the data are not the
leverage.**

**3 — It works best where it's hardest: sparse information.** The method's edge grows
as coverage thins — few sources, no consensus, cross-language — which is precisely the
emerging/frontier terrain this is built for. Animate a curve: x = information density
(sparse → saturated), y = advantage over plain RAG; the advantage is highest on the
sparse (left) side and compresses toward the right. Caption: "On saturated topics a
good analyst matches it. On thin ones — where the money is — it pulls away."

**4 — The engine (not "GraphRAG").** An **iterative research engine** — animate a
pipeline that flows and loops back:
`Question → Plan → Hypothesise → Search → Disconfirm → Confound-check → Coverage → Report`
with a backward arrow Coverage→Search (the loop). Caption: raw source text is the only
thing searched at inference — no knowledge graph in the loop.

**5 — Innovation 1: decompose to LEARN, retrieve the ORIGINAL.** LLM decomposition
(atomic facts, entities) distils *relationships and structure* — but strips *context*.
So we use decomposition two ways only: as **training supervision** for our projections
and as a **retrieval signal** — while we **preserve and cite the original source
chunk.** Animate: an article → a crisp "fact/relationship" crystal (structure) that
*fades the surrounding context grey*, then a "cite" arrow snapping back to the full
original passage. Caption: "Facts tell you what connects to what. The source tells you
what it *means.* Learn from the first; quote the second." This is why it beats **both**
plain chunk-RAG **and** decomposition-retrieval.

**6 — Innovation 2: the causal receptor (a targeted booster, honestly scoped).** A
single learned asymmetric matrix `W` maps a *cause* embedding into its *effect* region;
because `W` isn't symmetric, `score(a→b) ≠ score(b→a)` — that asymmetry IS the causal
arrow. Animate a vector rotating through a matrix into a new region. Stats (count up):
**direction accuracy 0.80 vs 0.50** floor; as a cause *finder* **Recall@10 +53%** over
cosine; **< 1 second on CPU, zero LLM calls**, on **free, already-extracted labels**;
**63% of the corpus reachable only this way**; upgradeable with new signals over time.
HONESTY (this is the credibility beat — a three-arm ablation): the receptor is **not**
what wins most of the battle — the process is. But head-to-head it adds a real edge
**concentrated exactly where it should**: on causal/driver/second-order questions it
improved the report **9 out of 9 times**; on plain descriptive questions it added
~nothing. It's a scalpel for the causal questions — which, in sparse markets, are the
ones that pay. Animate: two buckets, "causal Qs" lighting up 9/9, "descriptive Qs"
staying flat.

**7 — Innovation 2b: the semantic graph (multi-hop reach, paid for offline).** The
single-hop operator finds direct causes; some drivers sit *two hops upstream*
(cause-of-a-cause). We get to them WITHOUT paying the usual price of a knowledge
graph. The trick: the same operator `W` induces a **semantic causal graph** — nodes
are documents, edge `i→j` = "i causes j" scored by `doc·W`, each effect keeping its
top-k causes — and we **build that graph once, offline, at index time.** At query
time there is **no graph to construct, no LLM extraction, no entity pipeline** — just
a sub-second sparse power-iteration (personalised PageRank) that lets a query's causes
propagate back to *their* causes. Animate: a query lighting one node, then a pulse
propagating one more hop back along precomputed edges to a node cosine never reached;
a small "built offline · 0 inference-time graph cost" stamp. Stats (count up): on
held-out **2-hop chain** targets the fused retriever lifts recall **+29% over the
single hop** (R@10 0.044 vs 0.034; R@30 0.099 vs 0.083), in **< 0.5s on CPU**. And it
carries to the report: in a **blind 20-question head-to-head** vs the single-hop
method, adding the graph won the **causal/second-order questions 7 of 9** (near-even on
plain descriptive ones) — a real second-order edge, exactly where it should show up.
HONESTY (keep it): the lift is **specific and earned, not universal** — it shows up on
*2-hop chain* questions, not on direct causes (there the single hop already wins); and
raw PageRank alone is a **hub magnet** (85% of its top hits are macro "cause-of-
everything" boilerplate) — the usable signal needs hub-suppression + fusion with the
direct hop. So: a precomputed graph that buys real second-order reach for near-zero
query cost, scoped honestly to the questions that need it.

**8 — Innovation 3: a gate that can't be charmed by fluent prose.** Every claim gets a
**status × confidence** re-derived mechanically from the cited passages, and the gate
*overrules the writer*: **corroborated / supported / contradicted / no-evidence-found**.
The subtle move: it splits the old flat "unsupported" into **"searched, genuinely
absent" (high confidence)** vs **"didn't look hard enough" (low)** — different facts
for a decision. Plus **claim-specific corroboration** (counts distinct sources that
confirm the *exact* proposition, not topical neighbours) and **numeric verification**
(every figure — percent, quantity, **or price** — must appear in a cited passage, or the
claim can't be promoted). Honest scope: this catches *unsourced* numbers; a *wrong-context*
number that does sit in some cited source (baseline's **$4,525 gold** — a forecast quoted
as spot) slips the mechanical check and is caught by the blind judges instead. Two
independent nets, and the pitch names which one catches what. Animate claims dropping onto
a status grid; a couple auto-demote with a stamp.

**9 — Better inputs: intra-day capture.** Scraping intra-day (not end-of-day) preserves
**exact publish timestamps that are otherwise lost** — which lets us align news to
price moves at event-time resolution for quant signals, *widens the set of citable
sources*, and brings in **foreign-language sources** most pipelines drop. Animate a
timeline where vague "date-only" markers snap to precise timestamped ticks aligning
against a price series; flags of multiple languages feed in. Caption: "Better sources,
timed precisely, in more languages — the citation base the rest of the method stands
on."

**10 — The proof (the money slide).** A blind A/B: **20 questions** (the full benchmark
set), same raw-text search surface, same model. Baseline = plain cosine RAG. Method =
the whole process incl. the semantic graph. Two instruments: a mechanical citation gate
+ blind pairwise judges (labels randomised per question, told to ignore process and
judge content only, with corpus spot-checks of the numbers).
- **Blind judges: the method wins 20 / 20** (avg score **10.75 vs 8.10** out of 12;
  **13 of 20 decisive** by ≥3 points, only 2 within a point; on no question did the
  baseline score higher).
- Animated side-by-side bars, "method" racing past "baseline":

| metric | baseline | method |
|---|---|---|
| corroborated (≥2 aligned sources) | 14.5% | 42.4% |
| **counter-evidence surfaced** | **0** | **44** |
| evidence density (OK cites / claim) | 1.26 | 2.10 |
| **cross-vocabulary causal evidence** (causal-lift cites) | 1.5% | **4.4%** |
| hallucinated citations | 0 | 0 |
| mis-aligned cites (honest cost) | 4.2% | 5.5% |

These are **mechanical, arm-blind measurements** (the gate re-derives every claim's
support from the cited passages — no judge opinion). The headline: across **20 questions
and 256 baseline claims, plain RAG surfaced counter-evidence exactly ZERO times.** It
admits gaps, but it never goes looking for what would prove it wrong. Animate the "0"
holding while the method's "44" counts up.

Second objective proof — **the evidence a keyword system can't reach.** Scoring every
citation by topical similarity *and* by the learned causal operator, the method carries
**~3× more "causal-lift" cites** — sources the operator ranks high but plain cosine ranks
low — and **70% of them are load-bearing on corroborated/contradicted claims** (vs 20%
for baseline). That is the cross-vocabulary reach, measured at the citation level, not
asserted. Honest scope: it's a *targeted* 4.4% of citations — the differentiating layer,
not the backbone.
Small honest footnote (keep it): the method's mis-aligned-cite rate is *higher* (5.5 vs
4.2%) — the multi-hop reach cites lower-cosine cross-vocabulary sources on purpose, so
the gate flags more; hallucinated cites stay at zero. And the concrete saves: the gate +
judges caught **$4,525 gold**, a fabricated **−4.1% China FAI**, and an **IEA supply
figure mislabelled as demand** — all in the baseline, none in the method.

**11 — Output quality, shown not told.** Two real report snippets side by side, same
question, typed on scroll:
- Baseline (Q2 policy): a single "everyone is hawkishly tightening" narrative — **and
  it missed the upstream driver entirely** (the Strait-of-Hormuz energy shock).
- Full (Q2): a **split verdict** — "no uniform response": energy-importing Asia
  hawkish, India defending via FX intervention while holding rates, others dovish as
  the shock reversed ~30% — each claim tagged status + confidence, the driver surfaced
  by the receptor at causal-score 0.58 where cosine saw only 0.36.
Caption: "The difference isn't polish. It's being right about the shape of the world."

**12 — How to approach it (playbook) + cost.** Animated checklist that fills in:
1. Write the **process** (disconfirm + confound + coverage) — biggest lift, least code.
2. Build the **mechanical gate** — makes honesty a guarantee, not analyst luck.
3. **Decompose to learn / retrieve the original** — structure as signal, source as
   evidence.
4. Add the **learned causal operator** for hypothesis discovery — it proposes the
   upstream question the analyst wouldn't ask; upgrade its signals over time. For
   *second-order* drivers, precompute a **semantic causal graph** from the same
   operator and reach them multi-hop — but build it **offline**, so query cost stays
   flat.
5. Keep LLM decomposition **training-time only**, never in the inference loop.
Cost honesty: **~90% of the lift is steps 1–3 (cheap, fast). The full method is
~3–4× retrieval calls and only ~1.8× tokens**, and deliberately makes **fewer, harder
claims.** The asymmetry favours it: checking is cents; being wrong compounds.

**13 — The honest limits (this slide earns trust).** Plainly:
- **n = 20** questions on a ~2-week single-domain corpus — a real result, not a
  universal benchmark; the mechanical gate aggregates are the most robust part.
- **2 of the 20 wins were within a single point** — on saturated-coverage topics a
  good plain analyst nearly matches; the guarantee is the *floor*, not the ceiling.
- The **direction oracle** is a *lead*, not evidence — it agreed on some chains and
  dissented on others; we record it, we don't trust it as proof.
- It recovers **candidate** causal structure (directional association + time order),
  not proven causation — every candidate goes to the gate.
- The **semantic graph's** multi-hop lift is **narrow**: measured on 2-hop *chain*
  targets, not direct causes, and only after hub-suppression + fusion (raw PageRank
  alone is 85% macro-hub noise). We claim second-order *reach*, not a universal boost.
  Its 20-question report-level win (7/9 causal) was measured against an earlier-run
  single-hop arm, so run/protocol variance is not fully isolated — the causal-vs-
  descriptive split, not the headline margin, is the honest signal.

**14 — Close.** Big line: **"Plain RAG gives you a plausible report. This gives you an
auditable one — for a fraction more compute, and most for a fraction more code."**
Sub-line: *Proving a claim is cheap. Being wrong is not. This makes the cheap thing
automatic — and it's sharpest exactly where information is thin.*

## Hard rules on the data

- Use **ONLY** the numbers in this prompt. Do **not** invent, re-round, or extrapolate
  statistics. For any unlabeled visual, keep the label qualitative.
- Keep every caveat on slide 13 — the honesty is the pitch.
- Do **not** mention or invent any "polarity"/sentiment-direction probe; it is unproven
  and excluded.

## Design & motion direction

- **Aesthetic:** "Bloomberg terminal meets a well-set scientific paper." Near-black
  background (`#0a0e14`), ONE restrained accent (electric cyan `#22d3ee` OR amber
  `#f5a623` — pick one, commit), red only for "wrong/contradicted" (`#ef4444`), muted
  green for "verified" (`#10b981`). Generous whitespace. Monospace for numbers/stats
  (`JetBrains Mono`), a clean grotesque for prose (`Inter` or `Space Grotesk`).
- **Motion:** every stat **counts up** on entry; bars **grow/race**; the pipeline
  **flows** with a moving gradient; the receptor is a **vector rotating through a
  matrix** into a highlighted region (SVG/canvas); claims **drop and stamp**; the
  sparse-advantage curve **draws in** with `stroke-dashoffset`. Stagger ~80ms. Honour
  `prefers-reduced-motion` (snap to final, no motion).
- **Navigation:** vertical `scroll-snap`, a slim right-side progress rail (15 dots),
  arrow-key/space nav, and a fixed small credibility ribbon: "20 questions · blind-judged
  · mechanical gate · measured 2026-07".
- **Charts:** hand-built inline SVG/canvas (no chart library). GSAP + ScrollTrigger via
  CDN allowed; still degrade without it.
- **Performance:** one file, < 200KB excluding fonts, 60fps, animate only
  `transform`/`opacity`. Comment each section so copy is easy to edit.

Deliver the complete HTML in one code block. First give me a 4-line outline of the
visual system (palette, type, motion signature, nav) for approval.
