# Retrieval QA Report — market-color fact pipeline (2026-07-09/10)

Summary of a two-day investigation into graph-augmented retrieval over the
market-color fact layer, in Q&A form. Full experimental detail:
`docs/ppr-experiment-results.md`. All numbers measured on the 2026-06-27→07-01
crawl window (2,231 docs; 3,066 extracted facts / 2,789 entities / 808
resolved causal links from 734 docs).

---

**Q. The original hypothesis: does query-conditioned propagation over a
sparse entity graph (dense + sparse + learned traversal weights) beat dense
retrieval?**

No. Dense-seeded Personalized PageRank (≈HippoRAG) gained ~2 recall points
at k=10 and nothing at realistic budgets; query-conditioned edge weights,
hub damping, and deeper propagation were all flat. One-shot propagation
ranks by connectivity, not query relevance — it *reaches* the right facts
but cannot rank them into any budget.

**Q. What actually determines retrieval quality here?**

Two things, in order: (1) the extraction layer — because `decompose_facts`
fuses each claim with its stated cause, most "why" questions are single-hop
over facts; (2) the retrieval budget — see next.

**Q. Was top-k budget a real constraint?**

It was THE constraint. Sweeping k: dense recall on hard cross-document
causal cases goes 0.78 (k=10) → 0.85 (k=30) → 0.93 (k=50). Most "hard"
evidence sat at ranks 11–50. Nearly all k=10 differences between retrieval
strategies dissolve at k=50. Facts are ~70 tokens; k=50 is ~3.5k tokens.
**Production floor: k≈50. Judge all future retrieval work against dense@50.**

**Q. Is anything genuinely invisible to dense retrieval?**

Yes, but little: 2 of 16 hard cases at k=50 (e.g. "what pushed LNG to
multi-year highs?" → the Kiku attack / US airstrikes root events). These
chain-end facts share no semantics with the query at any k; only following
cause links finds them.

**Q. What is the winning retrieval shape?**

Trigger-gated assembly (`asm`): dense primaries; a primary whose stated
cause points outside its own fact gets attached context — the best
cross-document fact for its cause text (upstream) and the best causal
sibling from the same source document. Never worse than dense on any of 32
cases at any k; ahead on the easy set (+3 recall / +6 full-coverage at
k=50); closes part of the chain-end gap. Its safety is structural: when the
trigger doesn't fire, it IS dense.

**Q. Do the attachments change generated answers, or just retrieval
metrics?**

Measured by counterfactual ablation (24 cases, blind pairwise LLM judge):
attachments are cited in 71% of answers but net answer quality is a wash —
EXCEPT on chain-end cases, where they convert to clear wins. Assist value
is concentrated exactly where the retrieval gap is semantic.

**Q. What changed in the app as a result?**

`mcp/qdrant_facts_server.py::search_market_facts`:
- default `limit` 8 → 40 (cap 25 → 60) — the single highest-payoff change
  (+15 recall points on hard causal questions);
- new trigger-gated **cause-driver attachment**: seeds with an unresolved
  cause get the best cross-doc fact for their cause text, returned with
  `relation: "cause-driver"`;
- `market_facts` reseeded from the rebuilt parquet (3,066 facts, was stale
  at ~676).

Extraction prompts (`extract_template.md`, `extract_prompt.md`,
`CURSOR_DRAIN_PROMPT.md`) now require the MOST SPECIFIC stated cause
(concrete event/actor, never "the conflict") — pays off on the next
extraction run.

**Q. What was tried and rejected, with evidence?**

- Static PPR as ranker (all weightings): flat, structurally wrong objective.
- Walker refinements (novelty filter, downstream boost): neutral-to-negative.
- Naive expansion stacking (sib+iter): negative — slot competition.
- Story-cluster dedup: corpus is only 7.5% duplicated (3,066→2,836
  clusters); dedup HURT recall at this rate. Code kept behind `_dd` flags;
  revisit when duplication reaches ~25–30%.

**Q. What are the known scale risks (corpus 100×)?**

Wire-copy duplication (fix: story-clustering at extraction — re-measure the
facts→clusters ratio after each expansion); hub-entity explosion (graph
usable only for pruning, never ranking); generic cause texts (fixed at the
prompt); temporal windowing becomes mandatory (schema already carries
indexed timestamps). Sibling expansion is scale-immune (doc-local).

**Q. How does the tuned pipeline compare to plain semantic chunking over
raw articles?**

See final section below — blind head-to-head on 8 fresh causal questions,
matched context budget (same characters), answers blind-judged.

---

## Blind head-to-head: tuned facts vs chunk-RAG

Setup: 8 fresh causal questions (never used in any tuning —
`eval/blind_questions.jsonl`); baseline = classic semantic chunking over raw
article bodies (~1200-char chunks, 200 overlap, 7,242 chunks, same MiniLM
embeddings); tuned = asm@50 facts. **Matched context budget per question**
(same character count, ~8–10k chars ≈ 2–2.5k tokens: 50 facts vs 6–9 raw
chunks). Answers generated and blind-judged pairwise (opus-4-8, randomized
A/B). Harness: `eval/chunk_baseline.py`; outputs `eval/blind_out*/`.

| comparison | facts | chunks | ties |
|---|---|---|---|
| chunks over ALL 2,231 docs | 3 | **5** | 0 |
| chunks restricted to the 734 fact-extracted docs (coverage control) | 4 | 4 | 0 |

Findings (n=8 per run, single judge — directional only):

1. **Uncontrolled, plain chunking WON** — because the chunk index sees the
   ~1,500 docs the fact layer never extracted, and judges rewarded the
   specific figures/attributions found there. The fact layer's 33%
   extraction coverage is costing real answer quality **today**.
2. **Coverage-controlled, it's a dead heat — and forensics on the 4 chunk
   wins explain why.** Tracing the 9 judge-rewarded details through the
   pipeline: 4/9 were PRESENT in the facts context but unused by the
   generator (chunks embed the journalist's assembly — ordering, salience,
   juxtaposition — which the generator otherwise has to rediscover across
   50 shuffled atoms); 4/9 were extracted facts that didn't rank in top-50
   for the query (atomization removes document-level relevance smearing:
   a satellite fact like "MoF spent ¥11.73tn" must earn its rank alone,
   while its chunk rides the whole passage's relevance); only 1/9 was a
   true extraction loss. Over-distillation by the extraction model is NOT
   the dominant failure. Source diversity (facts contexts carried 2–3.5×
   more distinct sources — 5–14 vs 3–4) showed no effect on verdicts:
   every justification turned on specificity/mechanism, none on sources.
   Candidate fix, untested: small-to-big rendering — retrieve at fact
   granularity, render grouped by parent document (or attach the parent
   chunk), restoring the journalist's assembly without losing structure.
   Test blind against chunks before believing it.
3. **Implication for the product:** the extraction layer's justification is
   NOT raw answer quality — it is structure: per-claim provenance and
   citation, entity/desk/temporal filtering, dedup, aggregation
   (transmission maps, desk briefs), and the cause fields the app's new
   `cause-driver` expansion uses. Those capabilities don't exist over raw
   chunks. But for a bare "ask a question, get an answer" path, chunking is
   near-free to build and competitive — worth remembering before spending
   further on retrieval sophistication.
4. **Priority order confirmed:** (1) extraction coverage (drain the
   remaining 58 batches — the only intervention that moved ANSWER quality
   in this comparison), (2) budget (k≈50, applied to the app), (3)
   everything else at the margin.

### Fact-anchored chunk hybrid (fchunk) — beats both parents

User-proposed synthesis: **fact as retrieval key, parent passage as
payload.** Context = top-5 asm facts as anchors (one per doc), each rendered
as its best-matching parent chunk (claim-embedding match within the doc) +
KEY FACT + DRIVER lines; remaining budget filled with atomic facts for
breadth. Same char budget. Harness: `eval/hybrid_test.py`.

| matchup (8 fresh questions, blind judge) | hybrid | opponent |
|---|---|---|
| vs chunk-RAG (coverage-controlled) | **5** | 3 |
| vs pure facts (asm@50) | **6** | 2 |

11–5 combined (p≈0.1 — directional, n small). Mechanism confirmed by the
win pattern: passages restore the journalist's assembly (fixes the
4/9 unused-evidence failures), breadth fill keeps multi-source coverage
(why it also edges chunks). Residual losses = the same satellite-ranking
cases: anchors are chosen by fact retrieval, so a decisive fact that
doesn't rank never gets its passage anchored — a ranking problem, not a
rendering one. First intervention in the whole investigation that moved
blind answer quality. Production path: per-seed parent-passage lookup is
one filtered vector query against `market_color_chunks`; validate on a
bigger question set before wiring into `search_market_facts`.

### Hybrid v2 — the weak spot diagnosed and fixed (`eval/hybrid2.py`)

Diagnosis of the satellite-ranking failure (was it the primitive chunking?):
NO — two causes upstream of chunking, both measured:
1. **Facts were embedded naked.** Atomization strips document context from
   the embedding: "MoF spent ¥11.73tn" carries no USD/JPY signal alone.
   Prepending the doc title ("{title} — {claim}") moves the decisive facts
   from rank #194→#11 and #302→#9 for the USD/JPY query. Near-zero cost.
2. **Single-route passage nomination.** In v1 a passage could only enter via
   a ranking fact. Title context does NOT rescue every satellite fact
   (Barclays #442→#228; the 65% threshold got WORSE) — those passages must
   be nominated at passage granularity.

v2 = contextual fact embeddings + dual-granularity nomination (passages
nominated by fact-route AND direct chunk retrieval, fused by max score,
KEY FACT/DRIVER annotations, atomic-fact breadth fill; same char budget;
chunk nomination restricted to extracted docs for a clean mechanism test).

| matchup (blind, 8 questions) | v2 | opponent |
|---|---|---|
| vs chunk-RAG (restricted) | **6** | 2 |
| vs hybrid v1 | **5** | 2 (+1 tie) |

Both diagnosed cases (usdjpy, oil-forecast-cuts) flipped to v2 wins — each
via the mechanism predicted for it. Progression at matched budget vs the
same chunk baseline: pure facts 4–4 → v1 5–3 → v2 6–2. Residue: riot
(assembly/synthesis, not retrieval), jetfuel (flipped v1→chunks; single
case, likely noise). Production implications: (a) reindex `market_facts`
with title-contextualized vectors; (b) `search_market_facts` should fuse
chunk-nominated passages — requires indexing `market_color_chunks` in prod.
Validate on 30+ questions first.

### Coverage drain via Haiku — no answer-quality benefit; big causal-density gain

The remaining 58 batches (~678 docs) were drained through the claude CLI on
**claude-haiku-4-5** with the specific-cause prompt (~$3–4, ~40 min, 4
parallel; `facts_work/run_claude_batches.sh` + `validate_claude_batches.py`,
zero invalid outputs). Corpus: 3,066 → **4,615 facts**, 734 → ~1,400 docs
(63% of crawl).

**Methodology effect (old codex prompt vs new haiku prompt):** facts
carrying a stated cause **38.5% → 66.2%**; specificity metrics flat
(avg ~9 words, generic labels ~0%). Haiku yields fewer facts/doc
(~2.2 vs ~4) — spot-check new-batch claim quality before relying on them.

**Coverage effect on blind answers (the 2×2 that settles it):**

| vs FULL-corpus chunk-RAG (8 questions) | W–L–T |
|---|---|
| pure facts, pre-drain | 3–5–0 |
| hybrid v2, pre-drain (734 docs) | **5–2–1** |
| hybrid v2, post-drain (~1,400 docs) | 4–4–0 |

The v2 architecture had ALREADY closed the coverage gap before the drain —
passage rendering recovered the narrative value that made full-corpus
chunks win against bullet-list facts. Adding coverage changed nothing
measurable (4–4 vs 5–2–1 is two verdicts at n=8). The drain's real value is
structural: near-doubled causal density feeding cause-driver expansion /
graph features, and 2× filterable fact coverage for briefs — consistent
with the report's core thesis that the fact layer's payoff is structure,
not bare QA.

## FINAL: large-sample blind eval (24 questions, double-judged)

The definitive comparison (`eval/large_blind.py`,
`eval/blind_questions_large.jsonl`): 24 fresh causal questions across all
desks (many only answerable via newly-drained docs), hybrid v2 (post-drain,
4,615 facts) vs full-corpus chunk-RAG, matched char budgets, and **each pair
judged twice with flipped A/B order** — a win counts only if won in BOTH
orders.

| outcome | count |
|---|---|
| v2 wins (both orders) | 6 |
| chunks wins (both orders) | 8 |
| tie / split across orders | 10 |

**Statistical parity** (6–8 among decisive, p≈0.8). Two lessons with weight
beyond this comparison:

1. **10/24 pairs split across judging orders** — single-pass LLM verdicts
   carry heavy position/noise components. All earlier n=8 margins in this
   report (6–2, 5–2–1, 5–3) should be read with that discount; the
   double-judged protocol is the standard going forward.
2. **Bare causal QA is at an information ceiling.** Both systems re-serve
   the same articles; a short causal answer saturates what either can
   extract. Further retrieval optimization for single-question QA is
   measurably pointless on this corpus. The structured layer's case rests
   entirely on what chunks cannot do — per-claim provenance, entity/desk/
   temporal filters, cause chains, cross-document aggregation (briefs,
   transmission maps) — which this QA-shaped eval cannot and does not
   measure. Possible drag worth one future check: chunks' clean wins skew
   toward newly-drained stories where fact anchors are Haiku-thin
   (~2.2 facts/doc).

**Question-set bias caveat (user-caught):** the large-set questions were
mined from individual causal facts, so ~23/24 were single-article-
answerable — chunk-RAG's home turf by construction. Parity there measures
the floor, not the product. Hence the final benchmark below.

## STRATIFIED BENCHMARK — the product-shaped comparison (final)

`eval/strat_blind.py` + `eval/blind_questions_strat.jsonl`: 23 questions
across five categories mirroring actual product query types; matched
budgets; double-judged (win = both A/B orders). The five **chain**
questions use **canonical-document exclusion**: the article(s) explicitly
stating the endpoint⟵root link are resolved by pattern and removed from
BOTH systems — the chain provably exists but must be inferred from
surviving fragments. Coverage-oriented judge for agg/screen/pivot
(distinct correct items, attributed detail, organization).

| category | v2 | chunks | tie |
|---|---|---|---|
| aggregation (briefs) | **4** | 0 | 1 |
| chain (canonical excluded) | 2 | 1 | 2 |
| screening | 3 | 2 | 0 |
| temporal pivots | 2 | 1 | 1 |
| control (single-story) | 0 | 1 | 3 |
| **total** | **11** | **5** | 7 |

The category gradient is the result: v2's advantage grows monotonically
with how much cross-document structure the question demands — decisive on
aggregation (4–0), parity on single-story controls — exactly the mechanism
prediction. The earlier "parity" verdict was an artifact of testing only
single-article questions. On a representative mix, the structured
pipeline wins 11–5 (decisive verdicts; p≈0.1–0.2 — the aligned per-category
pattern carries more weight than the aggregate). Canonical-excluded chains:
v2 edges 2–1–2 — true chain inference is hard for both systems; DRIVER
fields help reassembly but aren't dominant.

**Final verdict of the investigation:** the fact layer earns its cost on
product-shaped work (briefs, screens, synthesis) and costs nothing on
single-story QA. Retrieval sophistication beyond hybrid v2 is not where
the next unit of value lies; question-set design and extraction quality
are.

### Postscript — was the cause-walk discounted on biased questions? (retest)

User challenge: the early evals that benched the cause-walk used the same
fact-mined (single-article-skewed) questions later found to bias the blind
sets. Retest (`eval/chain_retest.py`): **15 canonical-excluded chain
questions** (5 reused from the stratified set — v2 answers carried over
byte-identical; 10 newly mined), v2+cause-walk vs v2, same budget,
double-judged.

Result: **walk 5 – v2 6 – tie 4. The discount was earned** — even on the
walk's home distribution it is net neutral at fixed budget. Sub-pattern
(small n): on the 5 deep multi-link chains the walk went 3–2; on the 10
shallower mined chains 2–4–3 — walk lines pay for genuine depth and tax
everything else via budget displacement. Root cause: hybrid v2's fixes
(contextual embeddings + dual passage nomination) already recover most
upstream evidence the walk was built to reach. Depth-gated walking is
conceivable but at this margin would be fitting noise.

### Postscript 2 — the conditional gate (zero-compute test on existing verdicts)

A gated arm's answers are by construction v2walk-where-fired / v2-elsewhere,
so gates were testable purely against the existing 15 verdicts. Primary
gate PRE-DECLARED from the mechanism: **"unexplained driver"** — fire the
walk only when some top fact's DRIVER text has no semantic cover in the
already-selected context (max over drivers of 1 − max-cos(driver, context
items)).

Result: the three highest-gate questions are all three of the deep-chain
walk wins (0.78/0.71/0.69 — crude-ceasefire, btc, gas); implied gated
record **3W–0L–12T vs v2** (in-sample threshold ≈0.69). Ordering signal
p≈0.02 under the null. Both exploratory alternates failed (driver-query
distance weak; recursive depth non-discriminating) — specificity favors
the mechanism story. Same trigger philosophy as the asm arm: fire only on
provable gaps, no-regression shape.

Status: suggestive, in-sample, n=15, middle of the ranking noisy.
Validation recipe (cheap, automated): mine ~15 fresh exclusion chains,
FIX threshold 0.69 beforehand, check fired-walks keep winning. Adopt into
production retrieval only after that passes.

### Postscript 3 — gate validation FAILED; temporal walk emerged instead

Out-of-sample run (`eval/gate_validate.py`): 20 auto-mined fresh exclusion
chains (subjects excluded from all prior evals), frozen g1 threshold 0.69,
double-judged. **Fired (n=6): 2W–1L–3T vs unfired (n=14): 5W–2L–7T — no
separation, and the gate fired on a loss.** The in-sample 3W–0L was
threshold-fitting, as suspected. Structural features recorded (temporal
precedence saturated at 1.0 in a one-week corpus; graph distance showed no
alignment).

**Exploratory surprise: the walk itself went 7W–3L–10T overall on this
batch** (previously 5–6–4). Confounded change: this run's walk candidates
were TEMPORALLY FILTERED (upstream facts must predate the effect — the
arrow-of-time structural prior). If the filter is the cause, the correct
conclusion is not "gate the walk" but "fix the walk's candidates and walk
always." Isolation test (same 15 questions as the original retest, only
the temporal filter changed) run as Postscript 4.

### Postscript 5 — learned fusion on synthetic chains (FIRST POWERED POSITIVE)

User-proposed program: decompose queries into sparse+dense sub-retrievals
over independent indexes, fuse, and TUNE the weights on synthetic causal
chains with mechanical labels (`eval/fusion_train.py`). 325 auto-mined
exclusion chains, labels = endpoint-support + root-support fact sets;
metric = chain-coverage@30 (both sets represented in top-30 fused);
70/30 train/test; judge-free.

| config | train | TEST (n=98) |
|---|---|---|
| dense only | 0.643 | 0.592 |
| dense + lexical (BM25) | 0.718 | 0.643 |
| all 4 routes equal | 0.692 | **0.694** |
| tuned (600-config search) | 0.740 | 0.673 |

Findings: (1) **multi-route fusion +10pts held-out coverage** — the first
adequately-powered positive result of the program; (2) **the lexical leg is
the workhorse** (+5pts alone; highest tuned weight 1.31) — the one standard
component never previously tested, and the direct fix for the diagnosed
satellite-fact failures; (3) **tuning overfits at 227 train chains — equal
weights beat tuned on test**; the tuned route ordering (lex > dense >>
walk 0.39 > graph 0.16) is the useful residue, finally reconciling the
underpowered walk/graph judge results: small positive candidate sources,
never primary. Also: weighted-walk judge pilot on 26 VERIFIED questions =
9W–6L–11T (p=0.30, best walk showing but underpowered; no g1 gradient —
gate dead a third time). Judge-based power on a one-week corpus is
exhausted (~26 verifiable fresh chains); mechanical-label evaluation is
the scalable path. Remaining step: judge-validate equal-weight fusion
end-to-end vs v2 on the verified set.

### Postscript 6 — fusion end-to-end: coverage ≠ answer quality

Judged validation of the fusion champion (equal-weight RRF over 4 routes,
v2-style rendering over the FUSED order) vs v2 on the 26 verified
questions: **fusion 5 – v2 12 – tie 9 (p≈0.98)** — the retrieval-coverage
winner LOSES at the answer level. Mechanism: chain-coverage@30 rewards
tail presence; judged answers are built from the HEAD (the 5 passages),
and RRF reordering anchors passages to lexically-matched but contextually
weaker facts, displacing dense-best anchors.

Standing conclusions: (1) the production shape — dense head untouched,
lexical hits as an ADDITIVE appendix (`seed-lex`), cause-driver expansion —
is the correct integration of the fusion finding and is what shipped;
(2) chain-coverage is a valid tuning signal only for the appendix/tail;
any future weight-tuning needs rank-weighted or head/tail-split coverage.
Third demonstration this week of the retrieval-metric vs end-task gap.

### Postscript 7 — two-pass analyst-note generation + dissent modeling

User's thesis: news is aggregated collective reasoning; a report's value is
distinguishing consensus from potentially-relevant dissent, and answers
should form by reviewing the broad picture first, then targeted nuance.
Implemented (`eval/twopass.py`): breadth survey (haiku over 50 atomic
facts) -> nuance requests -> parent-passage fetch -> synthesis (opus) with
MECHANICALLY COMPUTED consensus/dissent annotations (conflicting
`direction` / diverging `cause` across `source_name`s — structure the
chunk world cannot compute) and an analyst-note format (consensus /
sourced disagreements / tail signals).

Evaluation: ANSWER-KEY GRADING — exclusion questions carry ground truth
(the excluded fact's cause); haiku grades 0/1/2. Cheap, objective, scales
past the judge ceiling. Calibration on existing answers: v2 1.46,
weighted-walk 1.42 (metric discriminates; corroborates judge parity).

| metric (26 verified questions) | v2 flat | two-pass |
|---|---|---|
| root-cause recovery (0-2 mean) | 1.46 | **1.58** |
| paired | — | 5 better / 3 worse / 18 same |
| dissent-awareness where disagreement existed | 3/25 | **24/25** |

**The dissent result (12% -> 96%, no accuracy tax) is the largest
product-relevant effect measured in the entire investigation.** Epistemic
structure does not emerge from retrieval quality — v2 had the same facts —
it must be computed and demanded. Production path: dissent detection in
the brief renderer / search response + the analyst-note synthesis shape.

**Control (fair fight): chunk-RAG given the SAME analyst-note prompt**
(matched budget, exclusions, same graders): root-cause **1.54**,
dissent-awareness **21/25 (84%)**. Honest decomposition: ~85% of the
two-pass effect is THE PROMPT — any retrieval backend gets it by demanding
the epistemic structure; the computed annotations + staged reading add a
small consistent edge (+0.04 accuracy, 96% vs 84% dissent — n=26, within
noise). The earlier "chunks cannot do epistemic structure" claim was
overstated and is corrected here. Ship the analyst-note shape regardless
of backend; keep mechanical dissent detection as a cheap enhancer. Open
question for scale: chunks carry 3–4 distinct sources per budget vs facts'
9–14 — the annotation edge plausibly widens on diverse corpora; unmeasured.

### Postscript 8 — regional source dependency + exogenous price joins

**Regional dependency (`eval` inline analysis):** single-source subject
coverage is 82–97% across ALL EM regions (Africa 97%, China 96% — Metal.com
commodity lens dominating, LatAm 42 facts total). Consequence: dissent
detection — the largest measured product effect — is structurally starved
outside the US/EU core (disagreement needs ≥2 sources per story).
Recommended feeds by region (add with `source_region` tags for
localized-query routing + local-vs-global dissent): bne IntelliNews /
Eurasianet (CIS), Caixin / Yicai / Nikkei Asia (China), Economic Times /
Mint (India), Valor / Bloomberg Línea (LatAm), BusinessDay NG / Moneyweb
(Africa), Al-Monitor / Zawya (MENA), Business Times SG / Jakarta Post (SEA).

**Exogenous join (`eval/exo_join.py`) — the operation chunk-RAG cannot
express** (requires entity→instrument, event_time, direction keys):
- 242 price_move facts joined to daily OHLC (29 instruments).
  **Market-confirmation rate 62%** (WTI 93%, Brent 73%, BTC 66%,
  gold 47% — intraday-claim vs daily-bar mismatch suspected).
- **Extraction confidence predicts reality: conf≥0.9 → 64% confirmed;
  conf<0.9 → 35%.** Enables: market-confirmation ranking prior;
  extraction-quality auditing (compare Haiku vs Codex batches by
  confirmation rate); continuous calibration loop.
- Reverse hypothesis exploration: top-|z| moves → candidate causes from
  the fact base. The week's biggest event (2yr yield +4.2σ, Jun 30) flags
  UNEXPLAINED — partly instrument-map gap (yields/FX unmapped; the
  explanation exists: July-hike odds 10%→35%), partly the feature working:
  "market moved, news doesn't account for it" is prime analyst signal.

### Postscript 9 — news-lag event study (do unexplained spikes predict?)

Second period crawled (July 2–10; sitemap-limited middle — random PAST
weeks are unreachable without archive backfill) and Haiku-drained: corpus
now **5,153 facts, Jun 27–Jul 10, two windows**. Prices refreshed through
Jul 10; instrument map extended to yields/FX/Asia indices.

Pre-registered design (`eval/newslag.py`): events = |z20|≥2 with T+2 news
available; explanation state publication-aware (0 none / 1 mention /
2 causal); outcomes frozen = catch-up rate and next-day continuation.

First results (n=7 events — machinery validated, statistically indicative
only):
- **6/7 extreme moves had NO causal explanation in same-day news** —
  unexplained is the NORM (incl. 2yr yield +4.2σ, USD/JPY −3.1σ, HSI +2.3σ).
- **Catch-up rate 1/6 (17%)**: news mostly does NOT arrive T+1/T+2 either
  (only Brent's Jul 8 spike got explained). Caveats: thin Jul 2–6 crawl
  coverage and keyword-matching recall make these LOWER BOUNDS on
  explanation.
- **Continuation: unexplained events REVERTED** (sign(retT)·retT+1 =
  −0.86% vs −0.06% for explained) — directionally the opposite of the
  informed-flow hypothesis; consistent with overshoot/liquidity noise, and
  in principle a fade signal — but n=6.
- 3/7 events regional (USD/JPY ×2, Hang Seng).

Standing value: the study machinery runs continuously as the corpus grows
(one command per window); power arrives with accumulation. Watch-list of
fixes that change the numbers: event-time matching recall (the 2yr-yield
explanation exists as FOMC-odds facts but isn't instrument-keyword-matched),
archive backfill for coverage gaps.

### Postscript 10 — full-corpus rebuild flips the event study; typed routes verdict

Full crawl drained (all 354 Haiku batches valid): corpus rebuilt to
**10,497 facts (6,600 with causes), Jun 27–Jul 10, no coverage gap**.
Both Qdrant collections re-indexed and bridged.

**News-lag study rerun (same frozen design): the headline REVERSED.**
- **5/7 extreme moves ARE causally explained in same-day news** (was 1/7
  on the gappy corpus). Postscript 9's "unexplained is the norm" was a
  corpus-coverage artifact — explanation rate is dominated by crawl
  completeness, which is itself the finding: any real version of this
  study needs archive backfill before "unexplained" means anything.
- The 2 residual unexplained events are both the **2yr yield** (+4.2σ,
  −2.8σ) and never catch up — consistent with the known keyword-recall
  gap ("2-year" vs "front-end/short-end" phrasing), not news absence.
- Continuation: unexplained reverted (−2.83% sign-adjusted, yield-scale
  inflated) vs explained flat (+0.10%) — same fade direction as before,
  n=2, anecdote.

**Typed routes + Haiku reprojection (task-22 close-out).** Tested on 324
mechanically-labeled chains (70/30 split), head-weighted metric = mean
reciprocal rank of first endpoint/root hit. Two earlier runs were invalidated
and redone: row-index labels went stale after the corpus rebuild (all-zero
scores → relabeled with FULL support sets, no 60-row truncation), and
fastembed proved **not thread-safe** — calling it from pool workers caused
the intermittent deadlocks/spins that stalled every long run today (fixed:
embed queries up front, workers only do numpy + subprocess).

Clean result (hw_TEST / cov10_TEST):

| config | hw_TEST | cov10 |
|---|---|---|
| **dense only** | **0.183** | **0.102** |
| dense+reproj | 0.139 | 0.092 |
| dense+walk | 0.134 | 0.092 |
| dense+lex_ent (typed sparse) | 0.114 | 0.071 |
| dense+lex_full | 0.108 | 0.071 |
| all_equal | 0.079 | 0.031 |

Verdict (head only): **contextual dense alone wins the head on both
metrics; every fused route dilutes it when used for final ordering.**

**Framing correction (user challenge, upheld):** the head metric answers
"can fusion replace dense as primary ranker" — a question the program
never asked. The typed/reprojection routes were proposed as suppliers of
*complementary multi-step evidence* for a reasoning stage (alt-hypothesis
exploration), not as rankers. Measured on THAT question (marginal
root-evidence recall beyond dense@50, same rankings):

| route | recovers root evidence dense@50 missed |
|---|---|
| lex_ent (typed sparse) | **79% of chains** |
| lex_full | 78% |
| walk | 69% |
| reproj | 64% |
| any route | 92% (94% of dense's misses) |

Dense@50 misses root-side evidence on 98% of chains — it almost never
carries the inference-end material alone. The entity-scoped sparse route,
worst under the head metric, is the BEST single extras-supplier. Two
different jobs, two different winners: **dense ranks; typed routes
supply.** Untested remainder: feed route-supplied extras into the
two-pass nuance stage and answer-key grade vs dense-only supply — the
end-to-end test of the alt-hypothesis strategy.

**Temporal-consistency rerun (user challenge #2, upheld — supersedes the
head table above).** Labels re-mined with a ±2-day event-time window
around the canonical event (`norm_event_time`; 212/324 chains survive;
corpus duplication measured at 18.6% same-subject/same-day cross-doc).
Result: **dense's head score HALVES (0.183 → 0.085) — about half its
"head wins" were temporally-inconsistent subject matches from the wrong
window — and the fusion dilution disappears entirely** (dense 0.085 vs
dense+lex_full 0.088, dense+reproj 0.088 = parity, n=64 test, all noise-
level), while fused configs lead top-10 chain coverage (0.078 vs 0.062).
Marginal root recall is robust to the window (98% missed by dense@50,
91% recovered by routes). Standing lesson: **the non-temporal labels
systematically flattered dense** (it excels at retrieving cross-window
paraphrases, which the old labels credited); any eval on a multi-window
corpus must window its labels by event time. The scale-up demanded
adaptation the first pass didn't give it — temporal windows and dedup
are now preconditions, not options, for corpus-growth work.

### Postscript 11 — extras-supply e2e: the hypothesis meets the cause field

The end-to-end test of the extras-supply hypothesis (`eval/extras_e2e.py`):
matched 50-fact budget, identical analyst-note instructions, only supply
differs — dense top-50 vs dense top-40 + 10 route-recovered missed-root
facts flagged as alternative-route retrievals. Answer-key graded (Haiku,
0–2 vs the excluded canonical's cause).

- **Main sample (n=30): NULL.** dense 1.67 vs extras 1.57; 23 ties;
  discordant 2–5 (sign test ns). Ceiling: both arms perfect on 20/30.
- **Scarce stratum (root evidence ≤4 docs, n=9): same shape.** 1.56 vs
  1.44, 6 ties, discordant 1–2; still 6/9 perfect in both arms.

Why the retrieval-layer win (91% marginal root recovery) doesn't convert:
**the cause-fused extraction already collapsed the hop.** Endpoint facts
carry their causes inline as text, so dense's head "names the root" even
when the root FACTS are absent — the answer key leaks through the cause
field, and corpus redundancy (18.6%) multiplies the leak. The routes
recover facts *about* the cause; the grading only requires *naming* it.

Where the hypothesis still lives (untested regimes, not refuted ones):
(1) questions requiring detail ABOUT the upstream event, not just its
name — depth grading, not naming grading; (2) corpora where cause fields
are vague/absent (weaker extraction, raw chunks); (3) low-redundancy
coverage — EM/regional (82–97% single-source) and archive scale. On THIS
corpus, at this budget, with this extractor, dense + cause-fused facts
saturate root-cause naming, and supplementary routes add retrieval-layer
recall the generation task cannot use.

### Postscript 12 — clean prospective validation (PFCA): timestamps fixed, and news arrives pre-explained

The clean-question instrument (`eval/pfca_mint.py`, all gates mechanical
and pre-registered): in-window event → multi-source consensus ground
truth (≥2 independent sources, cause cos ≥0.70) → hour-level
prospectivity (move in print ≥2h before first attribution) → semantic
leak scan over everything visible at the cutoff → pre-cutoff
answerability → plausible distractors → forced-choice scoring.

**Timestamp fix (no LLM):** trafilatura's `date` is date-only (htmldate)
and overrode even sitemap lastmod — 92% of the corpus was floored to
midnight. `extract_pub_ts` (regex over JSON-LD/meta tags) now takes
priority in the crawler, and `backfill_timestamps.py` re-fetched head
metadata for the floored docs (date-match safety rule): corpus went
**8% → 71% real time-of-day** (misses = client-rendered SPAs).
Floored stragglers get asymmetric conservatism: possible leak at 00:00,
usable evidence at 23:59 — both errors discard questions, neither
contaminates one.

**Result: zero clean prospective questions, at day AND hour granularity
— and that is the finding.** Funnel: 1,707 in-window move-groups → 199
multi-source → 36 consensus attributions → 29 with unmeasurable timing
(discarded, not faked) → 7 measurable → 0 where a bare move report
precedes the first attribution by ≥2h. In this coverage universe the
first print of a move already carries its cause; bare "X moved" reports
are follow-ups, not precursors. **News arrives pre-explained** — the
same-day-explanation result from the news-lag study, confirmed at hour
resolution from the minting side.

Paths to a non-empty prospective benchmark, in leverage order:
1. **Intraday price anchor**: take the move from hourly OHLC (spike at
   time t) instead of requiring a bare news report — G2 becomes price
   spike at t, first attribution at t+lag. The move needn't be in print
   at all; attribution timestamps (now 71%) are the only text needed.
2. EM/regional feeds, where attribution lag is structural.
3. Archive backfill for accumulation of rare lagged cases.

### Postscript 4 — isolation verdict: distribution artifact; walk retired

Temporal walk on the ORIGINAL 15 questions (`chain_out_t`, v2 answers
reused byte-identical): **2W–6L–7T — strictly worse than the plain walk's
5–6–4 on the same questions.** The fresh batch's 7–3 was question-mix
artifact number three. Walk record across three controlled distributions:
5–6–4, 2–6–7, and one non-replicating positive. **Retired.**

Why the temporal filter backfired — the useful residue: **publication time
≠ event time.** The best causal-chain statements in news are RETROSPECTIVE
(wrap-up articles published after the effects); publish-date precedence
filtering excludes exactly the most informative upstream sources. A valid
temporal adjacency must be built on normalized EVENT time — the extracted
`time` field (currently free text) needs timestamp normalization at
extraction. That is the concrete precondition for any temporal-matrix
work, and the third demonstration that the auto-mined exclusion-chain +
double-judge harness kills spurious signals for a few dollars each.

---

## Appendix — blind-test questions and answers (verbatim)

All answers generated by the same model from retrieved context only,
at matched character budgets. Winner = blind pairwise judge verdict
(randomized A/B; judge saw answers only, never labels).


### Why did the Bank of Japan signal readiness to tighten policy further?

*Verdicts: hybrid-vs-chunks → **HYBRID** · hybrid-vs-facts → **HYBRID***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **HYBRID2** · v2-vs-v1 → **HYBRID2**)*:

> The BoJ signaled readiness to tighten further as part of its ongoing normalisation process, exiting the ultra-loose policy it had maintained since 2013; after hiking its policy rate to 1.00% at its June meeting, its forward guidance reaffirmed that it would keep raising rates and adjusting monetary accommodation "in response to developments in economic activity and prices as well as financial conditions," with Deputy Governor Uchida reiterating willingness to raise rates further if economic conditions align [1][2][4]. Underlying drivers include inflation exceeding the BoJ's 2% target, fueled by a weaker Yen and rising energy prices, plus the prospect of rising wages [4]. One board member argued the policy rate must be brought closer to the estimated neutral rate of around 2% at a pace of once every few months, and analysts at Mitsubishi UFJ Morgan Stanley expected the BoJ to reduce accommodation as much as possible by next summer [1][6]. Missing: the context does not give a single explicit BoJ statement of intent beyond these normalisation-and-price-stability rationales, and it notes countervailing pressures (a board dissent citing Middle East conflict risks, and government pressure to keep borrowing costs low) rather than reasons to tighten [5][8].

**Hybrid (facts anchored to parent passages):**

> The Bank of Japan signalled readiness to tighten policy further because it was focused on taming price pressures linked to an Iran-driven energy shock and a weak yen [3]. Global energy prices had skyrocketed during the economic warfare between the United States and Iran, feeding those pressures [6]. This came after the BOJ raised rates to a 31-year high of 1% at its June meeting as part of its broader normalisation effort [3][14]. Supporting the concern over inflation, a BOJ measure of potential inflation (excluding fresh food and government subsidies) has averaged roughly 3% over the past four years [23], and a former BOJ executive expected the next hike before December citing inflation near 3% [16]. The context does not detail the BOJ's internal deliberations beyond these energy- and yen-driven price pressures, and it notes some countervailing signals (e.g., the BOJ's own assessment that price trends remain slightly below 2% [10] and a board member dissenting over risks to output and jobs [19]).

**Chunk-RAG (raw article passages):**

> The BoJ signaled readiness to tighten further largely because of a narrow political window: with PM Takaichi set to appoint several dovish board members (one replacement imminent and two more by year-end), market experts believe the BoJ could accelerate its tightening pace before those doves arrive [7]. Analysts at Mitsubishi UFJ Morgan Stanley Securities noted the BoJ "may not have much time left to continue raising rates toward the neutral rate" and may try to reduce accommodation as much as possible by next summer [7]. This is reinforced by the June Summary of Opinions, in which one board member argued the policy rate must be moved "closer to the estimated neutral rate of around 2% at a pace of once every few months" [1][7]. The underlying economic justification is that a weaker Yen and rising energy prices pushed Japanese inflation above the 2% target, with prospective wage increases fueling more sustained, demand-driven inflation [3]. Note that the context frames this partly as a race against political pressure rather than purely economic conditions, and it does not give the BoJ's own official rationale beyond these analyst and board-member views [4][8].

**Pure facts (asm@50 atomic facts):**

> The Bank of Japan signalled its readiness to tighten policy further in response to energy-driven price pressures stemming from the Middle East conflict and a weak yen [3]. These energy pressures reflected a broader surge in global energy prices during the economic warfare between the United States and Iran [6]. This tightening signal came alongside the BOJ having lifted its benchmark rate to 1%, a multi-decade high [4], and is consistent with a BOJ measure of potential inflation (excluding fresh food and government subsidies) that has averaged approximately 3% over the past four years [23]. Notably, the context contains some tension on the inflation picture, as the BOJ's own assessment is that price trends remain slightly below 2% [10]. The context does not provide a detailed official BOJ statement of reasoning beyond the price pressures cited in [3].


### Why are gold and silver falling despite elevated geopolitical tension?

*Verdicts: hybrid-vs-chunks → **HYBRID** · hybrid-vs-facts → **HYBRID***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **HYBRID2** · v2-vs-v1 → **HYBRID2**)*:

> Gold and silver are falling largely because the geopolitical tension is actually easing rather than escalating—the U.S. and Iran agreed to halt attacks and meet in Doha over the Strait of Hormuz, which improved risk appetite and undercut gold's safe-haven bid [10][11][5]. The dominant driver is Fed tightening risk: markets are pricing a more hawkish Federal Reserve (roughly 31–35% odds of a July hike) even as PCE inflation held at 4.1% year-on-year, which weighs on yield-less precious metals [1][2][4][17]. This hawkish repricing lifted the U.S. Dollar Index above 101.5 from a 52-week low near 95.5, and a stronger dollar makes dollar-priced metals more expensive and controlled [4][15]. The associated collapse in oil prices—Brent falling below $75 from over $100 during the conflict—has pushed energy back to pre-war levels, easing inflation fears but removing a support for metals [1][2][18]. A bearish technical backdrop reinforced the slide, with gold breaking below key moving averages and the $4,000 level and silver dropping below $60 support [2][3]. Notably, the context frames the tension as de-escalating, so it does not directly address a scenario of genuinely *elevated* ongoing tension.

**Hybrid (facts anchored to parent passages):**

> Gold and silver are falling primarily because a hawkish repricing of Federal Reserve rate-hike expectations has lifted bond yields and the US dollar, and higher interest rates reduce the appeal of non-yielding metals relative to interest-bearing assets [4][11][6]. Ironically, the geopolitical tension itself contributed: Middle East tensions triggered an energy-driven inflation shock, which reinforced a "higher-for-longer" rates outlook and drove the hawkish Fed repricing that weighs on gold [4][10]. Tightening financial conditions have pushed gold sharply lower alongside other risk assets [1], and for silver specifically, higher oil prices and firmer Treasury yields kept traders focused on Fed tightening risk rather than safe-haven demand [16]. There are also signs the geopolitical premium is unwinding — war fears collapsed and oil fell to pre-war levels as tensions eased, removing safe-haven support [1][2][9]. Notably, structural demand for gold as geopolitical protection remains strong among reserve managers [8], so the current weakness reflects rates and the dollar overpowering safe-haven demand rather than its disappearance.

**Chunk-RAG (raw article passages):**

> Precious metals are under pressure primarily because the market is pricing in a more hawkish Federal Reserve—with implied rate-hike odds for the July FOMC rising to around 35% (from under 10% before the meeting) and further tightening priced for September—which weighs on yield-less gold [1][2]. The prospect of higher rates has bolstered the US dollar to its strongest level since May 2025 (DXY above 101.5), and a strong dollar tends to keep gold prices controlled [2][3]. Notably, the geopolitical turmoil itself has been a driver of the weakness rather than a support: since the war began in late February, surging energy prices raised expectations of higher interest rates to combat inflation—a scenario that hurts gold—leaving bullion down more than 22% [5]. A recent hawkish peak and the sharp selloff in oil toward pre-war levels have prompted a slightly dovish repricing, suggesting pullbacks may ease absent a fresh catalyst [1].
> 
> What's missing: the context focuses almost entirely on gold; it offers no silver-specific data beyond grouping silver with gold as pressured precious metals [2].

**Pure facts (asm@50 atomic facts):**

> Gold and silver are falling primarily because of a hawkish shift in Federal Reserve expectations, which has lifted US Treasury yields and strengthened the dollar, making non-yielding metals less attractive [11][14][19][33]. Persistent inflation—with PCE at 4.1% year on year—has reinforced a "higher-for-longer" rate outlook, pushing gold to a monthly loss of around 11% and its worst quarter since 2013 [4][10][25][27]. On the geopolitical front, the tension premium has actually faded rather than supported prices: the easing of US-Iran hostilities and lower oil prices shifted trader focus toward Fed tightening risk over safe-haven demand, with silver specifically pressured by "the lack of a geopolitical risk premium" [9][16][23][24][44]. Investor outflows compound the move, as gold ETF holdings fell more than 40 tons in the quarter and silver faces continued selling from ETF investors and coin flows [18][22]. Note the picture is not uniform—renewed US-Iran tensions and Middle East risks are still cited as capping recoveries and posing upside inflation risks—so geopolitical tension is contributing indirectly through the inflation/rates channel rather than lifting prices directly [14][30].


### Why is India planning to buy more oil from suppliers like Guyana and Brazil?

*Verdicts: hybrid-vs-chunks → **HYBRID** · hybrid-vs-facts → **HYBRID***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **HYBRID2** · v2-vs-v1 → **HYBRID2**)*:

> India is diversifying its oil sourcing to reduce its heavy reliance on the Middle East following the supply shock of the Iran war, which exposed the country's vulnerability and led to shortages, higher prices, and heavy losses for refiners [1][2][3]. This dependence is especially acute because India imports about 90% of its crude needs (roughly 5 million barrels a day) and has thin inventory cover compared with buffered peers like Japan [6][10]. By taking advantage of new supply from countries such as Guyana, Brazil, and the US, India aims to secure steadier deliveries that can withstand another major disruption [1][2]. The context does not give specific volume or pricing details for the planned Guyana and Brazil purchases.

**Hybrid (facts anchored to parent passages):**

> India is diversifying its oil suppliers to reduce heavy reliance on the Middle East following the supply shock caused by the Iran war, which disrupted regional supplies and forced a diplomatic scramble to keep energy flowing [1][7][8]. This vulnerability is acute because India imports roughly 90% of its crude—about 5 million barrels a day—and has thinner inventory cover than peers like Japan, leaving it exposed to Middle Eastern supply disruptions [6][10]. Tapping new supply from countries such as Guyana, Brazil, and the US is part of that diversification strategy, alongside leaning more on spot-market purchases and trimming long-term Middle East contract volumes [1][7][8]. While US waivers permitting Russian oil purchases have offered some relief, that relief is described as only temporary, reinforcing the push toward new sources [1][11].

**Chunk-RAG (raw article passages):**

> India is planning to buy more oil from suppliers like Guyana and Brazil as part of a diversification strategy to reduce its reliance on the Middle East following the supply shock of the Iran war [1][2]. That conflict exposed India's heavy dependence on the Middle East for energy imports — from oil to LPG — leading to shortages, higher prices, and heavy losses for refiners, and even forcing a diplomatic scramble to keep energy flowing through the Strait of Hormuz [1][2]. India imports about 90% of its crude needs (roughly 5 million barrels a day), so tapping new supply from countries such as Guyana, Brazil, and the US helps insulate refiners against another major disruption [1][2]. The context does not give specific volumes, prices, or timelines for these Guyana and Brazil purchases.

**Pure facts (asm@50 atomic facts):**

> India is diversifying its oil supply toward newer sources like Guyana, Brazil, and the United States as part of a strategy to reduce dependence on the Middle East [1]. This shift was driven by the supply shock of the Iran war, which disrupted oil supplies and prompted India's state refiners to cut back on Middle East purchases—including trimming volumes under long-term contracts—and lean more on alternatives such as spot-market buys [3][4][6][7]. India is especially exposed here because it imports about 90% of its crude (around 5 million barrels a day) and has thin inventory cover combined with high reliance on Middle Eastern supply, leaving it more vulnerable than better-buffered buyers like Japan [2][10]. The context does not specify the volumes India intends to buy from Guyana and Brazil specifically, or the timeline for the shift.


### Why did USD/JPY break above its 2024 highs?

*Verdicts: hybrid-vs-chunks → **CHUNKS** · hybrid-vs-facts → **HYBRID***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **HYBRID2** · v2-vs-v1 → **HYBRID1**)*:

> USD/JPY broke above its 2024 highs primarily because of the widening interest rate differential between the U.S. and Japan, which continues to favor the dollar [2][5]. Even after the Bank of Japan lifted its benchmark rate to a 30-year high of 1%, that gradual tightening did little to offset the dollar's yield advantage, keeping the yen defensive [2][5][10]. Broad US dollar strength added to the pressure—the dollar index was on track for its fourth straight positive quarter and firmed ahead of the June Nonfarm Payrolls release, with BNY noting the renewed dollar strength pushed investors to prioritize portfolio protection over fundamentals [8][9][13]. Once the pair broke above the 2024 high near 161.95/97, the move was reinforced by strong buying momentum, with buyers repeatedly defending the breakout area as new support [1][7]. The context also cites geopolitical friction between Beijing and Tokyo and persistent capital outflows as contributing factors [3], though it does not quantify their relative weight versus the rate differential.

**Hybrid (facts anchored to parent passages):**

> USD/JPY broke above its 2024 highs primarily because of the wide policy gap between the Federal Reserve and the Bank of Japan, whose gradual tightening failed to offset the dollar's yield advantage, keeping the yen under pressure [5][6]. The dollar was further supported after a more hawkish-than-expected FOMC dot plot triggered a repricing toward higher US rates, with additional strength as the greenback outperformed ahead of the June US Nonfarm Payrolls release [2][8]. Broad US dollar strength was compounded on the yen side by intensifying geopolitical friction between Beijing and Tokyo and persistent capital outflows, which overrode otherwise strong Japanese trade data [3]. Once the 2024 high near 161.95–161.97 gave way, strong buying momentum and repeated defense of the breakout area as new support carried the pair to fresh multi-decade highs above 162 [1][7].
> 
> One gap: the context explains the fundamental and technical drivers but does not give a single triggering catalyst dated to the exact break, so the "why" is a combination of the rate differential, dollar strength, and yen-specific pressures rather than one discrete event.

**Chunk-RAG (raw article passages):**

> USD/JPY broke above its 2024 highs primarily because of the wide monetary policy gap between the Federal Reserve and the Bank of Japan, which kept the Dollar's yield advantage firmly in place—even after the BoJ lifted its policy rate to a 30-year high of 1%, the widening U.S.–Japan interest rate differential continued to favor the greenback [3][6]. Broad US Dollar strength acted as a systemic trigger, forcing investors to prioritize portfolio protection over Japanese fundamentals, while geopolitical friction between Beijing and Tokyo and persistent capital outflows added downward pressure on the Yen despite otherwise strong domestic data [2]. Japanese authorities' intervention efforts provided only fleeting relief—Tokyo spent a record 11.73 trillion yen (over $70 billion) between late April and late May with little lasting effect—so intervention fears capped momentum but could not reverse the trend [5][6]. The break past the ~161.95 level pushed the pair to its highest since 1986 [2][5][6].

**Pure facts (asm@50 atomic facts):**

> USD/JPY broke above its 2024 highs (near 161.95–161.97) chiefly because of the wide monetary policy gap between the Federal Reserve and the Bank of Japan, which left the yen under sustained pressure [5][1]. The widening US–Japan interest rate differential continued to favor the dollar, and this was reinforced by steady bets that the Fed might hike rates in the second half of the year [6][45]. Broader dollar strength also played a role, with the US dollar outperforming ahead of the June US Nonfarm Payrolls release, while the breakout itself was accompanied by strong buying momentum and buyers repeatedly defending the breakout area as support [8][1][7]. Notably, intervention fears from Tokyo had temporarily capped the pair before it broke through [38][50].
> 
> One limitation: the context describes the drivers (rate differential, Fed hike bets, dollar momentum) but does not detail *what specifically* shifted around June 29–30, 2026 to trigger the exact break beyond these standing conditions.


### Why did Riot Platforms sell more than $250 million of Bitcoin?

*Verdicts: hybrid-vs-chunks → **CHUNKS** · hybrid-vs-facts → **HYBRID***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **CHUNKS** · v2-vs-v1 → **TIE**)*:

> Riot Platforms sold more than $250 million of Bitcoin in the first quarter of 2026 as part of a "strategic evolution" into data center development [4]. The context frames this alongside a broader trend of Bitcoin treasury and mining firms shedding holdings to reposition their businesses—for example, MARA Holdings sold roughly 15,000 BTC to repurchase convertible debt and fund its pivot into an "energy and digital infrastructure company" [4].

**Hybrid (facts anchored to parent passages):**

> Riot Platforms sold over $250 million in Bitcoin in the first quarter of 2026 as part of a "strategic evolution" into data center development [1]. The context frames this as following a broader trend among Bitcoin mining/treasury firms repositioning their businesses—for example, MARA Holdings sold about 15,000 BTC to repurchase convertible debt and finance its pivot into an "energy and digital infrastructure company" [1]. Beyond the stated goal of funding its data center pivot, the context does not provide further detail on Riot's specific financial motivations or use of proceeds [1].

**Chunk-RAG (raw article passages):**

> Riot Platforms sold over $250 million of Bitcoin as part of a "strategic evolution" into data center development [3]. The sale was announced as covering BTC sold in the first quarter of the year, and it followed a similar move by fellow miner MARA Holdings, which had sold roughly 15,000 BTC a month earlier to shore up its finances and fund its own pivot toward energy and digital infrastructure [3]. The context does not give further detail on how Riot specifically deployed the proceeds or the exact strategic rationale beyond the data center pivot [3].

**Pure facts (asm@50 atomic facts):**

> Riot Platforms sold more than $250 million of Bitcoin in the first quarter of 2026 as part of a strategic evolution into data center development [1]. The context does not provide further detail on the reasoning behind this pivot (for example, financing needs or specific projects), so the driver beyond the data center strategy is missing [1].


### Why are banks cutting their oil price forecasts despite the war?

*Verdicts: hybrid-vs-chunks → **CHUNKS** · hybrid-vs-facts → **HYBRID***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **HYBRID2** · v2-vs-v1 → **HYBRID2**)*:

> Banks are cutting oil price forecasts because the U.S.-Iran memorandum of understanding has reopened the Strait of Hormuz and effectively ended the war premium, shifting the market from fear of a prolonged closure toward expectations of returning supply [1][5][9]. Flows through Hormuz are recovering faster than expected—reaching roughly 80% of pre-war levels—which is prompting repeated downgrades: Barclays cut its 2026 Brent estimate from $100 to $96 and 2027 from $88 to $85, Wood Mackenzie cut 2027 Brent to $78, and Morgan Stanley cut forecasts twice in about two weeks [2][5][3][13]. Analysts note flows only need to recover to about 65% of pre-conflict levels for a glut to form, and the market has "come full circle back to surplus," with near-zero geopolitical risk premium now priced in at around $70/bbl [3][14]. Additional bearish pressure comes from strong U.S. supply, weak Chinese demand, and Fed tightening risk weighing on demand [13][10]. The context cautions this is fragile—the ceasefire is treated as permanent despite renewed hostilities, so the cuts rest on an incomplete normalization that could reverse if the Strait shuts again [1][2][4].

**Hybrid (facts anchored to parent passages):**

> Banks are cutting their oil price forecasts because the U.S.-Iran memorandum of understanding (and the ceasefire it accompanies) shifted market sentiment away from a prolonged Strait of Hormuz closure, prompting investment banks to rush to slash forecasts even as hostilities technically continued [1][14]. The core driver is that oil flows through the Strait of Hormuz are returning faster than expected, with crude previously trapped in the Persian Gulf now reaching the market [6][9][15]. Morgan Stanley, for example, cut its forecasts twice in about two weeks—citing not only recovering Hormuz flows but also strong U.S. supply and weak Chinese demand—while Wood Mackenzie lowered its 2027 Brent forecast to $78 [9][14]. The result is a huge selloff that returned prices to pre-war levels, effectively pricing out the geopolitical risk premium (ING notes near-zero premium at ~$70/bbl) [3][7][1]. That said, analysts caution this optimism may be premature, since the "reopening" is treating a temporary ceasefire as permanent and the situation could deteriorate at any moment [1][5].

**Chunk-RAG (raw article passages):**

> Banks are cutting forecasts because the feared worst-case of the war — a prolonged closure of the Strait of Hormuz — has not materialized; instead, flows through the strait are recovering faster than expected, reaching about 80% of pre-war levels after an interim US-Iran peace deal allowed traffic to resume [1][6]. Barclays cited this Hormuz recovery in lowering its 2026 Brent estimate from $100 to $96 and its 2027 estimate from $88 to $85 [1], while Wood Mackenzie cut its 2027 Brent forecast to $78, saying a US-Iran Memorandum of Understanding shifted sentiment away from a prolonged closure [3]. Beyond the strait reopening, analysts point to fundamental oversupply: Morgan Stanley cut forecasts twice in two weeks, warning that flows need only recover to ~65% of pre-conflict levels for a glut to form, with "twin solvers" of high US exports and weak Chinese demand keeping the market oversupplied and heading "back to surplus" for 2027 [2][6]. In short, the war's supply-disruption premium is unwinding as barrels flood back from the Persian Gulf while demand stays soft, driving prices down despite the ongoing conflict [5][6]. The context does not detail forecast changes from Goldman Sachs, which is mentioned as also paring its outlook but without specific figures [6].

**Pure facts (asm@50 atomic facts):**

> Banks are cutting forecasts because the war's supply threat is unwinding faster than expected, not intensifying. After the U.S.–Iran memorandum of understanding, banks lowered their oil price forecasts as market sentiment shifted away from a prolonged Strait of Hormuz closure [1][14]. Morgan Stanley cut its forecast—for the second time in about two weeks—because flows through the Strait of Hormuz are returning faster than expected, while U.S. supply is strong and Chinese demand is weak [6][9]. With the ceasefire holding, more tankers resuming Hormuz transits, and crude previously trapped in the Persian Gulf reaching the market, prices have fallen sharply to pre-war levels and traders are pricing in rising supply [15][45][18]. In short, the easing of the geopolitical disruption—rather than the war itself—is driving the downgrades.


### How did the Iran war affect jet fuel prices?

*Verdicts: hybrid-vs-chunks → **HYBRID** · hybrid-vs-facts → **FACTS***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **CHUNKS** · v2-vs-v1 → **HYBRID2**)*:

> Early in the Iran war, jet fuel prices shot to a record as the conflict blocked cargo flows through the Strait of Hormuz—the corridor Europe had long relied on for Persian Gulf imports—sparking fears of shortages, some flight cancellations, and an IEA warning of possible demand destruction [1][3][6]. Rather than causing sustained shortages, the blockage prompted refiners to adjust operations while Europe ramped up local jet fuel production and increased imports from non-Persian Gulf suppliers, replacing the Middle East volumes that had plunged during the conflict [1][3][14]. As a result, airline executives and fuelmakers expressed confidence about supply through the summer, though prices remained relatively high and their primary residual concern was price rather than availability [1][3].
> 
> The context does not give specific jet fuel price figures (only that prices hit a "record") or indicate whether jet fuel prices later fell back toward pre-war levels the way crude and gasoline reportedly did after the Strait reopened [7][8][12][13][17].

**Hybrid (facts anchored to parent passages):**

> Early in the Iran war, jet fuel prices shot to a record [1]. The spike was driven by the closure of the Strait of Hormuz, which cut off Middle East supply, though the market limited the impact by replacing missing volumes with imports from other sources and by ramping up EU production [1]. In response, European refiners pushed jet fuel and kerosene output above seasonal norms, with Shell reporting that every European refinery was maximizing jet fuel output two months into the conflict and Repsol raising its jet fuel yields by up to 25% [19][20][16]. The context does not give specific jet-fuel price figures beyond the "record" characterization, nor does it indicate whether jet fuel prices later retreated to pre-war levels the way crude oil did [1][5].

**Chunk-RAG (raw article passages):**

> The Iran war disrupted jet fuel supply chains by blocking cargo flows through the Strait of Hormuz, which Europe had relied on for Persian Gulf imports; early in the conflict airlines canceled some flights, jet fuel prices spiked to a record, and the IEA warned of possible demand destruction [1]. Rather than causing sustained shortages, the closure prompted refiners to adjust operations while Europe boosted local production and imports from non-Middle East suppliers, partially replacing the missing Gulf volumes [1][5]. As a result, feared shortages were largely averted and airline executives expressed confidence about supply through the summer, though their primary remaining concern was prices, which stayed relatively high [1][5]. A clear signal that the market had not normalized is that jet crack spreads were running at almost double pre-war levels, and the full supply chain was expected to take the better part of a year to fully recover [6].
> 
> The context describes supply disruption, a record price spike, and elevated crack spreads, but does not give specific jet fuel price figures or a full timeline of how prices moved after the initial spike.

**Pure facts (asm@50 atomic facts):**

> Early in the conflict, jet fuel prices spiked to a record high [1], driven by the war's disruption of oil supplies and the blockage of cargo flows through the Strait of Hormuz [2][49]. In response, European refiners ramped jet fuel and kerosene output above seasonal norms, with Shell reporting that every European refinery was maximizing jet fuel output two months in, and Repsol increasing jet fuel yields by up to 25% [19][20][16]. Europe also replaced missing Middle East volumes by boosting jet fuel arrivals from the United States and Nigeria, which soared during the conflict [23][30]. This surging local production and increased non-Middle East imports helped Europe avoid a jet fuel crisis heading into the summer-vacation season [33]. The context covers the initial price record and the supply-side response but does not give figures on where jet fuel prices settled after the ceasefire, so the post-war price path for jet fuel specifically is missing.


### Why did the Philadelphia Semiconductor Index rebound?

*Verdicts: hybrid-vs-chunks → **HYBRID** · hybrid-vs-facts → **FACTS***

**Hybrid v2 (contextual embeddings + dual-nominated passages)** *(v2-vs-chunks → **HYBRID2** · v2-vs-v1 → **HYBRID1**)*:

> The Philadelphia Semiconductor Index (SOX) rebounded 3.83% (closing at 13,709.66, up 537.301 points or 3.919%) as a bounce-back after it had just posted its worst week since the post-Liberation Day sell-off last April, having lost 7.9% the prior week [1][2][8]. The move coincided with a broader recovery in technology stocks that day, with the Magnificent 7 rising 2.58% and the S&P 500 ending a run of five consecutive declines [9][12]. The context frames the rebound mainly as a recovery from that steep preceding-week decline rather than citing a specific new positive catalyst for semiconductors, so a clear fundamental driver for the semiconductor rebound itself is missing [1].

**Hybrid (facts anchored to parent passages):**

> The Philadelphia Semiconductor Index rebounded 3.83% (closing at 13,709.66, up 537.30 points or 3.919%), following its worst week since the post-Liberation Day sell-off last April—so the context frames the rebound largely as a recovery from that sharp preceding drop rather than attributing it to a specific new catalyst [1][2]. That prior weakness was steep: the PHLX Semiconductor Index had lost 7.9% the week before, after jumping 7.3% the week prior [5][6]. The rebound also fit a broader risk-on move, with megacaps like Tesla (+8.46%), Alphabet (+4.79%), and Amazon (+3.20%) rallying [1].
> 
> What's missing: the context does not identify a concrete fundamental driver for the specific rebound day (e.g., earnings, guidance, or policy news); it only notes the rebound came after an unusually bad week [1].

**Chunk-RAG (raw article passages):**

> The Philadelphia (Philly) semiconductor index rebounded +3.83%, recovering after posting its worst week since the post-Liberation Day sell-off the prior April [2]. This rebound occurred on a day when megacap tech names rallied strongly, including Tesla (+8.46%), Alphabet (+4.79%), and Amazon (+3.20%) [2]. The context does not give an explicit driver for the semiconductor index's rebound specifically—it presents the move as part of a broader tech rally rather than attributing it to a named catalyst [2].

**Pure facts (asm@50 atomic facts):**

> The context does not give a single explicit cause for the rebound, but several items point to a bounce-back after a sharp pullback. The index rebounded 3.83% (closing up ~3.92% at 13,709.66) after posting its worst week since the post-Liberation Day sell-off last April, having lost 7.9% the prior week [1][2][6]. This recovery coincided with broad tech outperformance following the recent AI-spending selloff and capital rotating into semiconductor and memory-chip stocks tied to the AI boom [12][11]. Underlying demand also remained strong, with surging semiconductor and memory-chip prices driven by AI data-center demand outstripping supply [41]. What's missing is any item that directly attributes the specific one-day rebound to a named catalyst—the sources describe the move and the surrounding backdrop rather than a stated trigger, and some notes are cautionary (e.g., the index trading ~65% above its 200-day average and warnings of possible underperformance) [3][20].
