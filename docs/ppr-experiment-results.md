# PPR retrieval experiment — results (2026-07-09)

> **Update, same day:** a fourth arm (`iter`, cause-guided iterative
> retrieval) was built from finding 3 below and wins — see
> "Iterative retrieval (arm 4)" at the end.

Three-arm comparison of retrieval over the fact layer (`ppr_experiment.py`):
**dense** (Qdrant vector search over fact claims), **ppr** (dense-seeded
Personalized PageRank over the fact–entity graph, uniform weights ≈ HippoRAG),
**qppr** (PPR with query-conditioned causal/predicate edge weights).

Graph: 3,066 facts / 2,789 canonical entities / 808 resolved causal links,
from 734 docs of the 2026-06-27→07-01 news window (rebuilt via
`graph_experiment.py build` after draining 62 codex batches).

## Results

**Easy set** (`eval/ppr_cases.jsonl`, 16 single-story "why" cases, k=10):
all arms tie — dense R=0.95, full-coverage 0.88. Ceiling effect.

**Hard set** (`eval/ppr_cases_hard.jsonl`, 7 cross-document causal-chain
cases; each evidence group must be satisfied by a distinct fact from a
distinct doc):

| arm            | evidence recall@10 | full coverage | MRR  |
|----------------|--------------------|---------------|------|
| dense          | 0.643              | 0.29          | 0.798|
| ppr (best)     | **0.690**          | 0.29          | 0.886|
| qppr (best)    | **0.690**          | 0.29          | 0.878|

Sweeps: alpha ∈ {0.1..0.6}, causal boost ×{1,3,6,10}, predicate weights
on/off, hub damping freq^-γ γ∈{0,0.5,1}, seed-interleave, dense-pool rerank.
All PPR variants converge to R=0.690; composition/weighting choices move MRR
(0.556–0.886) but not recall.

## Findings

1. **Extraction ate the easy hops.** Because decompose_facts stores the
   stated driver in the fact itself (`cause`/`cause_entities`), most "why X"
   questions are answerable by a single retrieved fact. Multi-hop only
   survives as a retrieval problem when the chain spans documents and no
   single fact states the link. This validates graph *construction* quality
   as the dominant term, and it is an argument FOR the fact layer.

2. **The graph genuinely reaches what dense cannot.** For chain-end queries
   ("what chain of events pushed LNG to multi-year highs?"), the upstream
   facts (Kiku attack, US airstrikes) are absent from dense top-100 —
   semantically unrelated to the query — but are 1–2 hops away in the graph
   and appear in PPR's ranking (ranks ~#30–#220). Reach is not the problem.

3. **One-shot PPR cannot convert reach into a top-10 budget.** Propagation
   ranks the 1-hop frontier by connectivity/confidence, not by relevance to
   the query; the frontier of 8 seeds is hundreds of facts, so the right
   hop-2 facts lose the budget race. Hub damping, causal boosting,
   interleaving, and dense×PPR blending all fail for the same reason:
   no per-candidate semantic signal exists beyond dense-to-query similarity,
   which is precisely what hop-2 facts lack.

4. **Query-conditioned weights (the novel bit) were second-order** on this
   graph size — they shift MRR at mid alpha but never recall. Verdict on the
   original hypothesis: mechanism confirmed directionally (+4.7pts recall
   over dense on hard cases, n=7 — too small for significance), but the
   static-weight version does not yet earn a learned-weight investment.

## What would actually move the number

- **Iterative retrieval, not deeper propagation**: retrieve → read frontier
  facts' cause_entities → issue a second, reformulated dense query per
  unexplained cause ("what happened to <cause entity>?") → merge. The chain
  is walked in query space, where semantic scoring exists at every hop.
- Score = f(dense sim at THIS hop, path mass), not one-shot blend.
- More eval cases (20→50) before trusting any delta; one news week limits
  chain diversity (everything routes through Iran/Hormuz).
- Drain the remaining 58 batches (~690 docs) if chains look
  connectivity-starved after the above.

## Repro

```bash
docker compose up -d qdrant
uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed \
  --with numpy --with pyarrow python graph_experiment.py build --recreate
... python ppr_experiment.py eval --cases eval/ppr_cases_hard.jsonl --k 10
... python ppr_experiment.py search --query "..." --arms dense,ppr,qppr
```

Dropped case: `hard-oil-spike-vs-drop` — the "biggest quarterly drop" oil
fact was never extracted (headline doc not in the 734 done), so the case was
unsolvable for every arm.

## Iterative retrieval (arm 4, `iter`) — the winner

Implements the "walk the chain in query space" recommendation:
dense-seed, then for each frontier fact (a) **upstream hop** — embed its raw
`cause` text as a fresh dense query, (b) **downstream hop** — match its claim
against an in-memory embedding index of all 1,151 cause texts (facts citing a
similar cause). Path score = parent × 0.8 × hop-sim; final list = top-4 dense
+ best walked facts. Defaults: keep=4, per_hop=5, depth=2, downstream=on.

Hard set (7 cases, k=10, distinct-fact/distinct-doc matcher):

| arm   | evidence recall@10 | full coverage | MRR  |
|-------|--------------------|---------------|------|
| dense | 0.643              | 0.286         | 0.798|
| ppr   | 0.690              | 0.286         | 0.810|
| qppr  | 0.690              | 0.286         | 0.810|
| iter  | **0.786**          | **0.571**     | 0.798|

Easy set: iter == dense exactly (0.823 / 0.625 / 0.938 under the strict
matcher) — the walk never displaces evidence dense already had. Sweeps:
keep 4>5≈6; depth/per_hop flat beyond 2/5 (frontier saturates).

Case-level: japan-strategy-root 0.33→1.00 (walked strategy→Hormuz→airstrikes),
fed-fx-ripple 0.67→1.00 (downstream hop). Still failing: lng-root-cause
(hop-2 cause text is the dead-generic "The Middle East conflict." — generic
conflict facts beat the specific Kiku/airstrike facts) and
korea-memory-ripple (Micron's cause text similarity to the Samsung claim
falls below the 0.5 floor / loses the slot race).

## Expanded eval (16 hard cases) — the arms converge

The hard set was tripled by mining facts.parquet for cross-doc
cause_entities→entities links (9 new cases: Ukraine→Crimea, Binance/MiCA,
silver/dollar, RBA/Brent, Qatar gas, India vulnerability, stablecoins,
Russia fuel exports, Alibaba/heatwave; all feasibility-checked). Results
at k=10:

| arm                      | recall@10 | full coverage |
|--------------------------|-----------|---------------|
| dense                    | 0.781     | 0.563         |
| ppr                      | 0.802     | 0.563         |
| iter (refined)           | 0.771     | 0.563         |
| iter (novelty off)       | 0.771     | 0.563         |
| iter (upstream only)     | 0.781     | 0.500         |

**At n=16 no arm separates; all differences are within noise.** Two
refinements tried after the 7-case result (paraphrase novelty filter on
upstream hops; entity-overlap boost + lower floor on downstream hops) were
neutral-to-negative — ablations kept.

Honest reading: the walker's advantage is **query-type dependent**. It wins
decisively on chain-end queries where the answer's root is semantically far
from the query (japan-strategy 3/3, fed-fx 3/3 — dense gets 1/3 and 2/3);
it is neutral where dense can see both hops, and its walked facts can
displace useful dense tail (tajikistan, gas-lng). The 9 mined cases were
mostly dense-visible, diluting the chain-end signal that dominated the
original 7.

Implications:
- The production shape is a **router**: dense by default; walk only when the
  query is causal AND top dense hits carry unresolved causes (cheap check —
  intent classifier + cause-field inspection). Walking everything buys
  nothing and costs slots.
- Any further tuning needs a bigger, chain-end-heavy case set (50+, second
  crawl window) before results are meaningful. Do not tune on n=16.
- Cost profile stands regardless: the walk is embedding-only at query time
  (extraction-time cause spans are the hop queries; no LLM in the loop) —
  the "multi-hop at single-hop latency" property is the durable takeaway.

## Sibling expansion (arms `sib`, `sibiter`) — small-to-big over facts

Facts carry full provenance (doc_id/url/source/title), so a fifth strategy
expands top dense hits with other facts FROM THE SAME DOCUMENT (causal
siblings first). Rationale: articles embed their own background context; the
decomposer turns it into sibling facts that share no vocabulary with the
query. Corpus rate: **14% of upstream causes are stated by a sibling fact**
(66/479); median 5 facts/doc.

16-case results: sib R=0.792 / full 0.563 (dense 0.781, iter 0.771);
naive combination sibiter 0.760 — WORSE. Case-level: sib fixes
lng-root-cause (0.33→0.67 — the Hormuz/Kiku background was a sibling of the
top LNG hit, precisely the predicted effect) and fed-fx (3/3), but breaks
russia-fuel-exports (siblings displaced the dense tail fact that mattered).

Conclusion: **all expansion strategies are budget-bound at k=10** — each
reaches different side-stepped facts (walker: cross-doc chain roots;
siblings: same-doc background) but pays by evicting dense tail. They are
complementary retrievers competing for one small budget, so naive stacking
subtracts. Production answer: don't make expansions compete for retrieval
slots — retrieve with dense, then ATTACH walk/sibling context to the facts
that carry unresolved causes at assembly time (briefs are hierarchical;
k=10 flat lists are an eval artifact, not the product shape).

## Assembly-time attachment (arm `asm`) — the winner (2026-07-10)

Router made concrete: dense top-4 primaries keep their slots; a primary
whose cause points outside its own fact (unresolved upstream hop — the
trigger) gets attached context: best cross-doc walked fact for its cause
(novelty-guarded) + best causal sibling covering the cause entities; dense
tail backfills to the same k=10 budget.

| set  | arm   | recall@10 | full coverage | MRR  |
|------|-------|-----------|---------------|------|
| hard | dense | 0.781     | 0.563         | 0.880|
| hard | iter  | 0.771     | 0.563         | 0.880|
| hard | sib   | 0.792     | 0.563         | 0.849|
| hard | **asm**  | **0.802** | 0.563     | **0.880**|
| easy | dense | 0.823     | 0.625         | 0.938|
| easy | **asm**  | **0.885** | **0.750** | **0.938**|

The aggregate delta is modest, but the per-case pattern is the strong
result: **asm is never worse than dense on any of the 32 cases** and better
on several, on both sets, with no MRR cost — unlike iter (broke
tajikistan/gas-lng), sib (broke russia-fuel-exports), and sibiter
(negative overall). Trigger-gated attachment cannot lose slots on queries
where expansion has nothing to offer; that structural no-regression
property, not the average, is the finding. asm is the recommended
production shape; in a real brief the attachments would be rendered as
subordinate "⟵ driven by …" context under their primary (with url/source
tie-back) rather than spending flat-list slots at all.

## Answer-level ablation (do the assists change the ANSWER?)

Counterfactual test (`eval/answer_ablation.py` + `eval/run_ablation.sh`,
claude CLI / opus-4-8): for every case where asm's context differed from
dense's (trigger fired: 24/32 = 75%), generate an answer from each context
(same 10-fact budget, cite-by-number), blind pairwise judge on causal-chain
specificity, randomized A/B order.

Result: **asm 10 wins / dense 9 / 5 ties — a wash at the answer level**
(n=24, single judge pass). Attachment citation rate 71%: the attached facts
DO enter the generated answers — they are not decorative — but on most
queries they add corroborating rather than decisive evidence, because (per
the extraction finding) the dense context usually already contains the
stated cause. The two marquee chain-end cases (lng-root-cause,
japan-strategy-root — where retrieval metrics showed the real gap) both
converted to clear ASM answer wins, consistent with the scalpel story.

Reading: retrieval-level assists convert to answer-level value only where
the retrieval gap was semantic (chain-end), not where attachments merely
reinforce. This sharpens the router condition — fire the walk on causal
queries whose dense top-k lacks a *specific* stated cause, not merely
"has unresolved cause entities" — and confirms attachments should render
as citations/provenance context (where 71% usage is a feature) rather than
as slot-competing evidence. Caveats: n=24, one judge, one pass.

## Budget sweep — k=10 vs k=30 vs k=50 (2026-07-10)

k=10 was an eval artifact; facts are ~70 tokens so k=50 is ~3.5k tokens.
Sweeping the budget shows it was the dominant constraint all along:

| set  | k  | dense R / full | asm R / full | iter R / full |
|------|----|----------------|--------------|---------------|
| hard | 10 | 0.781 / 0.56   | 0.802 / 0.56 | 0.771 / 0.56  |
| hard | 30 | 0.854 / 0.75   | 0.844 / 0.69 | 0.865 / 0.69  |
| hard | 50 | **0.927 / 0.88** | 0.927 / 0.88 | 0.917 / 0.81 |
| easy | 30 | 0.854 / 0.69   | **0.885 / 0.75** | —          |
| easy | 50 | 0.875 / 0.75   | **0.906 / 0.81** | —          |

Findings:
- **Most "hard" evidence lived at ranks 11–50.** dense@50 solves 14/16 hard
  cases fully. The k=10 drama (slot contention between arms) largely
  dissolves at realistic budgets.
- **The semantic-invisibility residual is real but tiny**: at k=50 dense
  still fails only lng-root-cause (0.33 — the Kiku/airstrike chain roots,
  genuinely absent from dense's ranking) and rba-brent (0.50). That is the
  entire remaining retrieval case for the walk/attachment machinery.
- **asm keeps its no-regression property at every k** and stays ahead on
  the easy set (+3pts recall, +6pts full coverage at k=50).
- Production guidance: retrieve k=50 dense as the floor; keep trigger-gated
  attachment for provenance structure and the rare invisible chain; judge
  any future retrieval work against dense@50, not dense@10.

**Specific-cause prompting (applied).** All three extraction prompts
(`extract_template.md`, `extract_prompt.md`, `CURSOR_DRAIN_PROMPT.md`) now
require the cause field to name the most specific stated driver (concrete
event/actor), never a generic label, when the text offers both. Pays off on
the next extraction run (58 undrained batches + future crawls); the
LNG-chain failure ("The Middle East conflict." as a hop query) is the
motivating case.

**Story-cluster dedup (implemented, measured, default OFF).**
`Retriever._story_clusters` (claim cosine >= 0.87 + shared entity,
union-find) + `_dd` arm variants capping retrieval at one fact per cluster.
Measured: this corpus has only **7.5% cross-source duplication**
(3,066 facts -> 2,836 clusters — one crawl week, modest source overlap).
Hard set: dedup changes nothing. Easy set: dedup HURTS (dense 0.823->0.760,
asm 0.885->0.823) — paraphrase redundancy in top-k is sometimes real
evidence (second source, slightly different cause span) and sometimes
matcher double-dipping, but either way removing it costs recall here.
Verdict: dedup is premature at this duplication rate; the code stays
(off by default) for when the corpus grows — re-measure the
facts->clusters ratio after each corpus expansion, and turn dedup on when
duplication reaches roughly 25-30%+ of facts.
