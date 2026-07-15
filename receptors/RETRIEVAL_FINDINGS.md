# Retrieval / receptor findings — driver discovery, stacking, generalization (2026-07-15)

Consolidated notes for the causal-retrieval investigation. Detailed per-experiment
metrics + figures live under `data/` (gitignored: large npy/embeddings + generated
eval artifacts); this is the tracked summary. Scripts referenced are under `scripts/`.

## Headline (read this first)
**"Finding drivers not in the baseline" is a PROCESS property, not a learned-operator
property.** The discovery engine is: *decompose the effect into upstream causal
hypotheses → cosine-search each → union the novel drivers → gate-verify.* Plain cosine
on the reformulated hypotheses does the finding; the LLM reasoning step (forming the
right upstream questions) is what creates the value. The learned receptor / transport
operator / multi-hop graph is **not** the discovery engine and, tested in isolation,
mostly adds noise.

## Driver-discovery, content-verified (the corrected evaluation)
Earlier retrieval experiments (below) measured the WRONG construct: LLM judges scoring
isolated snippets 0/1/2 for topical relevance, then ranking a pooled candidate set. That
metric rewards cosine-overlap and is structurally **blind to novelty** — a redundant doc
and a genuinely novel driver both score 2/2. "Drivers not in the baseline" is a
set-difference-then-verify question that requires *reading*, not snippet scoring.

Reading the actual content (q19 heatwave→EM appliance makers):
- **Baseline** = cosine on the effect phrase → the demand story (Midea, Thai AC, China
  cooling) + the EU-trade-gap doc.
- **Learned operator** (causal_find on the effect) novel-over-baseline docs = mostly JUNK
  (OpenAI copyright, Kenya EV football, AMAT index removal, Vietnam aquaculture). It does
  NOT discover drivers.
- **Hypothesis-driven cosine** (decompose effect → upstream questions → cosine each)
  surfaces the genuine baseline-missed drivers:
  - "copper input costs for AC makers" → *Copper Cathode −2.09%*, *Copper Market near
    100k Yuan/mt supply tight*, *Section 232 copper probe* — the copper margin-squeeze.
  - "EU anti-dumping on Chinese ACs" → *EU anti-dumping probe*, *trade-war-with-China*.
  - "refrigerant/energy input" → *European natural gas: tight storage supports prices*.
  - "exporter FX" → *Yuan volatility pressures export quotes*, *Thai Baht*.
  ≈ 4 genuine upstream drivers baseline misses — reproducing the original A/B win, via
  cosine-on-hypotheses, NOT the operator.

This reconciles the whole thread: the original process-level win (report A/B 20/0;
ablation: **9/13 novel load-bearing drivers came from analyst hypothesis queries, the
causal operator 0/6**) and the recent negative operator results are both correct — I had
conflated "the operator" with "the method," and measured ranking-relevance instead of
novel-driver-discovery.

**Corrected metric going forward:** count genuine, load-bearing drivers surfaced vs
baseline, by reading (hypothesis-vs-baseline set difference), not pooled snippet-relevance.

## Signal stacking (Exp E–F) — the one measured retrieval-level win
Reframed from SIGNALS.md: direction has nonlinear headroom and belongs as a *reranker*,
not a retriever. Reproduced the nonlinear direction head (HistGB on
`[e_a, e_b, e_a−e_b, e_a·e_b]`, temporal AUC **0.782** vs 0.696 linear).
- **cos + nonlinear-direction stack** significantly improves driver ranking under
  leave-one-query-out CV over 828 judged pairs: mean-grade@10 1.55→1.63,
  **Δ+0.086, 95% CI [+0.018, +0.158]** (excludes 0), 9/12 queries. scripts/stack_multi.py.
- **Standalone** direction rerank / abstention-gating / deviation all LOSE to cosine —
  the signal only helps *fused* with cosine (its own strong-driver AUC is 0.505 = chance;
  it works by correcting cosine's errors, i.e. orthogonality).
- The **typed mechanism tensor** (per-predicate W, a strong direction *classifier* at
  0.742) does NOT transfer to ranking (Δ+0.001; hurts the nonlinear stack). Direction-task
  accuracy ≠ retrieval value — the recurring lesson.
- NB: this is a *precision* tweak on an already-retrieved list, orthogonal to the
  *discovery* question above.

## Chain / multi-hop optimization (Exp A–D) — reverted
- Optimized cross-vocab chain recall (union RRF, concentrated seed, γ=0, K=25): +64%
  mechanical recall over the deployed abductive. But under blind relevance judging it was
  **worse** (nDCG 0.676 vs cosine 0.794) — reverted. Graph/tools left at the
  quality-validated defaults (K=10, γ=1, abductive).
- Hypothesis-at-depth (Exp C) + multi-step isolation (Exp D): the pure 2-hop PPR chain is
  the **weakest** source of drivers (0.83 mean, 54% relevant), worse than the 1-hop
  operator and much worse than reading deeper in cosine. The multi-hop graph does not
  justify itself for hypothesis generation.

## Generalization / substrate
- Operator micro-stats are **stale** on the grown 3433-doc corpus: direction 0.80→0.649;
  standalone finder R@10 +53%→+21% (random) and −23% (temporal). Value survives only at
  depth / in fusion. `receptors generalize` (src/generalize.rs).
- **BGE-768 substrate**: +10–15% mechanical finder recall over MiniLM-384, but **no**
  judged top-10 quality gain (cosine Δ+0.012, abductive −0.022). Not worth switching for
  quality. data_bge_base/ preserved.

## What to build (actionable)
1. **Invest in hypothesis decomposition** — better upstream-question generation is the
   real discovery engine (copper/EU/FX hypotheses above). Cheap, no operator needed.
2. **Keep the gate** — it converts discovered candidates into verified drivers.
3. **Optionally** add the nonlinear-direction cos-stack reranker for a small precision
   lift; do NOT add the typed tensor, confidence, deviation, or the multi-hop graph
   (each measured not to help).
4. **Measure with content review** (novel load-bearing drivers vs baseline), not pooled
   snippet-relevance.

## Reusable harness
`scripts/`: stack_experiment.py, stack_multi.py, reach_experiment.py, reach_multistep.py,
emit_uncovered.py, gen_candidates.py, build_judge_packets.py, score_relevance.py,
score_reach.py, score_ms.py. Judged label pool (828 query,doc driver-relevance grades)
under `data/eval/` (local, gitignored).
