# Driver-discovery eval — protocol (the corrected measurement)

**Purpose.** Settle, on the *right* construct, whether driver discovery needs graph
traversal of endogenous data — and whether feature extraction earns its keep as a
reranker. RETRIEVAL_FINDINGS.md concluded "hypothesis decomposition is the engine, no
operator/graph needed," but (a) it measured discovery with pooled snippet-relevance,
which its own analysis calls *structurally blind to novelty*; (b) it seeds hypotheses
only from the cosine top-5 of the effect — the baseline it must beat; and (c) the
content-verified re-eval that overturned the operator was **n = 1** (q19). This
protocol fixes all three: a novelty-aware content metric, an explicit graph-conditioned
hypothesis arm, and ≥15 queries with a bootstrap CI.

This supersedes pooled snippet-relevance (`score_relevance.py`) **for the discovery
question**. Snippet-relevance stays valid for precision/ranking questions; it is the
wrong tool for "drivers not in the baseline."

---

## The construct

> **A driver is discovered when it is (1) a genuine UPSTREAM cause of the effect and
> (2) NOT already represented in the baseline driver set.**

Baseline = cosine-on-the-effect-phrase top-K (arm-independent, the same reference for
every arm). Discovery is therefore a **set-difference-then-verify** count, decided by
*reading* the candidate against the baseline — never by topical similarity.

### Grade rubric (per candidate doc, judged against the shown baseline set)
| grade | meaning | test the judge applies |
|---|---|---|
| **0** | not a driver | off-topic, OR it restates/records the *effect* rather than a cause of it |
| **1** | driver, redundant | a genuine upstream cause, but the **same** driver is already in the baseline set (different article, same causal factor) |
| **2** | driver, **novel** | a genuine upstream cause whose **causal factor is absent** from the baseline set |

"Load-bearing" = grade ≥ 1 (it really is a driver). "Novel" = grade 2. Discovery
counts grade 2. Redundancy (grade 1) is judged at the level of the **causal factor**,
not the document: two different articles both about "copper input costs" are one
driver. When unsure whether a factor is in the baseline, the judge re-reads the
baseline snippets before deciding — that read is the whole point of the metric.

### Point-in-time
Every arm's hypothesis context and every candidate respects the query's `asof` epoch
(`--asof`): only docs knowable as-of T are visible. No look-ahead leakage.

---

## Arms (the single controlled variable = hypothesis SOURCE)

All arms run identical downstream machinery — LLM/template distils upstream queries →
cosine each → RRF-union → optional cos+ndir rerank → gate. Only the generation context
differs:

| arm | context fed to hypothesis generation |
|---|---|
| **A** prior | effect phrase only (world-knowledge floor) |
| **B** cosine5 | effect + cosine top-5 snippets — **the current pipeline** (`gen_hyde_input.py`) |
| **C** graph | effect + `cause_entities` graph drivers (endogenous traversal, `hyp_sources.graph_drivers`) — names drivers whose vocabulary does not overlap the effect |
| **D** hybrid | union of B and C |

Reference arm for deltas = **B**. The decisive result is **C or D beats B on
discovery@K with a bootstrap CI excluding 0.**

### Reranker (feature extraction, measured the right way)
The validated `cos+ndir` stack (`direction_head.py`, Exp E, temporal AUC 0.782) is
layered on each arm's union pool (`--rerank ndir`), i.e. as a **novelty-aware precision
layer fused with cosine** — never as a standalone first-stage retriever (the slot in
which prior experiments wrongly benchmarked it and reported it "loses to cosine").
Run each arm both with and without the reranker to isolate its contribution.

---

## Pipeline & artifacts

```
queries.json
   │  gen_hypotheses.py            (assemble_context per arm; template or LLM)
   ▼  hyps_<ARM>.json
   │  gen_hyp_candidates.py        (cosine-per-hypothesis → RRF union → cos+ndir rerank)
   ▼  cand_<ARM>.json + prov_<ARM>.json     (+ cand_baseline.json = cosine-on-effect)
   │  build_novelty_packets.py     (baseline-as-context + arm-blind pooled candidates)
   ▼  packet_disc_<qid>.json  (to judge)   /   key_disc.json  (arm ranks, hidden)
   │  ← judge grades 0/1/2 → rating_disc_<qid>.json
   ▼  score_discovery.py           (discovery@K, drivers@K, precision, novel_yield; CI vs B)
```

Runner: `scripts/run_discovery_eval.sh` (`COMPOSE=template` for a no-LLM smoke run,
`COMPOSE=llm` for the real generation; `RERANK=ndir|none`, `K`, `ARMS`, `STAGE`).

## Metrics (`score_discovery.py`)
- **discovery@K** — mean # grade-2 (novel load-bearing) in the arm's top-K. *Headline.*
- **drivers@K** — mean # grade-≥1 (any real driver).
- **precision@K** — drivers@K / K.
- **novel_yield** — discovery@K / drivers@K (share of real drivers that are net-new).
- **Δ discovery@K vs B**, bootstrap 95% CI over queries; W/L query split.

## Judging discipline (so the result is trustworthy where n=1 was not)
- **≥ 15 queries**, spanning desks and both "obvious-driver" and "hidden-driver" effects.
- **Blind to arm**: candidate order is `hash(qid|doc_id)`; the key is never shown.
- **Pooled**: arms share one judged pool per query → no arm is judged on a private set.
- **Two judges** on a ≥ 20% overlap sample; report Cohen's κ on the 3-way grade. If
  κ is low on the 1-vs-2 boundary (the redundancy call), tighten the "same causal
  factor" definition before trusting deltas.
- **Adversarial default**: when a candidate is plausible but the judge cannot point to
  the specific upstream mechanism in the text, grade 0 — discovery must be earned by
  the document, not by the reader's prior (the exact failure the whole project fights).

## What each outcome means
- **C/D > B (CI excludes 0):** graph-conditioned hypotheses find drivers the current
  pipeline misses → the graph is a hypothesis conditioner, not dead weight; revise
  RETRIEVAL_FINDINGS.md (which only tested it as a retriever).
- **C ≈ B, D ≈ B:** the cosine-snippet context already saturates hypothesis quality →
  the recent finding stands, now on the right metric and n ≥ 15.
- **rerank(ndir) lifts discovery@K:** feature extraction adds novelty-aware precision
  when fused — keep it; if it only lifts precision@K but not discovery@K, it is a
  ranking tweak orthogonal to discovery (as SIGNALS.md already suggests).
