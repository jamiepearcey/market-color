# receptors — cause→effect structure from embeddings via BLAS, no LLM

A small experiment testing one idea: **can pure matrix algebra over article
embeddings recover directed cause→effect structure — using only labels we
already paid for — well enough to beat plain cosine, with zero per-chunk LLM
inference?**

The "receptor" framing: instead of one sticky surface (global cosine, which
binds everything by topical gist), learn an **asymmetric linear operator `W`**
that maps a *cause* embedding into the region of its *effect*. Because `W` is
not symmetric, `score(a→b) ≠ score(b→a)` — and that asymmetry is the causal
arrow.

## Where the labels come from (no new enrichment cost)

The market-color pipeline already extracted atomic facts, 62.9% of which carry
`cause_entities`. We mine directed article pairs for free:

> Edge **A → B** exists when an entity B *attributes as a cause*
> (`B.cause_entities`) is something A is *about* (`A.entities`), and A precedes
> B in time. Pairs are weighted by entity IDF; we require ≥1 reasonably
> specific shared entity so we don't link on generic tokens like "strike".

This yields ~24.7k directed pairs over 2,653 fact-bearing docs, split
train/test by effect-doc hash (leak-free: all pairs for a given effect land in
one split).

## Pipeline (all BLAS / LAPACK-class ops)

1. **Embed** once with the project's model (`all-MiniLM-L6-v2`, 384-d) — see
   `scripts/export_for_receptors.py`. Re-embedded (not pulled quantised from
   Qdrant) for clean float32.
2. **(optional) De-gist** — strip the top-k topical directions via a symmetric
   eigendecomposition of `EᵀE`, so binding happens on the orthogonal-to-topic
   residual. *Finding: this hurt on this data — off by default.*
3. **Fit transport `W`** on train pairs by ridge: `W = (AᵀA + λI)⁻¹ AᵀB`
   (one linear solve). `A` = cause rows, `B` = effect rows.
4. **Score** `s(i→j) = cos(row_i · W, row_j)`, time-masked to `t_i < t_j`.

Heavy matmuls go through Apple Accelerate BLAS (ndarray `blas` feature); the
two small 384×384 kernels (eigendecomposition, inverse) use nalgebra. Whole run
is <1s on CPU.

## Results (held-out test pairs)

**Direction** — given an *unordered* {a,b}, guess which is the cause using
content only (sign of `score(a→b) − score(b→a)`):

| method                          | accuracy |
|---------------------------------|----------|
| symmetric cosine (baseline)     | 0.500 (by construction) |
| transport operator, best config | **0.804** |

Robust across λ∈{0.1..100}: 0.72–0.80. Symmetric cosine *cannot* do this task.

**Retrieval** — rank the true later effect of each cause among all later docs:

| scorer                       | Recall@10 | MRR   |
|------------------------------|-----------|-------|
| raw cosine                   | 0.131     | 0.160 |
| transport alone              | 0.044     | 0.071 |
| cosine + transport           | 0.133     | 0.172 |
| cosine + transport + entity  | **0.18**  | **0.20** |

## RAG reranking (`receptors rerank`)

Does W improve relevance when used to rerank a naive chunk-RAG shortlist?
Query = an effect claim; gold = earlier articles about its `cause_entities`.
Stage 1 = cosine top-N chunks (21.4k chunks over the full 5,530-doc corpus);
rerank = same shortlist reordered by `cos(q,chunk) + β·cos(chunk·W, q)`,
max-pooled to docs. 1,090 held-out test queries.

| config                 | nDCG@10 | Recall@10 | MRR   |
|------------------------|---------|-----------|-------|
| baseline cosine        | 0.122   | 0.117     | 0.189 |
| rerank +W (β=0.5)      | **0.131** | **0.126** | **0.197** |

- **Net-positive and consistent**: +0.009 nDCG (~7% relative), peaking at
  β≈0.5–1.0, decaying at β=2 (over-weighting direction hurts topical match).
  Per query at β=0.5: **226 improved, 138 hurt, 726 unchanged** (~1.6:1).
- **The real bottleneck is stage-1 recall, not reranking.** Only **26%** of gold
  cause-articles are even in the cosine top-100 shortlist (41% at top-300), so a
  reranker's headroom is small by construction. W helps at that margin but can't
  surface causes the first stage never retrieved.

**Takeaway:** the causal operator is a useful reranking *signal* on top of
cosine, but the larger lever is first-stage retrieval (hybrid sparse+dense, or
using W/entity signal *in* stage 1) to raise the recall ceiling.

### Strengthening the operator: contrastive refit (`RECEPTORS_FIT=margin`)

Ridge minimises `‖b − aW‖²` — it predicts the *mean* effect and never sees a
negative, so it can't push down co-topical near-misses or the wrong direction.
Refitting W with a **margin/contrastive objective** (warm-started from ridge;
per positive pair, require `aᵀWb` to beat the reversed pair `bᵀWa` and a random
later effect by a margin; full-batch subgradient, all BLAS, ~150 iters, <1s)
strengthens every metric:

| metric (N=300, β=1)  | baseline | ridge W | contrastive W |
|----------------------|----------|---------|---------------|
| direction accuracy   | 0.500    | 0.804   | **0.831**     |
| Recall@10            | 0.117    | 0.131   | **0.138**     |
| Hit@10               | 0.351    | 0.372   | **0.387**     |
| Hit@30               | 0.519    | 0.525   | **0.539**     |
| MRR                  | 0.189    | 0.198   | **0.204**     |

Top-30 lift roughly triples (+1.1% → +3.9%); per-query Hit@30 win/loss flips from
a wash (39/37) to net-positive (43/26); robust across β (no β=2 collapse).

### Bigger lever: in-distribution training (`RECEPTORS_INDIST=1`)

Article-level W is trained on article→article pairs but *scored* on chunk
embeddings vs claim queries. Training instead on `(cause_chunk → effect_claim)`
pairs — the exact objects it ranks, built from TRAIN queries only (~46k pairs) —
is the largest single gain. Progression at N=300, β=1:

| config                 | Rec@10 | Hit@10 | Hit@30 | MRR   |
|------------------------|--------|--------|--------|-------|
| baseline cosine        | 0.117  | 0.351  | 0.519  | 0.189 |
| article + ridge        | 0.131  | 0.372  | 0.525  | 0.198 |
| article + contrastive  | 0.138  | 0.387  | 0.539  | 0.204 |
| **in-dist + ridge**    | **0.152** | **0.415** | **0.561** | **0.228** |
| in-dist + contrastive  | 0.147  | 0.406  | 0.563  | 0.221 |

vs baseline, in-dist+ridge: Recall@10 **+30%**, Hit@10 **+18%**, MRR **+21%**,
Hit@30 **+8%**. In-distribution training subsumes most of the contrastive gain
(both realign W to the task), so ridge-in-dist is the simplest strong config.

**Caveat (honest):** split is by effect-doc hash (random) over a 2-week corpus
with overlapping stories, so a cause article gold for a test query may have
trained against a *similar* train-query claim — some in-dist gain reflects
corpus-specific regularities. Validate with a *temporal* split (train early,
test later) before trusting the magnitude. In-dist pair selection sorts gold
before capping so runs are deterministic.

### Biggest finding: W is a FINDER, not just a reranker (`RECEPTORS_MODE=retrieve`)

Every recall number above is *cosine's* — cosine does the finding (shortlist),
W only reorders it, so W was trapped under cosine's 0.41 ceiling. Using in-dist W
as a **standalone first-stage retriever** over the full corpus beats cosine at
finding, at every cutoff:

| retriever            | R@10  | R@30  | R@100 | R@300 |
|----------------------|-------|-------|-------|-------|
| cosine (naive)       | 0.117 | 0.220 | 0.391 | 0.631 |
| **W only (in-dist)** | **0.179** | **0.321** | **0.528** | **0.751** |
| cosine + W           | 0.153 | 0.275 | 0.477 | 0.719 |
| W only (article W)   | 0.130 | 0.229 | 0.410 | 0.645 |

R@10 +53%, and the recall ceiling itself rises 0.63→0.75 @300. Two notes:
(1) **W-only beats cosine+W for finding** — cosine injects co-topical non-causes;
pure directional retrieval is best for *finding causes*. (Fusion still wins for
reranking, where cosine is the finder.) (2) Only *in-distribution* training makes
W a viable retriever; article-level W ≈ cosine. This inverts the design: the best
use of the operator is **as the retriever**, not reranking cosine's output.

**Temporal validation (`RECEPTORS_SPLIT=temporal`).** Train on earlier effects,
test on the latest ~30% — the honest generalization test. The finder advantage
survives, reduced (random split was optimistic, as expected):

| retriever (temporal) | R@10  | R@30  | R@100 | R@300 |
|----------------------|-------|-------|-------|-------|
| cosine               | 0.080 | 0.158 | 0.307 | 0.508 |
| W only               | 0.107 | 0.197 | 0.373 | 0.594 |
| lift                 | +34%  | +25%  | +21%  | +17%  |

Absolute recall drops for both (later-period stories are genuinely novel), but W
stays a materially better finder at every cutoff — so it's learning transferable
cause→effect structure, not just memorising the corpus.

### Multi-hop causal retrieval (`RECEPTORS_MODE=multihop`)

Single-hop W finds direct causes; a full *explanation* often needs the chain
(root cause is 2–3 hops upstream). Build a directed doc-level causal k-NN graph
from W (edge i→j = "i causes j"), then propagate the 1-hop seed with personalized
PageRank: `s = α·seed + (1−α)·P·s` (restart anchors mass to the direct causes).

The result is **conditional on what you're retrieving for**:

*vs 1-hop gold (direct causes only)* — multi-hop always hurts (diffusion adds
non-direct docs, and the causal graph has hubs that over-amplify): 1-hop R@100
0.528 → PPR α=0.9 0.443. Naive Katz accumulation is catastrophic (0.199→0.03).

*vs chain gold (direct ∪ 2-hop upstream causes)* — multi-hop **wins**, robustly
across random and temporal splits. Temporal split, chain gold:

| config       | R@10  | R@30  | R@100 |
|--------------|-------|-------|-------|
| 1-hop        | 0.054 | 0.106 | 0.227 |
| PPR α=0.9    | 0.058 | **0.150** | **0.317** |
| lift         | +7%   | +42%  | +40%  |

Takeaways: (1) the win concentrates at deeper cutoffs (R@30/@100), not R@10 —
the top slot is still best held by the direct cause; multi-hop fills in the
upstream chain below it. (2) α=0.9 (light leak) is the sweet spot — heavy
diffusion (α≤0.5) over-spreads onto hubs and hurts even chain recall. (3) This
matches the LLM use case: you feed top-k (k=10–30) context to the model, and
multi-hop gives fuller causal-chain coverage there → more complete "why behind
the why" explanations. Use it only when the explanation needs upstream causes;
for direct-cause retrieval, stay 1-hop.

Remaining untried levers: label denoising via fact `confidence` + semantic
`cause`-text matching, a per-relation operator bank (`direction` field), and a
stronger embedder than MiniLM-384.

## What it demonstrates

- **Yes**: causal *direction* is linearly recoverable from embeddings by a
  single asymmetric operator — ~80% vs a 50% floor — trained on free,
  already-extracted labels, no LLM in the loop, sub-second on CPU.
- **Honest limits**: (1) transport is a poor *retriever* on its own; it's a
  direction/rerank signal that pays off *fused* with cosine + entity overlap.
  (2) De-gisting (my a-priori "weight orthogonal relations" idea) *hurt*
  direction here — the topical subspace carried causal signal. Kept as a toggle,
  off by default.
- The recovered structure is candidate edges (directional association +
  time-order), not proven causation — feed low-confidence edges to a gated
  verifier, don't trust the algebra to have proven cause.

## Run

```bash
uv run scripts/export_for_receptors.py     # writes data/embeddings.npy + docs.jsonl
cargo build --release
./target/release/receptors                 # or set RECEPTORS_{K,LAMBDA,SPECIFIC,TEST}
```

`data/` holds the exported matrix; the raw corpus/facts stay in the parent
project. Requires brew (Accelerate is system-provided on macOS).
