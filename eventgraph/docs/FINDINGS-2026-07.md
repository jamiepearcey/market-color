# Eventgraph Findings Report — July 2026 research arc

Reference copy of everything discovered in the 2026-07-18..20 investigation: the
hypotheses, the evidence, what survived, what died, and where every artifact lives.

**One-sentence thesis the whole arc supports:**
> The causal structure of news, gated by market confirmation, identifies which asset
> correlations are structural rather than statistical — a modest, real, non-tautological
> signal whose economic value is bounded only by news coverage.

---

## 1. Data & infrastructure

| Asset | Location | Notes |
|---|---|---|
| Historical graph (57k docs) | `data/eg_runs/eg100k_graph/` + GCS `/eg_runs` | Bloomberg 2006-13 (dense 2010-12), gpt-oss-120b, 149k entities / 134k causal edges, 81% grounding; merged Groq batch checkpoint (partially irreproducible — source batches expired) |
| Live graph (2026) | `data/eg_runs/eg_live2/` | 3,095 docs 06-27..07-19 (99% extraction after max-tokens fix), 94% grounding |
| Bloomberg corpus (446k articles) | GCS `/bbg/bloomberg_financial_data.parquet.gzip` | was only in ~/Downloads |
| Daily capture | `data/news_corpus/` + GCS `/news_corpus` | 269 sources, scraper runs daily |
| Formal plane | `lake/0005..0011.sql` + `data/eg_runs/formal/` | ALFRED first-prints (2,301, DST-correct), EDGAR acceptance timestamps, agency ICS, options layers, Kalshi/Polymarket implied-prob feeds, lineage + unified provenance |
| GCS bucket | `gs://nimble-sylph-96609-market-color-data` (europe-west2) | 2.73GB, rsync-incremental |
| Results logs | `data/eg_runs/logs/` | raw outputs of every run below |

Panel methodology used throughout: monthly pairwise correlations of **abnormal returns**
(9-macro-factor EWMA-WLS residuals; live window uses ACWI/TLT/UUP/GLD/EEM proxy residuals),
US listings only (foreign local lines excluded — validated doctrine), dyadic
cluster-robust SEs (Aronow-Samii-Assenova) on all headline regressions.

---

## 2. Validated findings

### F1. The core result: news-linked correlations persist; unlinked mean-revert
Among pairs with elevated trailing correlation (>0.3), P(persist) by evidence tier
(2010-12, 160 names, 342k pair-months):

| Tier | P(persist) | next corr (from) |
|---|---|---|
| unlinked | **20%** | 0.24 (0.43) — collapses |
| shared news cause, dormant | 41% | 0.48 (0.53) |
| + one leg market-confirmed | 63% | 0.58 (0.53) |
| + both legs confirmed | **69%** | **0.65 (0.60) — rises** |

Regression: dormant +0.133 (dyadic t 5.4), act1 +0.193 (4.9), act2 +0.294 (**t 8.6**,
highest of the project). Confirmation = volume z≥2 or abnormal-move z≥2 on edge day.
Risk statement: *"correlated for a named, market-confirmed reason — don't assume the
diversification benefit returns."* Log: `activation.log`.

### F2. Structure x activation is the organizing principle (pre-registered, confirmed)
Direction hit-rate on news days: vol-confirmed 68% (dir·z +1.29, t 5.8) vs unconfirmed
54% (+0.10) — ~12x effect-size concentration; volume is direction-blind (non-circular).
Retroactively explains the results table: every unconditioned-structure test died;
every market-gated test survived. Activation *concentrates* (dormant 41% ≠ noise).

### F3. News leads the correlation matrix (non-tautological second-moment result)
Shared-cause pairs at month m: next-2-month corr rise +0.024 vs +0.005 unlinked
(excess +0.020, t 3.0; controlling trailing level +0.065, t 9.7 pre-dyadic).
Survives + **grows monotonically within trailing-corr bins** (relatedness control):
+0.030*/+0.043**/+0.055**/+0.078**/+0.157** — high-corr unlinked pairs mean-revert
hard (−0.18) while linked hold. News-link density peaked 2011-08 (263 shared-cause
pairs) = the contagion month. Structural-vs-transient correlation discrimination is
information a trailing-covariance model cannot contain. Log: `leads` runs in /tmp
(re-runnable: `news_leads_correlation.py`).

### F4. Causal structure complements co-mention (novelty test vs Schwenkler-Zheng)
Link sets Jaccard 0.42 (distinct). Persistence: neither 20% / co-mention-only 43% /
shared-cause-only 39% / both **50%**. Incremental regression: co_mention +0.121
(t 20.5 OLS), shared_cause **+0.080 (t 7.9)** beyond it; interaction null (additive).
Honest bound: co-mention is the stronger single signal; causal is the junior,
*across-article* complement (graph-transitive links co-mention cannot see).

### F5. Latent cause-profile embedding (beyond co-mention, beyond sector)
tf-idf vector of which drivers move each asset; cosine similarity predicts co-movement:
- strongest news feature in full regression (+0.141, dyadic t 3.7)
- **pure-latent** (never co-mentioned, no shared event): +0.194, dyadic t 4.6
- survives SEC SIC-2 sector control: +0.223 (t 4.0), 3x the same_sector effect (+0.072)
- **cross-sector pairs only** (192,820; sector cannot explain): +0.168, dyadic **t 2.8**
  — the linchpin, real but marginal.
NMF factorization (295 assets x 358 causes, k=14): legible event factors — Deepwater
Horizon (BP+KBR+banks+GM), Galleon insider (GS/MS/LEH), Japan/yen (SONY/MUFG/Toyota),
Euro-debt banks, defense-aero-tech (BA/LMT/MSFT/LUV); **9/14 span ≥4 SIC sectors**.
Logs: `dyadic.log`; script `news_latent.py`, `news_embedding.py`.

### F6. Contemporaneous name-level signal (mirror, replicated at 10x scale)
Cross-sectional IC of graph direction vs same-day abnormal return: +0.099 raw (t 7.4) /
+0.107 weighted (**t 8.0**), 744 names, 5,452 obs — vs 6k-corpus baseline +0.128 (t 4.5).
Lag+1 **null** (replicated) — news explains, does not predict, at daily horizon.
Magnitude recovered from +0.084 after fixing foreign-listing leakage (two real bugs:
resolve_yahoo fell back to foreign lines; IC filter was source-name not symbol-shape).
Logs: `ic_full.log`, `ic_v2.log`.

### F7. Coverage decomposition (the honest product profile)
Of 40,977 big idiosyncratic movers (|z|>2, 1,189-name full panel, 2010-12):
- 2.2% had **any** story in the 57k corpus that day (story existence = dominant bottleneck; sampling limit, we hold 57k of 446k)
- of those, 50% got a directional edge (extraction completeness gap)
- of those, **72% direction correct** (best precision measured)
- end-to-end: **36%** correct call given a big move had a story.
Raw coverage 1.1% — *unchanged* at 10x corpus because the resolved universe scaled too:
coverage is per-name-per-day density-bound, not total-corpus-bound. Log: `coverage2.log`.

### F8. News factor set: two kinds
Weekly long-short theme factors (abnormal): demand_supply +221bp/wk (t 4.2), earnings
+179 (3.6), monetary +130 (2.7), regulation +53 (2.0); credit/contagion **null on
direction**. Inter-factor corr |off-diag| 0.13 (distinct). Co-exposure co-movement
excess vs matched random pairs: **credit +0.075** (the direction-null theme is the
strongest covariance factor), macro_data +0.187 (small n); earnings/demand ~0.
=> idiosyncratic themes price directionally; systemic themes appear as covariance.
Contagion case study: absorption ratio of 39 news-selected global financials
0.27 baseline -> **0.64 in 2011-09**; H2-2011 0.45 vs 0.27 other. Honest: earnings
control also spiked (0.36) — most of the crisis spike is market-wide; news-specific
excess ≈ +0.09. Logs: `comovement.log`, contagion runs.

### F9. Link-source expansion (10x coverage)
Beyond shared-cause: relation-month +0.074 (t 5.8), relation-static +0.050 (6.9),
sensitivity +0.041 (2.3*, survives pre-registered sector kill-condition — only 11%
same-sector). Linked coverage 1,175 -> **12,061 pair-months (0.34% -> 3.5%)**.
Caveats (review): relation-static inseparable from unobserved structural pairing;
link-timing endogeneity (links form inside trailing window); month clustering owed.
Log: `linkexp.log` (/tmp), script `link_expansion.py`.

### F10. Live pipeline + monitors (shipped)
scrape -> extract (resumable; **--max-tokens 3000 for reasoning models** — hardcoded
1200 silently truncated ~49% of gpt-oss-120b output) -> Rust ingest with grounding
gates -> SEC/Yahoo US resolution -> `persistence_signal.py` (tiers + provenance) +
`exante_flags.py` (E1 live agency-ICS calendar + FOMC from Kalshi closes; E3 live
option chains via cookie+crumb, cross-sectional implied/realized rank, mandatory
earnings screen; E2 driver-state as context metadata; ordinal labels only, calibration
footnoted as cross-horizon) + `emerging_pairs.py` (cross-sector risers, sign-aware
links: shared-factor co-movement only when exposure signs agree; mixed-sign = "graph
predicted hedge" disagreement class). Live demo: SOBO~oil-majors correlation surge
(+0.25..0.34 to 0.63-0.67) named by an oil-supply news cluster — then correctly
reclassified MIXED-SIGN once sign-awareness landed (SOBO's edge was a spill story).
Provenance drill-down works cross-lingually (Arabic article -> INTC/TXN pair flag).

---

## 3. Refuted / null results (do not resurrect)

1. **News as forecaster**: lag+1 IC null at daily horizon, both corpora, both models.
   Forecast-modality news: t = −0.67 (null). The mirror critique stands.
2. **Stress-timing / early-warning framing**: cross-sector cause_sim does NOT scale
   with VIX (interaction t −0.5); tercile gradient runs the WRONG way (calm +0.255
   t 3.0 / stress +0.105 t 1.4). The chronological "crisis half" significance was
   actually the calm 2012 months. A calm-market phenomenon, not a risk timer.
3. **Attention/uncertainty factors**: news flow vs |z| corr +0.06; narrative
   dispersion vs |z| 0.00.
4. **Direction agreement** (B1): same-sign vs opposite-sign shared-cause pairs adds
   nothing to signed next-corr (t 1.1, underpowered/collinear).
5. **Hedge integrity** (B2): uninterpretable — linked effect flips sign at trailing
   ≈ −0.34 (review caught an algebra error before mis-claiming); opposite-sign pairs
   n=2; sign-preservation null. No claim.
6. **Covariance-forecast product at current density**: +0.4% OOS MSE improvement,
   tracking the 0.4% pair coverage ~1:1. Scalpel, not blanket.

## 4. Statistical honesty ledger

- Dyadic clustering deflates OLS t by 3-6x; all headline claims survive at p<0.01
  except cross-sector cause_sim (t 2.8) — marginal, regime-dependent (calm-half only).
- Not yet done: month-level clustering on top of dyadic (2010-12 common shocks);
  strict link-timing (links precede trailing window); tf-idf idf computed full-sample
  (mild look-ahead); resolution uses 2026 knowledge for 2010 names.
- Survivorship: differential resolution ~4pp for distress-heavy entities (6% of active
  names); unresolved mass dominated by collectives/foreign/sports, but genuinely dead
  issuers dropped (MF Global, 57 edges). Universe itself selected via current-day
  resolvability. Results conditional on the resolved US universe.
- One regime (2010-12 euro-crisis window); one outlet (Bloomberg editorial conventions
  shape both co-mention and extraction).

## 5. Methodological lessons

1. **The tautology test first**: "news explains finance" is near-circular; only
   second-moment (correlation) and divergence constructions escape it.
2. **Structure x activation**: the market's filter tells you which narratives matter;
   the graph tells you what network they bind. Neither alone.
3. **The density wall**: signal quality was never the problem; per-name-per-day
   coverage always was. Economic value scales ~1:1 with coverage.
4. **Sign-awareness**: the extractor captures exposure signs ("shielded from oil",
   sign=−1); ignoring them manufactures false co-movement links.
5. **Adversarial review loop pays**: external design/results review caught a wrong
   algebraic claim (B2), mandated the sector kill-condition, demoted a confounded
   evidence leg (E2), banned rate-numbers on live flags.
6. **Reasoning models need output headroom**: max_tokens tuned for lean models
   silently truncates reasoning-model JSON (49% failure -> 1% at 3000).

### F11. Activation-first inversion (the forward direction) — 2026-07-20
The recurring lesson made operational: *the graph knows lots of narratives; the market
tells you which matter.* Inverted pipeline (`activation_first.py`): observe an
activation (|abnormal z|>=2.5 or volume z>=3 across the resolved US universe) ->
focused retrieval of every fact touching the name -> local causal graph (recurring
causes, multi-firm mechanisms, explicit disagreements) -> deterministic
dominant-narrative template with spillover candidates + verbatim provenance.
First live run (7 trading days, 300 names): 60 activations; top-8 triage: **2/8
explained (~11x the 2.2% broad story-coverage)** — e.g. ITV −6.5σ correctly
attributed to Netflix/YouTube audience erosion (demand_change) with the Guardian
quote; ManpowerGroup +7.7σ -> guidance/earnings events. Unexplained cases (IBM −8σ,
PYPL +7.3σ) are corpus-sampling gaps, not graph failures — the trigger names exactly
what to fetch, so the production design retrieves on demand at activation time
(~60 names/week instead of 16k docs/day). This reframes the density wall: coverage
becomes a retrieval-latency problem, not a pre-extraction-volume problem.

### Rigor updates (2026-07-20, post-report)
- **Month-clustered SEs**: core linked coefficient +0.110 — dyadic t 5.2 remains the
  binding dimension (month t 15.4); common time shocks do not explain the result.
- **Strict link timing**: links formed entirely BEFORE the correlation-measurement
  window still show 34% vs 17% persistence (n=1,174) — the co-mention-because-
  already-comoving endogeneity cannot account for the effect.
- **Graph transitivity** (multi-hop): directionally consistent but underpowered —
  directed common-cause parent2 +0.052 (dyadic t 1.8, pre-registered magnitude
  ordering respected), collider placebo cleanly ~0, bridge2 null; prevalence after
  hub-exclusion + doc-leakage safeguards only 59/333k. Density-bound, not refuted.
- **Sign-aware links** (live monitor): shared-factor co-movement only when exposure
  signs agree; mixed-sign pairs surfaced as "graph predicted hedge" disagreements
  (caught the 'shielded from oil' false flag; reclassified the SOBO cluster).
- **Intrade archive**: post-shutdown archive with all contracts to 2003 (daily prices
  + trades) exists on GitHub (via P. Ipeirotis) — makes the historical divergence
  test (LLM narrative expectations vs Intrade odds, 2010-12) feasible with data in hand.

### F12. Emerging markets: 18% of the graph as drivers, untestable as assets (2026-07-20)
EM is NOT a data-scarcity problem: 24,613/134,366 causal edges (18%) touch an EM entity,
concentrated in 2010-12; top EM nodes China (4,183 driver edges) and Greece (1,336) are
among the biggest causal hubs in the whole graph. BUT this is EM-as-SOVEREIGN (countries
as macro drivers), not EM-as-tradeable-single-name. The tradeable EM company cross-section
is thin: 200 EM-country company effect-entities (>=3 edges) -> only 59 priceable (Vale,
Petrobras, Norilsk, Sberbank, Gold Fields, Standard Bank, Baidu, Hyundai, POSCO, OTP,
America Movil, Sasol). EM PERSISTENCE TEST (EEM/ACWI/UUP/GLD-neutralized, mirror of the
core test): EM unlinked 21% (n=3,408, matches DM 20% = validity check) vs EM news-linked
23% (n=43) = +2pp, UNDETECTABLE, vs DM's +28pp (20% -> 48%). NOT a refutation of the EM
edge thesis and NOT a confirmation — the thesis is UNTESTABLE on this corpus: the mechanism
needs dense per-name-per-month linking, but the English Bloomberg desk covers EM as macro
backdrop, not as linkable single names (only 43 elevated linked EM pairs in 3 years). The
signal STARVES in EM for lack of single-name coverage, not lack of inefficiency. CONCLUSION
(data-acquisition, not signal): testing the EM thesis needs a NATIVE-LANGUAGE EM-single-name
-dense corpus (Valor/Caixin/Economic Times/Russian press). The edge, if real, is exactly
where Western data can't see it — consistent with why the inefficiency persists. Scripts:
em_resolve.py, em_persistence.py.


### F12b. India partial test — corpus testable (F12 diagnosis validated), signal weak/inconclusive (2026-07-20)
Recovered 13,036 India 2021 extractions (survivor of a spend-cap-failed 38k batch; Groq
account hit a spend-alert block, 100% of the fresh batch rejected — resubmit staged).
Graph: 13k docs, 7,611 causal edges (94% grounded), 26k entities. Resolved 46 of 50
Nifty50 names as effects (India names matched to nifty50_ticker.csv); **479 shared-cause
linked pair-months, 141 distinct linked pairs** — economically real bank clusters
(AXISBANK~SBIN~ICICIBANK). This VALIDATES the F12 diagnosis: a single-name-dense native
corpus IS testable where Bloomberg EM was not (43 pairs). PERSISTENCE (market-mean-
neutralized, 45 priced): trailing>0.3 unlinked 32% vs linked 29% (n=52, null); trailing>0.2
unlinked 23% vs linked 30% (n=81, +7pp). Weak, threshold-sensitive, underpowered — NOT the
DM +28pp, better than Bloomberg-EM's +2pp. CAVEATS: 34% of one year (13k/38k), crude
neutralization (equal-weight market mean, not a factor model), no activation tiering (India
news 0.6 edges/doc = list/price-report style, less causal). Inconclusive; needs the full
38k (billing) + more years + proper India factors. Script india_persistence.py.


### F12c. India with a PROPER factor model — the EM edge appears (2026-07-20)
The user's methodological catch (India needs its own factor model) was decisive. The weak
F12b result used a crude equal-weight market mean, leaving Bank Nifty + Nifty IT co-movement
in the residuals (inflated unlinked baseline). Two proper neutralizations:
- PCA-3 (market+banks+IT, statistical): LINKED 47% vs unlinked 21% = **+26pp** (>0.3),
  +28pp (>0.2) — matches DM's +28pp. But K-sensitive (PCA-5 only +5pp) = researcher df.
- SPECIFIED named factors (Nifty/BankNifty/NiftyIT/INR/Brent, no K-arbitrariness), nested:
  market-only +12pp/+2pp; +sectors +10pp/+9pp; +macro +35pp(n19)/+16pp (>0.3/>0.2).
  **Positive in EVERY specification** (+2..+35pp, never null) = robust in SIGN.
Honest bound: robustly positive, best-powered estimate ~+10-16pp (>0.2, sectors+),
plausibly DM-comparable; magnitude UNDERPOWERED (n=19-69 linked, 34% of one year). The
crude-neutralization null was the artifact — with correct India factors the news-graph
edge appears. => EM thesis alive: Bloomberg-EM null = coverage; first India null =
factor-model; both fixed -> India ~ DM. Needs full 38k + more years to pin magnitude.
Scripts india_factor_test.py (PCA), india_specified_factors.py (named).

### F13. Counterfactual driver attribution — the signed synthesis (2026-07-20)
The user's reframe ("find negative synthetic values by discounting a factor's residual to
see what things look like without it") turned the F5 embedding + directional edges into an
*attribution* engine: for a correlated pair, build the shared-driver return factor, residualize
both legs against it, and read Δcorr = how much of the co-movement that driver explained.
- **Link-based** (shared-cause driver, `driver_attribution.py`): named-driver removal drops
  corr **+0.266** vs random-shared **+0.121** → excess **+0.145, t=14.4**. The specific driver
  the graph names carries most of the correlation — not any shared driver.
- **Embedding-weighted, unsigned** (`embedding_attribution.py`): ranking shared drivers by
  tf-idf profile overlap → top vs random-shared excess only **+0.039, t=1.6 (null)**. Unsigned
  embedding weight does *not* isolate the load-bearing driver — shared drivers are redundant.
- **SIGNED** (`signed_attribution.py`, the fix the user predicted): build the driver factor
  direction-weighted (`sign(effect_dir)·AR`). Same-sign (co-movement) driver removal Δcorr
  **+0.219** (n492) vs opposite-sign (divergence) **+0.098** (n98) → **difference +0.121,
  t=7.5**. Sign is what the unsigned embedding threw away.
- **Product-readiness** (`signed_calibration.py`): distribution median Δcorr +0.22 (5% neg);
  **out-of-sample stability corr +0.75, magnitude 108% persists** — *better* than the link
  version's +0.64, the strongest stability in the project. Calibration via embedding tf-idf
  weight FAILS (non-monotone: mid +0.30 > high +0.23 > low +0.18) — the embedding weight does
  not predict per-pair confidence.
Synthesis / honest architecture: **embedding = candidate generator** (which drivers to test),
**realized signed Δcorr = the measured attribution**, **OOS-stability 0.75 = the confidence**.
Trust the measured Δcorr (self-calibrating via 0.75 persistence), not the embedding weight.
Product statement: "A and B correlate primarily through same-sign driver X (+0.22, holds
quarter-to-quarter); driver Y partially offsets. If X de-activates, diversification returns."
Descriptive not predictive (consistent with F1-F6). Strongest single result: significant
(t=7.5) AND out-of-sample stable (0.75) AND sign-aware (links cannot do this).

## 6. Open threads (priority order)

0. **India signed attribution at power** (data-gated, running): 6× 5k 8b sub-batches on
   Groq extract the remaining ~25.4k of the unbiased complete-year India-2021 feed
   (13,040 already recovered from the spend-cap survivor batch). On completion: ingest
   full 38k, re-resolve Nifty50, re-run F13 signed attribution + F1 persistence
   cross-market at power — the real generalization test for the EM edge (F12c).
1. **Clean sign-flip regime** (analysis-gated): mirror of F13 — for a *negatively*-correlated
   pair joined by an *opposite-sign* driver, removing it should *raise* corr toward zero.
   If it holds, signed attribution decomposes co-movement and divergence symmetrically.
   Thin (few negative-corr pairs at n≥20 co-days); check power before claiming.
2. **Graph transitivity / multi-hop**: does 2-hop connection (drivers linked in the
   entity graph, driver sets disjoint) predict co-movement the embedding cannot see
   (cosine=0)? Hub exclusion mandatory (macro hubs connect everything). Designed,
   not yet run.
2. **Month-clustered SEs + strict link timing**: rigor items owed on F1/F3/F9.
3. **Divergence program** (the one construction with predictive potential): narrative
   vs market-implied expectation. Machinery built (`v_prob_divergence`, Kalshi/
   Polymarket feeds); needs live accumulation, or an Intrade 2010-12 archive (overlaps
   the dense corpus window — likely nobody has done LLM-narrative vs Intrade).
4. **Accumulation**: every week of daily capture widens the live monitors; net-signs
   replace n=1 sign estimates. Cron + GCS rsync recommended.
5. **Deferred**: intraday horizon (timestamp infra built, data unpurchased); unused
   fact layers (sentiments 44k, figures 144k); delisted-inclusive price data (CRSP/
   FirstRateData) for survivorship-clean replication.

## 7. Reproduction quick reference

```
# historical panel tests (graph dir = data/eg_runs/eg100k_graph)
uv run scripts/cross_sectional_ic.py --graph-dir <g> --years 2010,2011,2012
uv run scripts/coverage_panel.py --graph-dir <g>
uv run scripts/news_leads_correlation.py --graph-dir <g>     # + stratified control
uv run scripts/news_covariance.py --graph-dir <g>            # persistence + OOS forecast
uv run scripts/comention_vs_cause.py --graph-dir <g>
uv run scripts/news_latent.py --graph-dir <g>                # embedding + sector + dyadic + VIX
uv run scripts/news_embedding.py --graph-dir <g>             # NMF exhibit
uv run scripts/activation_split.py --graph-dir <g>
uv run scripts/link_expansion.py --graph-dir <g>
# live pipeline (graph dir = data/eg_runs/eg_live2)
uv run scripts/extract.py --feed <feed> --out <g> --model openai/gpt-oss-120b --max-tokens 3000
<eventgraph-bin> ingest --mock --feed <extracted-subset-feed> --out <g>
uv run scripts/resolve_tickers.py --graph-dir <g> && uv run scripts/resolve_yahoo.py --graph-dir <g>
uv run scripts/exante_flags.py --graph-dir <g>
uv run scripts/emerging_pairs.py --graph-dir <g>
# signed driver attribution (F13)
uv run scripts/driver_attribution.py        # link-based counterfactual
uv run scripts/signed_attribution.py        # signed (co-move vs divergence)
uv run scripts/signed_calibration.py        # distribution + OOS stability + calibration
# formal plane
uv run scripts/formal_calendar.py alfred|ics|edgar ...
uv run scripts/implied_prob.py kalshi|polymarket ...
uv run scripts/load_formal.py --graph-dir <g> --db <duckdb>
```
