# Eventgraph Findings Report — July 2026 research arc

Reference copy of everything discovered in the 2026-07-18..20 investigation: the
hypotheses, the evidence, what survived, what died, and where every artifact lives.

**One-sentence thesis the whole arc supports:**
> News-event embeddings carry a small but genuine, leakage-free FORWARD signal about asset
> co-movement, incremental to price history — a predictive *covariance/risk* signal (not returns,
> not contemporaneous exposure, which is mostly recoverable from returns). Modest (+0.03 IC) but real.
> [Corrected 2026-07-23 after Fable rigor review — see F30/F31; earlier F22-F24 magnitudes were price-history-inflated.]

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

### F14. Lead-lag test — news does NOT lead correlation (decisive negative, 2026-07-22)
The sharpest question: is the news channel a LEADING, non-price observable (the one thing a
price-defined beta regression cannot be), or just an interpretable re-description of one? Test
(`lead_lag.py`): for each (driver, week) does a 1-SD surge in the driver's news over the trailing
window predict its connected names co-moving MORE over the FORWARD 15 trading days, controlling
for mean-reversion (C_past) and market-wide co-movement change (crisis confound)?
- NEWS coef = **-0.0044, t=-3.9** (negative, not positive). C_past -0.33; dC_market **+0.96** (the
  driver names' co-movement change is ~1:1 the market's -> the confound does the work).
- Intuitive lead-lag peak cross-correlation: **median lag 0 wk**, 47% lead / 13% same / 40% lag = symmetric.
CONCLUSION: news activation is **coincident, not leading** (news spikes during crises when co-movement
is already high, then mean-reverts). The attribution (F13) is confirmed to be, mechanically, a BETA
REGRESSION to a text-defined factor: novelty is factor CONSTRUCTION from news (named/signed/no-price-
history-needed), NOT a leading signal. RETRACTS any "sits ahead of the trailing covariance / early
warning" framing. Value = interpretable text-discovered factor model (descriptive), not prediction/
alpha. Caveats: daily/weekly granularity (an intraday lead is untested & unseen here); market control
is aggressive. But symmetric lead-lag argues against a rescue. `lead_lag.py`.

### F15. In-sample vs out-of-sample attribution — real structure, not a relabeled fit (2026-07-22)
Concern (correct): the attribution fits the factor AND measures the correlation drop on the SAME
window -> in-sample, near-mechanical, and the news basket ~ sector. Strict test (`oos_attribution.py`):
fit each leg's beta to the driver factor on W1, apply the FIXED W1 betas to the non-overlapping
FORWARD W2, measure the drop there; compare named-in-sample / named-OOS / random-driver-OOS.
- named in-sample +0.250 (t24.6); named **OOS +0.247 (t23.0) = 99% retained** -> NOT overfit, the
  factor structure is stable across windows.
- **random-driver OOS +0.062** -> named beats random **4:1 OOS** -> the news GROUPING is informative,
  not arbitrary relabeling. named-OOS − random-OOS = **+0.184**.
CONCLUSION: the decomposition is real, out-of-sample-stable, and non-arbitrary — the "GS-MS co-move
through the Moody's channel" statement is genuine structure, not an in-sample artifact. Splits the
"relabeled regression" worry: it IS a beta regression to a text-defined factor (conceded), but it is
NOT circular/in-sample/arbitrary. Boundary that HOLDS: still DESCRIPTIVE not predictive (F14: news
doesn't lead). Sector share handled separately (F13 sector-neutralization: 70% survives). `oos_attribution.py`.

### F16. India at power — the signed attribution GENERALIZES to an emerging market (2026-07-22)
Full India-2021 corpus re-ingested (45,428 causal edges, 6x the 13k version). First pass was
UNDERPOWERED (signed n=20) — diagnosed NOT as a real null but as an unfinished resolution: only
46 of 3,580 company entities mapped (Nifty50-only, 6% edge coverage). Expanded resolution via
LLM+Yahoo-verified gate (`india_resolve_expand.py`): 46 -> **140 NSE tickers** (107 new, each gated
on real 2020-22 price history + issuer name-match). Re-run at power (fixed a global common-day
intersection bug that newly-listed 2021 IPOs exposed — residualize each stock on its own days):
- **Signed attribution (F13) GENERALIZES**: same-sign (co-move) removal +0.113 (t6.9, n68) vs
  opposite-sign (divergence) +0.021 (t2.4, n13); **DIFFERENCE +0.091, t4.9** — same pattern,
  significance, and comparable magnitude to US (+0.121, t7.5). 
- **OOS (F15) holds**: same-sign OOS (W1 betas -> W2) retains **80%** of in-sample (t3.7) vs US 99%.
- **Persistence (F1)**: +19/+19/+25pp (>0.3) across market/sectors/macro specs, n=192-420. Robust, powered.
CONCLUSION: every component of the US result (persistence, signed co-move-vs-divergence attribution,
out-of-sample stability) reproduces in a structurally different emerging market once the universe is
properly resolved. The earlier India "null" was a 15%-complete resolution, NOT a failure to generalize.
This is the finding's broadening — it is not a US-only artifact. Caveat: single year (2021), so temporal
(not cross-sectional) power is still limited. Scripts india_resolve_expand.py, india_signed_attribution.py,
india_specified_factors.py.
- **Descriptive verdict retested in India (F14 replication, india_lead_lag.py)**: news does NOT
  lead forward co-movement — NEWS coef +0.0021 (t+1.7, not significant), controlling for mean-
  reversion (C_past -0.57) + market ΔC. US was -0.0044 (t-3.9). India leans mildly positive but
  below significance; not a leading/tradeable signal in either market. Descriptive-not-predictive holds cross-market.

### F17. Factor decomposition — the news channel is a ~0-2% incremental sliver (2026-07-22)
Full orthogonalized covariance decomposition of a pair's correlation (`factor_decomp.py`): split
cov(A,B) EXACTLY into blocks in economic order macro(9)->sector(9 GICS SPDRs)->named news drivers->
idiosyncratic (Gram-Schmidt makes it additive, checks to 100%).
- GS-MS (corr +0.815): MACRO 46% + SECTOR 41% + news drivers ~2% + idiosyncratic 12%.
- BP-GM (+0.528): macro 77% + sector 20% + news 1% + idio 3%. AAPL-GS: macro 78% + sector 26% + news ~0%.
CONCLUSION: once macro AND sector are separated FIRST, the named news channels make up only ~0-2% of
pair co-movement. Reconciles the earlier "Moody's +0.13 (19%)" / "70% survives sector": those removed
the driver BASKET (which IS mostly sector) from sector-containing returns, CREDITING sector-overlapping
variance to the driver; giving sector priority collapses the unique news share to ~1-2% (finer sub-sector
clustering the 9 broad ETFs miss). Confirms the standing skepticism: most "news explains co-movement" is
macro+sector wearing news labels; the news-specific INCREMENTAL factor is small. The cross-market-replicated
STRUCTURE (F16) is real but sector-adjacent, not a large independent factor. CAVEAT: order-dependent (news-
last = conservative); a Shapley decomposition (avg over orderings) is the order-independent version. `factor_decomp.py`.

### F18. Shapley (order-independent) decomposition — news ≈ sector, unique add ~1-3% (2026-07-22)
Order-independent version of F17 (`factor_shapley.py`): Shapley value of each block (MACRO/SECTOR/NEWS)
= avg marginal covariance-explained over all orderings; the 7 subset values v(S) = cov of A,B projected
onto that subset's factor span (order-free). GS-MS: MACRO 15% / SECTOR 39% / NEWS 34% / idio 12%.
BUT the block bounds expose the truth: NEWS-FIRST (block alone) 81%, Shapley 34%, NEWS-LAST (incremental)
~1%. The 80pp spread = NEWS and SECTOR are nearly the SAME SUBSPACE (driver baskets ARE same-sector stocks).
Shapley's 34% is the FAIR SPLIT of variance news SHARES with sector, NOT independent importance; news's
UNIQUE orthogonal contribution (news-last) is ~1-3% (BP-GM 1%, AAPL-GS ~0%). CONCLUSION: the news graph is
largely a text-derived sector/sub-sector classifier — it reconstructs groupings a sector model already has;
you cannot cleanly separate news from sector because they are the same factor; forced to a unique-contribution
answer, news is a couple of percent. This is the quantified end of the "half a trend relabeled" skepticism:
the cross-market-replicated structure (F16) is real but ≈ sector; the independent news factor is small. `factor_shapley.py`.

### F19. The residual — idiosyncratic on average, but a small genuine news signal survives (2026-07-22)
`residual_view.py` — what's left after macro+sector removed. (A) universe residual is IDIOSYNCRATIC:
mean pairwise residual corr +0.014, absorption ratio (top-eigenvalue share) 0.45 raw -> 0.06 residual
= no missing common factor; macro+sector captured the systematic co-movement. (B) BUT the residual is
NEWS-shaped: news-linked pairs mean residual corr +0.073 vs unlinked +0.011 = **+0.062, t6.2** — a small
but significant SECTOR-ORTHOGONAL news component that survives macro+sector removal. Reconciles F18: news
is mostly sector, but has a real thin non-sector residual (+0.06) — the part that replicated in India (F16).
HONEST FLAG: pair-level residual magnitude is method-sensitive — GS-MS residual 0.10 (Gram-Schmidt pair
decomp F17) vs 0.38 (per-name macro+sector OLS here); one method mishandles collinear sector/news blocks,
reconciliation OWED before trusting absolute pair-level residuals. Aggregate B/C use one consistent method,
so the linked-vs-unlinked +0.062 is robust. `residual_view.py`.

### F20. What the residual CONTAINS — event-driven sub-sector linkages (2026-07-22)
`residual_content.py` characterizes the sector-orthogonal news signal (F19's +0.062). (A) it lives more
in SAME-sector pairs (linked +0.103 vs unlinked +0.014, excess +0.090) than cross-sector (+0.035) = the
broad 9 GICS factors miss FINE SUB-SECTOR clusters the news identifies. (B) the top drivers of residual
co-movement are SPECIFIC EVENTS, not factors: UK/EU bank regulation (Project Merlin, Vickers/Independent
Commission on Banking, EU Competition -> Lloyds-NatWest-Barclays-DB-UBS), SA gold-miner strikes (National
Union of Mineworkers -> AngloGold-Gold Fields), the mortgage-agency complex (Maiden Lane III/Sherry Hunt/
SEC -> Freddie-Fannie), German autos (BMW -> Mercedes-VW), US derivatives reg (CFTC -> GS-MS).
CONCLUSION: the residual contains EVENT-DRIVEN, sector-orthogonal co-movement — firms bound by a specific
shared catalyst (regulation, litigation, strike, national policy, crisis complex) forming fine sub-sector/
national clusters that broad sector betas cannot encode. The news graph's genuine informational value = it
NAMES the catalyst. Small magnitude (~+0.06 aggregate) but real and interpretable; it is an event-driven
sub-sector linkage detector, not a sector relabel and not a factor. `residual_content.py`.
RATE (residual_content count): named event-cluster structures (specific driver linking >=2 priceable
names) are SPARSE+BURSTY — 507 in 3yrs = ~1.1/day mean but MEDIAN 0 (clustered on event days, max 16);
>=3 names 194 total, 0.4/day. On 57k/446k sample; F7 (coverage density-bound, not total-bound) implies
full-corpus rate is low single digits/day, NOT ~8. So the unique news signal is small (~+0.06) AND sparse
(~1/day, median 0, bursty) — an occasional event-linkage detector, valuable on event days, quiet otherwise.

### F21. "Most events look sector-driven" — true for membership, FALSE for the co-movement (2026-07-22)
User observation: the event clusters (F20 viz) are mostly same-sector. Confirmed at MEMBERSHIP level:
61% of strong events single-sector, Financials-dominated (51/67); apparent "cross-sector" ones are
mostly sector-tagging noise (Freddie/Fannie both mortgage GSEs, Honda/Toyota both autos, AAL/UAL both
airlines). BUT decisive control (`residual_subsector.py`): add 12 SUB-INDUSTRY ETFs (KBE/KRE/KIE/IAI/
XHB/OIH/SMH/IYT/IBB/ITB/XRT/IYR) to the broad-9 sector block and re-measure the news-linked residual
excess: +0.063 (t6.6) -> +0.056 (t6.0) — barely moves. The co-movement SURVIVES a fine sub-industry
taxonomy. INTERPRETATION: membership is sectoral (banks with banks) but the co-movement is NOT sector-
explained — a bank ETF captures "banks co-move on average"; the news links the SPECIFIC banks bound by a
SPECIFIC event (the ones downgraded, the UK banks under Vickers) that move together MORE than the sub-
industry factor. The news graph's irreducible contribution = the DYNAMIC, catalyst-specific selection of
which same-industry names are bound right now — not a new sector, but time-varying within-industry
clustering no static taxonomy (however fine) provides. Small (+0.056) and descriptive, but real. `residual_subsector.py`.

### F22. Event-anchored (out-of-mention) exposure — description beats the ticker-mention (2026-07-22)
Reframe (user): everything prior is MENTION-ANCHORED (article names X -> resolve to ticker -> measure
co-movement among named tickers), conflating 'mentioned' with 'exposed' and conditioning all results on
the journalist's list. Test the event-anchored alternative (`descriptive_exposure.py`): represent each
event by its verbatim quotes+catalyst+mechanism (tf-idf), each company by its OWN news text (independent
of the event); exposure=cosine. Among firms the article NEVER named, do high-descriptive-exposure ones
co-move with the event's mentioned basket more than low-exposure ones, after macro+sector removal?
RESULT: non-mentioned HIGH-exposure +0.029 vs LOW +0.007 -> **excess +0.023, t9.4**. YES — rich event
description surfaces genuinely exposed firms the mention missed, co-moving beyond sector. So the mention
anchor UNDERCOUNTS exposure; the event-anchored/descriptive frame adds real (small) sector-orthogonal signal
and a full-universe exposure landscape (not just named firms). Caveats: small magnitude (~residual scale);
company profile still news-derived (not fully mention-free — 10-Ks/business-desc would be cleaner); first-cut
tf-idf + coarse quartile split. EXTENDS not overturns. Next: event-as-factor betas + real embeddings. `descriptive_exposure.py`.
LANDSCAPE (`exposure_landscape.py`): full-universe exposure map per event + validation that exposure
predicts realized co-movement for NON-mentioned firms: pooled corr +0.047, t6 (n15651) — CONCEPT holds in
aggregate. BUT per-event ranking is NOISY with tf-idf: LIBOR/Almunia landscape surfaces REAL unnamed banks
(JPM realized +0.20, GS, MFG) mixed with vocabulary-coincidence NOISE (Vale mining, Mastercard). tf-idf on
own-news too coarse for a trustworthy per-event exposure list. CLEAN product needs (a) real semantic
embeddings (not tf-idf) + (b) mention-independent company profile (10-K business descriptions, not own news).
Validated concept, not-yet-shippable ranking. `exposure_landscape.py`.
EMBEDDING UPGRADE (`embed_exposure.py`, all-MiniLM-L6-v2 on Metal/MPS): aggregate corr(exposure,realized)
+0.047(tfidf)->+0.057(embed), t7. PER-EVENT ranking MUCH cleaner: LIBOR/Almunia top surfaced non-mentioned
firms go from [Vale,Mastercard,BAE noise] -> [GS,JPM,HSBC,BAC] — the actual unnamed banks (HSBC realized
+0.50). Semantic embeddings fix the vocab-coincidence noise; ranking now dominated by the right firms.
Residual noise (Google/airline) remains -> the last upgrade is mention-independent profiles (10-Ks). The
event-exposure landscape is now trustworthy enough to ship as a full-universe exposure map per event.

### F23. Mechanism-factor exposure (step 4) — return-anchoring beats embedding cosine (2026-07-22)
User's progression (named -> tf-idf -> embedding -> latent factors). Tested (`mechanism_factors.py`):
K=24 KMeans clusters over EVENT embeddings give INTERPRETABLE economic-mechanism factors (US financial
legislation/Dodd-Frank, UK banking reform/Vickers-Merlin-ICB, antitrust-LIBOR/Almunia, industrial metals,
German autos/BMW, natural disasters, credit downgrades/Moody's). Bonus: embedding clustering semantically
merges the Vickers/Merlin/ICB catalyst fragments the LEXICAL canonicalizer (F21) could not.
EXPOSURE vs realized co-movement (non-mentioned, n15651): raw embedding cosine +0.057 (baseline);
factor-mediated SEMANTIC +0.058 (NO gain — factorizing the embedding is a rotation of the same info);
**RETURN-ANCHORED mechanism factors +0.076, t10 (+33%)** — company loads on each mechanism via its RETURN
BETA to the factor's mimicking portfolio (magnitude from prices, not text). CONFIRMS: latent decomposition
per se adds nothing over cosine; the gain comes from grounding exposures in RETURNS (the direction/magnitude
the embedding lacks). Breaks past the ~0.057 ceiling that held across mention/tfidf/embedding -> ceiling was
model structure, not purely data. Still small (residual-scale). CAVEATS: mimicking-portfolio self-inclusion
(tighten), and DIRECTION (signed effect) not yet added. `mechanism_factors.py`.

### F24. Mechanism model + DIRECTION — magnitude works, direction is chance (2026-07-22)
Completed the step-4 framework (`mechanism_direction.py`): signed event shock D_E (net effect_dir of named
firms) x signed company mechanism betas (leakage-clean: firm excluded from its own mimicking portfolio).
(A) MAGNITUDE/co-movement: |mechanism-factor exposure| vs realized co-move = **+0.101, t13** (leakage-clean),
up from cosine +0.057 and self-included +0.076 — nearly 2x the baseline; best exposure model built. WHO is
exposed is now well-predicted. (B) DIRECTION: corr(D_E*exposure, realized SIGNED abnormal return) = +0.039
(t5 only from n=15651); **directional hit-rate 51.4%** (chance=50%). Predicting UP vs DOWN is essentially a
coin flip. CONCLUSION on the user's 3 ingredients: pathway (mechanisms) ✓ interpretable; magnitude (return
betas) ✓ +0.101; DIRECTION ✗ near-chance. Direction is exactly where the signal runs out — reaffirming the
whole project's through-line: a DESCRIPTIVE exposure lens (who/how-strongly), NOT predictive (which-way/when).
`mechanism_direction.py`.

### F25. Sentiment — best contemporaneous direction, still not predictive (2026-07-22)
Does SENTIMENT recover DIRECTION (mechanism-model failed at 51.4%)? 44k signed-polarity annotations (mostly
+-1, intensity, immediate horizon), 6.8k on priceable firms. Test polarity vs signed abnormal return
(`sentiment_direction.py`): CONTEMPORANEOUS [D..D+5] IC **+0.125 t8** (beats causal effect_dir F6 +0.10 and
mechanism-dir +0.039 — the strongest directional signal in the project); LEAD [D+6..D+15] IC +0.013 (NULL);
PRE [D-5..D] IC +0.052 (positive -> sentiment partly LAGS price, reporter tone follows the move). Directional
sign hit-rate ~48% (chance) — graded correlation real but not a clean up/down call (base-rate skew ~60% neg).
CONCLUSION: sentiment enriches DESCRIPTIVE direction (best contemporaneous polarity signal) but adds NO
predictive direction — same wall as F6/F14/F24. Fourth independent directional signal (mention-dir, mechanism-
dir, sentiment) all landing on: the news layer DESCRIBES, does not PREDICT. `sentiment_direction.py`.

### F26. Exposure accuracy varies by EVENT TYPE — route by channel, not one embedding (2026-07-22)
User's insight: different event types propagate through different channels; one universal embedding conflates
them. Tested (`category_exposure.py`): LLM-classify 116 events into 6 types, per-event exposure-vs-realized IC.
RESULT — accuracy swings hugely: **regulatory +0.183** (embedding's home turf: named-set+peers, e.g. LIBOR),
company-specific +0.095, macro-surprise +0.089, policy +0.050, **supply-chain -0.027** (embedding is the WRONG
channel — semantics can't capture supplier->customer linkage; needs the NETWORK). Breadth ~constant so it's the
CHANNEL not diffuseness. CONCLUSION: the universal embedding is one channel forced onto 6 physics — right for
regulatory, wrong for supply-chain/policy. Route by type: regulatory/company->embedding, supply-chain->relation
network (relation_edge.jsonl), macro/policy->return beta, geopolitical->country/commodity. NB reconciles F23:
recombining latent EMBEDDING factors = rotation (no gain), but routing to independent CHANNELS by type IS
justified (+0.18 to -0.03 spread). The demo's best cases (LIBOR) are regulatory = the model's best type. `category_exposure.py`.

### F27. Event classifier (top of the hierarchy) — routing + per-event direction-determinacy (2026-07-22)
User's hierarchy: News event -> decomposition {mechanism|factor|regime} -> exposure -> response, classify BEFORE
ranking. Built (`event_classifier.py` -> event_routing.json): tags each event with mech + domain + dominant LAYER
+ **dir (factor_signed | idiosyncratic)**. Correct on samples: LIBOR/Almunia & Dodd-Frank & BMW = mechanism/
idiosyncratic; Growth-Forecasts & Earthquake & Project-Merlin = factor/factor_signed. ROUTING VALIDATED (embedding
IC by mech): regulatory +0.13, legal +0.12, macro +0.11, policy +0.09, company +0.07 vs supply_shock +0.01,
credit_rating -0.06, geopolitical -0.06 (embedding is the WRONG channel for the last three). DIRECTION DIAGNOSIS
(the key): F24's chance-direction was from AVERAGING factor_signed events (sign recoverable via firm's signed factor
beta) with idiosyncratic ones (not). The classifier lets us route: factor_signed -> signed factor-beta channel (and
STOP residualizing the macro factors that carry the sign); idiosyncratic -> no direction claim. This is the concrete
path to fixing direction for the subset where it's knowable. `event_classifier.py`, `event_routing.json`.

### F28. Routed direction — factor-event direction is the FACTOR MODEL, not news (2026-07-22)
Tested F27's hope that direction is recoverable for factor_signed events (`direction_routed.py`): does the
event shock D_E propagate to NON-mentioned firms, market-relative (drift removed), split by routing tag?
RESULT (reversed): factor_signed **48.9%** (n1129, AT chance), idiosyncratic **53.3%** (n2418, weakly>chance);
by layer factor 48.7% vs mechanism 53.2%. INTERPRETATION: for factor/macro events the direction lives in the
MARKET MOVE itself — remove the factor (market-relative) and nothing news-specific remains -> chance. So the
'recoverable direction' for factor events is just the FACTOR MODEL (firm beta x shock), well-known, NOT news.
Idiosyncratic events keep a tiny sub-industry directional residual (53%, the F21 structure). CONCLUSION: direction
is not a news-actionable signal in ANY bucket (F24 chance / F25 sentiment contemporaneous-only / F28 factor=factor-
model). The routing insight was architecturally right and PROVED where direction comes from (factor model, not news).
Routing's real payoff is EXPOSURE (F26/F27), not direction. `direction_routed.py`.

### F29. Information-gap / timing alpha — no exploitable lag (2026-07-23)
User's reframe: the trade isn't DIRECTION, it's TIMING — a firm with HIGH latent exposure that is NOT mentioned
and has NOT reacted (abnormal return ~0) is the 'market hasn't caught up' gap. Tested (`information_gap.py`):
split each event window into early (reacted?) and late (catch-up?); for non-mentioned firms measure late drift
toward the event basket, conditioned on exposure and early reaction. RESULT: high-exposure + not-reacted late
drift +0.010 (t2.2) — BELOW the low-exposure-unreacted control +0.012; corr(exposure, late drift | unreacted)
= -0.010 (t-0.8, NULL). Only the already-reacted high-exposure group keeps moving (+0.037, contemporaneous
momentum, not a gap). CONCLUSION: the exposure model does NOT lead price — if a genuinely-exposed firm hasn't
moved early, exposure says nothing about a late move. The market prices this exposure CONTEMPORANEOUSLY; no lag
to harvest. 5th independent null on the predictive side (F6/F14/F24/F25/F29). Closes the alpha search: this is a
real contemporaneous EXPOSURE map (explanation/risk), not alpha — the market is efficient wrt the structure it
recovers. `information_gap.py`.

### F30. RIGOR RECHECK (Fable review) — text adds little over a price-history baseline (2026-07-23)
Fable review flagged the +0.101 exposure headline as partly mechanical (in-window betas, no price baseline,
inflated pooled t). Decisive recheck (`rigor_recheck.py`, event-level inference, T2-broadened non-mention):
add the MISSING baseline = pre-event trailing correlation with the event basket. CONTEMPORANEOUS: trailing
+0.134 (t11) >> embedding +0.032 (t3.3); text INCREMENT over trailing +0.018 (t1.9, 95%CI [-0.000,+0.036] —
marginal). FORWARD: trailing +0.127; embedding +0.034; text increment +0.021 (t2.5, CI [+0.005,+0.037]).
CORRECTIONS: (1) the ~+0.10 exposure IC was MOSTLY price history (trailing corr), not text; (2) embedding's
UNIQUE contribution is ~+0.02, contemporaneously marginal; (3) naive pooled t=13 was inflated ~4-6x (event-
level t=1.9-3.8). SILVER LINING: text adds a small SIGNIFICANT FORWARD increment (+0.021) = predictive-covariance
(R3, consistent with F3) beyond price history — the first survivor pointing at a predictive (risk) signal. CAVEAT:
forward increment may still be profile-leakage (company profiles include cross-month quotes) -> F31/T3 pending.
Retro-deflates F22-F24 headline magnitudes; the exposure STRUCTURE is real but its text-specific size was overstated.
`rigor_recheck.py`.

### F31. Leakage-free (pre-event profiles) — a small genuine FORWARD covariance signal survives (2026-07-23)
Fable T3/#3 decisive test: rebuild firm profiles from ONLY the firm's news published BEFORE the event month
(`rigor_recheck_t3.py`) — kills cross-document retrieval leakage, directly answering the research question
(latent inference vs retrieval). CONTEMPORANEOUS text increment over trailing corr: +0.016 (t1.5, 95%CI
[-0.004,+0.036] = NOT significant — dies). FORWARD text increment: **+0.028 (t2.7, CI [+0.007,+0.048]) —
SURVIVES, if anything stronger than the leaky +0.021.** CONCLUSION: contemporaneous 'exposure' is mostly price
history (text adds ~0 clean); but a small, significant, LEAKAGE-FREE, FORWARD-predictive covariance signal is
real — pre-event text predicts next-window co-movement beyond trailing correlation, and it is genuine latent
inference (pre-event-only), not retrieval. The corrected thesis: not a descriptive exposure map, but a modest
(+0.03 IC) PREDICTIVE covariance/RISK signal (consistent with F3 corr-forecasting + F13 OOS-stability). More
defensible AND more valuable (forward risk > contemporaneous explanation). Next (Fable): sharpen into a proper
realized-covariance / hedge-failure forecast; run per-type routing at event-level inference. `rigor_recheck_t3.py`.

### F32. Hedge-failure overlay — text predicts correlation PERSISTENCE, not rising (2026-07-23)
Upgraded the F31 survivor into a forward-risk tool (`hedge_overlay.py`). Text-exposure does NOT forecast
correlation RISING: IC(exposure, Δcorr=forward-trailing) = -0.008 (t-0.8, null) — no new-link formation.
But the forward-LEVEL signal survives (+0.028, t2.7): text nudges the mean-reversion-adjusted forecast (β<1),
not the raw change. ECONOMIC payoff = a DIVERSIFICATION TRAP: among ELEVATED pairs (trailing>0.3, n395), high-
text-exposure ones retain correlation (trailing 0.40 -> forward 0.21) while low-exposure revert (0.36 -> 0.08);
forward gap +0.13 is ~3x the trailing gap +0.04 -> text flags which already-correlated pairs PERSIST vs decouple.
CORRECTED FINAL THESIS: not rising correlation, not returns, not a big contemporaneous exposure map — a modest,
leakage-free, forward PERSISTENCE signal on covariance (which elevated correlations won't mean-revert = which
hedges fail). Consistent with F1 (linked persist, unlinked revert), now leakage-free + incremental to price
history. The one result that survived the full Fable rigor review. Bounds: modest (+0.03 IC), elevated-subset,
persistence-not-discovery. `hedge_overlay.py`.

### F33. Implied move BY NEWS TYPE — empirical prior, huge variation (2026-07-23)
Answering 'what move does each news type imply' (`news_type_moves.py`): for every causal edge, measure the
firm's IDIOSYNCRATIC abnormal return on the article day, aggregate by mechanism. Random-firm-day baseline
|move| ~1.07%. RESULT (mean |move| / direction-aligned / n): restructuring 5.07%/+5.07%/53; earnings 3.56%/
+3.15%/72; rate_decision 2.18%/+1.41%; data_surprise 1.96%/+1.63%; guidance 1.82%/+1.27%/244; M&A 1.53%/+0.63%/
1021; regulation 1.28%/+0.02%/1483; leadership 1.91%/-0.49%; competition 0.85%. KEY: (1) news types differ
hugely — restructuring/earnings move 3-5% directionally-aligned (real catalysts), regulation/competition barely
clear the 1.07% baseline w/ NO direction (context/noise). (2) direction-aligned column is the tell: positive =
extracted up/down tracks the move (trustworthy); ~0/negative (regulation +0.02, leadership -0.49, factor -0.41)
= mention/reaction not driver. (3) high-VOLUME != high-impact: the two biggest buckets (regulation 1483, M&A 1021)
are among the weakest movers -> most edges are context. This is the empirical implied-move-by-type prior; before
this the per-event number was realized-from-price only and the type label decorative. `news_type_moves.py`.

### F34. Unified driver pipeline + Fable judge (fix-first -> fixed) (2026-07-23)
Unified process (`hypothesis_engine.py`), REPLACING embedding ranking: activation -> residual DECOMPOSITION
(macro/sector/idio) as the driver detector -> ROUTED retrieval (macro catalysts | same-sector catalysts |
firm's event-study-scored edges w/ F33 type-trust) -> hypothesis (LLM w/ deterministic fallback; Groq key
currently EXPIRED). FABLE JUDGE verdict: architecture sound (right replacement), FIX-FIRST on 4 defects:
(a) intercept-in-macro bug — secular drift booked to macro, corrupting routing (VWSYF -32% "macro"); (b)
trust-prior as hard gate — false-low on best hit (ACGBY/Huijin +14% = 13x noise marked low) and false-high
on worst (BIDU "Google" -1.5% sub-noise high); (c) no naming floor (BB/"Dolby" +0.4% named for a -47% move);
(d) 8-anecdote evaluation. FIXES APPLIED: (a) leave-one-month-out betas + intercept EXCLUDED from attributed
components (drift->idio); (b) evidence-combination confidence — z-vs-1.07%-noise x coverage-of-idio x sign,
trust MODULATES not vetoes, ABSTAIN below noise, never high on mixed. POST-FIX: BP->high (10.7x noise, 30%
cov, right family), ACGBY->med w/ correct catalyst, BIDU no false-high, BB/NWG/EBKOF clean abstentions —
no confidently-wrong outputs remain. PENDING (Fable #3): scaled scoring vs labeled ground truth; same-day
multi-event disentangling; ADR/FX factors. `hypothesis_engine.py`, unified_attributions.json.

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
- **SECTOR-NEUTRALITY (the decisive confound test, `sector_neutral_attribution.py`):** re-run
  the same-sign attribution on returns residualized against 9 GICS sector SPDRs (XLF/XLK/…)
  ON TOP of the 9 macro factors. Baseline mean Δcorr +0.222 (t26.7) → sector-neutral +0.156
  (t23.0) = **70% survives**. The sector confound is real (~30% of the level) but the majority
  is genuine news-specific covariance beyond macro AND sector. This is what upgrades the daily/
  pair monitor from "interesting" to trustworthy at ~70% of the shown magnitude.
- **Daily/pair monitor (productization, `attribution_daily.py` + `pair_attribution.py` + viz):**
  rolling 63d attribution replayed one trading day at a time -> interactive HTML. Moody's
  downgrade channel spikes +0.44 in 2011Q4 (euro sovereign/bank downgrade wave); GS-MS raw
  corr 0.70, Moody's channel +0.13 (19%), counterfactual corr collapses to ~0.25 at the peak.
  Attribution BUILDS ahead of the peak and DECAYS after. Levels ~70% news-specific per above.

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

---

## 8. Taxonomy correction + sector x event-class risk (2026-07-26)

`scripts/taxonomy.py` (substring fallback) · `scripts/sector_event_risk.py`

### F35. The `other` bucket was half a mapping failure and half the extractor refusing
Exact-match-only `EVENT_TYPE_MAP` left **26.4%** of eg100k events (52.9% india2021,
75.9% eg_live2) in `other`. Decomposing it:

- **~12.5% was a mapping failure** spread over **1,540 distinct strings** — variants
  (`court_case` / `court_ruling` / `court_hearing` / `court_decision`), so no
  realistic number of dict keys fixes it. An ordered substring fallback (same
  technique `SECTOR_TEXT_MAP` already used) recovered about half: eg100k
  **26.4% -> 19.8%**.
- **The rest is irreducible**: 69% of the remaining `other` is the extractor
  emitting the literal string `"other"`/null. No taxonomy work touches that; it
  needs re-extraction or a second-pass classifier into the controlled vocab.
- The largest unmapped strings are **not classes at all** — `event` (1,271),
  `meeting` (1,171), `announcement`, `report`. These are deliberately kept in
  `other` via `EVENT_TYPE_NON_INFORMATIVE`; mapping them anywhere would
  manufacture classification that does not exist.

### F36. Coverage share and immediate risk are close to inversely related
Sector-matched lift (each event divided by its OWN sector's non-event control):

| class | n | lift | 95% CI |
|---|---|---|---|
| earnings | 350 | **1.97x** | [1.73, 2.23] |
| guidance | 343 | 1.48x | [1.31, 1.66] |
| monetary_policy | 230 | 1.35x | [1.19, 1.54] |
| legal_regulatory | 197 | 1.31x | [1.10, 1.56] |
| rating_action | 141 | 1.30x | [1.07, 1.54] |
| **m_and_a** | **907** | **1.21x** | [1.11, 1.31] |

**M&A is the largest class by volume (2.6x earnings) and nearly the weakest by
risk.** A "what is this name's news made of" panel that weights by share is
therefore weighting by almost the wrong thing.

Sector-matching turned out to matter little in practice — control sigma runs
0.734-0.774 across all 12 sectors, because names are already standardised by
their own trailing residual vol. Correct in principle, ~5% correction in fact.

### F37. Earnings risk varies ~2x by sector; the low-vol sectors take the biggest jolt
Consumer Staples **3.16x**, Health Care 2.67x, Info Tech 2.66x, Industrials 2.22x
vs Energy 1.73x, Materials 1.48x. Same class, different risk regime — the
conditional structure F-series keeps finding. Guidance splits equally hard:
Info Tech 1.99x vs Communication Services 0.99x.

### F38. A missing class, found by the data and then confirmed — but underpowered
Health Care's `other` ran **2.40x** while every other sector's sat near 1.0-1.3x.
Inspecting it showed FDA reviews, clinical trials and study results with no home
in the 30-class vocab. Pre-registered prediction: split them out and HC `other`
should fall toward the other sectors.

**Confirmed in mechanism:** HC `other` **n=17 -> n=8, 2.40x -> 1.62x**, and the
nine cells that left became `clinical_trial` at **3.09x — the highest lift in the
sector, above earnings (2.67x)**.

**NOT confirmed in magnitude.** n=9 in Health Care, n=24 globally, and the global
CI is **[0.84, 2.86] — it crosses 1**. The class is real and separable; its risk
is not estimated.

### The binding constraint is density, again
All of Health Care rests on **161 event cells**. At this corpus density the
taxonomy can *identify* classes but cannot *estimate* their risk. This is the
density wall from section 5 lesson 3, arriving at the same conclusion from a new
direction: the fix is the 446k BBG corpus, not a better classifier.

**Caveats:** 23 classes tested at 95%, so ~1 false positive is expected by
construction — treat n<50 rows as suggestive. Contemporaneous risk description,
not forecast (F17/F18 bound the channel at 0-3%; F24/F28 killed direction). One
regime (2008-2014), one outlet.

### F39. The event-class attribution is doc-union, and that is a bug
Every consumer of event classes builds cells as:

    doctypes[doc_id] = {every event_type anywhere in the document}
    cell[(effect_entity_ticker, date)] |= doctypes[doc_id]

so **every entity on the receiving end of any causal edge inherits every event
class in that document**. A blind sample of 320 `m_and_a` cells (dumped with no
sigma attached — `scripts/subclass_dump.py`) shows what that produces:

- **11% are ETFs / index proxies** (HYG, TLT, FEZ, RSX, GLD, FXI) tagged to deals
- **3% are roundup columns** — "Medtronic, Monsanto, Warner Music, Coach, Eaton:
  Intellectual Property", "CMPC, Itau, Pao de Acucar, Vale: Latin America Equity
  Preview"
- resolution errors: **RGCO** tagged to "Ex-**RBG** Resources Chairman Ordered to
  Pay $44m"; **ORIC** (IPO'd 2020) tagged to a 2010 Astellas bid for **OSI**
  Pharmaceuticals — the 2026-knowledge-on-2010-names anachronism the honesty
  ledger warns about
- the bulk of the remainder are executive hires, patent suits and sector
  commentary that are not M&A for that name at all

**The correct path exists and is unused.** `event.issuer_entity` names the entity
an event is about (32.5% populated, 10,214 resolving to a ticker).
`causal_event_edge.event_id` would be better still but is **0% populated**
(134,366 rows, none linked) — a schema field that was never filled.

Re-measuring with issuer attribution (`scripts/attribution_compare.py`):

| class | union n | lift | issuer n | lift | delta |
|---|---|---|---|---|---|
| earnings | 350 | 1.97x | 531 | **2.17x** | +0.20 |
| guidance | 343 | 1.48x | 196 | **1.61x** | +0.13 |
| rating_action | 141 | 1.30x | 117 | **1.43x** | +0.13 |
| `other` | 296 | 1.19x | 75 | **0.86x** | −0.32 |
| debt_issuance | 46 | 1.48x | 46 | 0.91x | −0.57 |
| m_and_a | 907 | 1.21x | 621 | 1.18x | −0.03 |

`other` falling **below** the control (CI [0.73, 1.00]) is the reassuring
direction: unclassified events attributed to their real issuer are genuinely
uninformative, which is what an honest residual bucket should look like.

**The fix must be hybrid.** Macro classes collapse under issuer attribution
(monetary_policy 230 -> 1, employment 56 -> 0, disaster 21 -> 0) because a Fed
decision has no corporate issuer. Corporate classes should attribute via
`issuer_entity`; macro classes need the exposure/edge path they have now.

### F40. There is no hidden M&A signal — the targets are not in the universe
> **CORRECTED by F41 (same day).** The conclusion "the signal was never sampled"
> is right that the population is tiny, but the attribution to SURVIVORSHIP is
> wrong. Measured across 574 takeover headlines, delisting/price-availability is
> the SMALLEST of three leaks; extraction misses and entity resolution dominate.
> And the signal is not absent — where it reaches a priceable name it runs
> **1.80x vs the 1.21x pooled**, with a 9.55-sigma tail. Read F41 first.
The pre-registered hypothesis was that `m_and_a`'s flat 1.21x concealed a
takeover-premium sub-bucket (announced targets gap 20-40%). **Refuted twice.**

Fixing the attribution moved it **1.21x -> 1.18x**. And scanning the blind sample
for explicit deal language finds **14 of 320 cells (4%)** — of which, on
inspection, essentially **one or two** are the tagged name being an announced
target (ARES/GS/JPM all appear as advisers on the same GNC story; NIKA is not
Ista; EC is not a party to Bridas/BP).

So the signal is not diluted — **it was never sampled**. Takeover targets get
acquired and delisted, and the universe is defined by current-day resolvability,
so the names that would show 3x are structurally absent. This is the
survivorship caveat in section 4 turning up as a hard ceiling rather than a
footnote: **no amount of classification or LLM sub-labelling recovers a
population that is missing.** Testing it properly needs delisted-inclusive
price data (CRSP / FirstRateData), already on the deferred list.

**Method note.** The LLM contribution here was not sub-classification — it was
*inspection*. Reading 320 blind headlines found a structural attribution bug and
a survivorship ceiling, both of which dominate anything a finer taxonomy could
have delivered. Blindness was enforced mechanically (the dump script never loads
the price panel) so this is not a post-hoc story fitted to the outcome.


### F41. Takeover signal exists at 1.80x — 95.5% of it never reaches a priceable name
Two independent probes of F40's claim, one embedding and one lexical.

**Embedding (`scripts/embed_takeover_probe.py`).** No encoder is installed, so the
query is built BY EXAMPLE: seed on the 34 chunks in `receptors/data_bloomberg`
(11,100 all-MiniLM vectors, same `bbg_` doc_id namespace as the lake) carrying
explicit takeover language, take their centroid, rank every chunk by cosine. The
direction is real — top hits are the LME takeover-bid process, LSE/LCH.Clearnet,
NYSE Euronext. But only 22 of 172 retrieved docs overlap eg100k, so it is
underpowered; used only to confirm the lexical probe is not missing phrasing.

**Lexical, full corpus (574 takeover headlines).** The taxonomy is doing its job —
353 are labelled `m_and_a`. The pipeline is not:

| stage | share |
|---|---|
| no causal edge at all | **48.1%** |
| edge exists, no entity resolves to a ticker | **40.2%** |
| ticker but no price that day | 7.1% |
| **PRICEABLE** | **4.7%** |

Decomposing the 275 edgeless docs further: **66.2% have no event row either** — the
extractor emitted nothing at all for a document whose headline says "takeover".
The F39 issuer_entity path recovers only 3.3% of them.

**So the three leaks, ranked:** extraction misses (~32% of all takeover docs
produce nothing), entity resolution (~40%), and price availability/delisting
(~7%). **Survivorship is the smallest, not the largest** — which is the opposite
of what F40 asserted.

**And the signal is real where it lands.** The 28 priceable cells average
**1.80x** against the 0.753 control — well above the 1.21x pooled `m_and_a` — with
14% exceeding 2 sigma:

| sigma | name | headline |
|---|---|---|
| **9.55** | STX | Seagate's Default Swaps Surge to Highest Since 2008 Amid Takeover Report |
| 4.11 | CVI | Carl Icahn and CVR Energy Agree on Tender Offer |
| 2.90 | OVV | Encana Shares Soar After PetroChina Agrees to Buy Gas Assets |
| 1.60 | FSLR | First Solar Jumps Most in 17 Months on Takeover Speculation |

**Implication, and it is more optimistic than F40.** The answer to "is there higher
signal in the flat buckets" is **yes — 1.80x vs 1.21x, with fat tails**. It is not
blocked by a structural ceiling that needs CRSP; it is blocked by **pipeline
recall**, and ~88% of the loss (extraction + resolution) is addressable with work
already prototyped in this repo (`resolve_llm.py` for the resolution leg). The
density wall from F35-F38 is therefore not the only constraint — RECALL is a
second, separately fixable one.


### Re-run verification (2026-07-26, after regenerating the classification side-car and mcp_cache)

`classify_layer.py` had never been re-run after taxonomy.py gained its substring
fallback, so every stored artifact and the 60 MB panel cache were still on the old
vocab. Both were regenerated (eg100k `other` 26.4% -> 19.8%, 27 -> 31 classes
present, `clinical_trial`/`operations`/`fiscal_policy`/`fund_flows` now materialised)
and **every headline number above was re-derived on the fresh cache**. Nothing moved
materially:

| figure | as first reported | re-run |
|---|---|---|
| earnings, doc-union | 1.97x (n=350) | **1.96x** (n=359) |
| earnings, issuer | 2.17x (n=531) | **2.17x** (n=544) |
| guidance, issuer | 1.61x | **1.65x** |
| m_and_a, doc-union | 1.21x (n=907) | **1.20x** (n=919) |
| m_and_a, issuer | 1.18x | **1.17x** |
| `other`, issuer | 0.86x | **0.86x** |
| rating_action, issuer | 1.43x | **1.43x** |
| clinical_trial | 1.70x, CI [0.84, 2.86] | **1.70x**, CI [0.85, 2.81] |
| graph takeover reach | 4.5% | **4.7%** |
| primitive takeover reach | 56.4% | **56.4%** |
| primitive subject / mentioned | 1.38x / 1.05x | **1.36x / 1.05x** |
| m_and_a subject / mentioned | 1.41x / 1.04x | **1.39x / 1.05x** |

The cache rebuild dropped 123 tickers as stale and rebuilt 3,502 event cells
(from 3,473), so counts shift by ~1-3% throughout; no lift moved by more than
0.05x and no conclusion changes. `clinical_trial` still crosses 1.0 and remains
unestablished.

### F42. ~~F31's dead contemporaneous signal was profile THINNESS, not absence~~ — **RETRACTED, see F43**
> **RETRACTED same day.** Every number below is real but the interpretation is
> wrong: the subject-profile gain is **centroid proximity in an uncentred
> embedding space**, not information. Mean-centring the space destroys it
> (contemporaneous +0.048 -> +0.006 t0.4; forward +0.033 -> -0.017 t-1.4). F31's
> original result, by contrast, SURVIVES centring. Read F43.
`scripts/rigor_recheck_t3_profile.py` — one variable changed (where profile text
comes from); universe, events, baskets, windows and non-mention exclusion held
identical. `--profile edge` reproduces F31 exactly, to three decimals.

**Why look.** F31 builds firm profiles from causal-edge quotes only — the
high-precision/low-recall path. Measured: **median 3 documents per firm, and 81%
of firms rest on fewer than 10.** That is thin enough that a null could be
under-powering rather than absence.

| profile source | contemporaneous increment | forward increment |
|---|---|---|
| **edge** (F31 baseline) | +0.016 (t1.5) CI [−0.004, +0.036] — **not significant** | +0.028 (t2.7) |
| **both** (edge + headline-subject) | **+0.027 (t2.6)** CI [+0.006, +0.047] | +0.025 (t2.4) |
| **subject** (headline-named only) | **+0.048 (t3.8)** CI [+0.024, +0.072] | **+0.033 (t2.7)** |
| subject + `--strict-mention` | **+0.048 (t3.8)** | **+0.033 (t2.7)** |

**F31's headline conclusion — "contemporaneous exposure is mostly price history;
text adds ~0 clean" — does not survive better profile attribution.** The
increment triples and crosses firmly into significance.

**It is not leakage.** `--strict-mention` folds every firm named in an event
document's HEADLINE into the exclusion set (5,228 documents) — the original T2
exclusion covered causal-effect / sentiment / relation targets only, so
headline-named firms had been counted as "non-mentioned". The result does not move
by a single decimal. Profiles remain strictly pre-event throughout, so the
leakage-free property F31 established is preserved.

**Text quantity vs firm composition, honestly separated.** `subject` mode drops
firms that never appear in a headline, so its universe differs. `both` keeps
**every** edge firm and only adds text — and on that fixed firm set the
contemporaneous increment still goes **+0.016 (t1.5) -> +0.027 (t2.6)**, i.e. more
text alone moves it from null to significant. The additional lift to +0.048 in
subject-only mode plausibly reflects a cleaner firm subset as well, and that part
is not isolated here.

Curiosity worth flagging: **`both` scores below `subject` on both metrics**, so
folding the causal-edge quotes back in *dilutes*. Either the edge quotes are
noisier than headline+lede text, or n is small enough that this is chance.

**What this does and does not change.** It is still second-moment — covariance and
risk, not returns or direction. F24/F25/F28/F29 are untouched. But F30/F31's
"contemporaneous text adds nothing" was load-bearing in the strategic read that the
news channel is a 0-3% sliver, and it turns out to have been an artifact of how
little text each profile got. One regime (2010-2012), one outlet, 114 events.

**Method note.** The earlier claim that F17/F18/F30/F31 were contaminated by the
doc-union bug was WRONG — all four read `causal_event_edge.effect_entity`, never
`doctypes`. The exposure was recall/thinness, a different mechanism reaching a
similar place. Right conclusion, wrong reason, corrected before running.


### F43. The centring check — F31 survives it, F42 does not, and it reverses the causal-vs-primitive call
A skeptical question ("how can name and subject carry anything predictive?")
prompted the adversarial check neither F31 nor F42 had faced. Transformer
embedding spaces are strongly anisotropic: cosine in an UNCENTRED space is
dominated by proximity to the corpus centroid, and a heavily-covered firm has a
diverse profile whose mean sits near that centroid — hence near ANY event vector.
Such firms are also large, liquid, and co-move more with any basket. That is a
confound with nothing to do with news content.

`--center` mean-centres the embedding space before averaging. Forward text
increment over the price-history baseline:

| profile source | uncentred | **centred** |
|---|---|---|
| **edge** (causal-edge quotes — F31) | +0.028 (t2.7) | **+0.028 (t2.6) — SURVIVES** |
| **subject** (headline + lede — F42) | +0.033 (t2.7) | **−0.017 (t−1.4) — COLLAPSES** |

Contemporaneous is null in both arms once centred (+0.006 / +0.006), so **F31's
original contemporaneous conclusion stands and F42's "revival" is dead.**

**Three consequences.**

1. **F31 is strengthened, not weakened.** The one surviving predictive finding in
   the project passed an adversarial check it had never been given. The forward
   covariance signal is not an anisotropy artifact.
2. **F42 is retracted.** The profile-thinness story was measured correctly and
   interpreted wrongly.
3. **The causal-vs-primitive call reverses.** The claim two turns earlier — that
   causal extraction adds nothing and the cheap headline path beats it — is
   **wrong**. Centred, it is the exact opposite: **causal-edge quotes carry the
   signal that survives; headline+lede text carries none.** The plausible
   mechanism is selection — a causal-edge quote is the specific span asserting a
   relation, which is high-information and low-redundancy, whereas headline+lede
   is generic text dominated by the space's common component.

**Doctrine.** Every embedding-cosine result in this repo predating this check is
suspect until re-run with `--center`; anisotropy inflates uncentred cosine
similarity systematically and in the direction of size/coverage. `--sector-control`
is worth keeping too, though it exonerated the text here: a same-sector dummy is
worth +0.011 (t0.8) contemporaneously, and the returns are already residualised on
macro plus the 9 SPDR sector ETFs, so broad sector is gone from the target before
the test begins.

### F44. The causal extraction earns its keep on DOCUMENT ATTRIBUTION, not span semantics
F43 showed causal-edge quotes survive mean-centring where headline+lede text does
not. Two explanations were live: (a) the extracted causal SPAN is the informative
unit, or (b) the causal edge merely identifies the right DOCUMENTS and any text
from them would do. `--profile edge_random` separates them: same firms, same
documents, same months, same counts — but a **deterministically chosen random
sentence** from the document instead of the extracted span.

Forward text increment over the price-history baseline, all mean-centred:

| arm | documents | text | forward increment |
|---|---|---|---|
| `edge` | causal-edge | the causal span | **+0.028 (t2.6)** CI [+0.007, +0.048] |
| `edge_random` | causal-edge | a random sentence | **+0.022 (t2.0)** CI [+0.001, +0.042] |
| `subject` | headline-matched | headline + lede | **−0.017 (t−1.4)** |

**A random sentence recovers ~79% of the signal.** The span premium (+0.006) sits
well inside overlapping CIs and is **not established**. Meanwhile changing the
DOCUMENT SET destroys the signal outright.

**So the extraction's contribution is retrieval, not semantics.** What matters is
identifying documents that assert an exposure relationship for a firm — not
parsing what the relationship says. A headline naming a firm indicates *coverage*;
a causal edge indicates a *claim about that firm's drivers*, and only the latter
predicts residual co-movement.

**Caveat not excluded:** the `subject` arm also has a narrower firm universe (201
firms with profile text vs edge's ~800), so document-set and universe composition
are not fully separated. The clean missing cell is causal-edge documents with
headline-only text.

**Implication.** If the value is document→firm attribution rather than causal
parsing, the expensive LLM step may be reducible to a much cheaper linking task —
but NOT to naive headline matching, which measurably fails. That is a sharper and
more useful specification than either "the graph works" or "the graph is
unnecessary".

### F45. Centring audit of every embedding-cosine result — two survive, two die, one is demoted
`EMB_CENTER=1` added to `event_embedding_risk.py`, `embed_exposure.py`,
`hedge_overlay.py`, `mechanism_factors.py`, `information_gap.py`; each run
baseline-then-centred. All baselines reproduced their recorded values first.

| result | script | uncentred | **centred** | verdict |
|---|---|---|---|---|
| **F31** forward covariance | `rigor_recheck_t3` | +0.028 (t2.7) | **+0.028 (t2.6)** | **SURVIVES** |
| **F32** hedge persistence | `hedge_overlay` | +0.028 (t2.7); revert −0.191 vs −0.280 | **+0.028 (t2.6); −0.184 vs −0.271** | **SURVIVES** |
| **edge-hunt #2** embedding risk baskets | `event_embedding_risk --st` | +0.091, placebo **p=0.005**, beats GICS +0.062 and random +0.004 | **−0.009, placebo p=0.690, random (+0.057) BEATS it** | **DIES** |
| **F42** subject profiles (mine) | `rigor_recheck_t3_profile` | +0.048 / +0.033 | **+0.006 / −0.017** | **DIES** |
| **F22** embedding exposure | `embed_exposure` | +0.057 (t7) vs tf-idf +0.047 | **+0.033 (t4) vs tf-idf +0.047** | **DEMOTED** |

**The risk-basket death is the notable one.** The edge-hunt doc already recorded
the *edge* as null OOS but retained the methodological claim that "the embedding is
a good basket constructor / risk-mapping tool". It is not. Uncentred, the `rate`
anchor selected a **Finance-dominated** basket; centred it selects
**Manufacturing**. The mechanism is now legible: heavily-covered names sit near the
corpus centroid, the centroid is near every anchor, banks are the most-covered
names in a 2010-12 Bloomberg corpus, and banks genuinely move on FOMC. The
apparent skill was size proxy -> banks -> FOMC sensitivity.

**F22 is demoted rather than killed.** The embedding still carries signal (t4) but
**no longer beats the tf-idf baseline it was credited with beating** (+0.033 vs
+0.047). tf-idf is sparse and non-anisotropic, so its number is unaffected — the
entire "embeddings beat bag-of-words" margin was the artifact.

**The pattern across the whole audit:** everything built on **causal-edge quotes**
survives centring untouched (F31, F32); everything built on **broader or
differently-selected text** dies or shrinks (risk baskets, F42, F22). That is F44's
conclusion arriving independently — the causal edge's contribution is identifying
*which documents pertain to a firm*, and results resting on that are robust while
results resting on generic text similarity are not.

**Not yet re-run:** `mechanism_factors` (F23), `information_gap` (F29),
`mechanism_direction`, and the `news_*latent/clusters` family. F23/F29 were already
recorded as null or negative, so the expected value is lower, but the flag is in
place for `mechanism_factors` and `information_gap`.

### F46. Second-order propagation from drug trials — NULL at sector resolution
`scripts/second_order_trials.py`. First run of the event-backward angle: take an
event, then ask which firms it should touch WITHOUT being named in it.

157 trial documents (110 from the `clinical_trial` class created in F38, plus 47
recovered by headline patterns for phase/FDA/endpoint language), 64 event days with
at least one named priceable firm, 29 naming a Health Care firm.

| group | n | event \|sigma\| | base | lift |
|---|---|---|---|---|
| **named** (first-order) | 72 | 1.119 | 0.757 | **1.48x** |
| **2nd-order**: unmentioned Health Care | 2,442 | 0.776 | 0.742 | **1.05x** |
| control: unmentioned other sectors | 21,264 | 0.798 | 0.753 | **1.06x** |

Circular-shift placebo on the second-order group: **p = 0.177**. Widening the
window to +/-2 sessions changes nothing (1.04x vs control 1.05x, p = 0.119), so
this is not a timing artifact.

**The sanity check passes and the test fails.** Named firms move 1.48x on trial
days, so the event set is real. But unmentioned Health Care firms move **no more
than unmentioned firms in any other sector** — 1.05x vs 1.06x is a dead heat. That
is the pre-registered "trial days are just noisy days" branch: a small
everyone-moves-slightly-more effect, not propagation.

**What actually died is the SECTOR PROXY, not necessarily the thesis.** "Unmentioned
Health Care firm" is a poor stand-in for "exposed to this trial". The economics that
motivated the test are indication-level — a same-indication rival gains when your
readout fails, a licensee moves with its partner's data — and a random biotech is
not exposed to an unrelated oncology readout at all. Diluting 39 Health Care names
into one bucket guarantees that a real effect on 2-3 of them is invisible.

**Which is the argument for `graph_transitivity.py`.** It defines second-order
exposure by SHARED DRIVERS IN THE CAUSAL GRAPH (`bridge2`: an edge connects a driver
of i to a driver of j; `parent2`: directed common cause) rather than by sector
membership, with hub exclusion as a kill condition and `child2` as a built-in
collider placebo. That is the sharper instrument for the same question, it is fully
implemented, and it has still never been run.

**Honest limits:** 39 Health Care names in the priceable universe, 29 usable event
days. Even a correct exposure map would be thin here.

### F47. Can the LLM be dropped? Labels yes, document selection no (so far)
Two separable questions, opposite answers.

**Classification — YES, without a model.** `EVENT_TYPE_PATTERNS` reads event_type
SLUGS; applied to prose it misses most of it, because newswires write verbs
("acquisit" matches "acquisition" but not "acquires", "buys", "bid for"). Measured
on the 26,362 eg100k headlines that name a verified firm, one iteration of prose
triggers (`taxonomy.classify_headline`, landed):

| | unclassified |
|---|---|
| slug patterns only | **61.6%** |
| + prose triggers | **41.1%** |

Recovered: market_move 1,919 · **roundup 1,334** · m_and_a 761 · operations 354 ·
guidance 306 · debt_issuance 226 · monetary_policy 177 · legal 142 · rest 206.

The two EXCLUSION classes matter as much as the event classes. `roundup`
("Colombian Stocks: Ecopetrol, Rubiales, Canacol") catches the bystander documents
F39 showed were poisoning every bucket via doc-union — at source, rather than
downstream. `opinion` catches bylined columns ("...: David Reilly"). Both are
classifications, not failures; callers drop them. The residual 41.1% is dominated
by more verb constructions a second iteration would catch, so parity with the LLM
path's 19.8% looks reachable.

**Document selection — NO, not yet.** F44 established the LLM's real contribution
is identifying WHICH DOCUMENTS assert an exposure relationship for a firm. Two
deterministic substitutes, both against F31's centred **+0.028 (t2.6)** benchmark,
same events, same machinery:

| document selector | firm-doc entries | forward increment (centred) |
|---|---|---|
| `edge` — LLM causal edges | — | **+0.028 (t2.6)** |
| `subject` — headline names the firm | 4,249 | **−0.017 (t−1.4)** |
| `causal_lex` — headline names the firm AND asserts a driver relation | 422 | **−0.017 (t−0.8)** |

Both fail identically, and **at very different text volumes** (4,249 vs 422
entries), so this is not the thinness story — headline-derived document sets simply
do not carry it, filtered or not.

**The likely reason, and the untested next candidate.** Causal edges are extracted
from document BODIES. A document can assert "X drives Y" in paragraph four with a
headline that says nothing of the sort, so headline-level filtering is
structurally blind to exactly the documents the edge is marking. The remaining
candidate is body-level causal-language filtering; untested.

**Process note.** The first `causal_lex` run returned +0.028 (t2.6) — identical to
the edge baseline to three decimals — because a string patch silently missed on
indentation and `_CAUSAL_LEX` was referenced but never defined, so the run was just
the baseline again. It looked like a clean positive. **A patch that fails silently
produces a perfect false negative**; the diagnostic print line being absent was the
only tell. Verify the patch landed before believing the number.

### F48. ~~Causal-chain search — the chain fails its kill condition~~ — **SUPERSEDED by F49**
> **The chain verdict below was produced by a DEFECTIVE driver construction** —
> driver sets aggregated over all of 2010-12 (median 3, mean 6.4, **max 129** per
> firm) instead of the trailing window, which both diluted them and leaked
> post-event edges. Fixed in F49; the conclusion reverses. The embedding results
> below stand.
Design: search the causal graph for a 2-hop chain from the event's cause to a
candidate firm's own drivers (`bridge`), then weight candidates by relevance in
the CAUSAL embedding space (mean-centred causal-quote profiles — the one
representation that survived F43/F45). Both features tested against the same
forward co-movement target, incremental to trailing correlation, on the same 114
events. `scripts/rigor_recheck_t3_profile.py --chain`.

**Hub-exclusion sensitivity is the whole result.** `graph_transitivity.py`
pre-registered the rule: *"KILL CONDITION: effect must survive hub exclusion"*,
because macro hubs (Greece in 2011, the Fed) connect everything.

| top-% entities excluded as hubs | bridge -> fwd | **bridge \| trail -> fwd** | events |
|---|---|---|---|
| 0.25% | +0.041 (t2.7) | **+0.032 (t2.1)** | 67 |
| 1.0% | +0.009 (t0.4) | **−0.002 (t−0.1)** | 36 |
| 2.0% | +0.009 (t0.3) | **−0.002 (t−0.1)** | 21 |

**The chain effect exists only when hubs are left in.** Excluding the top 1% by
degree removes it entirely. The point estimate COLLAPSES (+0.032 -> −0.002) rather
than merely widening its interval, which points at hub-driven rather than at power
loss — though n does fall to 36/21, so power is not fully excluded. Either way the
pre-registered kill condition fires: what looks like second-order causal
propagation is largely "both firms connect to the same high-degree macro entity",
i.e. shared macro beta that entity-level residualisation does not remove.

**The embedding weighting survives everything.** Incremental to trailing
correlation AND to the bridge feature:

| control set | emb increment |
|---|---|
| trail only | +0.028 (t2.6) |
| trail + sector dummy | +0.055 (t4.5) |
| trail + bridge (hub 1%/2%) | **+0.027 (t2.5)** |
| trail + bridge (hub 0.25%) | **+0.022 (t2.1)** |

At loose hub exclusion the two features overlap substantially — adding bridge cuts
the embedding increment 0.028 -> 0.022, and adding embedding cuts bridge 0.032 ->
0.024 (t1.7, CI now crossing zero). Neither subsumes the other, but only the
embedding is robust to the hub rule.

**Read:** the *ranking* half of the proposal works and is now the most heavily
defended result in the project — it has survived mean-centring, a same-sector
dummy, headline-mention exclusion, and a graph-structure control. The *search* half
does not survive its own pre-registered hub test. Second-order exposure, at this
corpus density, is not recoverable from graph topology once you remove the
entities that connect everything.


### F49. Driver SPECIFICITY was the defect — windowed, the chain strengthens and the kill condition no longer fires
Challenge raised: *are the drivers of high enough specificity?* They were not.

`graph_transitivity.py` specifies "features per pair-month, from the entity causal
graph over the same trailing window as the correlation" and "hubs: WITHIN-WINDOW
degree". F48's implementation used neither — driver sets and hub degree were built
over all of 2010-12.

| driver construction | drivers per firm |
|---|---|
| whole 2010-12 (F48, wrong) | median 3, mean 6.4, **max 129** |
| trailing window (F49, correct) | median 1, mean 1.8, max 26 |

A firm with 129 drivers bridges to nearly anything, and the set included edges from
months AFTER the event — dilution plus leakage.

**Rebuilt windowed, the chain gets STRONGER, not weaker:**

| | F48 (global drivers) | **F49 (windowed)** |
|---|---|---|
| bridge -> fwd, hub 0.25% | +0.041 (t2.7) | **+0.058 (t2.5)** |
| bridge \| trail, hub 0.25% | +0.032 (t2.1) | **+0.049 (t2.1)** |
| bridge \| trail, hub 1.0% | **−0.002 (t−0.1)** | **+0.033 (t1.3)** |
| bridge \| trail + emb, hub 0.25% | +0.024 (t1.7) | **+0.045 (t1.9)** |

**The kill condition no longer fires.** Under the global construction the point
estimate COLLAPSED and flipped sign at 1% hub exclusion (+0.032 -> −0.002), the
signature of a hub-driven artifact. Windowed, it HOLDS (+0.049 -> +0.033) and only
the interval widens as n falls 33 -> 22. That is power, not hubs.

**And the chain is now complementary to the embedding, not redundant with it.**
Controlling for the causal-embedding score, bridge still carries +0.045 (t1.9) —
against +0.024 (t1.7) under the defective construction. The embedding meanwhile is
unmoved by adding bridge (+0.028 -> +0.026, t2.4), as it has been by every other
control.

**Status: promising, not established.** At strict hub exclusion the bridge CIs
cross zero (t1.3-1.9), and n is 22-33 events. The specificity fix moved every
point estimate in the same direction, which is the encouraging sign; the sample is
too thin to call it. **This is the first result today where a correction made a
finding stronger rather than weaker**, and the honest next step is more events, not
more controls — which points back at the 446k corpus.

**Method note.** Two implementation defects in one day produced two wrong
conclusions (F42 interpretation, F48 chain verdict). Both were caught by external
challenge rather than by the harness. A spec existed in `graph_transitivity.py`'s
docstring and was not followed; reading the pre-registered design before
implementing it would have prevented this one.

### F50. INFORMATION SPECIFICITY — mean-pooling was averaging the signal away
Challenge raised: the causal descriptions may not be specific enough — "second
line trial success breakthrough" is the level at which exposure is actually
determined. F31 scores a firm by `cosine(MEAN of up to 80 causal quotes, MEAN of
up to 40 event quotes)`. **Mean-pooling is exactly the operation that turns
"second-line trial success, breakthrough designation" into "generic pharma".**

Replacing it with best-match pooling — does ANY of this firm's causal descriptions
match ANY of this event's, closely? — `scripts/rigor_recheck_t3_profile.py --pool`:

| pooling | contemporaneous | **forward** |
|---|---|---|
| `mean` (F31 original) | +0.006 (t0.5) | **+0.028 (t2.6)** |
| `max` (single best pair) | +0.022 (t1.9) | **+0.042 (t4.2)** |
| `top3` (mean of best 3) | +0.024 (t2.1) | **+0.042 (t4.2)** |

max and top3 agree to three decimals, so this is not one fluke match.

**But max-pooling has a mechanical coverage bias** — a firm with 80 quotes gets 80
chances at a high match, a firm with 2 gets 2 — and coverage tracks size, which
tracks co-movement. The same class of confound as anisotropy, so it must be
excluded before believing anything. Controlling for log(pre-event quote count),
with `--strict-mention` also on:

| | raw | **coverage-controlled** |
|---|---|---|
| log #quotes -> forward, alone | — | **+0.029 (t2.4)** — the confound is real |
| emb \\| trail, forward | +0.042 (t4.2) | **+0.032 (t3.0)** — SURVIVES |
| emb \\| trail, contemporaneous | +0.024 (t2.1) | **+0.017 (t1.5)** — DIES |

**Verdict, split.** The specificity insight is CORRECT and gives the only genuine
improvement of the day: forward **+0.028 -> +0.032 (t2.6 -> t3.0)**, after killing
a coverage confound that accounted for roughly a quarter of the raw gain. The
contemporaneous "revival" does NOT survive — **F31's original contemporaneous
conclusion stands**, and my F42 retraction of it remains correct.

**The forward result has now passed every adversarial control raised today:**
mean-centring (anisotropy), a same-sector dummy, headline-mention exclusion, a
graph-structure control, and coverage volume. It is the most defended number in
the project at **+0.032 (t3.0)**.

**And it explains the F44 anomaly.** A random sentence from a causal document
scored +0.022 against the causal span's +0.028 — suspiciously close if span
selection mattered. Under mean-pooling it wouldn't: averaging 80 quotes washes out
whichever sentence you picked. The specificity question and that anomaly have the
same answer.

### F51. LEXICAL beats dense — the embedding space was the wrong space
Challenge raised: max-pooling still searches *embedding* similarity, which is
fuzzy exactly where specificity lives; try a keyword/hybrid instead. Implemented
as idf-weighted term-overlap over the SAME causal quotes, same pooling, same
events, same controls (`--sim {dense,lexical,hybrid}`).

**Forward increment, fully controlled** (mean-centred, top3 pooling,
`--strict-mention`, and coverage-controlled for log #quotes):

| similarity space | contemporaneous | **forward** |
|---|---|---|
| dense (all-MiniLM cosine) | +0.017 (t1.5) — dies | **+0.032 (t3.0)** |
| **lexical (idf term overlap)** | **+0.036 (t3.0)** | **+0.048 (t5.2)** |
| hybrid (z-average of both) | +0.031 (t2.5) | +0.047 (t4.5) |

**Lexical wins outright, and hybrid is no better than lexical alone** — adding the
dense channel slightly dilutes it. The whole embedding apparatus is being beaten by
idf-weighted term overlap.

**Third independent confirmation in this repo that sparse beats dense.** F45 found
`embed_exposure`'s tf-idf baseline (+0.047) beating its centred embedding (+0.033);
F23's centred mechanism factors (+0.038) also landed below that same tf-idf +0.047;
now quote-level lexical (+0.048) beats quote-level dense (+0.032) under the full
control set. Three different harnesses, same verdict.

**And lexical is structurally more trustworthy.** Sparse idf vectors are not
anisotropic, so the centroid-proximity artifact that killed three findings today
(F43/F45) cannot arise in this space at all. `--center` is a no-op for it.

**F31's contemporaneous null was a property of the REPRESENTATION, not of text.**
It reported contemporaneous text adding ~0 over price history. That holds for dense
mean-pooled profiles and collapsed under every attempt to revive it (F42 retracted,
F50 died under coverage control). In lexical space with best-match pooling it is
**+0.036 (t3.0)**, surviving coverage control, strict-mention and centring.

**Cumulative effect of the two challenges** (specificity of pooling, then of
representation): forward **+0.028 (t2.6) -> +0.048 (t5.2)**. A 71% larger effect
and roughly double the t, from the same data, same events, same leakage discipline.

**Production implication.** `retrieval_router.py` builds hybrid SCOPING (metadata
filters + dense), not dense+sparse fusion, and the Qdrant collection is empty.
There is now a measured reason to build it with **sparse/BM25 as the primary
channel** rather than as an afterthought to the dense one.

### F52. In LEXICAL space the full corpus works — the graph adds ~70%, it is no longer a prerequisite
Challenge raised: if we are using sparse representations, why stay inside the
graph? Because F44/F47 said document selection was what the LLM earned — but
**both were measured in DENSE space**, which F51 established is the wrong space.
Re-running full-corpus (alias-matched) attribution with lexical similarity:

| document set | representation | forward, coverage-controlled |
|---|---|---|
| graph (causal-edge docs) | dense | +0.032 (t3.0) |
| graph (causal-edge docs) | **lexical** | **+0.048 (t5.2)** |
| full corpus (alias-matched) | dense | **−0.017 (t−1.4)** — null |
| full corpus (alias-matched) | **lexical** | **+0.028 (t2.3)** — works |

**F47's "document selection: NO" was a dense-space artifact.** Full-corpus
attribution goes from null/negative to positive and significant purely by changing
representation. The graph still wins — **+0.048 vs +0.028, about 70% more signal**
— so causal-edge selection earns its keep, but it is no longer a PREREQUISITE.

**The zero-LLM path now reproduces F31's original headline.** Alias matching plus
idf term overlap on the full corpus gives **+0.028 (t2.3)** — the same number F31
reported (+0.028, t2.6) using LLM causal extraction and dense embeddings. The
entire extraction pipeline can be replaced, at the cost of the 70% the graph adds.

**Coverage control is doing much heavier lifting here, as it must.** In the
full-corpus arm, log(#quotes) alone predicts forward co-movement at **+0.093
(t7.8)** versus +0.029 (t2.4) in the graph arm — because alias attribution gives a
firm ~70 documents against the graph's ~3, and the count tracks prominence. Raw
uncontrolled the arm reads +0.071 (t6.3); controlled it is +0.028. Anyone running
this path without the coverage control will measure firm size.

**What this unlocks.** The full-corpus lexical path needs **no extraction, no
embeddings, no GPU, no API** — so the 446k Bloomberg corpus becomes immediately
runnable rather than gated on extraction spend. n has been the binding constraint
on every result today (114 events overall, 22-33 for the chain). This is the first
route to lifting it that costs nothing.

**Caveat:** the two arms do not cover identical firm sets (the subject path drops
firms never named in a headline), so the +0.048 vs +0.028 gap conflates document
selection with universe composition. Isolating that needs the `both` arm re-run in
lexical space.

### F53. Recency decay HURTS, and the graph's advantage is not universe composition
Two loose ends, both negative-to-neutral, both informative.

**Recency weighting — tested and rejected.** `preprofile` treats a 2010-01 quote
identically to a 2012-11 one when scoring a 2012-12 event, and under best-match
pooling a stale perfect match beats a fresh good one. Exponential decay
(`--half-life`, months) makes it WORSE in both arms:

| arm | no decay | with decay |
|---|---|---|
| graph (edge) | +0.048 (t5.2) | **+0.036 (t3.9)** at 6-month half-life |
| full corpus (subject) | +0.028 (t2.3) | **+0.021 (t1.8)** at 12-month half-life |

The graph result is explicable — median 3 quotes per firm, so downweighting most of
them leaves nothing to match. The full-corpus arm has ~70 documents per firm and a
wide time spread, where decay *should* have helped, and it did not.

**So firm exposure profiles are STABLE over 1-3 years.** That is informative about
what the signal is: a durable structural signature (what this firm is exposed to),
not a decaying news-flow effect. It is finer than sector — a same-sector dummy is
worth only +0.011 (t0.8) — but it behaves like a characteristic, not like news.
Anyone reading this as "news predicts covariance" should read it as "text reveals a
stable exposure characteristic that price history alone estimates noisily".

**The graph's advantage survives the universe control.** F52's +0.048 vs +0.028 gap
conflated document selection with firm coverage, since the subject path drops firms
never named in a headline. The `both` arm (union of documents AND firms) settles it:

| arm (lexical, top3, fully controlled) | contemporaneous | forward |
|---|---|---|
| graph only | +0.036 (t3.0) | +0.048 (t5.2) |
| full corpus only | +0.016 (t1.5) | +0.028 (t2.3) |
| **both** | **+0.041 (t3.8)** | **+0.049 (t5.5)** |

`both` matches the graph arm on forward (+0.049 vs +0.048) on a strictly larger
universe, so the gap is **document selection, not composition** — the graph's ~70%
advantage is real. Adding full-corpus documents does not dilute it, and improves
the contemporaneous reading to its strongest value of the day (+0.041, t3.8).

**Best configuration found:** `--profile both --sim lexical --pool top3 --center
--strict-mention --nq-control`, no recency decay.

### F54. The signal is NOT interpretable — it lives in common vocabulary, not specific language
Challenge raised: *prove a signal I can infer something from, not statistics.* The
lexical representation makes that answerable, because the matching terms can be
printed. `--examples` dumps, per event, the firms ranked most exposed while never
being mentioned, and **the actual shared terms driving each match**.

**What is actually driving the matches:**

```
TM     ~ apple           : nissan, motor        Toyota "exposed" to an Apple event
BLK    ~ apple           : inc                  matched on the word "inc"
SHEL   ~ apple           : fuel
JPM    ~ goldman_sachs   : york-based, asked, new
BP     ~ bank_of_america : estimate
GOOGL  ~ goldman_sachs   : goldman, sachs       <- co-mention, not inference

most common match terms overall:
percent(16) · goldman(12) · sachs(11) · inc(6) · china(5) · much(4) · group(4)
```

**The single largest driver is "percent".** Where matches are not generic filler
they are largely CO-MENTION — the firm's own past coverage names the event's
subject.

**The decisive test.** If the signal were carried by specific exposure language
("second-line trial success, breakthrough"), restricting to rare, high-information
terms should PRESERVE or strengthen it. It does the opposite:

| min-idf | terms kept | forward increment | events |
|---|---|---|---|
| none | 8,598 | **+0.049 (t5.5)** | 114 |
| 3.0 | 8,588 | +0.052 (t5.8) | 114 |
| 4.5 | 8,394 | +0.046 (t4.8) | 114 |
| 6.0 | 7,135 | **+0.011 (t1.0)** | 89 |
| 7.0 | 6,047 | **+0.002 (t0.2)** | 46 |

**The entire effect lives in terms appearing in more than ~12 of 5,485 quotes.**
Restrict to genuinely distinctive vocabulary and it is zero. The point estimate
collapses monotonically (+0.049 → +0.011 → +0.002) rather than merely widening,
though n does fall as firms lose all overlap.

**So the specificity story is backwards.** F50/F51's gains were real as
measurements, but not for the stated reason: best-match pooling and lexical
matching improved the statistic while the *content* doing the work is common
newswire register, not specific exposure language.

**And it contradicts F31's core interpretive claim** — "genuine latent inference
(pre-event-only), not retrieval". Inspecting the matches, it IS retrieval:
co-mention plus vocabulary-register overlap. The most likely mechanism is
register similarity acting as a crude topic/sub-sector proxy — finer than a GICS
dummy (which was worth only +0.011), which is why the sector control did not
catch it, but the same family of thing F18 already identified: *"the news graph is
largely a text-derived sector/sub-sector classifier."*

**Status of the +0.049: statistically robust, interpretively empty.** It survived
mean-centring, a sector dummy, headline-mention exclusion, a graph-structure
control and a coverage control — and then failed the simplest check of all, which
was to look at it. **A statistic that survives five adversarial controls can still
be measuring nothing you would want to act on.** That is the lesson of the day.

### F55. Learned sparse selection — fits in-sample, transfers to nothing
Challenge raised: rather than a hand-set idf cut, TRAIN something to find which
sparse features matter in which cases. Done: L1 (LassoCV) over the 8,598-term
shared-vocabulary matrix, 13,489 firm-event rows, target residualised on trailing
correlation first so the model must find something price history does not already
have, and a strict **train-on-early / test-on-late** split (train ≤ 2012-03).

| | IC vs residualised target |
|---|---|
| in-sample | **+0.158** |
| **out-of-sample** | **+0.004** |

39 of 8,598 terms selected. **It fits and it does not transfer.**

**And the selected terms say exactly why:**

```
+0.298 banking   +0.287 european   +0.175 report    +0.150 billion
+0.147 concern   +0.121 america    +0.110 europe    +0.110 banks
+0.093 group     +0.090 securities +0.082 government
-0.180 inc       -0.154 companies  -0.139 quarter   -0.109 rose
                 +0.060 his        +0.057 five      +0.057 plc
```

Two kinds of word, neither of them an exposure characteristic: **period topic**
(banking, european, europe, banks, securities, government — the 2010-12 sovereign
crisis) and **newswire furniture** (inc, report, companies, quarter, rose, group,
his, five, plc, shares).

The model learned *what 2010-2012 was about*. The test period begins 2012-04, the
dominant topic moves, and the learned weights are worth +0.004.

**This also explains the +0.049 retrospectively.** None of the earlier tests had a
train/test split — the IC was cross-sectional within each event, pooled across
events, all inside one period. Period-specific topic vocabulary scores perfectly
well under that design. **Requiring the relationship to transfer across time is
what kills it**, and that requirement was never imposed until now.

**Would a neural network help? No, and the diagnostic says why.** Capacity is not
the binding constraint — L1 already fits at +0.158 in-sample. More capacity fits
the period topic *better* and transfers no further, probably worse. The
constraint is that the learnable structure is a time-local topic, not a
characteristic. That is the same conclusion `news_latent_signal.py` and
`news_raw_latent.py` reached by PLS ("the constraint is redundancy with price, not
representation or sample size") — now confirmed on sparse features with the terms
made visible.

**Closing status of this line.** The forward covariance signal is: statistically
robust within period, uninterpretable on inspection (F54), and non-transferable
across time (F55). Three independent ways of asking "is this real?" — controls,
content, and time — and only the first says yes.

### F56. Forward-looking bias, hybrid, and term downranking — all tested, all fail out-of-sample
Three proposed fixes, all evaluated on the **time-split OOS** metric F55 established
as the only meaningful one.

| variant | rows | in-sample | **OOS** |
|---|---|---|---|
| baseline (all terms, all modality) | 13,489 | +0.158 | **+0.004** |
| **forward-looking only** (`modality='forecast'`) | 1,432 | +0.214 | **undefined** |
| **stability-selected terms** | 13,489 | +0.080 | **−0.044** |
| sparse+dense hybrid (F51) | — | — | no better than lexical, which is +0.004 |

**Forward-looking bias.** The extractor already tags 34,736 of 134,366 edges
(26%) as `modality='forecast'` — genuine exposure assertions ("should drop more in
the fourth quarter due to lower crude prices"), and a well-motivated filter, since
exposure claims are what the task needs rather than event reporting. But it cuts
the sample 90% to 1,432 rows; the 22 selected terms are zero for nearly every test
row, so predictions have no variance and OOS is undefined. The selected terms are
still furniture: `five`, `inc`, `out`, `because`.

**Term downranking via stability selection.** Keep only terms whose sign is
consistent across both halves of the TRAINING period — aimed squarely at F55's
failure. **4 terms of 8,598 qualify (0.05%)**: `billion`, `climbed`, `inc`,
`companies`. All furniture. OOS **−0.044**.

**Read the 4-of-8,598 number.** It is not that we picked the wrong terms — it is
that **the vocabulary has essentially no temporally stable predictive structure at
all**, not even within one training period, let alone across the regime boundary.

**This line is exhausted, and the pattern is now unambiguous.** Across
representation (dense/lexical/hybrid), pooling (mean/max/top3), document selection
(graph/full-corpus/causal-lexical/forward-looking), term weighting (idf
threshold/L1/stability), and capacity (overlap score → L1): **every variant fits
in-sample and none transfers.** That is the signature of a sample-size and
single-regime problem, not a method problem — which means more method variants have
negative expected value, and the only moves with positive expected value are more
data (the 446k corpus) or a different market (`india2021`).

### F57. LLM-as-activation-pathway — pre-registered pilot on POST-CUTOFF data, and it fails
The one untested channel: lexical/embedding matching can only find what is
textually co-present, so it can never supply a link like "Hormuz closure ->
peripheral fuel costs -> airlines". An LLM reasoning over the event could. Tested
properly for the first time.

**Why this design is clean.** eg100k (2010-12) is disqualified — F55 showed the
failure mode is TEMPORAL TRANSFER, and hindsight over a period saturated in
training data manufactures exactly that appearance. `eg_live2` runs
**2026-06-27..07-19, after the reasoning model's May 2026 cutoff**. Blindness was
mechanical: `llm_exposure_dump.py` loads no prices; predictions were written to
`data/llm_exposure_predictions.json` from the events and universe alone;
`llm_exposure_score.py` was the first thing to touch a price.

**The shock.** All 12 extracted events collapse to one story — US-Iran, Strait of
Hormuz, oil +11% on the week, dollar and yields up, gold down on revived Fed hike
bets. Two transmission channels predicted: oil (producers benefit, fuel-intensive
transport suffers) and rates (long-duration assets hit).

**Result — mean |standardised abnormal move|, 2026-07-08..17:**

| basket | value |
|---|---|
| most_exposed (15 reasoned names) | **0.660** |
| least_exposed (15 reasoned names) | 0.764 |
| whole universe (264 names) | 0.755 |

**gap most−least −0.104, permutation p = 0.821.** The reasoned basket moved LESS
than the basket predicted to be inert, and less than the universe. Not merely
insignificant — the wrong sign.

**A pre-registered weakness did fire, but does not rescue it.** `least_exposed` was
flagged in advance as micro-cap-skewed (VANI 1.40, TRUP 1.09, CALM 1.04 top that
list), and standardising by trailing vol evidently does not neutralise fat-tailed
small-cap idiosyncratics. But the uncontaminated comparison — **most_exposed vs the
whole universe — is −0.095**, also the wrong sign.

**The reasoning was not uniformly wrong.** Within the exposed basket, the ordering
has some sense: EOG 0.84, FANG 0.84 (shale E&P, highest crude torque) and AAL 0.84
top it. The thesis failed on TLT (0.42) and LMT (0.40) — the rates and defense
channels, which I weighted and which did nothing.

**Honest limits.** One shock, not twelve — a 23-day corpus cannot deliver
independent events. Eight sessions. And residualising on the equal-weight universe
removes the market-wide component of an oil shock by construction, leaving only
cross-sectional differential exposure (which is what was predicted, so the test is
fair, but the residual signal is small).

**Status: the hypothesis is not supported, on a pilot too small to be a verdict.**
What now exists is a pre-registered, hindsight-free harness. The daily scraper is
running, so every additional month adds clean post-cutoff events. This is the one
question in the project where waiting genuinely improves the answer rather than
deferring it.

### F58. Stable 10-K text beats news text 16x — and three optimisation attempts all failed
Following the diagnosis in F53/F55 (we were estimating a stable characteristic with
a time-varying event stream), the input was swapped for **10-K Item 1 business
descriptions** — what a firm IS, not what happened to it. 162 names, 8,911 pairs,
period A 2014-2019 predicting period B 2020-2026.

| feature | predicts period-B correlation, incremental to period-A |
|---|---|
| same-sector dummy | +0.103 raw |
| **news text** (the whole day's work) | **+0.004** |
| **10-K business text** | **+0.064** |

**Unchanged by a same-sector control** (+0.064 -> +0.064), so unlike the news graph
this is NOT merely a text-derived industry classifier — F18's verdict does not
carry over. Permutation null (400 firm-label shuffles of the text matrix, the
correct null for dyadic data): **p = 0.002**.

**The anachronism check passes.** The filing is from 2025-26, i.e. the end of
period B, so it might describe what each firm BECAME. It predicts B-early
(2020-22) at +0.055 and B-late (2023-26) at +0.054 — flat, which is what a stable
characteristic does and a look-ahead artifact does not.

**And it is interpretable, which nothing else today was.** The top pairs:

```
IMO  ~ XOM   exxon, exxonmobil, mobil          Imperial Oil is Exxon-controlled
AMT  ~ SBAC  towers, tenants, rooftop, sites   both cell-tower REITs
BNY  ~ NTRS  custody, servicing, trust         both custody banks
MA   ~ V     four-party, prepaid, authorization
ARES ~ BX    secondaries, multi-asset          both alternative managers
HUT  ~ WULF  hpc, campus, compute, electrical  bitcoin miners pivoting to AI
                                               corr rose +0.08 -> +0.30
```

HUT~WULF is the case for the method: GICS separates them, the text does not, and
their correlation subsequently rose.

**THREE OPTIMISATIONS, ALL NEGATIVE — recorded so they are not retried.**

| attempt | result |
|---|---|
| aggressive boilerplate stoplist | **+0.065 -> +0.047** — also stripped `securities`/`exchange`/`commission`, which are furniture for most filers but genuine business vocabulary for financials |
| surgical stoplist (furniture only) | +0.064 — no change; the stoplist was never the lever |
| cut each doc at the "Available Information" marker | **+0.034** — mangles ~10 docs into short sparse vectors whose L2 normalisation distorts similarity to everything |

**Robustness: the result is NOT carried by a few names.** Jackknife by firm — drop
each in turn — moves it at most **−0.0068** (JPM), and the **top-5 names account
for 12% of total absolute influence**, with inflating and deflating names roughly
balanced.

**What would actually help, untried:** more text per firm (the current input is a
600KB range-slice truncated to 2,500 chars; full Item 1 runs to tens of thousands),
more names (ADRs file 20-F not 10-K, which is why 129 of 312 are missing), and
multiple split points rather than one. All are DATA improvements. Every
representation-side lever tried today has been neutral or negative.

**Caveat on novelty:** this substantially reproduces the Hoberg-Phillips
text-based-industry literature, which builds firm similarity from 10-K product
descriptions. The contribution here is not the method but the comparison — it is
the first correctly-specified baseline this project has had, and it shows the news
apparatus was underperforming a much simpler input by 16x the whole time.

### F59. Optimising the stable-text signal — extraction quality helps, more text partly imports look-ahead
Following F58's conclusion that the remaining levers are all DATA-side, the Item 1
extractor was improved (score the segment's OPENING rather than the whole window,
so a TOC followed by prose no longer wins; cut at the Item 1A/1B/2 header) and the
kept length was varied.

| text | incremental | permutation p | anachronism (early / late) |
|---|---|---|---|
| 2,500 chars, original extraction | +0.064 | 0.002 | +0.055 / +0.054 — flat |
| 2,500 chars, **fixed extraction** | **+0.073** | 0.005 | — |
| **6,000 chars** | **+0.090** | 0.005 | **+0.072 / +0.088 — gap opens** |
| 20,000 chars | +0.062 | 0.025 | — |

**Extraction quality is a genuine gain: +0.064 -> +0.073 at identical length.**

**The move to 6,000 chars is partly NOT a gain.** The anachronism check was flat at
2,500 chars and opens to +0.072 early / +0.088 late at 6,000 — longer extracts pull
in more recent-specific content (current products, recent acquisitions, the current
risk environment), which is look-ahead against a period-B window that starts in
2020. **The defensible number is the early-period one, ~+0.072**, not +0.090.

**Inverted U, and the reason is visible in the pairs.** Item 1 opens with the
business description and drifts into regulation, competition and human capital. At
6,000 chars the top pairs already include `AMD~DE` matching purely on SAFE-HARBOUR
boilerplate (*cautionary, forward-looking, uncertainties, differ*) — two firms with
corrB −0.12. At 20,000 that generic tail dominates and the signal falls to +0.062.

**Good pairs at 6,000, for the record:** `AAL~DAL` (*congress, peb, arbitration,
nmb* — Railway Labor Act machinery, airline-specific), `NTRS~OPY` (*high-net-worth,
fiduciary, custody, aum*), `AMZN~CPNG` (*app-based, omnichannel*), `PSKY~WBD`
(*warner, bros, theaters, netflix*), `AMT~SBAC`, `CVX~XOM`.

**Net of the whole optimisation pass: +0.064 -> ~+0.072 defensible** (+0.090
nominal, discounted for look-ahead). Still ~18x the news-text equivalent of +0.004.

**Next lever, and it is data again:** the safe-harbour and risk-factor language
should be excluded structurally (they are identifiable SECTIONS, not scattered
words), and the 130 missing names are mostly ADRs filing 20-F rather than 10-K.

### F60. Coverage pays, structural section-cutting does not fire
Two levers from F59, both data-side.

**20-F support — WORKED.** Foreign private issuers file 20-F, not 10-K, which is
why 130 of 312 names were missing; their business description sits in Item 4
"Information on the Company" / "Business Overview" rather than Item 1.

| | before | **after** |
|---|---|---|
| descriptions | 182 | **227** |
| usable names | 161 | **197** |
| pairs | 8,778 | **13,695** |
| incremental (6k chars) | +0.090 | +0.090 |
| permutation p | 0.005 | **0.002** |

Same effect size, 56% more pairs, better significance. Coverage buys power, not
magnitude — which is the expected and honest outcome.

**Structural section-cutting — DID NOT FIRE.** The intent was to cut safe-harbour
and risk sections where they live (they are SECTIONS, and the word-level stoplist
attempt already failed in F58). Median description length after the change is
19,999 of a 20,000 cap, i.e. **the end-marker regex essentially never matched**.
Widening the header regex to catch 20-F's Item 4 changed where extraction starts,
and the Item 1A/forward-looking markers do not appear within the window from there.
Recorded as not-yet-solved rather than as a null: the lever was never actually
pulled.

**Standing position on this signal:**

| | value |
|---|---|
| nominal, 6,000 chars | +0.090 (p 0.002) |
| **defensible, discounted for look-ahead** | **~+0.072** |
| news-text equivalent (F55) | +0.004 |
| same-sector dummy | control, and it changes nothing |

The anachronism gap (+0.072 early / +0.088 late) is stable across both fetches, so
the discount is a property of using 6,000 chars of a 2025-26 filing to predict a
window opening in 2020 — not noise.

### F61. Walk-forward across four disjoint regimes — the look-ahead concern is REFUTED
The strongest test available without a second market: walk the A/B split forward so
period B moves from nine years before the 10-K filing to contemporaneous with it. A
stable characteristic should hold everywhere; look-ahead from a 2025-26 filing must
DECAY the further back period B sits.

| period A | period B | distance from filing | pairs | incremental | p |
|---|---|---|---|---|---|
| 2014-01 .. 2016-07 | 2016-07 .. 2019-01 | ~9 yr before | 11,935 | **+0.101** | 0.005 |
| 2016-07 .. 2019-01 | 2019-01 .. 2021-07 | ~6 yr before | 13,366 | +0.049 | 0.005 |
| 2019-01 .. 2021-07 | 2021-07 .. 2024-01 | ~3 yr before | 16,471 | **+0.119** | 0.005 |
| 2021-07 .. 2024-01 | 2024-01 .. 2026-08 | contemporaneous | 19,306 | +0.068 | 0.005 |

**All four significant. No decay with distance from the filing — the pattern is
non-monotone and if anything runs the wrong way for the look-ahead story** (the
earliest window is the second highest; the contemporaneous one is near the bottom).

**So the F59 discount to ~+0.072 was too conservative.** That discount rested on
the within-B early/late gap (+0.072 / +0.088), which the full walk shows is regime
variation rather than distance-from-filing. Mean across the four disjoint splits is
**~+0.084**, every one of them out-of-time by construction.

**Magnitude is regime-dependent (+0.049 to +0.119) but the sign and significance
never are.** That is the profile of a real but time-varying relationship, not an
artifact — artifacts do not survive four disjoint windows at p=0.005 while varying
2.4x in size.

**Final standing of this signal:**

| property | status |
|---|---|
| effect | **~+0.084 mean across four disjoint out-of-time splits** |
| significance | p = 0.005 in every split (firm-label permutation, the correct dyadic null) |
| vs news text | +0.004 (F55) — **~20x** |
| vs same-sector dummy | control changes nothing (+0.090 -> +0.090) |
| look-ahead | refuted by walk-forward |
| single-name dependence | refuted by jackknife (top-5 = 12% of influence) |
| interpretability | pairs readable and economically correct |
| universe | 197 names, up to 19,306 pairs |

**This is the only result from the entire session that survives every check applied
to it.** Its limitation is not rigour but novelty: it substantially reproduces the
Hoberg-Phillips text-based-industry literature. The contribution is the comparison
— the news apparatus this project was built around is beaten ~20x by 10-K text that
takes an afternoon to fetch.

### F62. The +0.084 is real, robust, and ECONOMICALLY REDUNDANT
"What does the output mean, is there usable information in it" — the right
question, and the honest answer is no. Two tests convert the partial correlation
into something a user could act on. Both fail.

**(1) Does it improve a correlation FORECAST?** Firms split in half; the model
fitted on one half's pairs and tested on the other half's — out-of-sample by FIRM,
not just by time.

| forecast | RMSE |
|---|---|
| price history only | 0.1115 |
| **+ business-text similarity** | **0.1120** |
| improvement | **−0.45%** |

**Adding text makes the forecast slightly worse.** A partial correlation of +0.084
explains 0.084² ~ 0.7% of residual variance; net of estimation error on the
coefficient, that is not enough to help.

**(2) Does it flag usable pairs?** Conditioning on "price history says unrelated,
text says same business" — the diversification-trap screen, which is where such a
signal would earn its keep:

```
INFY ~ MAN   sim 0.11    Infosys ~ ManpowerGroup
EOG  ~ LMT   sim 0.13    shale E&P ~ defense prime
BIIB ~ KO    sim 0.14    biotech ~ Coca-Cola
GLD  ~ JPM   sim 0.13    gold ETF ~ a bank
```

Economically meaningless. And the apparent convergence is mostly mean reversion:
ALL low-correlation pairs converged +0.046, the text-flagged ones +0.076 — a
+0.030 edge on pairs that make no sense.

**THE MECHANISM, and it is the whole answer.** Where text similarity is high AND
meaningful, price history already knows: AMT~SBAC corrA **+0.64**, CVX~XOM **+0.63**,
BAC~JPM **+0.74**, AAL~DAL **+0.70**. Where price history is uninformative (low corrA),
the text flags are noise. **The signal is redundant precisely where it would need
to add value.**

HUT~WULF (bitcoin miners pivoting to AI, corrA +0.08 -> corrB +0.30) is the one
genuine catch — and one pair is an anecdote.

**Final verdict on the whole day.** The 10-K signal is statistically the most
robust thing in this project: ~+0.084 across four disjoint out-of-time regimes, all
p=0.005, look-ahead refuted, jackknife-robust, sector-independent, interpretable.
**And it is worth nothing**, because it re-describes relationships a correlation
matrix already contains. That is a more useful thing to know than another
representation variant, but it should be stated plainly rather than dressed as a
result.
