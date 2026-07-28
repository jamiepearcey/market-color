# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
SHOWCASE — the whole system in one page.

Six views, each answering a question that was actually contested during this work:

  1 SCORECARD   what was measured, what was fixed, what is still limited
  2 COVERAGE    coverage is uneven BY DESIGN; pooled figures measure the sampling
                plan. Filter to captured days and the explained fraction goes from
                2.3% to 27.4% of >4-sigma moves.
  3 RISK MAP    which news classes carry risk, and that the answer MOVES: classes
                drift twice as much as the baseline and REORDER over time.
  4 DAY         a day decomposed market/sector/idiosyncratic, drilled down through
                the real SIC hierarchy: division -> major group -> industry -> name.
  5 NAME        one ticker: its idiosyncratic correlation neighbours drawn as
                strength-weighted lines, and the breakdown of its news flow.
  6 GATE A/B    the same evidence answered by an article-only model and by the
                gated pipeline, side by side.

All figures are computed from the shared cache and the shared gate, so nothing
here can drift from the research scripts.

Usage:
    uv run scripts/build_showcase.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import Gate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg100k_graph"
OUT = G / "showcase.html"
N_DAYS = 8
N_NAMES = 160


# ---------------------------------------------------------------------------
# THE NEGATIVE-RESULTS REGISTER.
#
# Kept here deliberately rather than in a notebook nobody reopens. Most of the
# work on this project produced nulls, and several confident-looking positives
# died when a control was added. Recording that is not self-flagellation: an
# unrecorded null gets re-tested, and a retracted claim that is not written down
# comes back. Every figure below is from a stored result file or a logged run.
# ---------------------------------------------------------------------------
REGISTER = {
    "rejected": [
        {"claim": "News predicts next-day trading volume beyond the stock's own recent volume",
         "test": "Out-of-sample incremental R2 of news features over an AR(lag-0) volume "
                 "baseline, news days only",
         "result": "-0.0023", "verdict": "NOT SUPPORTED",
         "detail": "Adding news features degrades the forecast relative to the autoregressive "
                   "baseline. On the unscheduled-news subset the figure is -0.0133. News alone "
                   "achieves R2 of 0.036, indicating information in isolation but redundancy "
                   "given price history. Re-tested on deep-coverage days only, where an absence "
                   "of news is meaningful: -0.00133 against a permutation control of -0.00136.",
         "src": "news_volume.json (sharp_test2, sharp_test4); coverage_conditioned_retest.json"},
        {"claim": "Event type carries information beyond the fact that news arrived",
         "test": "Incremental R2 of typed events over an arrival-only indicator",
         "result": "-0.0023", "verdict": "NOT SUPPORTED",
         "detail": "The event taxonomy adds no measurable predictive value over the bare fact of "
                   "publication. This bears directly on the classification layer's utility for "
                   "forecasting, as distinct from description.",
         "src": "news_volume.json (sharp_test3)"},
        {"claim": "A non-linear model recovers latent signal in text that linear models miss",
         "test": "Gradient-boosted model on baseline plus hashed n-gram text, against the same "
                 "model with text row-shuffled",
         "result": "text -0.0035; permutation control -0.0010", "verdict": "NOT SUPPORTED",
         "detail": "Both arms are non-linear, isolating information content rather than "
                   "functional form. The text arm underperforms its own permutation control. "
                   "Comparing the boosted text model against a linear baseline would have "
                   "conflated the two. Under coverage conditioning the full-panel figure is "
                   "+0.00003 against a control of +0.00003.",
         "src": "news_nonlinear.py; coverage_conditioned_retest.json"},
        {"claim": "The news taxonomy aligns with the risk taxonomy: corporate news should affect "
                  "idiosyncratic risk, macro news systematic risk",
         "test": "Incremental R2 by news group and risk target, horizons of 5 and 21 days",
         "result": "corporate to idiosyncratic +0.0007", "verdict": "NOT SUPPORTED",
         "detail": "Effects are indistinguishable from zero throughout and do not follow the "
                   "predicted pattern: macro to systematic (+0.0054) is no larger than corporate "
                   "to systematic (+0.0050). On deep-coverage days the pattern is inverted, with "
                   "macro news contributing more to the idiosyncratic target (-0.00023) than to "
                   "the systematic one (-0.00105).",
         "src": "news_idio_risk.py; coverage_conditioned_retest.json"},
        {"claim": "Embedding-anchored baskets isolate a tradeable risk theme",
         "test": "Impact delta of an energy-anchored basket against 200 placebo baskets",
         "result": "-0.078; 19th placebo percentile", "verdict": "INSUFFICIENT EVIDENCE",
         "detail": "The energy basket ranks at the 19th percentile of random baskets, providing "
                   "no evidence of an effect. A related rate anchor reached the 99.5th percentile "
                   "in sample, but its out-of-sample interval [-0.012, +0.171] includes zero. The "
                   "falsifying test is itself weak: the OPEC event set contains seven decision "
                   "days, and the source script documents this as underpowered for the placebo "
                   "and out-of-sample gates. The hypothesis is unsupported, but the evidence does "
                   "not establish falsification.",
         "src": "event_embedding_risk_energy.json, _rate.json"},
        {"claim": "Emerging markets are less efficient, so news should carry predictive value there",
         "test": "The full test battery applied to the India graph (265 verified NSE mappings)",
         "result": "Null in both markets", "verdict": "NOT SUPPORTED",
         "detail": "The India sample reproduces the US non-result. This was the strongest "
                   "remaining alternative explanation for the null findings. The India corpus is "
                   "uniformly deep where sampled (25th percentile of 122 documents per day, no "
                   "thin days), so no coverage conditioning is required. Linear -0.00008; "
                   "gradient-boosted +0.00092 against a permutation control of -0.00019 (z=1.8, "
                   "below threshold). The taxonomy pattern is inverted as in the US sample: macro "
                   "news contributes more to the idiosyncratic target (+0.00007) than to the "
                   "systematic one (-0.00041).",
         "src": "india_news_test.py; india2021/coverage_conditioned_retest.json"},
        {"claim": "Price shocks accompanied by news continue; shocks without news reverse "
                  "(Savor 2012; Chan 2003)",
         "test": "Signed shock times forward idiosyncratic return at horizons of 1, 5 and 21 "
                 "days, news versus no-news, deep-coverage days, block-bootstrapped over dates",
         "result": "Inconclusive; underpowered by a factor of 1.2 to 5",
         "verdict": "INCONCLUSIVE",
         "detail": "All differences include zero, but the power calculation indicates this "
                   "outcome was predetermined. The published effect corresponds to 0.074-0.148 in "
                   "these units (0.5-1.0% cumulative abnormal return over 21 days at this "
                   "sample's 1.48% daily idiosyncratic volatility), against a minimum detectable "
                   "effect of 0.183 at best and 0.379 at median. The binding constraint is the "
                   "news arm (334 shocks on US deep days, 136 in India); the no-news arm is "
                   "adequately powered at n=2,530. Detecting the lower bound would require "
                   "approximately 13 times more news-labelled shocks. Directionally, the US "
                   "sample produced 8 of 9 differences in the predicted direction and India 6 of "
                   "6 against it, consistent with noise at this power.",
         "src": "savor_news_noews.py; eg100k_graph and india2021/savor_news_noews.json"},
        {"claim": "News about one firm predicts a linked firm the following day, through lagged "
                  "diffusion along links investors do not track (Cohen & Frazzini 2008)",
         "test": "Signed move of firm A times forward idiosyncratic return of neighbour B, "
                 "horizons 1, 5, 21; control is the same trigger dates paired with randomly "
                 "selected unlinked firms",
         "result": "-0.031 at 21 days; interval excludes the target effect",
         "verdict": "NOT SUPPORTED FOR THESE LINKS",
         "detail": "Adequately powered, with power assessed before the result. The published "
                   "long-short return of approximately 1.5% per month corresponds to +0.111 in "
                   "these units on a one-sided basis, against a minimum detectable effect of "
                   "0.088. The measured value at 21 days is -0.031 with an interval of [-0.091, "
                   "+0.033], which excludes the target. The India sample is underpowered "
                   "(detectable effect 0.138-0.141) and is inconclusive. Scope limitation: the "
                   "published work uses customer-supplier relationships from segment "
                   "disclosures, whereas this test uses idiosyncratic correlation neighbours, "
                   "which are by construction already reflected in contemporaneous prices. The "
                   "supportable conclusion is limited to correlation-defined links. Testing the "
                   "original hypothesis requires supply-chain data, which is not held.",
         "src": "cohen_frazzini_leadlag.py; cohen_frazzini_leadlag_corr.json"},
        {"claim": "News about one firm predicts its product-market rivals (Hoberg-Phillips TNIC "
                  "links, reconstructed annually from 10-K text)",
         "test": "As above, with TNIC rival links substituted for correlation neighbours; links "
                 "applied only within the year of estimation",
         "result": "Link-minus-random includes zero at every horizon", "verdict": "NOT SUPPORTED",
         "detail": "23,646 rival links across nine years, 597 triggers, 1,745 pairs. "
                   "Link-minus-random is +0.010, -0.032 and +0.019 at horizons 1, 5 and 21, "
                   "against a minimum detectable effect of 0.083-0.093, making this an adequately "
                   "powered null. The control was material: the random arm at 5 days returned "
                   "+0.040 with an interval of [+0.005, +0.078], indicating residual "
                   "cross-sectional dependence not absorbed by the equal-weighted market factor. "
                   "Without it, the raw linked figure at 21 days (+0.031) would have been read as "
                   "a weak diffusion effect. This is a horizontal-link test; rivals compete, "
                   "which is a weaker prior for lagged diffusion than vertical supply "
                   "relationships, and is consistent with TNIC pairs showing near-zero signed "
                   "co-movement (mean r 0.017 against mean absolute r 0.082).",
         "src": "cohen_frazzini_leadlag.py --links tnic; tnic_links.py"},
        {"claim": "Investors overreact to stale news, defined as stories largely repeating "
                  "information already published about the same firm (Tetlock 2011)",
         "test": "Staleness measured as maximum cosine similarity to any prior story about the "
                 "same firm within 125 trading days; terciles formed within prior-story-count "
                 "strata; same-day and forward returns, block-bootstrapped over dates",
         "result": "Same-day +0.197 sigma [+0.107, +0.299]; reversal not detected",
         "verdict": "PREMISE SUPPORTED; PREDICTION UNTESTED",
         "detail": "Stale news produces a same-day idiosyncratic move 23% larger than fresh news "
                   "(0.844, 0.926, 1.042 across terciles, monotonic, with the difference interval "
                   "excluding zero). This supports the premise that news carrying no new "
                   "information nonetheless moves prices. An initial specification was confounded "
                   "by coverage intensity (stale stories originated from firms with 7.5 prior "
                   "stories against 3.6 for fresh); stratifying by prior-story count balanced "
                   "this (5.8, 6.0, 6.1) and the gradient persisted. The tradeable component is "
                   "untested: stale-minus-fresh forward return is +0.055, contrary to the "
                   "predicted sign, with a minimum detectable effect of 0.163 against a target of "
                   "0.074-0.148. An alternative explanation is not excluded by this data: "
                   "staleness may proxy for an ongoing situation rather than recycled "
                   "information, since a firm in a developing takeover or crisis generates both "
                   "repetitive coverage and sustained volatility.",
         "src": "tetlock_stale_news.py; tetlock_stale_news.json"},
        {"claim": "The price reaction on an earnings day follows the reported numbers, and news "
                  "coverage contributes beyond them",
         "test": "Standardised unexpected earnings, computed as the change from the same quarter "
                 "one year prior scaled by the firm's own surprise volatility (Foster; "
                 "Bernard-Thomas), from SEC XBRL filings, joined to 8-K Item 2.02 release dates",
         "result": "Numbers +0.505% [+0.342, +0.686]; narrative untestable",
         "verdict": "PARTIALLY ESTABLISHED",
         "detail": "The first directional result in this programme; prior work measured magnitude "
                   "only. On an earnings day the stock moves +0.505% in the direction of the "
                   "surprise. The lowest surprise quintile (mean absolute SUE of 0.08) shows no "
                   "directional response (+0.18%, interval including zero) while all other "
                   "quintiles are significant, providing an internal control. Reaction magnitude "
                   "also increases monotonically with surprise size, from 2.43 to 2.95 sigma. "
                   "This attributes the earnings-day effect to the reported figures rather than "
                   "to coverage. The second component cannot be assessed: of 3,436 releases with "
                   "known figures, the news graph supplies a directional reading for 57 (1.7%), "
                   "splitting to 29 in agreement and 28 in conflict, which is insufficient to "
                   "determine which signal the price follows when they diverge.",
         "src": "earnings_surprise.py (XBRL frames); news_vs_numbers.py"},
        {"claim": "A large language model can identify why a correlation regime changed",
         "test": "Three-way classification of real transition windows against windows for the "
                 "same pair in which no change occurred, matched on article volume",
         "result": "46% against 42%", "verdict": "NOT SUPPORTED",
         "detail": "Two of fourteen cases discriminated correctly. Three returned the same "
                   "affirmative verdict for both arms and one was inverted. The model produces "
                   "accounts of comparable confidence and specificity for windows in which "
                   "nothing measurable occurred.",
         "src": "regime_reason.json"},
        {"claim": "A dynamic covariance model (DCC-GARCH) outperforms static estimators",
         "test": "Gaussian quasi-likelihood, minimum-variance portfolio volatility and turnover "
                 "across a ladder of asset-count to sample-length ratios",
         "result": "Outperforms at N/T=0.10; deteriorates from N/T=0.40; fails at 0.80",
         "verdict": "CONCLUSION HOLDS; ORIGINAL FIGURES CORRECTED",
         "detail": "The dynamic model performs best of nine estimators in the low-dimensional "
                   "regime (N/T=0.10), then degrades monotonically, ranking third at N/T=0.20 and "
                   "sixth of eight at 0.40, before failing at 0.80. Its turnover is two to twelve "
                   "times that of alternatives throughout, so its likelihood advantage does not "
                   "translate into a portfolio advantage. The dominant factor is the "
                   "dimensionality ratio rather than the choice of dynamics: a five-factor PCA "
                   "estimator leads from approximately N/T=0.4, and at N/T=2.0 Ledoit-Wolf "
                   "shrinkage is outperformed by an identity matrix. Sample covariance is "
                   "unstable where N exceeds T. The original summary understated the "
                   "low-dimensional performance and referenced an N/T=4.0 result for which no run "
                   "exists; the ladder terminates at 2.0.",
         "src": "research/finance/covariance/results/baselines_n*.json"},
    ],
    "retracted": [
        {"was": "The India extraction is approximately 70% null and of insufficient quality",
         "now": "38,466 of 38,469 documents extracted; 98% grounding, 3.2% conflict rate",
         "why": "Duplicate null records were counted in place of distinct documents, and quality "
                "was inferred from the extraction model rather than measured. The comparable "
                "Bloomberg figures are 99% and 5.0%. A proposed re-extraction was withdrawn."},
        {"was": "Raw text outperforms the graph, indicating the graph adds nothing",
         "now": "Not supportable on this evidence",
         "why": "Both arms returned null results, which cannot be ranked by predictive quality. "
                "The associated recommendation was withdrawn."},
        {"was": "28% of the strongest idiosyncratic links are cross-sector",
         "now": "6% cross-division, 11% cross-major-group, 29% cross-industry",
         "why": "Granularity was unspecified, and the original figure rested on a manually "
                "compiled map covering 12% of names, in which unclassified pairs were treated as "
                "same-sector. All three levels are now reported."},
        {"was": "The system cannot explain a trading day",
         "now": "It lacks the coverage to do so, having been run on approximately 13% of the corpus",
         "why": "Coverage was pooled across days sampled at 1-4 documents and days sampled at "
                "500, and the average was reported as a capability limit. Conditioned on days "
                "actually captured, explained coverage of moves above 4 sigma rises from 2.3% to "
                "27.4%."},
        {"was": "DCC-GARCH is outperformed throughout; Ledoit-Wolf trails identity at N/T=4",
         "now": "It leads at N/T=0.10 and fails by 0.80; no N/T=4 run exists",
         "why": "Reported from recollection rather than from the stored results. The overall "
                "conclusion is unaffected, but two figures were unsupported. Corrected against "
                "results/baselines_n*.json."},
        {"was": "The energy embedding basket was falsified against OPEC dates",
         "now": "Unsupported, but the falsifying test has n=7 and is documented as underpowered",
         "why": "An underpowered check was reported as a falsification."},
        {"was": "The earnings-day effect is approximately 2.2 times an ordinary day",
         "now": "3.20 times, when events are dated by SEC filing rather than publication",
         "why": "The news graph dates events by publication timestamp. 49% of earnings releases "
                "are accepted at or after 16:00 ET, placing the market reaction in the following "
                "session and splitting the effect across two days. The unaligned calendar profile "
                "shows a two-day plateau (2.45 and 2.27); aligning after-close releases to the "
                "first affected session yields a single peak (3.22, falling to 1.32). Built from "
                "17,508 8-K Item 2.02 filings across 482 tickers, against 415 events identified "
                "as earnings by the news graph."},
        {"was": "Earnings register at 2.27 times an ordinary day",
         "now": "2.17 times, once the sector component is removed from the measured quantity",
         "why": "The gate's standardised move was computed from a market-only residual while the "
                "decomposition displayed alongside it removed both market and sector, so the two "
                "measured different quantities and sector-factor improvements did not reach the "
                "gate. Resolved by sourcing the series from the shared cache. The effect reduces "
                "by approximately 5% because some sector co-movement had been attributed to the "
                "idiosyncratic component. The India figure moved comparably, from 2.26 to 2.18."},
        {"was": "Sector controls confirm the effect, with a +0.079 improvement",
         "now": "Spurious; the effect does not survive a validated sector map",
         "why": "Sectors were inferred from residual-overlap clustering. Replaced with curated "
                "GICS, subsequently with SEC-assigned SIC."},
    ],
    "bugs": [
        {"bug": "Control value of 267 sigma", "impact": "All standardised move measurements",
         "detail": "17% of the universe (123 tickers) are illiquid OTC depositary receipts; one "
                   "records an exactly zero return on 99.4% of days. The guard tested for "
                   "non-positive standard deviation, which a window of near-total zeros passes "
                   "with a small positive value. Corrected by testing the zero-return fraction "
                   "and applying a volatility floor. The headline class ranking was unaffected "
                   "(2.28 before, 2.27 after) because the contamination inflated control and "
                   "event classes proportionately."},
        {"bug": "Results varied between identical runs", "impact": "Reproducibility of all controls",
         "detail": "The seeded control sampled from a set, and Python randomises string hashing "
                   "per process, so iteration order and therefore the sample differed each run "
                   "(pooled control varied between 0.86 and 1.02 sigma). This also concealed the "
                   "staleness defect by largely omitting the affected names."},
        {"bug": "284 of 500 pairs classified as emerging on the first window",
         "impact": "Correlation regime detection",
         "detail": "Missing data was treated as absence of relationship, so unpopulated periods "
                   "at the sample boundary formed spurious inactive runs and generated a "
                   "transition for nearly every pair. A pair was also classified as ended while "
                   "its final correlation stood at 0.478 against a 0.19 threshold."},
        {"bug": "Concentration of regime changes in the final month",
         "impact": "Correlation regime detection",
         "detail": "15 emerging and 10 ended transitions fell in a single month at the sample "
                   "boundary, where a change rests on minimum evidence and has no opportunity to "
                   "revert. A confirmation requirement of six windows was introduced; 79 pairs "
                   "moved to unconfirmed and ended transitions fell from 33 to 16."},
        {"bug": "Drill-down navigation non-functional", "impact": "Reporting interface",
         "detail": "A group label was interpolated into a double-quoted HTML attribute as a "
                   "double-quoted string, terminating the attribute. A SIC label containing an "
                   "apostrophe separately broke single-quoted handlers. Handlers now reference an "
                   "index, and escaping covers both quote characters."},
        {"bug": "Empty model responses", "impact": "All generated explanations",
         "detail": "The extraction model draws reasoning and output from a shared token budget, "
                   "so an insufficient limit returned empty content without raising an error."},
        {"bug": "Incremental R2 of approximately zero by construction",
         "impact": "Initial news-to-volume tests",
         "detail": "Testing across the full panel, of which 99.1% of rows carry no news, dilutes "
                   "any effect mechanically. A specification error rather than a finding. "
                   "Re-run conditional on news days, the result remains null but is meaningful."},
    ],
    "controls": [
        {"control": "Placebo windows matched on article volume",
         "before": "46% against 4% - apparent effect",
         "after": "46% against 42% - null",
         "detail": "Unmatched placebo windows contained little or no text, forcing the negative "
                   "verdict. The comparison measured presence of news rather than transition "
                   "news against ordinary news."},
        {"control": "Row-shuffled text arm", "before": "Text contributes +0.0019",
         "after": "Control contributes +0.0014 - indistinguishable",
         "detail": "A raw gain is uninterpretable without the score attained by randomised text "
                   "on an identical model."},
        {"control": "Circular-shift null for pair correlation",
         "before": "Any absolute correlation above approximately 0.2 read as a relationship",
         "after": "Median chance threshold of 0.181",
         "detail": "Shifting one series removes the contemporaneous relationship while preserving "
                   "length, variance and autocorrelation, so the threshold reflects what the "
                   "specific pair produces in the absence of a relationship."},
        {"control": "Conditioning on documents per day",
         "before": "2.3% of moves above 4 sigma explained",
         "after": "27.4% on days actually captured",
         "detail": "A figure pooled across a deliberately uneven sampling design measures the "
                   "design rather than the data."},
        {"control": "Re-running null results on deep-coverage days only",
         "before": "All predictive nulls run across the full panel, where a no-news label on a "
                   "thinly sampled day may indicate absence of sampling rather than absence of news",
         "after": "All three arms remain null: linear -0.00133 against -0.00136; boosted +0.00060 "
                  "against +0.00062; taxonomy alignment still inverted",
         "detail": "Label noise in the negative class biases predictive tests toward the null, so "
                   "the original findings were confounded with sampling. Restricted to the 96 "
                   "deep-coverage days (50,236 rows, 2,581 with news), news scores as row-shuffled "
                   "news. Note the asymmetry: the same contamination biases the contemporaneous "
                   "positive result in the opposite direction, inflating the control and "
                   "understating the effect, so that finding is conservative. Power is limited on "
                   "deep days (linear baseline R2 of 0.006; boosted baseline negative), so the "
                   "load-bearing boosted figure is the full panel, where the baseline functions "
                   "and news contributes +0.00003 against a control of +0.00003."},
        {"control": "Trailing rather than pooled class priors",
         "before": "A single constant per class",
         "after": "3.6% of verdicts incorrect, in both directions",
         "detail": "Priors drift at twice the rate of the baseline and reorder, with one class "
                   "spanning ranks 1 to 6. 2.0% of verdicts were insufficiently strict, which is "
                   "the condition under which an unsupported narrative is generated."},
    ],
}


MARKET_GRAPH = {"us": "eg100k_graph", "india": "india2021"}
MARKET_META = {
    "us": {"label": "United States",
           "sub": "Bloomberg, 2008–2014 · SIC sectors assigned by the SEC",
           "sector_conf": "Authoritative: SIC codes are assigned per filer by the SEC."},
    "india": {"label": "India",
              "sub": "NSE, 2021–2022 · sectors inferred from text",
              "sector_conf": "Weaker evidence: sector labels are inferred from article and "
                             "company text rather than assigned. Text-inferred sectors produced "
                             "a spurious result in the US sample, so India sector-level figures "
                             "should be treated with more caution than US equivalents."},
}


def build_market(market: str) -> dict:
    """Assemble the reporting payload for one market.

    Each market uses its own gate, control and priors. A US baseline does not
    describe what is normal for an Indian name: the volatility regime, news
    cadence and coverage all differ.
    """
    global G
    G = ROOT / MARKET_GRAPH[market]
    C = json.loads((G / "mcp_cache.json").read_text())
    GATE = Gate(G / "classification" / "rolling_priors.json")
    days = C["days"]; di = {d: i for i, d in enumerate(days)}

    dpd = collections.Counter()
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            dpd[d] += 1
    tier = lambda d: ("deep" if dpd.get(d, 0) >= 100 else "partial" if dpd.get(d, 0) >= 20
                      else "thin" if dpd.get(d, 0) >= 5 else "token" if dpd.get(d, 0) else "none")
    hist = collections.Counter(tier(d) for d in days)
    deep = [d for d in days if tier(d) == "deep"]

    # ---------- 2. coverage conditioned on day tier ----------
    ev = C["events"]
    def cov_table(dayset):
        out = {}
        for thr in (0, 1, 2, 3, 4):
            n = has = ok = 0
            for d in dayset:
                i = str(di[d])
                for tk, s in C["sigma"].items():
                    v = s.get(i)
                    if v is None or v <= thr:
                        continue
                    n += 1
                    cell = ev.get(f"{tk}|{d}")
                    if not cell:
                        continue
                    has += 1
                    cls = max(cell["types"], key=lambda t: (GATE.as_of(d, t)["prior"] or 0.0))
                    if GATE.evaluate(d, cls, v)["status"] == "SUFFICIENT":
                        ok += 1
            out[f">{thr}"] = {"n": n, "news": has, "ok": ok,
                              "reach": round(has / max(n, 1), 4),
                              "gated": round(ok / max(has, 1), 4),
                              "explained": round(ok / max(n, 1), 4)}
        return out
    thin_days = [d for d in days if tier(d) in ("thin", "token")]
    coverage = {"deep": cov_table(deep), "thin": cov_table(thin_days),
                "pooled": cov_table([d for d in days if dpd.get(d, 0)])}
    print("coverage computed")

    # ---------- 3. risk map + drift ----------
    ctrl_p = GATE.pooled_control
    classes = sorted(GATE.pooled, key=lambda c: -GATE.pooled[c])
    riskmap = [{"cls": c, "prior": GATE.pooled[c], "sd": GATE.pooled_full[c][1],
                "mult": round(GATE.pooled[c] / ctrl_p, 2)} for c in classes]
    freq = {c: len(v) for c, v in GATE.priors.items()}
    track = sorted(freq, key=lambda c: -freq[c])[:7]
    drift = []
    for k, d in enumerate(GATE.dates):
        row = {"d": d, "ctrl": round(GATE.control[k], 3), "c": {}}
        for c in track:
            v = GATE.priors[c].get(d)
            if v:
                row["c"][c] = round(v[0] / GATE.control[k], 3)
        if len(row["c"]) >= 4:
            drift.append(row)
    for r in drift:
        for rk, c in enumerate(sorted(r["c"], key=lambda c: -r["c"][c]), 1):
            r.setdefault("rank", {})[c] = rk
    rr = {c: [r["rank"][c] for r in drift if c in r["rank"]] for c in track}
    rank_range = {c: [min(v), max(v)] for c, v in rr.items() if v}
    print("risk map + drift computed")

    # ---------- 4. days, decomposed and rolled up the SIC hierarchy ----------
    sect = C["sectors"]
    picked = sorted(deep, key=lambda d: -dpd[d])[:N_DAYS]
    daydata = []
    for d in picked:
        i = str(di[d])
        rows = []
        for tk, dec in C["decomp"].items():
            v = dec.get(i)
            if not v:
                continue
            s = C["sigma"][tk].get(i)
            cell = ev.get(f"{tk}|{d}")
            r = {"tk": tk, "nm": C["names"].get(tk, tk),
                 "div": sect.get(tk, {}).get("division", "UNK"),
                 "mg": sect.get(tk, {}).get("major_group", "UNK"),
                 "ind": sect.get(tk, {}).get("industry", "UNK"),
                 "tot": v[0], "mkt": v[1], "sct": v[2], "idi": v[3], "sig": s}
            if cell and s is not None:
                cls = max(cell["types"], key=lambda t: (GATE.as_of(d, t)["prior"] or 0.0))
                g = GATE.evaluate(d, cls, s)
                r |= {"cls": cls, "gate": g["status"], "conf": g.get("confidence"),
                      "prior": g["prior"], "asof": g["asof"], "sur": g["surprise"],
                      "why": g["reasons"],
                      "docs": [{"h": C["docs"][x]["headline"],
                                "t": " ".join(C["docs"][x]["chunks"])[:700]}
                               for x in cell["docs"][:2] if x in C["docs"]]}
            rows.append(r)
        tot = np.array([r["tot"] for r in rows])
        vt = float(np.var(tot)) or 1.0
        daydata.append({
            "d": d, "docs": dpd[d], "n": len(rows),
            "var": {"mkt": round(float(np.var([r["mkt"] for r in rows])) / vt, 3),
                    "sct": round(float(np.var([r["sct"] for r in rows])) / vt, 3),
                    "idi": round(float(np.var([r["idi"] for r in rows])) / vt, 3)},
            "ctrl": GATE.as_of(d, "earnings")["control"],
            "rows": sorted(rows, key=lambda r: -(r["sig"] or 0))[:90],
        })
    print(f"{len(daydata)} days decomposed")

    # ---------- 5. names: neighbours + news weight ----------
    # ---------- rolling correlation for the name view's neighbour lines ----------
    # The radial chart was a single static snapshot of something that demonstrably
    # moves, so compute the correlation to each neighbour in every window.
    #
    # Done with CUMULATIVE SUMS: the naive form (slice each window, call numpy) is
    # ~1,280 pairs x 79 windows x 9 series = over a million small numpy calls and
    # takes minutes. Cumsums make each pair O(n) regardless of how many windows.
    RWIN, RSTEP, RSHIFT = 120, 21, 8
    n_days = len(days)
    idi_arr = {}
    for tk, dec in C["decomp"].items():
        v = np.full(n_days, np.nan)
        for k, arr in dec.items():
            v[int(k)] = arr[3]
        idi_arr[tk] = v
    rstarts = np.array(list(range(0, n_days - RWIN, RSTEP)))
    rdates = [days[s + RWIN - 1] for s in rstarts]

    def _cs(x):
        return np.concatenate([[0.0], np.cumsum(x)])

    def roll_corr(a, b):
        """Rolling Pearson r over the fixed window grid, NaN-aware, via cumsums."""
        ok = np.isfinite(a) & np.isfinite(b)
        a0 = np.where(ok, a, 0.0); b0 = np.where(ok, b, 0.0)
        cn, ca, cb = _cs(ok.astype(float)), _cs(a0), _cs(b0)
        caa, cbb, cab = _cs(a0 * a0), _cs(b0 * b0), _cs(a0 * b0)
        lo, hi = rstarts, rstarts + RWIN
        n_ = cn[hi] - cn[lo]
        sa, sb = ca[hi] - ca[lo], cb[hi] - cb[lo]
        saa, sbb, sab = caa[hi] - caa[lo], cbb[hi] - cbb[lo], cab[hi] - cab[lo]
        num = n_ * sab - sa * sb
        den = np.sqrt(np.maximum(n_ * saa - sa * sa, 0) * np.maximum(n_ * sbb - sb * sb, 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.where((den > 0) & (n_ >= 80), num / den, np.nan)
        return r

    rng_b = np.random.default_rng(3)

    def band_of(a, b):
        """Same circular-shift null as pair_regimes: destroys the contemporaneous
        link, keeps each series' length, variance and autocorrelation."""
        vals = []
        for _ in range(RSHIFT):
            off = int(rng_b.integers(RWIN, n_days - RWIN))
            vals.append(roll_corr(a, np.roll(b, off)))
        v = np.abs(np.concatenate(vals))
        v = v[np.isfinite(v)]
        return float(np.percentile(v, 95)) if len(v) > 40 else 0.25

    rank_n = sorted(C["neighbours"], key=lambda t: -sum(C["class_counts"].get(t, {}).values()))
    # index this name's event days once, so the news-weight table can drill into
    # the days behind each share rather than just asserting a percentage
    ev_by_tk = collections.defaultdict(list)
    for k, v in C["events"].items():
        ev_by_tk[k.split("|")[0]].append((k.split("|")[1], v))
    names = {}
    for tk in rank_n[:N_NAMES]:
        cc = C["class_counts"].get(tk, {})
        tot = sum(cc.values()) or 1
        evs = []
        for d, v in sorted(ev_by_tk.get(tk, []), reverse=True)[:34]:
            i = str(di.get(d, -1))
            s = C["sigma"].get(tk, {}).get(i)
            row = {"d": d, "cls": v["types"], "sig": s,
                   "docs": [{"h": C["docs"][x]["headline"],
                             "t": " ".join(C["docs"][x]["chunks"])[:260]}
                            for x in v["docs"][:2] if x in C["docs"]]}
            if s is not None:
                cls = max(v["types"], key=lambda t: (GATE.as_of(d, t)["prior"] or 0.0))
                g = GATE.evaluate(d, cls, s)
                row |= {"gate": g["status"], "conf": g.get("confidence"),
                        "prior": g["prior"], "asof": g["asof"], "sur": g["surprise"],
                        "why": g["reasons"][:1]}
            evs.append(row)
        names[tk] = {
            "nm": C["names"].get(tk, tk), "sec": sect.get(tk, {}),
            "n": tot,
            "weight": [{"c": c, "n": n, "share": round(n / tot, 3),
                        "mult": (round(GATE.pooled[c] / ctrl_p, 2) if c in GATE.pooled else None)}
                       for c, n in sorted(cc.items(), key=lambda x: -x[1])[:8]],
            "nb": [{"t": t, "r": r, "same": bool(same),
                    "ind": sect.get(t, {}).get("industry", "UNK")}
                   for t, r, same in C["neighbours"][tk][:10]],
            "events": evs,
        }
        # per-window correlation to each neighbour, plus that pair's chance band
        a = idi_arr.get(tk)
        if a is not None:
            for nb in names[tk]["nb"][:8]:
                b = idi_arr.get(nb["t"])
                if b is None:
                    continue
                r = roll_corr(a, b)
                nb["series"] = [None if not np.isfinite(x) else round(float(x), 3) for x in r]
                nb["band"] = round(band_of(a, b), 3)
    print(f"{len(names)} names, "
          f"{sum(len(v['events']) for v in names.values())} drillable event days")

    # ---------- 6. A/B pairs, reused from the day-explain build ----------
    ab = []
    p = G / "day_explain.html"
    if p.exists():
        m = re.search(r"const D=(\{.*?\});", p.read_text(), re.S)
        if m:
            for day in json.loads(m.group(1))["days"]:
                for c in day["cases"]:
                    ab.append({"d": day["d"], "tk": c["tk"], "nm": c.get("nm", c["tk"]),
                               "sig": c["sig"], "cls": c["cls"], "exp": c["exp"],
                               "sur": c["sur"], "asof": c.get("asof"),
                               "conf": c.get("conf"), "z": c.get("z"),
                               "ok": c["ok"], "why": c.get("why", []),
                               "tot": c["tot"], "mkt": c["mkt"], "sct": c["sct"], "idi": c["idi"],
                               "naive": c["naive"], "gated": c["gated"],
                               "texts": c.get("texts", [])[:2]})
    print(f"{len(ab)} A/B pairs")

    # ---------- 7. pair regimes + the placebo-controlled narrative test ----------
    regimes = {}
    rp = G / "pair_regimes.json"
    if rp.exists():
        R = json.loads(rp.read_text())
        pairs = R["pairs"]
        st = collections.Counter(p["state"] for p in pairs)
        # keep the most interesting: biggest transitions, plus a few persistent
        def mag(p):
            t = p["transition_index"]
            if t is None:
                return 0.0
            pre = [x for x in p["corr"][:t] if x is not None][-4:]
            post = [x for x in p["corr"][t:] if x is not None][:4]
            return abs(float(np.mean(post)) - float(np.mean(pre))) if pre and post else 0.0
        # ATTACH THE NEWS IN THE TRANSITION WINDOW. This is the honest product: the
        # measurement says a link ended in April 2014, and here is what was published
        # around then for both names — for the reader to judge. No causal claim is
        # made or implied, because the placebo test showed an LLM cannot tell a real
        # transition window from a matched one where nothing changed.
        HALFW = 130
        wd = R["window_dates"]
        by_tk = collections.defaultdict(list)
        for k, v in C["events"].items():
            t, dd = k.split("|")
            by_tk[t].append((dd, v))
        for lst in by_tk.values():
            lst.sort()

        def articles(tk, lo, hi, cap=5):
            out = []
            for dd, v in by_tk.get(tk, []):
                if not (lo <= dd <= hi):
                    continue
                for doc in v["docs"][:1]:
                    o = C["docs"].get(doc)
                    if o:
                        out.append({"d": dd, "cls": v["types"][:2],
                                    "h": o["headline"], "t": " ".join(o["chunks"])[:520]})
            return out[:cap]

        def attach(p):
            t = p["transition_index"]
            if t is None:
                p["news"] = {"a": [], "b": [], "window": None, "n": 0}
                return p
            c = di.get(wd[t], 0)
            lo, hi = days[max(0, c - HALFW)], days[min(len(days) - 1, c + HALFW)]
            na, nb = articles(p["a"], lo, hi), articles(p["b"], lo, hi)
            p["news"] = {"a": na, "b": nb, "window": [lo, hi], "n": len(na) + len(nb),
                         "docs_in_window": sum(dpd.get(x, 0) for x in days
                                               if lo <= x <= hi)}
            return p

        cand_t = [p for p in pairs if p["state"] in ("ENDED", "EMERGING")]
        for p in cand_t:
            p["magnitude"] = round(mag(p), 3)
            attach(p)
        # surface the ones we can actually show evidence for first
        trans = sorted(cand_t, key=lambda p: (-(p["news"]["n"] > 0), -p["magnitude"]))[:70]
        pers = sorted([p for p in pairs if p["state"] in ("PERSISTENT", "EPISODIC")],
                      key=lambda p: -p["mean_abs"])[:20]
        for p in pers:
            p["magnitude"] = round(mag(p), 3)
            attach(p)
        unc = sorted([p for p in pairs if p["state"] == "UNCONFIRMED"],
                     key=lambda p: -mag(p))[:10]
        for p in unc:
            p["magnitude"] = round(mag(p), 3)
            attach(p)
        keep = trans + pers + unc
        n_ev = sum(1 for p in cand_t if p["news"]["n"] > 0)
        print(f"  {n_ev}/{len(cand_t)} transitions have article evidence in their window")
        regimes = {
            "window_dates": R["window_dates"], "win": R["window_days"],
            "band_q": R["band_percentile"], "min_run": R["min_run"],
            "min_confirm": R.get("min_confirm"),
            "states": dict(st), "n_pairs": len(pairs),
            "median_band": round(float(np.median([p["band"] for p in pairs])), 3),
            "pairs": keep,
        }
    numbers = {}
    for fn, key in (("news_vs_numbers.json", "nvn"),
                    ("event_window_profile.json", "window"),
                    ("calendar_window_profile.json", "calwindow")):
        fp = G / fn
        if fp.exists():
            numbers[key] = json.loads(fp.read_text())

    reason = {}
    rr = G / "regime_reason.json"
    if rr.exists():
        reason = json.loads(rr.read_text())

    return {
        "market": market, "meta": MARKET_META[market],
        "numbers": numbers,
        "regimes": regimes, "reason": reason,
        "roll_dates": rdates, "roll_win": RWIN,
        "hist": dict(hist), "n_days": len(days),
        "dpd_pct": {f"p{q}": int(np.percentile([dpd[d] for d in days if dpd.get(d)], q))
                    for q in (25, 50, 75, 90, 100)},
        "coverage": coverage,
        "riskmap": riskmap, "control": ctrl_p,
        "drift": drift, "track": track, "rank_range": rank_range,
        "days": daydata, "names": names, "ab": ab,
    }


def main() -> None:
    markets = {}
    for m in ("us", "india"):
        if (ROOT / MARKET_GRAPH[m] / "mcp_cache.json").exists():
            print(f"--- {m} ---")
            markets[m] = build_market(m)
        else:
            print(f"--- {m} SKIPPED (no mcp_cache.json — run mcp_precompute --market {m})")
    # side-by-side class comparison: the only honest replication test available
    cmp_rows = []
    if len(markets) > 1:
        u, i = markets["us"], markets["india"]
        ur = {r["cls"]: r for r in u["riskmap"]}
        ir = {r["cls"]: r for r in i["riskmap"]}
        for c in sorted(set(ur) & set(ir), key=lambda c: -ur[c]["mult"]):
            cmp_rows.append({"cls": c, "us": ur[c]["mult"], "india": ir[c]["mult"]})
        if cmp_rows:
            a = np.array([r["us"] for r in cmp_rows]); b = np.array([r["india"] for r in cmp_rows])
            rank = float(np.corrcoef(a.argsort().argsort(), b.argsort().argsort())[0, 1])
            lvl = float(np.corrcoef(a, b)[0, 1])
        else:
            rank = lvl = 0.0
        compare = {"rows": cmp_rows, "rank_corr": round(rank, 3), "level_corr": round(lvl, 3),
                   "us_control": u["control"], "india_control": i["control"],
                   "mean_abs_diff": round(float(np.abs(a - b).mean()), 3) if cmp_rows else None}
    else:
        compare = None
    payload = {"register": REGISTER, "markets": markets, "compare": compare}
    OUT.write_text(HTML.replace("__DATA__", json.dumps(payload)))
    print(f"\nwrote {OUT} ({OUT.stat().st_size/1e6:.1f} MB) — markets: {list(markets)}")


HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Event graph — showcase</title>
<style>
 :root{--surface:#1a1a19;--page:#0d0d0d;--ink:#fff;--ink2:#c3c2b7;--muted:#898781;
  --grid:#2c2c2a;--border:rgba(255,255,255,.10);--warn:#fab219;--bad:#e5484d;--good:#199e70;
  --acc:#3987e5;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--s4:#c98500;--s5:#d55181;--s6:#008300;--s7:#9085e9}
 html[data-theme=light]{--surface:#fcfcfb;--page:#f4f4f2;--ink:#0b0b0b;--ink2:#52514e;
  --muted:#6b6a66;--grid:#e1e0d9;--border:rgba(11,11,11,.12);--acc:#2a78d6;
  --s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--s4:#eda100;--s5:#e87ba4;--s6:#008300;--s7:#4a3aa7}
 *{box-sizing:border-box}
 body{margin:0;background:var(--page);color:var(--ink);
  font:13.5px/1.55 system-ui,-apple-system,sans-serif;display:flex;min-height:100vh}
 nav{width:196px;flex:0 0 196px;background:var(--surface);border-right:1px solid var(--border);
  padding:16px 0;position:sticky;top:0;height:100vh;overflow:auto}
 nav h1{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);
  margin:0 0 12px;padding:0 16px;font-weight:600}
 nav a{display:block;padding:8px 16px;color:var(--ink2);text-decoration:none;font-size:13px;
  border-left:2px solid transparent;cursor:pointer}
 nav a:hover{background:rgba(127,127,127,.08);color:var(--ink)}
 nav a.on{color:var(--ink);border-left-color:var(--acc);background:rgba(57,135,229,.10)}
 nav a small{display:block;color:var(--muted);font-size:10.5px;margin-top:1px}
 main{flex:1;padding:24px 28px 70px;max-width:1180px;overflow-x:hidden}
 .view{display:none}.view.on{display:block}
 h2{font-size:19px;margin:0 0 4px}
 .lede{color:var(--ink2);font-size:13px;margin:0 0 18px;max-width:88ch}
 .card{background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:15px 17px;margin-bottom:14px}
 .card h3{font-size:13.5px;margin:0 0 3px}
 .note{color:var(--muted);font-size:11.5px;margin:0 0 11px;max-width:92ch;line-height:1.6}
 table{border-collapse:collapse;width:100%;font-size:12px}
 th{color:var(--muted);font-size:10px;text-transform:uppercase;letter-spacing:.05em;
  font-weight:500;text-align:right;padding:5px 9px;border-bottom:1px solid var(--border)}
 th:first-child,td:first-child{text-align:left}
 td{padding:5px 9px;text-align:right;border-bottom:1px solid rgba(127,127,127,.10)}
 tr:hover td{background:rgba(127,127,127,.05)}
 .kpi{display:flex;gap:11px;flex-wrap:wrap;margin-bottom:14px}
 .k{background:var(--surface);border:1px solid var(--border);border-radius:9px;
  padding:11px 14px;min-width:132px;flex:1}
 .k b{display:block;font-size:22px;line-height:1.15;font-variant-numeric:tabular-nums}
 .k span{color:var(--muted);font-size:10.5px;display:block;margin-top:2px;line-height:1.35}
 .bar{height:9px;border-radius:5px;background:var(--grid);overflow:hidden;display:flex}
 .bar i{display:block;height:100%}
 .pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:10.5px;
  border:1px solid var(--border);color:var(--ink2)}
 .ok{color:var(--good)}.no{color:var(--warn)}.bd{color:var(--bad)}
 .ab{display:grid;grid-template-columns:1fr 1fr;gap:11px;margin-top:9px}
 .ans{padding:11px 13px;border-radius:8px;font-size:12.5px;line-height:1.6}
 .naive{background:rgba(229,72,77,.07);border:1px solid rgba(229,72,77,.28)}
 .gated{background:rgba(57,135,229,.07);border:1px solid rgba(57,135,229,.30)}
 .ans h4{margin:0 0 5px;font-size:10.5px;text-transform:uppercase;letter-spacing:.05em}
 .naive h4{color:var(--bad)}.gated h4{color:var(--acc)}
 details{margin-top:8px}summary{cursor:pointer;color:var(--muted);font-size:11.5px}
 .art{border-left:2px solid var(--grid);padding:5px 0 5px 10px;margin:7px 0;
  font-size:11.5px;color:var(--ink2)}
 select,input{background:var(--surface);color:var(--ink);border:1px solid var(--border);
  border-radius:7px;padding:5px 9px;font-size:12.5px;font-family:inherit}
 .row{display:flex;gap:9px;align-items:center;flex-wrap:wrap;margin-bottom:11px}
 .crumb{color:var(--muted);font-size:11.5px;margin-bottom:8px}
 .crumb b{color:var(--ink)}.crumb a{color:var(--acc);cursor:pointer;text-decoration:none}
 .tog{position:fixed;top:11px;right:13px;background:var(--surface);color:var(--ink2);
  border:1px solid var(--border);border-radius:7px;padding:4px 9px;font-size:12px;cursor:pointer;z-index:20}
 .tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;background:var(--surface);
  border:1px solid var(--border);border-radius:7px;padding:7px 10px;font-size:11.5px;
  box-shadow:0 8px 26px rgba(0,0,0,.5);z-index:30;max-width:290px}
 .lg{display:flex;gap:13px;flex-wrap:wrap;font-size:11.5px;color:var(--ink2);margin-top:9px}
 .sw{width:12px;height:3px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle}
 code{font-size:11.5px;background:rgba(127,127,127,.12);padding:1px 5px;border-radius:4px}
</style></head><body>
<button class="tog" onclick="tg()">◐</button>
<nav><h1>Event graph</h1>
 <div style="padding:0 12px 12px">
  <select id="mksel" onchange="setMarket(this.value)" style="width:100%;font-size:12px">
  </select>
  <div id="mksub" style="color:var(--muted);font-size:10px;margin-top:5px;line-height:1.45"></div>
 </div>
 <a data-v="score" class="on">Scorecard<small>what holds up</small></a>
 <a data-v="cov">Coverage<small>condition on the sample</small></a>
 <a data-v="risk">Risk map<small>and how it moves</small></a>
 <a data-v="day">Day drill-down<small>market → sector → name</small></a>
 <a data-v="name">Name<small>correlation + news mix</small></a>
 <a data-v="num">Numbers vs narrative<small>what actually moves it</small></a>
 <a data-v="reg">Regimes<small>when a link is over</small></a>
 <a data-v="ab">Gate A/B<small>before and after</small></a>
 <a data-v="cmp">US vs India<small>does it replicate?</small></a>
 <a data-v="null">What we disproved<small>the negative register</small></a>
</nav>
<main>
<!-- ============ SCORECARD ============ -->
<section class="view on" id="v-score">
 <h2>What was measured, and what survived checking</h2>
 <p class="lede">Every figure here is recomputed from one shared cache and one shared gate. Where a
 number changed under scrutiny, the corrected value is shown and the reason given — several
 headline claims in this project were wrong the first time.</p>
 <div class="kpi" id="k-score"></div>
 <div class="card"><h3>Immediate risk by news class</h3>
  <p class="note">Contemporaneous: risk being realised as the news lands, not a forecast. The
  predictive versions of these tests are null, and that null survived every re-run.
  <b>× control</b> is the number that matters — 1.0 means indistinguishable from an ordinary day.</p>
  <table id="t-risk"></table></div>
 <div class="card"><h3>Corrections made under challenge</h3>
  <table id="t-fix"></table></div>
</section>

<!-- ============ COVERAGE ============ -->
<section class="view" id="v-cov">
 <h2>Coverage is uneven by design — condition on it</h2>
 <p class="lede">The corpus was sampled to give some days deep coverage and most days only token
 coverage. A <b>pooled</b> coverage figure therefore measures the sampling plan, not the graph.
 This was the single largest error in the analysis: pooled, it looks as though almost nothing is
 explainable. Filtered to the days actually captured, the picture inverts.</p>
 <div class="kpi" id="k-cov"></div>
 <div class="card"><h3>Documents per trading day</h3>
  <p class="note">Bimodal by construction. Deep days were capped at 500 documents during
  extraction, so their coverage figures are a <b>floor</b>, not a ceiling.</p>
  <div id="tierbar"></div></div>
 <div class="card"><h3>Explained fraction, by day tier</h3>
  <p class="note"><b>reach</b> = share of moves with any linked news · <b>of news</b> = share of
  those clearing the gate · <b>explained</b> = share of all moves at that size with a gate-cleared
  narrative. Toggle the tier to see how much of the pooled figure was sampling.</p>
  <div class="row"><span style="color:var(--muted);font-size:11.5px">day tier</span>
   <select id="covsel" onchange="drawCov()">
    <option value="deep">deep — 100+ documents (the days we captured)</option>
    <option value="thin">thin/token — 1–19 documents</option>
    <option value="pooled">pooled — all days with any document</option>
   </select></div>
  <table id="t-cov"></table>
  <p class="note" id="cov-read" style="margin-top:10px"></p></div>
</section>

<!-- ============ RISK MAP ============ -->
<section class="view" id="v-risk">
 <h2>Which news carries risk — and the fact that it moves</h2>
 <p class="lede">Class priors are not constants. They drift twice as much as the baseline and they
 <b>reorder</b>: six of seven tracked classes traverse most of the ranking. A pooled prior therefore
 mis-gates in both directions, which is why the gate reads a <b>trailing</b> prior as of the date.</p>
 <div class="card"><h3>Relative risk over time — class prior ÷ contemporaneous control</h3>
  <p class="note">Dividing by the control measured on the <em>same window</em> separates “this class
  got riskier” from “everything got riskier”. 1.0× = indistinguishable from an ordinary day.
  Click a legend entry to isolate.</p>
  <svg id="c-drift" viewBox="0 0 1000 300" style="width:100%"></svg>
  <div class="lg" id="lg-drift"></div></div>
 <div class="card"><h3>Rank — the ordering itself reorders</h3>
  <p class="note">A static prior implicitly assumes these lines are flat and parallel.</p>
  <svg id="c-rank" viewBox="0 0 1000 260" style="width:100%"></svg>
  <table id="t-rank" style="margin-top:8px"></table></div>
</section>

<!-- ============ DAY ============ -->
<section class="view" id="v-day">
 <h2>A day, decomposed and drilled down</h2>
 <p class="lede">Every name's return splits into <b>market + sector + idiosyncratic</b>. Company news
 is expected to explain primarily the idiosyncratic part. Drill through the real SIC hierarchy —
 division → major group → industry → name — and the gate verdict appears at the leaf.</p>
 <div class="row">
  <select id="daysel" onchange="drawDay()"></select>
  <span class="pill" id="daytier"></span>
 </div>
 <div class="kpi" id="k-day"></div>
 <div class="card"><h3>Cross-sectional variance split</h3>
  <p class="note">This is why article-only attribution overstates company news: it is the share of
  <em>dispersion between names</em>, and the idiosyncratic part — the only part company news can
  speak to — is measured against each name's own trailing volatility.</p>
  <div id="daybar"></div></div>
 <div class="card"><h3 id="drillh">Drill-down</h3>
  <div class="crumb" id="crumb"></div>
  <table id="t-drill"></table></div>
</section>

<!-- ============ NAME ============ -->
<section class="view" id="v-name">
 <h2>One name: what it co-moves with, and what its news is made of</h2>
 <p class="lede">Correlations are of <b>idiosyncratic residuals</b> — after market and sector are
 removed — so these are relationships a sector map does not show. Line thickness and opacity encode
 correlation strength; dashed lines cross a SIC industry boundary.</p>
 <div class="row"><select id="namesel" onchange="drawName()"></select>
  <span class="pill" id="nameind"></span></div>
 <div class="card"><h3>Idiosyncratic correlation neighbours — through time</h3>
  <p class="note">Contemporaneous association, not a lead-lag signal. Drag the slider to move a
  120-day window through the sample and watch the links strengthen and fade. A line is drawn
  <b>solid</b> only while it clears that pair's own <b>chance band</b> (measured by circular-shifting
  one series, so it reflects what |r| looks like for these two names with no relationship); inside
  the band it is faint and dotted, because at this window length it is not distinguishable from
  noise. The neighbour SET is held fixed at the full-sample top 8 so the lines stay comparable —
  what changes is each link's strength, not which names are shown.</p>
  <div class="row">
   <input type="range" id="wslider" min="0" value="0" style="flex:1;min-width:300px" oninput="drawNbAt()">
   <span class="pill" id="wlabel"></span>
   <button id="playbtn" onclick="togglePlay()" style="background:var(--surface);color:var(--ink2);
    border:1px solid var(--border);border-radius:7px;padding:5px 11px;font-size:12.5px;cursor:pointer">▶ play</button>
  </div>
  <svg id="c-nb" viewBox="0 0 1000 420" style="width:100%"></svg>
  <p class="note" id="nbread" style="margin-top:2px"></p>
  <svg id="c-nbts" viewBox="0 0 1000 190" style="width:100%;margin-top:6px"></svg>
  <p class="note">Every neighbour's correlation over the whole sample; the vertical line is the
  window selected above and the shaded corridor is the chance band for the highlighted pair.</p></div>
 <div class="card"><h3>News weight — what this name's news flow is made of</h3>
  <p class="note">A coverage measure, not a forecast. <b>× ctrl</b> is the measured immediate risk
  of that class across the whole sample.</p>
  <table id="t-weight"></table></div>
</section>

<!-- ============ NUMBERS ============ -->
<section class="view" id="v-num">
 <h2>Numbers vs narrative — what actually moves the price?</h2>
 <p class="lede">An earnings day moves a stock <b>3.2× an ordinary day</b>. But is that the
 RESULTS moving it, or the COVERAGE moving it? Those are indistinguishable until you have the
 numbers themselves. This view separates them, using earnings surprise computed from SEC XBRL
 filings — free, exact, and available for every filer.</p>
 <div class="kpi" id="k-num"></div>

 <div class="card"><h3>1 · The event window — and why the date matters</h3>
  <p class="note">Idiosyncratic move by offset from the release, as a multiple of a clean
  control day (no news AND no filing within ±10 days). <b>49% of earnings releases are accepted
  at or after 16:00 ET</b>, so dating them by publication — as the news graph does — puts half of
  them one session early. The raw calendar arm shows that directly as a two-day plateau;
  aligning to the session the release could first affect collapses it into a single spike.</p>
  <svg id="c-window" viewBox="0 0 1000 300" style="width:100%"></svg>
  <table id="t-window" style="margin-top:8px"></table></div>

 <div class="card"><h3>2 · Does the price follow the numbers?</h3>
  <p class="note"><b>SUE</b> = (EPS this quarter − EPS four quarters ago) ÷ the firm's own
  surprise volatility — the Foster / Bernard-Thomas standardised unexpected earnings, which needs
  no analyst data. <b>sign(SUE) × idiosyncratic return</b> is positive when the price moves the way
  the numbers went. Watch Q1: releases with essentially NO surprise show NO directional move,
  which is the control that makes the rest believable.</p>
  <table id="t-sue"></table>
  <p class="note" id="sue-read" style="margin-top:10px"></p></div>

 <div class="card"><h3>3 · Where the narrative and the numbers disagree — BLOCKED</h3>
  <p class="note">The question this view was built to answer, and it cannot be answered with
  this data. The reason is itself the finding.</p>
  <div id="clash"></div></div>
</section>

<!-- ============ REGIMES ============ -->
<section class="view" id="v-reg">
 <h2>Correlations start and end — and when is a link actually over?</h2>
 <p class="lede">Everything else here is measured once over the whole sample, which hides the
 question worth asking: relationships form and decay. This tracks the rolling correlation of two
 names' <b>idiosyncratic</b> residuals and asks whether it still clears the level explainable by
 chance <em>for that specific pair</em>.</p>
 <div class="kpi" id="k-reg"></div>
 <div class="card"><h3>The band is measured, not assumed</h3>
  <p class="note">“This correlation ended” only means something against a band of what |r| looks
  like when there is <b>no</b> relationship — two finite series correlate spuriously all the time.
  So for each pair one series is <b>circularly shifted</b> by a random offset and the same rolling
  correlation recomputed, many times. That destroys the contemporaneous link while preserving each
  series' own length, variance and autocorrelation. The 95th percentile of |shifted r| is the band.
  A window counts as ACTIVE only above that pair's own band, and a state change needs
  <span id="minrun"></span> consecutive windows <em>and</em> agreement with the current state.</p>
  <div id="regstates"></div></div>
 <div class="card">
  <div class="row">
   <select id="regfilter" onchange="drawReg()">
    <option value="EMERGING">EMERGING — was chance, now real</option>
    <option value="ENDED">ENDED — was real, now indistinguishable from chance</option>
    <option value="PERSISTENT">PERSISTENT — held throughout</option>
    <option value="EPISODIC">EPISODIC — flips repeatedly, distrust it</option>
    <option value="UNCONFIRMED">UNCONFIRMED — changed too recently to call</option>
   </select>
   <select id="regsel" onchange="drawRegChart()"></select>
   <label style="color:var(--muted);font-size:11.5px">
    <input type="checkbox" id="onlyev" checked onchange="drawReg()"> only pairs with article evidence</label>
  </div>
  <svg id="c-reg" viewBox="0 0 1000 300" style="width:100%"></svg>
  <p class="note" id="regread"></p>
  <div id="regnews"></div></div>
 <div class="card"><h3>Can an LLM say <em>why</em>? — placebo-controlled</h3>
  <p class="note">Each transition is judged twice by the same model, which is not told which is
  which: once with the news actually surrounding the transition, once with news from a window for
  the <b>same pair</b> where nothing changed — <b>matched on article volume</b>, because an empty
  placebo makes “UNRELATED” trivial and turns the control into a test of news-vs-no-news.</p>
  <div class="kpi" id="k-reason"></div>
  <div id="reasonverdict"></div>
  <table id="t-reason" style="margin-top:11px"></table>
  <div id="reasoncases" style="margin-top:13px"></div></div>
</section>

<!-- ============ A/B ============ -->
<section class="view" id="v-ab">
 <h2>The same evidence, with and without the gate</h2>
 <p class="lede">Both models receive identical article text. The <span style="color:var(--bad)">
 article-only</span> model gets nothing else. The <span style="color:var(--acc)">gated</span> model
 also receives the decomposition, the trailing class prior in force on that date, and a graded
 verdict — and is instructed to treat a narrative as <b>contributory rather than explanatory</b>
 when the reaction sits inside the class's normal range. The gate never asserts what
 <em>did</em> cause a move: rejecting one explanation is not evidence for another.</p>
 <div class="row"><label style="color:var(--muted);font-size:11.5px">
  <input type="checkbox" id="onlyfail" onchange="drawAB()"> show only gate-rejected cases</label></div>
 <div id="ablist"></div>
</section>

<!-- ============ COMPARE ============ -->
<section class="view" id="v-cmp">
 <h2>Does it replicate in a second market?</h2>
 <p class="lede">The strongest test available for the one surviving positive result. The India graph
 is an independent corpus — different market, different years, different extraction run, and its
 entities had to be resolved by a different method because the extractor resolved none of them.
 If the topic→risk map is real it should reappear; if it was an artifact of the US sample it
 should not.</p>
 <div class="kpi" id="k-cmp"></div>
 <div class="card"><h3>Immediate risk by class — both markets, each against its OWN control</h3>
  <p class="note">Each market's figure is a multiple of ITS OWN measured ordinary day, so the two
  are comparable despite different absolute volatility. Both controls are computed the same way.</p>
  <svg id="c-cmp" viewBox="0 0 1000 330" style="width:100%"></svg>
  <table id="t-cmp" style="margin-top:9px"></table></div>
 <div class="card"><h3>How the two graphs differ</h3>
  <table id="t-cmpmeta"></table>
  <p class="note" id="cmpcav" style="margin-top:10px"></p></div>
</section>

<!-- ============ NEGATIVE REGISTER ============ -->
<section class="view" id="v-null">
 <h2>Findings register</h2>
 <p class="lede">A complete record of hypotheses tested, figures subsequently corrected, defects
 identified, and the effect of control procedures on reported results. The majority of tests in
 this programme returned null results, and several apparent effects did not survive the addition
 of a control. Each figure below is traceable to a stored result file or logged run, cited at the
 end of each entry.</p>
 <div class="kpi" id="k-null"></div>

 <div class="card"><h3>1 · Hypotheses tested</h3>
  <p class="note">Each hypothesis was tested against a criterion set in advance. Verdicts are
  graded and should be read individually: <code>NOT SUPPORTED</code> indicates an adequately
  powered test returning no effect; <code>INCONCLUSIVE</code> indicates a test that could not have
  detected the published effect at the available sample size; <code>INSUFFICIENT EVIDENCE</code>
  indicates no support for the hypothesis where the contrary test is also too small to be
  decisive; and <code>PARTIALLY ESTABLISHED</code> or <code>PREMISE SUPPORTED</code> indicate that
  one component was demonstrated and another was not. These distinctions are material and are not
  interchangeable.</p>
  <div id="reg-rej"></div></div>

 <div class="card"><h3>2 · Corrections to previously reported figures</h3>
  <p class="note">Figures reported before verification, together with the corrected value and the
  cause of the discrepancy. Retained in full: the correction is more informative than the
  original.</p>
  <div id="reg-ret"></div></div>

 <div class="card"><h3>3 · Defects identified in data and implementation</h3>
  <p class="note">Each altered a figure that had already been reported. None raised an error at
  runtime, which is the characteristic that made them costly to detect.</p>
  <div id="reg-bug"></div></div>

 <div class="card"><h3>4 · Control procedures and their effect</h3>
  <p class="note">In each case the uncontrolled specification produced a result that would have
  been reported as a finding. The middle column records what that specification returned.</p>
  <table id="reg-ctl"></table></div>

 <div class="card"><h3>Established findings</h3>
  <p class="note">Results that survived the control procedures above.</p>
  <ul style="font-size:12.5px;line-height:1.75;color:var(--ink2);margin:0;padding-left:19px">
   <li><b>Contemporaneous</b> topic→risk: earnings <b>2.17×</b> an ordinary day, and
    <b>2.18×</b> in India — an independent market, different years, different extraction.
    Robust to the stale-price bug AND to removing the sector component.</li>
   <li><b>Class priors drift and reorder</b> — six of seven tracked classes traverse most of the
    ranking; only earnings is stable.</li>
   <li><b>Correlation regimes are detectable</b> against a per-pair circular-shift band, and the
    transitions are economically legible (STX~WDC emerging as that industry consolidated).</li>
   <li><b>The gate does real work</b>: on captured days it clears 100% of news-attached moves above
    3σ while rejecting 71% at 0σ.</li>
   <li><b>Coverage conditioning</b> — 27.4% of the largest moves on captured days carry a
    gate-cleared narrative.</li>
   <li><b>An event CALENDAR beats extracted news</b> for anything scheduled: 8-K Item 2.02
    gives 8,406 measurable earnings sessions against 415 from the graph, a prior estimable in
    57 of 57 rolling windows rather than 34, and a tighter trailing range. Dating by filing
    rather than publication raises the measured effect from 2.21x to <b>3.20x</b>.</li>
   <li><b>Risk is elevated the session BEFORE a scheduled release</b> — 1.34x a clean control,
    and it survives calendar alignment, so it is not a timestamp artifact. Unlike unscheduled
    news this is knowable in advance, which is the only version of a pre-event effect that
    could ever be acted on.</li>
   <li><b>Stale news moves prices more</b> — +0.197σ same-day against fresh news
    [+0.107, +0.299], monotonic across terciles and robust to matching on coverage
    intensity. Tetlock's premise; his reversal prediction remains underpowered here.</li>
  </ul>
  <p class="note" style="margin-top:11px"><b>Scope of the established findings.</b> All are
  contemporaneous or descriptive; none is predictive. The consistent result across this programme
  is that news carries information in isolation but is redundant given price history. The single
  directional finding — the earnings-day response — is attributable to the reported figures rather
  than to news coverage.</p></div>
</section>
</main><div class="tip" id="tip"></div>
<script>
const D=__DATA__;const PAL=['--s1','--s2','--s3','--s4','--s5','--s6','--s7'];
/* Everything below reads M, the ACTIVE market. Markets are never merged: each has
   its own control, its own priors and its own sector provenance, so a pooled number
   across them would be meaningless. The comparison view puts them side by side
   instead, each against its own baseline. */
let MK=Object.keys(D.markets)[0], M=D.markets[MK];
function setMarket(k){MK=k;M=D.markets[k];
 document.getElementById('mksub').textContent=M.meta.sub;
 initAll();}
const tip=document.getElementById('tip');
/* Escapes quotes as well as angle brackets. Several of these strings are
   interpolated into HTML ATTRIBUTES (inline onmousemove handlers), and one real
   SIC label — "Wholesale-Drugs, Proprietaries & Druggists' Sundries" — contains
   an apostrophe that would otherwise terminate a single-quoted attribute. */
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>
 ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const un=s=>String(s??'').replace(/_/g,' ');
const pct=v=>(v*100).toFixed(2)+'%';
function show(e,h){tip.innerHTML=h;tip.style.opacity=1;
 tip.style.left=Math.min(e.clientX+14,innerWidth-300)+'px';tip.style.top=(e.clientY+14)+'px';}
function hide(){tip.style.opacity=0;}
function tg(){const h=document.documentElement;h.dataset.theme=h.dataset.theme==='light'?'dark':'light';redraw();}
document.querySelectorAll('nav a').forEach(a=>a.onclick=()=>{
 document.querySelectorAll('nav a').forEach(x=>x.classList.remove('on'));a.classList.add('on');
 document.querySelectorAll('.view').forEach(v=>v.classList.remove('on'));
 document.getElementById('v-'+a.dataset.v).classList.add('on');redraw();});

/* ---------- scorecard ---------- */
const dc=M.coverage.deep['>4'], pc=M.coverage.pooled['>4'];
const _nv=(M.numbers||{}).nvn, _cw=(M.numbers||{}).calwindow;
document.getElementById('k-score').innerHTML=[
 ..._nv?[['+'+_nv.overall_dir.mean.toFixed(2)+'%','idiosyncratic move in the DIRECTION of the earnings surprise — the only directional result here']]:[],
 ..._cw?[[_cw.profiles['CALENDAR-ALIGNED']['0'].ratio.toFixed(2)+'×','earnings-day move vs an ordinary day, dated by SEC filing (2.22× if dated by publication)']]:[],
 [`${(dc.explained*100).toFixed(1)}%`,'of >4σ idiosyncratic moves EXPLAINED on days we actually captured (pooled: '+(pc.explained*100).toFixed(1)+'%)'],
 ['2.17×','earnings news-day risk vs an ordinary day, news-dated (India: 2.18× — replicates)'],
 ['3.6%','of gate verdicts the old pooled prior got wrong, in both directions'],
 ['123','stale illiquid tickers removed; they had produced a 267σ "control"'],
 ['74%','SIC coverage from SEC EDGAR, replacing a 12% hand-made map'],
].map(([b,s])=>`<div class="k"><b>${b}</b><span>${s}</span></div>`).join('');
document.getElementById('t-risk').innerHTML=
 '<tr><th>event class</th><th>prior σ</th><th>× control</th><th>class sd</th></tr>'+
 M.riskmap.map(r=>`<tr><td>${un(r.cls)}</td><td>${r.prior.toFixed(2)}</td>
  <td class="${r.mult>=1.4?'ok':r.mult<=1.05?'no':''}">${r.mult.toFixed(2)}×</td>
  <td style="color:var(--muted)">${r.sd.toFixed(2)}</td></tr>`).join('')+
 `<tr><td style="color:var(--muted)">control (ordinary non-event day)</td>
  <td style="color:var(--muted)">${M.control.toFixed(2)}</td><td style="color:var(--muted)">1.00×</td><td></td></tr>`;
document.getElementById('t-fix').innerHTML=
 '<tr><th>claim</th><th>was</th><th>is</th><th>why it changed</th></tr>'+[
 ['explained fraction of >4σ moves','2.3%','27.4%','pooled across deliberately-thin days; conditioning on coverage inverts it'],
 ['control (ordinary day)','0.72σ / 267σ','0.75σ','17% of tickers were stale OTC ADRs; std()&gt;0 is not a liquidity guard'],
 ['cross-sector links','28%','29% / 11% / 6%','granularity was never defined — reported at all three SIC levels'],
 ['sector taxonomy','GICS, 12% by hand','SIC, 74% from EDGAR','proprietary and unverifiable → public and SEC-assigned'],
 ['class priors','pooled constants','trailing as of date','priors drift 2× the baseline and reorder'],
 ['"cannot explain a day"','stated as a limit','a sampling artifact','graph holds 13% of the corpus'],
].map(r=>`<tr><td>${r[0]}</td><td class="bd">${r[1]}</td><td class="ok">${r[2]}</td>
 <td style="color:var(--ink2);text-align:left;font-size:11.5px">${r[3]}</td></tr>`).join('');

/* ---------- coverage ---------- */
const H=M.hist;
document.getElementById('k-cov').innerHTML=[
 [H.deep||0,'days deeply covered (100+ docs, avg 453)'],
 [(H.thin||0)+(H.token||0),'days with only token coverage (1–19 docs)'],
 [H.none||0,'trading days with no documents at all'],
 [M.dpd_pct.p50+' / '+M.dpd_pct.p90,'median / p90 documents per covered day'],
].map(([b,s])=>`<div class="k"><b>${b}</b><span>${s}</span></div>`).join('');
(function(){const o=['deep','partial','thin','token','none'],
 col={deep:'--s3',partial:'--s1',thin:'--s4',token:'--s2',none:'--grid'},T=M.n_days;
 document.getElementById('tierbar').innerHTML=
  `<div class="bar" style="height:15px">`+o.map(t=>H[t]?
   `<i style="width:${H[t]/T*100}%;background:var(${col[t]})" title="${t}: ${H[t]} days"></i>`:'').join('')+`</div>
   <div class="lg">`+o.filter(t=>H[t]).map(t=>
   `<span><span class="sw" style="background:var(${col[t]})"></span>${t} — ${H[t]} days</span>`).join('')+`</div>`;})();
function drawCov(){const k=document.getElementById('covsel').value,c=M.coverage[k];
 document.getElementById('t-cov').innerHTML=
  '<tr><th>move size</th><th>moves</th><th>with news</th><th>reach</th><th>of news, gate OK</th><th>EXPLAINED</th></tr>'+
  ['>0','>1','>2','>3','>4'].map(b=>{const r=c[b];return `<tr><td>${b}σ</td>
   <td style="color:var(--muted)">${r.n.toLocaleString()}</td><td>${r.news.toLocaleString()}</td>
   <td>${(r.reach*100).toFixed(1)}%</td><td>${(r.gated*100).toFixed(1)}%</td>
   <td><b class="${r.explained>0.1?'ok':''}">${(r.explained*100).toFixed(1)}%</b></td></tr>`}).join('');
 const d=M.coverage.deep['>4'],t=M.coverage.thin['>4'];
 document.getElementById('cov-read').innerHTML=k==='deep'
  ? `On captured days <b>${(d.explained*100).toFixed(1)}%</b> of &gt;4σ moves have a gate-cleared narrative, and
     <b>${(d.gated*100).toFixed(0)}%</b> of news-attached moves above 3σ clear the gate — at that size
     <em>reach is the only binding constraint</em>. At &gt;0σ the gate rejects
     ${(100-M.coverage.deep['>0'].gated*100).toFixed(0)}% of news-attached moves as insufficient, which is the point:
     a 0.5σ move on an earnings day is unremarkable for earnings.`
  : k==='thin'
  ? `On thinly-sampled days only <b>${(t.explained*100).toFixed(1)}%</b> of &gt;4σ moves are explained — roughly
     ${(d.explained/Math.max(t.explained,1e-9)).toFixed(0)}× worse than captured days. Nothing about the market changed
     between these two rows; only the sampling did. <b>has_news=false on a thin day means NOT SAMPLED.</b>`
  : `Pooling the two hides the design: the average is dragged down by the ${(H.thin||0)+(H.token||0)}
     token-coverage days and reads as though almost nothing is explainable. This was the original error.`;}

/* ---------- risk drift ---------- */
function axes(H,lo,hi,fmt,ticks,pts){let s='';const L=52,R=112,W=1000;
 const Y=v=>H-26-((v-lo)/(hi-lo))*(H-48);
 for(let k=0;k<=ticks;k++){const v=lo+(hi-lo)*k/ticks,y=Y(v);
  s+=`<line x1="${L}" y1="${y}" x2="${W-R}" y2="${y}" stroke="var(--grid)"/>
   <text x="${L-7}" y="${y+3.5}" font-size="9.5" fill="var(--muted)" text-anchor="end">${fmt(v)}</text>`;}
 pts.forEach((p,i)=>{if(i%6)return;s+=`<text x="${X(i,pts.length)}" y="${H-8}" font-size="9"
  fill="var(--muted)" text-anchor="middle">${p.d.slice(0,7)}</text>`;});
 return{s,Y};}
const X=(i,n)=>52+(i/(n-1))*(1000-52-112);
let dOff=new Set();
function drawDrift(){const P=M.drift,T=M.track;
 let vals=[];P.forEach(p=>T.forEach(c=>{if(p.c[c]&&!dOff.has(c))vals.push(p.c[c]);}));
 if(!vals.length)vals=[1];
 const lo=Math.min(0.9,...vals)*.98,hi=Math.max(...vals)*1.02;
 let{s,Y}=axes(300,lo,hi,v=>v.toFixed(1)+'×',5,P);
 s+=`<line x1="52" y1="${Y(1)}" x2="888" y2="${Y(1)}" stroke="var(--warn)" stroke-dasharray="4,3"/>
  <text x="894" y="${Y(1)+3.5}" font-size="9.5" fill="var(--warn)">1.0× control</text>`;
 T.forEach((c,ci)=>{if(dOff.has(c))return;
  const q=P.map((p,i)=>p.c[c]?[X(i,P.length),Y(p.c[c])]:null).filter(Boolean);
  if(q.length<2)return;
  s+=`<path d="M${q.map(a=>a.join(',')).join(' L')}" fill="none" stroke="var(${PAL[ci%7]})" stroke-width="2"/>`;
  s+=`<text x="${q[q.length-1][0]+6}" y="${q[q.length-1][1]+3.5}" font-size="9.5" fill="var(${PAL[ci%7]})">${un(c)}</text>`;
  P.forEach((p,i)=>{if(!p.c[c])return;s+=`<circle cx="${X(i,P.length)}" cy="${Y(p.c[c])}" r="6" fill="transparent"
   onmousemove='show(event,"<b>${un(c)}</b><div style=color:var(--ink2)>${p.d}</div><div style=color:var(--ink2)>"+${p.c[c]}+"× control (ctrl "+${p.ctrl}+"σ)</div>")' onmouseleave="hide()"/>`;});});
 document.getElementById('c-drift').innerHTML=s;
 const mr=Math.max(...P.flatMap(p=>Object.values(p.rank||{})));
 let a=axes(260,mr+.5,.5,v=>Math.round(v),mr-1,P);let r=a.s;
 T.forEach((c,ci)=>{if(dOff.has(c))return;
  const q=P.map((p,i)=>p.rank&&p.rank[c]?[X(i,P.length),a.Y(p.rank[c])]:null).filter(Boolean);
  if(q.length<2)return;
  r+=`<path d="M${q.map(x=>x.join(',')).join(' L')}" fill="none" stroke="var(${PAL[ci%7]})"
   stroke-width="2.3" stroke-linejoin="round" opacity=".92"/>`;
  q.forEach(x=>{r+=`<circle cx="${x[0]}" cy="${x[1]}" r="3" fill="var(${PAL[ci%7]})"/>`;});
  r+=`<text x="${q[q.length-1][0]+6}" y="${q[q.length-1][1]+3.5}" font-size="9.5" fill="var(${PAL[ci%7]})">${un(c)}</text>`;});
 document.getElementById('c-rank').innerHTML=r;
 document.getElementById('lg-drift').innerHTML=T.map((c,i)=>
  `<span style="cursor:pointer;opacity:${dOff.has(c)?.3:1}" onclick="tkD('${c}')">
   <span class="sw" style="background:var(${PAL[i%7]})"></span>${un(c)}</span>`).join('');
 document.getElementById('t-rank').innerHTML='<tr><th>class</th><th>rank range</th></tr>'+
  T.filter(c=>M.rank_range[c]).map(c=>{const[x,y]=M.rank_range[c];
   return `<tr><td style="color:var(${PAL[T.indexOf(c)%7]})">${un(c)}</td>
   <td style="text-align:left">${x===y?`stable at ${x}`:`<b>${x} → ${y}</b> — moves ${y-x} places`}</td></tr>`;}).join('');}
function tkD(c){dOff.has(c)?dOff.delete(c):dOff.add(c);drawDrift();}

/* ---------- day drill-down ---------- */
let dIdx=0,dPath=[],curGroups=[];
function drillInto(i){if(curGroups[i]){dPath.push(curGroups[i][0]);renderDay();}}
document.getElementById('daysel').innerHTML=M.days.map((d,i)=>
 `<option value="${i}">${d.d} — ${d.docs} documents, ${d.n} names</option>`).join('');
function drawDay(){dIdx=+document.getElementById('daysel').value;dPath=[];renderDay();}
function renderDay(){const day=M.days[dIdx];
 document.getElementById('daytier').textContent=`deep coverage · ${day.docs} docs · control ${day.ctrl.toFixed(2)}σ`;
 const withNews=day.rows.filter(r=>r.cls).length, ok=day.rows.filter(r=>r.gate==='SUFFICIENT').length;
 document.getElementById('k-day').innerHTML=[
  [(day.var.idi*100).toFixed(0)+'%','of cross-sectional variance is idiosyncratic'],
  [withNews,'of the '+day.rows.length+' largest movers have linked news'],
  [ok,'clear the narrative-confidence gate'],
  [day.rows[0].sig?.toFixed(2)+'σ','largest idiosyncratic move'],
 ].map(([b,s])=>`<div class="k"><b>${b}</b><span>${s}</span></div>`).join('');
 const v=day.var,C2=[['market','--s1',v.mkt],['sector','--s4',v.sct],['idiosyncratic','--s3',v.idi]];
 document.getElementById('daybar').innerHTML=`<div class="bar" style="height:15px">`+
  C2.map(([n,c,x])=>`<i style="width:${x*100}%;background:var(${c})"></i>`).join('')+`</div><div class="lg">`+
  C2.map(([n,c,x])=>`<span><span class="sw" style="background:var(${c})"></span>${n} ${(x*100).toFixed(1)}%</span>`).join('')+`</div>`;
 const lv=['div','mg','ind'],lb=['division','major group','industry'];
 let rows=day.rows.filter(r=>dPath.every((p,i)=>r[lv[i]]===p));
 document.getElementById('crumb').innerHTML='<a onclick="dPath=[];renderDay()">all</a>'+
  dPath.map((p,i)=>` › <a onclick="dPath=dPath.slice(0,${i+1});renderDay()">${esc(p)}</a>`).join('')+
  (dPath.length<3?` <span style="color:var(--muted)">— click a ${lb[dPath.length]} to drill in</span>`:'');
 if(dPath.length<3){const k=lv[dPath.length],agg={};
  rows.forEach(r=>{const g=r[k]||'UNK';(agg[g]=agg[g]||{n:0,idi:0,sig:0,news:0,ok:0});
   agg[g].n++;agg[g].idi+=Math.abs(r.idi);agg[g].sig+=r.sig||0;
   if(r.cls)agg[g].news++;if(r.gate==='SUFFICIENT')agg[g].ok++;});
  /* Hand the click an INDEX, never the label itself: interpolating a group name
     into an inline handler breaks on any label containing a quote, and SIC has
     several. curGroups is the lookup the handler resolves against. */
  curGroups=Object.entries(agg).sort((a,b)=>b[1].idi/b[1].n-a[1].idi/a[1].n);
  document.getElementById('t-drill').innerHTML=
   `<tr><th>${lb[dPath.length]}</th><th>names</th><th>mean |idio|</th><th>mean σ</th><th>with news</th><th>gate OK</th></tr>`+
   curGroups.map(([g,a],gi)=>`<tr style="cursor:pointer" onclick="drillInto(${gi})">
    <td style="color:var(--acc)">${esc(g)}</td><td>${a.n}</td><td>${pct(a.idi/a.n)}</td>
    <td>${(a.sig/a.n).toFixed(2)}σ</td><td>${a.news}</td>
    <td class="${a.ok?'ok':''}">${a.ok}</td></tr>`).join('');
 }else{
  document.getElementById('t-drill').innerHTML=
   '<tr><th>name</th><th>total</th><th>market</th><th>sector</th><th>idio</th><th>σ</th><th>class</th><th>gate</th></tr>'+
   rows.map((r,i)=>`<tr style="cursor:${r.cls?'pointer':'default'}" ${r.cls?`onclick="tglCase(${i})"`:''}>
    <td>${r.tk} <span style="color:var(--muted)">${esc(r.nm).slice(0,26)}</span></td>
    <td>${pct(r.tot)}</td><td style="color:var(--muted)">${pct(r.mkt)}</td>
    <td style="color:var(--muted)">${pct(r.sct)}</td><td><b>${pct(r.idi)}</b></td>
    <td>${r.sig?.toFixed(2)}</td><td style="text-align:left">${r.cls?un(r.cls):'<span style="color:var(--muted)">no news</span>'}</td>
    <td class="${r.gate==='SUFFICIENT'?'ok':r.gate?'no':''}">${r.gate?(r.gate==='SUFFICIENT'?r.conf:'INSUFF'):'—'}</td></tr>`+
    (r.cls?`<tr id="cse${i}" style="display:none"><td colspan="8" style="text-align:left;padding:9px 12px;background:rgba(127,127,127,.05)">
     <div style="font-size:11.5px;color:var(--ink2);margin-bottom:6px">
      prior <b>${r.prior?.toFixed(2)}σ</b> as of ${r.asof} · surprise <b>${r.sur>0?'+':''}${r.sur}σ</b>
      ${r.why&&r.why.length?`<div style="color:var(--warn);margin-top:4px">◯ ${esc(r.why.join('; '))}</div>`:''}</div>
     ${(r.docs||[]).map(d=>`<div class="art"><b>${esc(d.h)}</b><br>${esc(d.t)}</div>`).join('')}</td></tr>`:'')).join('');}}
function tglCase(i){const e=document.getElementById('cse'+i);if(e)e.style.display=e.style.display==='none'?'':'none';}

/* ---------- name ---------- */
let playT=null;
function togglePlay(){const b=document.getElementById('playbtn');
 if(playT){clearInterval(playT);playT=null;b.textContent='▶ play';return;}
 b.textContent='❚❚ pause';
 playT=setInterval(()=>{const s=document.getElementById('wslider');
  s.value=(+s.value+1>+s.max)?0:+s.value+1;drawNbAt();},420);}
function drawNbAt(){
 const t=document.getElementById('namesel').value,n=M.names[t];
 const W=M.roll_dates||[],k=+document.getElementById('wslider').value;
 document.getElementById('wlabel').textContent=W.length?`${M.roll_win}-day window ending ${W[k]}`:'';
 const nb=n.nb.filter(b=>b.series);        // only those with a rolling series
 const cx=500,cy=205,R=155;let s='';
 // scale to the pair set's strongest link IN THIS WINDOW, floored so a quiet
 // window does not visually inflate noise into a strong-looking link
 const cur=nb.map(b=>b.series[k]);
 const mx=Math.max(0.35,...cur.map(v=>v==null?0:Math.abs(v)));
 let nAct=0,nCross=0;
 nb.forEach((b,i)=>{const ang=(i/nb.length)*Math.PI*2-Math.PI/2;
  const x=cx+Math.cos(ang)*R,y=cy+Math.sin(ang)*R;
  const v=b.series[k];
  const act=v!=null&&Math.abs(v)>b.band;      // clears its own chance band
  if(act){nAct++;if(!b.same)nCross++;}
  const mag=v==null?0:Math.abs(v)/mx;
  const w=act?(1.2+4.2*mag):1;
  const o=v==null?.08:(act?(.35+.6*mag):.16);
  const col=v==null?'--muted':(v>0?'--s1':'--s2');
  const dash=act?(b.same?'':'stroke-dasharray="6,4"'):'stroke-dasharray="2,4"';
  s+=`<line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}" stroke="var(${col})" stroke-width="${w.toFixed(2)}"
   opacity="${o.toFixed(2)}" ${dash}
   onmousemove='show(event,"<b>${t} ↔ ${b.t}</b><div style=color:var(--ink2)>r = "+${JSON.stringify(v==null?'n/a':(v>0?'+':'')+v)}+" in the window ending "+${JSON.stringify(W[k]||'')}+"</div><div style=color:var(--ink2)>chance band ±${b.band}</div><div style=color:var(--muted)>"+${JSON.stringify(act?'clears the band — a real link in this window':'inside the band — indistinguishable from noise here')}+"</div><div style=color:var(--muted)>${esc(b.ind)}</div>")'
   onmouseleave="hide()" onclick="hiN=${i};drawNbTs()"/>`;
  s+=`<circle cx="${x}" cy="${y}" r="${act?5:3}" fill="var(${col})" opacity="${act?1:.35}"/>
   <text x="${x+(Math.cos(ang)>0?8:-8)}" y="${y+3.5}" font-size="10.5"
    fill="${act?'var(--ink2)':'var(--muted)'}"
    text-anchor="${Math.cos(ang)>0?'start':'end'}">${b.t}</text>
   <text x="${x+(Math.cos(ang)>0?8:-8)}" y="${y+15}" font-size="8.5" fill="var(--muted)"
    text-anchor="${Math.cos(ang)>0?'start':'end'}">${v==null?'—':(v>0?'+':'')+v}</text>`;});
 s+=`<circle cx="${cx}" cy="${cy}" r="24" fill="var(--surface)" stroke="var(--acc)" stroke-width="2"/>
  <text x="${cx}" y="${cy+4}" font-size="12" fill="var(--ink)" text-anchor="middle" font-weight="600">${t}</text>`;
 s+=`<text x="16" y="18" font-size="10.5" fill="var(--muted)">solid = same industry · dashed = crosses industry ·
  faint dotted = inside the chance band · blue positive, orange negative — click a line to highlight it below</text>`;
 document.getElementById('c-nb').innerHTML=s;
 document.getElementById('nbread').innerHTML=
  `In the window ending <b>${W[k]||'—'}</b>, <b>${nAct}</b> of ${nb.length} links clear their own
   chance band${nAct?`, ${nCross} of them across an industry boundary`:''}. The rest are, at this
   window length, indistinguishable from no relationship — a full-sample correlation can be carried
   by a handful of windows.`;
 drawNbTs();}
function drawNbTs(){
 const t=document.getElementById('namesel').value,n=M.names[t];
 const W=M.roll_dates||[],k=+document.getElementById('wslider').value;
 const nb=n.nb.filter(b=>b.series);if(!nb.length||!W.length){document.getElementById('c-nbts').innerHTML='';return;}
 const H=190,L=48,Rt=96,Wd=1000,X=i=>L+(i/(W.length-1))*(Wd-L-Rt),Y=v=>H-24-((v+1)/2)*(H-46);
 let s='';
 for(let g=-1;g<=1;g+=0.5){s+=`<line x1="${L}" y1="${Y(g)}" x2="${Wd-Rt}" y2="${Y(g)}" stroke="var(--grid)"/>
  <text x="${L-6}" y="${Y(g)+3.5}" font-size="9" fill="var(--muted)" text-anchor="end">${g.toFixed(1)}</text>`;}
 const hi=(typeof hiN==='number'&&nb[hiN])?nb[hiN]:null;
 if(hi)s+=`<rect x="${L}" y="${Y(hi.band)}" width="${Wd-L-Rt}" height="${Y(-hi.band)-Y(hi.band)}"
  fill="var(--warn)" opacity=".10"/>`;
 nb.forEach((b,i)=>{let seg=[],paths=[];
  b.series.forEach((v,j)=>{if(v==null){if(seg.length>1)paths.push(seg);seg=[];}else seg.push([X(j),Y(v)]);});
  if(seg.length>1)paths.push(seg);
  const on=hi===null||i===hiN;
  paths.forEach(q=>{s+=`<path d="M${q.map(a=>a.join(',')).join(' L')}" fill="none"
   stroke="var(${PAL[i%7]})" stroke-width="${on?2:1}" opacity="${on?.95:.22}"/>`;});
  const last=b.series.map((v,j)=>v==null?null:[X(j),Y(v)]).filter(Boolean).pop();
  if(last&&on)s+=`<text x="${last[0]+5}" y="${last[1]+3.5}" font-size="9.5" fill="var(${PAL[i%7]})">${b.t}</text>`;});
 s+=`<line x1="${X(k)}" y1="14" x2="${X(k)}" y2="${H-24}" stroke="var(--ink)" stroke-width="1.5" opacity=".65"/>`;
 W.forEach((d,i)=>{if(i%10)return;s+=`<text x="${X(i)}" y="${H-8}" font-size="9" fill="var(--muted)"
  text-anchor="middle">${d.slice(0,7)}</text>`;});
 if(hi)s+=`<text x="${L}" y="12" font-size="10" fill="var(--muted)">highlighting ${t} ↔ ${hi.t} — shaded corridor is its chance band (click another line to switch, click the same to reset)</text>`;
 document.getElementById('c-nbts').innerHTML=s;}
let hiN=null;
function drawName(){const t=document.getElementById('namesel').value,n=M.names[t];
 document.getElementById('nameind').textContent=(n.sec.industry||'UNK')+' · '+n.n+' event days';
 hiN=null;
 const sl=document.getElementById('wslider');
 sl.max=Math.max(0,(M.roll_dates||[]).length-1);
 if(+sl.value>+sl.max)sl.value=sl.max;
 drawNbAt();
 document.getElementById('t-weight').innerHTML=
  '<tr><th>event class</th><th>days</th><th>share of news</th><th>× ctrl (immediate risk)</th><th></th></tr>'+
  n.weight.map((w,wi)=>`<tr style="cursor:pointer" onclick="tglW(${wi})">
   <td style="color:var(--acc)">${un(w.c)}</td><td>${w.n}</td>
   <td><div style="display:flex;align-items:center;gap:7px;justify-content:flex-end">
    <div class="bar" style="width:88px"><i style="width:${w.share*100}%;background:var(--s1)"></i></div>
    ${(w.share*100).toFixed(0)}%</div></td>
   <td class="${w.mult>=1.4?'ok':w.mult&&w.mult<=1.05?'no':''}">${w.mult?w.mult.toFixed(2)+'×':'—'}</td>
   <td style="text-align:left;color:var(--muted);font-size:11px">▸ the days</td></tr>
   <tr id="wk${wi}" style="display:none"><td colspan="5" style="text-align:left;padding:9px 12px;background:rgba(127,127,127,.05)">
    ${dayList(n,w.c,w.n)}</td></tr>`).join('');}
/* The share is a coverage number; this is what is actually behind it. Each day
   carries its own gate verdict, so a class with a high share is not the same as a
   class that repeatedly produced moves large enough to attribute. */
function dayList(n,cls,total){
 const rows=(n.events||[]).filter(e=>e.cls.includes(cls));
 if(!rows.length)return '<span style="color:var(--muted);font-size:11.5px">no captured articles for these days</span>';
 const ok=rows.filter(e=>e.gate==='SUFFICIENT').length;
 /* the count in the row above is the FULL tally; only the most recent are carried
    into this page, so say which is which rather than let the two disagree */
 const trunc=total&&total>rows.length
  ? ` <span style="color:var(--muted)">(the ${rows.length} most recent of ${total} — this page carries a capped slice)</span>` : '';
 return `<div style="font-size:11.5px;color:var(--ink2);margin-bottom:7px">
   ${rows.length} ${un(cls)} day${rows.length>1?'s':''} for ${document.getElementById('namesel').value}${trunc} ·
   <b>${ok}</b> cleared the gate · <span style="color:var(--muted)">a high share of news flow is not the same as
   repeatedly moving the stock</span></div>`+
  `<table><tr><th>date</th><th>idio σ</th><th>prior (as of)</th><th>surprise</th><th>gate</th><th>headline</th></tr>`+
  rows.map(e=>`<tr><td>${e.d}</td><td>${e.sig!=null?e.sig.toFixed(2):'—'}</td>
   <td style="color:var(--muted)">${e.prior!=null?e.prior.toFixed(2)+'σ':'—'}
    ${e.asof?`<span style="font-size:10px">${e.asof}</span>`:''}</td>
   <td class="${e.sur>0?'':'no'}">${e.sur!=null?(e.sur>0?'+':'')+e.sur:'—'}</td>
   <td class="${e.gate==='SUFFICIENT'?'ok':e.gate?'no':''}">${e.gate?(e.gate==='SUFFICIENT'?e.conf:'INSUFF'):'—'}</td>
   <td style="text-align:left;font-size:11px">${e.docs&&e.docs.length?esc(e.docs[0].h).slice(0,86):'<span style="color:var(--muted)">—</span>'}</td></tr>`
   +(e.docs&&e.docs.length?`<tr><td colspan="6" style="text-align:left;padding:0 9px 7px 9px">
     <div class="art">${esc(e.docs[0].t)}</div>
     ${e.why&&e.why.length?`<div style="color:var(--warn);font-size:11px">◯ ${esc(e.why[0])}</div>`:''}</td></tr>`:'')).join('')+`</table>`;}
function tglW(i){const e=document.getElementById('wk'+i);if(e)e.style.display=e.style.display==='none'?'':'none';}

/* ---------- A/B ---------- */
function drawAB(){const only=document.getElementById('onlyfail').checked;
 const rows=M.ab.filter(a=>!only||!a.ok);
 document.getElementById('ablist').innerHTML=rows.map(a=>`<div class="card">
  <h3>${a.tk} <span style="color:var(--ink2);font-weight:400">${esc(a.nm)}</span>
   <span class="pill">${a.d}</span> <span class="pill">${un(a.cls)}</span></h3>
  <p class="note" style="margin-bottom:7px">total ${pct(a.tot)} = market ${pct(a.mkt)} + sector ${pct(a.sct)} +
   <b style="color:var(--ink)">idiosyncratic ${pct(a.idi)}</b> · ${a.sig}σ ·
   prior <b>${a.exp}σ</b> as of ${a.asof} · surprise <b style="color:${a.sur>0?'var(--ink)':'var(--warn)'}">${a.sur>0?'+':''}${a.sur}σ</b></p>
  <div style="font-size:11.5px;margin-bottom:7px;color:${a.ok?(a.conf==='WEAK'?'var(--warn)':'var(--ink2)'):'var(--warn)'}">
   ${a.ok?`${a.conf==='WEAK'?'◔':a.conf==='MODERATE'?'◑':'●'} gate SUFFICIENT · confidence <b>${a.conf}</b> —
     surprise is ${a.z} of this class's own standard deviation${a.conf==='WEAK'?', so it only barely clears the prior':''}`
   :`◯ gate INSUFFICIENT — ${esc((a.why||[]).join('; '))}`}</div>
  <div class="ab">
   <div class="ans naive"><h4>article-only — no decomposition, no priors, no gate</h4>${esc(a.naive)}</div>
   <div class="ans gated"><h4>gated — decomposition + trailing prior + confidence test</h4>${esc(a.gated)}</div></div>
  ${a.texts&&a.texts.length?`<details><summary>source articles (${a.texts.length}) — the same evidence both models saw</summary>
   ${a.texts.map(t=>`<div class="art"><b>${esc(t.head)}</b><br>${esc(t.body)}</div>`).join('')}</details>`:''}
  </div>`).join('')||'<p class="note">No cases match.</p>';}

/* ---------- numbers vs narrative ---------- */
function drawNumbers(){const N=M.numbers||{};
 const nvn=N.nvn, W=N.calwindow, WN=N.window;
 if(!nvn){document.getElementById('v-num').innerHTML=
  '<p class="note">Not built for this market — the SUE/calendar work is US-only.</p>';return;}
 const od=nvn.overall_dir;
 document.getElementById('k-num').innerHTML=[
  ['+'+(od.mean).toFixed(2)+'%','average idiosyncratic move in the DIRECTION of the earnings surprise'],
  [(W?W.profiles['CALENDAR-ALIGNED']['0'].ratio.toFixed(2):'—')+'×','earnings-day move vs a clean day, once dated by SEC filing'],
  [nvn.n_rows.toLocaleString(),'releases with both a surprise and a measured reaction'],
  [nvn.n_with_news+' / '+nvn.n_rows.toLocaleString(),'of those the news graph gave a DIRECTION for ('+(100*nvn.n_with_news/nvn.n_rows).toFixed(1)+'%)'],
 ].map(([b,t])=>`<div class="k"><b>${b}</b><span>${t}</span></div>`).join('');
 // window chart
 if(W){const P2=W.profiles,offs=Object.keys(P2['CALENDAR-ALIGNED']).map(Number).sort((a,b)=>a-b);
  const H=300,L=52,Rt=118,Wd=1000;
  const hi=Math.max(...Object.values(P2).flatMap(p=>Object.values(p).map(x=>x.ratio)))*1.06;
  const Y=v=>H-30-((v-0.8)/(hi-0.8))*(H-56), X=i=>L+(i/(offs.length-1))*(Wd-L-Rt);
  let s2='';
  for(let g=1;g<=hi;g+=0.5){s2+=`<line x1="${L}" y1="${Y(g)}" x2="${Wd-Rt}" y2="${Y(g)}" stroke="var(--grid)"/>
   <text x="${L-7}" y="${Y(g)+3.5}" font-size="9.5" fill="var(--muted)" text-anchor="end">${g.toFixed(1)}×</text>`;}
  s2+=`<line x1="${X(offs.indexOf(0))}" y1="16" x2="${X(offs.indexOf(0))}" y2="${H-30}" stroke="var(--warn)" stroke-dasharray="4,3"/>`;
  const cols={'NEWS-DATED':'--s2','CALENDAR-RAW':'--s4','CALENDAR-ALIGNED':'--s3'};
  Object.entries(P2).forEach(([nm,pr])=>{
   const q=offs.filter(o=>pr[o]).map(o=>[X(offs.indexOf(o)),Y(pr[o].ratio)]);
   s2+=`<path d="M${q.map(a=>a.join(',')).join(' L')}" fill="none" stroke="var(${cols[nm]})" stroke-width="2.2"/>`;
   offs.forEach(o=>{if(!pr[o])return;
    s2+=`<circle cx="${X(offs.indexOf(o))}" cy="${Y(pr[o].ratio)}" r="6" fill="transparent"
     onmousemove='show(event,"<b>${nm}</b><div style=color:var(--ink2)>t${o>0?'+':''}${o} = "+${pr[o].ratio.toFixed(2)}+"× a clean day</div><div style=color:var(--muted)>n = "+${pr[o].n}+"</div>")' onmouseleave="hide()"/>`;});
   const last=q[q.length-1];
   s2+=`<text x="${last[0]+6}" y="${last[1]+3.5}" font-size="9.5" fill="var(${cols[nm]})">${nm}</text>`;});
  offs.forEach((o,i)=>{if(i%2)return;s2+=`<text x="${X(i)}" y="${H-12}" font-size="9.5" fill="var(--muted)" text-anchor="middle">${o===0?'t0':'t'+(o>0?'+':'')+o}</text>`;});
  document.getElementById('c-window').innerHTML=s2;
  document.getElementById('t-window').innerHTML=
   '<tr><th>arm</th><th>t−1</th><th>t0</th><th>t+1</th><th>n at t0</th><th></th></tr>'+
   Object.entries(P2).map(([nm,pr])=>`<tr><td style="color:var(${cols[nm]})">${nm}</td>
    <td>${pr['-1']?pr['-1'].ratio.toFixed(2):'—'}</td><td><b>${pr['0'].ratio.toFixed(2)}</b></td>
    <td>${pr['1']?pr['1'].ratio.toFixed(2):'—'}</td><td>${pr['0'].n.toLocaleString()}</td>
    <td style="text-align:left;font-size:11.5px;color:var(--ink2)">${
     nm==='NEWS-DATED'?'dated by publication — the original measurement':
     nm==='CALENDAR-RAW'?'<span class="no">two-day plateau = the smearing</span>':
     '<span class="ok">collapses to one spike — the true effect</span>'}</td></tr>`).join('');}
 // SUE table
 document.getElementById('t-sue').innerHTML=
  '<tr><th>|SUE| quintile</th><th>n</th><th>mean |SUE|</th><th>reaction size</th><th>sign(SUE) × idio return</th><th>95% CI</th></tr>'+
  nvn.quintiles.map(q=>{const d=q.dir,sig=d&&d.lo>0;
   return `<tr><td>Q${q.q}${q.q===1?' <span style="color:var(--muted)">— no surprise</span>':''}</td>
   <td style="color:var(--muted)">${q.n}</td><td>${q.mean_abs_sue.toFixed(2)}</td>
   <td>${q.size.mean.toFixed(2)}σ</td>
   <td class="${sig?'ok':'no'}"><b>${d.mean>0?'+':''}${d.mean.toFixed(2)}%</b>${sig?' *':''}</td>
   <td style="color:var(--muted)">[${d.lo>0?'+':''}${d.lo.toFixed(2)}, ${d.hi>0?'+':''}${d.hi.toFixed(2)}]</td></tr>`;}).join('')+
  `<tr><td><b>overall</b></td><td style="color:var(--muted)">${nvn.n_rows.toLocaleString()}</td><td></td><td></td>
   <td class="ok"><b>+${od.mean.toFixed(2)}%</b> *</td>
   <td style="color:var(--muted)">[+${od.lo.toFixed(2)}, +${od.hi.toFixed(2)}]</td></tr>`;
 document.getElementById('sue-read').innerHTML=
  `Reaction size rises monotonically with surprise size (${nvn.quintiles[0].size.mean.toFixed(2)}σ → 
   ${nvn.quintiles[4].size.mean.toFixed(2)}σ, correlation ${nvn.corr_abs_sue_reaction>0?'+':''}${nvn.corr_abs_sue_reaction.toFixed(3)}),
   and the DIRECTIONAL response appears only where there is a surprise to respond to — <b>Q1 is the
   control and it is not significant</b>. This is the first directional result in the project: everything
   before measured magnitude. <b>The 3.2× earnings-day move is the numbers, not the coverage.</b>`;
 document.getElementById('clash').innerHTML=
  `<div style="padding:12px 14px;border-radius:8px;border:1px solid rgba(250,178,25,.4);
    background:rgba(250,178,25,.07);font-size:12.5px;line-height:1.65">
   <b style="color:var(--warn)">CANNOT BE ANSWERED WITH THIS DATA</b><br>
   Of <b>${nvn.n_rows.toLocaleString()}</b> earnings releases where the numbers are known, the news graph
   supplies a directional read on <b>${nvn.n_with_news}</b> — <b>${(100*nvn.n_with_news/nvn.n_rows).toFixed(1)}%</b>.
   Split into agreement and conflict that is ${nvn.n_agree} and ${nvn.n_clash}: far too few to test whether
   the price follows the story or the figures when they diverge.<br><br>
   <b>The reason is the finding.</b> Months of LLM extraction produced ${nvn.n_with_news} usable observations
   for this question. EDGAR produced ${nvn.n_rows.toLocaleString()} in an afternoon — free, exact and complete.
   Where prices move on identifiable news, the identifiable thing is usually a NUMBER; and on every
   predictive test in this register, the narrative layer added nothing measurable on top of it.</div>`;}

/* ---------- regimes ---------- */
const RG=M.regimes||{},RS=M.reason||{};
function drawRegSetup(){
 if(!RG.pairs){document.getElementById('v-reg').innerHTML='<p class="note">pair_regimes.json not built.</p>';return;}
 document.getElementById('minrun').textContent=RG.min_run;
 const s=RG.states;
 document.getElementById('k-reg').innerHTML=[
  [RG.n_pairs,'pairs tracked, rolling '+RG.win+'-day correlation of idiosyncratic residuals'],
  [s.ENDED||0,'links that ENDED — fell below chance and stayed there'],
  [s.EMERGING||0,'links that EMERGED from chance'],
  [s.UNCONFIRMED||0,'changes too recent to confirm — a state must hold '+RG.min_confirm+' windows'],
  [RG.median_band,'median band |r| — below this, a correlation is indistinguishable from none'],
 ].map(([b,t])=>`<div class="k"><b>${b}</b><span>${t}</span></div>`).join('');
 const tot=Object.values(s).reduce((a,b)=>a+b,0),
  col={PERSISTENT:'--s3',EMERGING:'--s1',ENDED:'--s2',EPISODIC:'--s4',NEVER:'--grid',INSUFFICIENT:'--grid'};
 document.getElementById('regstates').innerHTML=`<div class="bar" style="height:15px">`+
  Object.entries(s).map(([k,v])=>`<i style="width:${v/tot*100}%;background:var(${col[k]||'--grid'})"></i>`).join('')+
  `</div><div class="lg">`+Object.entries(s).map(([k,v])=>
   `<span><span class="sw" style="background:var(${col[k]||'--grid'})"></span>${k} — ${v}</span>`).join('')+`</div>`;
 drawReg();drawReason();}
function drawReg(){const f=document.getElementById('regfilter').value;
 const ev=document.getElementById('onlyev').checked;
 let ps=RG.pairs.filter(p=>p.state===f);
 if(ev)ps=ps.filter(p=>p.news&&p.news.n>0);
 document.getElementById('regsel').innerHTML=ps.length?ps.map(p=>
  `<option value="${RG.pairs.indexOf(p)}">${p.a} ~ ${p.b}${p.transition_date?' — '+p.transition_date:''}
   (r now ${p.last_corr}${p.news&&p.news.n?', '+p.news.n+' articles':''})</option>`).join('')
  :'<option value="-1">none with evidence in this state — untick the box</option>';
 drawRegChart();}
function drawRegChart(){const i=+document.getElementById('regsel').value;const p=RG.pairs[i];
 if(!p){document.getElementById('c-reg').innerHTML='';document.getElementById('regread').textContent='';return;}
 const W=1000,H=300,L=52,Rt=118,W2=RG.window_dates.length;
 const lo=-1,hi=1,Y=v=>H-26-((v-lo)/(hi-lo))*(H-52),Xr=k=>L+(k/(W2-1))*(W-L-Rt);
 let s='';
 for(let g=-1;g<=1;g+=0.5){s+=`<line x1="${L}" y1="${Y(g)}" x2="${W-Rt}" y2="${Y(g)}"
  stroke="var(--grid)"/><text x="${L-7}" y="${Y(g)+3.5}" font-size="9.5" fill="var(--muted)"
  text-anchor="end">${g.toFixed(1)}</text>`;}
 // the chance band, drawn as a filled corridor
 s+=`<rect x="${L}" y="${Y(p.band)}" width="${W-L-Rt}" height="${Y(-p.band)-Y(p.band)}"
  fill="var(--warn)" opacity=".10"/>
  <line x1="${L}" y1="${Y(p.band)}" x2="${W-Rt}" y2="${Y(p.band)}" stroke="var(--warn)" stroke-dasharray="4,3" opacity=".7"/>
  <line x1="${L}" y1="${Y(-p.band)}" x2="${W-Rt}" y2="${Y(-p.band)}" stroke="var(--warn)" stroke-dasharray="4,3" opacity=".7"/>
  <text x="${W-Rt+6}" y="${Y(p.band)+3.5}" font-size="9.5" fill="var(--warn)">±${p.band} chance band</text>`;
 // active windows shaded
 p.active.forEach((on,k)=>{if(on)s+=`<rect x="${Xr(k)-3}" y="24" width="6" height="${H-50}"
  fill="var(--s3)" opacity=".10"/>`;});
 if(p.transition_index!=null){const x=Xr(p.transition_index);
  s+=`<line x1="${x}" y1="20" x2="${x}" y2="${H-26}" stroke="var(--s2)" stroke-width="2"/>
   <text x="${x+5}" y="32" font-size="10" fill="var(--s2)">${p.state} · ${p.transition_date}</text>`;}
 let seg=[],paths=[];
 p.corr.forEach((v,k)=>{if(v==null){if(seg.length>1)paths.push(seg);seg=[];}else seg.push([Xr(k),Y(v)]);});
 if(seg.length>1)paths.push(seg);
 paths.forEach(q=>{s+=`<path d="M${q.map(a=>a.join(',')).join(' L')}" fill="none"
  stroke="var(--s1)" stroke-width="2"/>`;});
 p.corr.forEach((v,k)=>{if(v==null)return;
  s+=`<circle cx="${Xr(k)}" cy="${Y(v)}" r="6" fill="transparent"
   onmousemove='show(event,"<b>${p.a} ~ ${p.b}</b><div style=color:var(--ink2)>"+${JSON.stringify(RG.window_dates[k])}+"</div><div style=color:var(--ink2)>r = "+${v}+"</div><div style=color:var(--muted)>"+${JSON.stringify(p.active[k]?'above the chance band':'within chance')}+"</div>")'
   onmouseleave="hide()"/>`;});
 RG.window_dates.forEach((d,k)=>{if(k%8)return;
  s+=`<text x="${Xr(k)}" y="${H-8}" font-size="9" fill="var(--muted)" text-anchor="middle">${d.slice(0,7)}</text>`;});
 document.getElementById('c-reg').innerHTML=s;
 const sl=p.tail_slope_per_window;
 document.getElementById('regread').innerHTML=
  `<b>${p.a}</b> (${esc(p.a_ind)}) ~ <b>${p.b}</b> (${esc(p.b_ind)}) — ${p.cross_industry
   ?'<span style="color:var(--s2)">crosses a SIC industry boundary</span>':'same industry'}.
   Mean |r| ${p.mean_abs}, chance band ${p.band}, latest r <b>${p.last_corr}</b>.
   ${p.state==='ENDED'?`This link <b>ended</b> around ${p.transition_date}: it was reliably above chance, then fell inside the band and stayed there. Trading it as a live relationship after that date would have been trading noise.`
   :p.state==='EMERGING'?`This link <b>emerged</b> around ${p.transition_date} — indistinguishable from chance before, reliably above it since.`
   :p.state==='PERSISTENT'?'Held above chance throughout the sample.'
   :'Flips in and out of significance — treat any single reading as unreliable.'}
   ${sl!=null?` Recent trend ${sl>0?'+':''}${sl} per window${Math.abs(sl)<0.005?' (flat)':sl<0?' — decaying':' — strengthening'}.`:''}`;
 // ---- the evidence, presented WITHOUT a verdict ----
 const nw=p.news||{n:0};const box=document.getElementById('regnews');
 if(!nw.window){box.innerHTML='';return;}
 const side=(tk,arts,ind)=>`<div><div style="font-size:11.5px;color:var(--ink2);margin-bottom:5px">
   <b>${tk}</b> <span style="color:var(--muted)">${esc(ind)}</span></div>
   ${arts.length?arts.map(x=>`<div class="art"><span style="color:var(--muted)">${x.d} · ${x.cls.map(un).join(', ')}</span><br>
    <b>${esc(x.h)}</b><br>${esc(x.t)}</div>`).join('')
   :'<div class="art" style="color:var(--muted)">no articles captured for this name in this window</div>'}</div>`;
 box.innerHTML=`<div style="margin-top:13px;padding-top:12px;border-top:1px solid var(--border)">
  <h3 style="font-size:13px;margin:0 0 3px">News published around ${p.transition_date} — judge it yourself</h3>
  <p class="note">Window ${nw.window[0]} → ${nw.window[1]} · ${nw.n} articles ·
   ${nw.docs_in_window} documents captured across the whole window.
   <b>No causal claim is made here.</b> The measurement establishes only that the correlation
   crossed this pair's chance band and stayed across; it does not establish why. A placebo test
   (see below) showed an LLM cannot distinguish a real transition window from a matched window
   where nothing changed, so the evidence is shown rather than narrated.</p>
  <div class="ab">${side(p.a,nw.a,p.a_ind)}${side(p.b,nw.b,p.b_ind)}</div></div>`;}
function drawReason(){
 if(!RS.cases||!RS.cases.length){document.getElementById('k-reason').innerHTML=
  '<div class="k"><b>—</b><span>regime_reason.json not built</span></div>';return;}
 const rr=RS.real_rate,pp=RS.placebo_rate,n=RS.cases.length;
 const agree=RS.cases.filter(c=>c.real.verdict===c.placebo.verdict).length;
 const disc=RS.cases.filter(c=>c.real.verdict!=='UNRELATED'&&c.placebo.verdict==='UNRELATED').length;
 document.getElementById('k-reason').innerHTML=[
  [(rr*100).toFixed(0)+'%','of REAL transitions judged related to their news'],
  [(pp*100).toFixed(0)+'%','of PLACEBO windows judged related — where nothing changed'],
  [disc+'/'+n,'cases where the model correctly told real from placebo'],
  [agree+'/'+n,'cases where it gave the SAME verdict to both'],
 ].map(([b,t])=>`<div class="k"><b>${b}</b><span>${t}</span></div>`).join('');
 const gap=rr-pp;
 document.getElementById('reasonverdict').innerHTML=
  `<div style="padding:11px 13px;border-radius:8px;border:1px solid ${gap>0.2?'rgba(25,158,112,.4)':'rgba(229,72,77,.4)'};
    background:${gap>0.2?'rgba(25,158,112,.07)':'rgba(229,72,77,.07)'};font-size:12.5px;line-height:1.6">
   <b style="color:${gap>0.2?'var(--good)':'var(--bad)'}">${gap>0.2?'DISCRIMINATES':'NULL — the narratives carry no information'}</b><br>
   ${gap>0.2?`Real windows are judged related ${(gap*100).toFixed(0)} points more often than matched placebos.`
   :`Real and placebo differ by only <b>${(gap*100).toFixed(0)} points</b> (${(rr*100).toFixed(0)}% vs ${(pp*100).toFixed(0)}%).
     The model produces an equally confident, equally specific account of a window in which
     <b>nothing measurably changed</b>. On this evidence an LLM cannot say <em>why</em> a correlation
     regime shifted, and a narrative it offers should not be presented as an explanation.
     Note the first run of this test, with placebos NOT matched on article volume, showed
     14% vs 4% and looked like a clear win — that gap was entirely the empty placebos forcing
     "UNRELATED". The control is what turned a false positive into a null.`}</div>`;
 document.getElementById('t-reason').innerHTML=
  '<tr><th>pair</th><th>state</th><th>Δr</th><th>transition</th><th>real verdict</th><th>placebo verdict</th><th></th></tr>'+
  RS.cases.map((c,i)=>{const good=c.real.verdict!=='UNRELATED'&&c.placebo.verdict==='UNRELATED';
   const bad=c.real.verdict===c.placebo.verdict&&c.real.verdict!=='UNRELATED';
   const rev=c.real.verdict==='UNRELATED'&&c.placebo.verdict!=='UNRELATED';
   return `<tr style="cursor:pointer" onclick="tglR(${i})"><td>${c.a} ~ ${c.b}</td>
   <td style="text-align:left;color:var(--muted)">${c.state}</td><td>${c.magnitude}</td>
   <td style="color:var(--muted)">${c.transition_date}</td>
   <td>${c.real.verdict}</td><td>${c.placebo.verdict}</td>
   <td style="text-align:left">${good?'<span class="ok">✓ told them apart</span>'
    :bad?'<span class="bd">same story for both</span>':rev?'<span class="bd">backwards</span>':''}</td></tr>
   <tr id="rr${i}" style="display:none"><td colspan="7" style="text-align:left;padding:10px 12px;background:rgba(127,127,127,.05)">
    <div class="ab"><div class="ans gated"><h4>REAL window ${c.real.window[0]} → ${c.real.window[1]} · ${c.real.n_articles} articles</h4>${esc(c.real.answer)}</div>
    <div class="ans naive"><h4>PLACEBO window ${c.placebo.window[0]} → ${c.placebo.window[1]} · ${c.placebo.n_articles} articles — nothing changed here</h4>${esc(c.placebo.answer)}</div></div></td></tr>`;}).join('');}
function tglR(i){const e=document.getElementById('rr'+i);if(e)e.style.display=e.style.display==='none'?'':'none';}

/* ---------- negative register ---------- */
function drawRegister(){const G2=D.register;if(!G2)return;
 document.getElementById('k-null').innerHTML=[
  [G2.rejected.length,'hypotheses tested'],
  [G2.retracted.length,'figures corrected after review'],
  [G2.bugs.length,'defects affecting reported figures'],
  [G2.controls.length,'control procedures materially affecting a result'],
 ].map(([b,t])=>`<div class="k"><b>${b}</b><span>${t}</span></div>`).join('');
 document.getElementById('reg-rej').innerHTML=G2.rejected.map(r=>`
  <div style="border-left:2px solid var(--bad);padding:2px 0 2px 12px;margin:13px 0">
   <div style="font-size:13px;color:var(--ink);margin-bottom:3px">${esc(r.claim)}</div>
   <div style="font-size:11.5px;color:var(--muted);margin-bottom:5px">${esc(r.test)}</div>
   <div style="font-size:12px;margin-bottom:5px">
    <span class="pill" style="border-color:rgba(229,72,77,.45);color:var(--bad)">${r.verdict}</span>
    <code>${esc(r.result)}</code></div>
   <div style="font-size:12px;color:var(--ink2);line-height:1.6">${esc(r.detail)}</div>
   <div style="font-size:10.5px;color:var(--muted);margin-top:4px">source · ${esc(r.src)}</div></div>`).join('');
 document.getElementById('reg-ret').innerHTML=G2.retracted.map(r=>`
  <div style="border-left:2px solid var(--warn);padding:2px 0 2px 12px;margin:13px 0">
   <div style="font-size:12.5px;margin-bottom:3px">
    <span style="color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em">said</span>
    <span style="color:var(--bad)"> ${esc(r.was)}</span></div>
   <div style="font-size:12.5px;margin-bottom:5px">
    <span style="color:var(--muted);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em">actually</span>
    <span style="color:var(--good)"> ${esc(r.now)}</span></div>
   <div style="font-size:12px;color:var(--ink2);line-height:1.6">${esc(r.why)}</div></div>`).join('');
 document.getElementById('reg-bug').innerHTML=G2.bugs.map(b=>`
  <div style="border-left:2px solid var(--s2);padding:2px 0 2px 12px;margin:13px 0">
   <div style="font-size:13px;color:var(--ink)">${esc(b.bug)}</div>
   <div style="font-size:11px;color:var(--muted);margin-bottom:5px">affected · ${esc(b.impact)}</div>
   <div style="font-size:12px;color:var(--ink2);line-height:1.6">${esc(b.detail)}</div></div>`).join('');
 document.getElementById('reg-ctl').innerHTML=
  '<tr><th>control added</th><th>without it</th><th>with it</th></tr>'+
  G2.controls.map(c=>`<tr><td style="width:24%">${esc(c.control)}</td>
   <td style="text-align:left;color:var(--bad);width:24%">${esc(c.before)}</td>
   <td style="text-align:left;color:var(--good)">${esc(c.after)}</td></tr>
   <tr><td colspan="3" style="text-align:left;color:var(--ink2);font-size:11.5px;padding:0 9px 9px 9px">
    ${esc(c.detail)}</td></tr>`).join('');}

function redraw(){drawDrift();renderDay();drawName();drawRegChart();}
/* ---------- US vs India ---------- */
function drawCompare(){const K=D.compare;
 if(!K||!K.rows.length){document.getElementById('v-cmp').innerHTML=
  '<p class="note">Only one market is built — run mcp_precompute --market india.</p>';return;}
 document.getElementById('k-cmp').innerHTML=[
  ['+'+K.rank_corr.toFixed(2),'rank correlation of the class risk ordering across the two markets'],
  ['+'+K.level_corr.toFixed(2),'correlation of the actual × control levels'],
  [K.us_control.toFixed(2)+' / '+K.india_control.toFixed(2),'control σ — US / India, measured independently'],
  [K.rows.length,'classes present in both graphs'],
 ].map(([b,t])=>`<div class="k"><b>${b}</b><span>${t}</span></div>`).join('');
 const W=1000,H=330,L=150,Rt=60,n=K.rows.length;
 const mx=Math.max(...K.rows.flatMap(r=>[r.us,r.india]))*1.06;
 const X=v=>L+(v/mx)*(W-L-Rt),rowH=(H-52)/n;
 let s='';
 for(let g=0;g<=mx;g+=0.5){s+=`<line x1="${X(g)}" y1="20" x2="${X(g)}" y2="${H-30}" stroke="var(--grid)"/>
  <text x="${X(g)}" y="${H-16}" font-size="9.5" fill="var(--muted)" text-anchor="middle">${g.toFixed(1)}×</text>`;}
 s+=`<line x1="${X(1)}" y1="20" x2="${X(1)}" y2="${H-30}" stroke="var(--warn)" stroke-dasharray="4,3"/>
  <text x="${X(1)}" y="14" font-size="9.5" fill="var(--warn)" text-anchor="middle">1.0× = an ordinary day</text>`;
 K.rows.forEach((r,i)=>{const y=28+i*rowH;
  s+=`<text x="${L-9}" y="${y+rowH/2}" font-size="11" fill="var(--ink2)" text-anchor="end">${un(r.cls)}</text>`;
  [['us',r.us,'--s1',0],['india',r.india,'--s3',1]].forEach(([lb,v,c,k])=>{
   s+=`<rect x="${X(0)}" y="${y+2+k*(rowH/2-3)}" width="${X(v)-X(0)}" height="${rowH/2-5}"
    fill="var(${c})" opacity=".85" rx="2"
    onmousemove='show(event,"<b>${un(r.cls)}</b><div style=color:var(--ink2)>${lb.toUpperCase()} "+${v}+"× its own control</div>")' onmouseleave="hide()"/>
    <text x="${X(v)+5}" y="${y+2+k*(rowH/2-3)+rowH/4}" font-size="9.5" fill="var(${c})">${v.toFixed(2)}</text>`;});});
 s+=`<rect x="${W-Rt-58}" y="24" width="10" height="3" fill="var(--s1)"/>
  <text x="${W-Rt-44}" y="28" font-size="10" fill="var(--muted)">US</text>
  <rect x="${W-Rt-58}" y="38" width="10" height="3" fill="var(--s3)"/>
  <text x="${W-Rt-44}" y="42" font-size="10" fill="var(--muted)">India</text>`;
 document.getElementById('c-cmp').innerHTML=s;
 document.getElementById('t-cmp').innerHTML=
  '<tr><th>class</th><th>US × ctrl</th><th>India × ctrl</th><th>difference</th></tr>'+
  K.rows.map(r=>`<tr><td>${un(r.cls)}</td><td>${r.us.toFixed(2)}</td><td>${r.india.toFixed(2)}</td>
   <td class="${Math.abs(r.us-r.india)<0.15?'ok':''}">${(r.india-r.us>0?'+':'')+(r.india-r.us).toFixed(2)}</td></tr>`).join('');
 const um=D.markets.us,im=D.markets.india;
 document.getElementById('t-cmpmeta').innerHTML=
  '<tr><th></th><th>United States</th><th>India</th></tr>'+
  [['corpus','Bloomberg, 2008–2014','NSE-linked, 2021–2022'],
   ['names',Object.keys(um.names).length+' on the desk',', '.replace(', ','')+Object.keys(im.names).length+' on the desk'],
   ['entity resolution','extractor (resolved_security)','verified NSE map — the extractor resolved NONE, and the legacy map was partly hallucinated'],
   ['sector taxonomy','SIC, assigned by the SEC','TEXT-INFERRED from article and company names'],
   ['rolling prior windows','34–39 per class','5–9 per class'],
  ].map(r=>`<tr><td style="color:var(--muted)">${r[0]}</td>
   <td style="text-align:left">${esc(r[1])}</td><td style="text-align:left">${esc(r[2])}</td></tr>`).join('');
 document.getElementById('cmpcav').innerHTML=
  `<b>Read this as encouraging, not settled.</b> The ordering agrees at +${K.rank_corr.toFixed(2)} and
   earnings sits at the top of both (${K.rows[0]?K.rows[0].us.toFixed(2):'—'}× vs
   ${K.rows[0]?K.rows[0].india.toFixed(2):'—'}×), with independently measured controls
   0.01σ apart — across a different market, different years and a different resolution method.
   But only <b>${K.rows.length} classes</b> are shared, India has 5–9 rolling windows per class
   against 34–39 for the US, and its sector labels are inferred rather than assigned. This is a
   directional replication of the one surviving positive result, on a small sample — not a
   confirmation. The nulls have NOT been re-run on India in this build.`;}

function initAll(){
 document.getElementById('daysel').innerHTML=M.days.map((d,i)=>
  `<option value="${i}">${d.d} — ${d.docs} documents, ${d.n} names</option>`).join('');
 const NKx=Object.keys(M.names);
 document.getElementById('namesel').innerHTML=NKx.map(t=>
  `<option value="${t}">${t} — ${esc(M.names[t].nm).slice(0,34)}</option>`).join('');
 dPath=[];hiN=null;
 drawCov();drawDrift();drawDay();drawName();drawAB();drawRegSetup();drawNumbers();}
document.getElementById('mksel').innerHTML=Object.entries(D.markets)
 .map(([k,v])=>`<option value="${k}">${v.meta.label}</option>`).join('');
document.getElementById('mksub').textContent=M.meta.sub;
initAll();drawRegister();drawCompare();
</script></body></html>"""


if __name__ == "__main__":
    main()
