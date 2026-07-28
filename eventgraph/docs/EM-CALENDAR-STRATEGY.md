# EM event-calendar strategy — building a calendar that beats the vendors

**Status:** proposed · 2026-07-26 · architecture reviewed by Fable
**Scope:** emerging & frontier markets an EM macro book actually trades — Sub-Saharan
Africa, MENA/Gulf, Turkey, frontier Asia, LatAm. Sovereign rates / FX / credit, not
DM equity earnings.

---

## 0. The thesis, and the ambition it licenses

The one calendar result this repo has *measured* is a **measurement** result:
`scripts/earnings_calendar.py` + `scripts/calendar_window_profile.py` showed that dating
events by publication timestamp instead of by the first tradeable session understated an
effect **2.2× → 3.2×** (49% of SEC Item 2.02 8-Ks land after the close). Meanwhile
`docs/FINDINGS-edge-hunt-2026-07.md` is a graveyard of directional edges that died
out-of-sample.

So the honest claim for this layer is **not** "the calendar predicts the market". It is:

> A bitemporal, session-aligned, self-validating event-knowledge asset whose payoffs are
> (1) correct measurement, (2) risk/de-diversification timing, and (3) a proprietary
> schedule-revision history that nobody else keeps and that **cannot be bought or
> backfilled**.

Point (3) is the one with a clock on it. Vendors overwrite dates in place; the Wayback
Machine covers a fraction of EM stats-office pages. Every week without a snapshotter is a
week of EM schedule-revision history permanently lost. That single fact should determine
the build order.

---

## 1. Architecture

### 1.1 Plane placement — still three planes, but redraw the boundary

Keep narrative / formal / realised. Amend the doctrine: `README.md` defines the formal
plane by **source type** ("external feeds"), which breaks immediately in EM, where the raw
material for "when does Nigeria's NBS publish CPI" is a PDF advance-release calendar, a
BusinessDay article, and a CBN post. Those are narrative *carriers* of formal *claims*.

**The formal plane is defined by epistemic role: claims about scheduled occurrences and
their outcomes, whatever the carrier.** Estimated and predicted dates don't break it —
they are *derived* rows, exactly the role `realised_*` already plays for the realised plane.

### 1.2 The defect in the current model

`lake/0007_lineage.sql` supersedes by `(series_id, event_time)` with "latest `ingested_at`
wins". Three problems, the first fatal:

1. **The key is wrong.** In EM, `event_time` *is the thing that changes*. When Nigeria CPI
   moves 15 Jul → 22 Jul, that is two rows under two different keys — no supersession
   fires, the abandoned date survives as "current", and `v_event_surprise` (0009/0010) can
   join a narrative event to a date that never happened. **The current model cannot
   represent a reschedule** — the single most important EM phenomenon — except as two
   unrelated events.
2. **`ingested_at` conflates fetch time with knowledge time.** Fine for live accrual; for
   any archival backfill everything stamps 2026 and "as of D" queries degenerate. ALFRED
   already gets this right (`realtime_start` = publication date); nothing else does.
3. **No source precedence.** A re-fetched Google-News mention supersedes yesterday's
   central-bank primary document.

### 1.3 The fix: the observation is the fact

Identity moves to the **occurrence**: `(series_id, period_ref)` — `NG-CPI:2026-06`,
`EG-MPC:2026-08`, `KE-TBILL-91:2026-W31`. `event_time` is demoted from key to *attribute
under belief revision*.

One new append-only DuckLake table becomes the formal-plane primitive:

```sql
CREATE TABLE eg.calendar_observation (
  -- occurrence identity
  series_id        VARCHAR,   -- canonical (event_series)
  period_ref       VARCHAR,   -- '2026-06' | 'MPC-2026-04' | auction id
  -- the claim
  event_time       TIMESTAMP, -- UTC, nullable
  event_time_local VARCHAR,   -- VERBATIM local literal ('23 Rajab 1448, 10:00 WAT')
  time_precision   VARCHAR,   -- exact | date | week | month | quarter | unknown
  status           VARCHAR,   -- scheduled | confirmed | postponed | cancelled | occurred
  expected DOUBLE, actual DOUBLE, prior DOUBLE, unit VARCHAR,
  -- epistemics: the bitemporal triple + authority
  source VARCHAR, source_ref VARCHAR,
  source_tier      VARCHAR,   -- primary | secondary | narrative | model
  method           VARCHAR,   -- ics | alfred | edgar | pdf:<sha> | llm:<model> | pattern:<v>
  observed_at      TIMESTAMP, -- when the source asserted it
  knowable_at      TIMESTAMP, -- earliest public availability — the PIT axis
  ingested_at      TIMESTAMP, -- when we fetched — the reproducibility axis
  doc_id VARCHAR, chunk_id VARCHAR, quote VARCHAR   -- narrative bridge
);
```

This **strengthens** "never UPDATE a fact row" rather than breaking it: an observation
("on 2026-07-10 the CBN site said the MPC meets 2026-07-22") stays true forever, even
after the meeting moves. `eg.calendar_event` survives as a compatibility **view** — the
same trick 0009 used for pre-0007 rows.

Two queries, both one-liners:

| Question | Query |
|---|---|
| *When is it, as known on D?* | latest observation per `(series_id, period_ref)` with `knowable_at <= D`, ranked by `(tier precedence, reliability, knowable_at DESC)` → `v_calendar_asof(D)` |
| *How did belief move?* | full observation history → `v_schedule_revision` (self-join of consecutive observations where `event_time` or `status` changed) |

**Date-change becomes a queryable fact table for free, by construction.** That is the
proprietary asset, and it only exists if we start recording now.

`doc_id/chunk_id/quote` mean a claim extracted from a scraped Ghanaian gazette PDF flows
through `eg.v_provenance` (0008) with zero new machinery.

### 1.4 Confidence: status × tier, **not** a distribution primitive

Rejected: making the primitive a distribution over datetimes. Sources assert *dates*, not
distributions; storing anything else launders model output into the evidence layer, and
every downstream join (`v_event_surprise` ±36h, `series_alias`) would become a convolution.

Also rejected: a single status lattice. "announced/confirmed/postponed/occurred" is event
**lifecycle**; "inferred-from-pattern/predicted" is **who says so**. Two orthogonal
columns — a pattern-inferred date is a `scheduled` claim at `source_tier='model'`.

Distributions belong in a derived sibling table:

```sql
CREATE TABLE eg.calendar_estimate (
  series_id VARCHAR, period_ref VARCHAR, as_of TIMESTAMP,
  p10_time TIMESTAMP, p50_time TIMESTAMP, p90_time TIMESTAMP,
  p_within_5d DOUBLE, model_ref VARCHAR
);
```

For "when-ready" statistics offices the sufficient model is the **empirical publication-lag
distribution per series** ("NBS publishes CPI at median M+14, IQR 12–17") — a byproduct of
the observation store. No ML.

### 1.5 Session assignment as a first-class table

The `±36h` tolerance in `v_event_surprise` is calibrated to US 08:30 ET prints. It is wrong
for date-precision EM claims and wrong for Sun–Thu venues. Replace with precision-aware
joining, and promote the session mapping — today inlined ad hoc as `next_session()` in
`calendar_window_profile.py` — into a derived table:

```sql
eg.event_session (series_id, period_ref, first_tradeable_session, venue, rule_ref)
```

computed from `(event_time, time_precision, venue_tz, venue_calendar)`.

| precision | join behaviour |
|---|---|
| `exact` | session assignment (the 2.2×→3.2× rule, generalised) |
| `date` | venue trading session(s) for that date |
| `week`/`month`/`quarter` | **must not join for surprise** — anticipation-window analytics only |

This institutionalises the repo's own measured lesson instead of re-implementing it per
script.

---

## 2. Where the edge actually is — ranked honestly

| # | Claimed advantage | Verdict |
|---|---|---|
| **1** | **Session alignment / exact timestamping** | **Real, already measured here (3.2× vs 2.2×), cheap.** Vendors publish local dates with placeholder times for frontier venues. We publish *first tradeable session, per venue calendar*. |
| **2** | **Date-change detection** | **Real and genuinely proprietary.** Vendors overwrite in place; revision history exists only if someone snapshots, and it cannot be backfilled. Whether delay-as-signal *pays* is unproven (Argentina/INDEC 2007–15 is the canonical prior); the asset has option value regardless. |
| **3** | **Frontier coverage** | **Real but an ops grind, with a capacity caveat.** For Turkey/SA/Egypt/Saudi the vendors are better than the pitch assumes. The genuinely uncovered tier — Zambia/Angola/Ethiopia auctions, FX auctions, IMF board dates — is exactly where liquidity is thinnest. F12's lesson cuts both ways. |
| **4** | **Self-validation vs realised reaction** | **Real as machinery, not as edge.** A quality flywheel vendors structurally don't run; it is what makes 1–3 trustworthy. |
| 5 | Source fidelity (primary vs aggregator) | Marginal standalone — an *input* to 1–3, not an edge. |
| 6 | Probabilistic prediction of unannounced dates | **Mostly fantasy as an edge.** Local desks already carry lag intuition. Build the free empirical-lag version; nothing fancier. |

---

## 3. The source ladder — doctrine for any EM

Every observation carries a tier, and tier precedence is a **total order** that reliability
weights can reorder only *within* a tier, never across it.

| tier | what qualifies | invariant enforced |
|---|---|---|
| `primary` | the institution's own publication: CB MPC calendar, DMO issuance calendar, stats-office ARC, auction results notice, IMF press release | must have a fetchable content-addressed raw artifact |
| `secondary` | exchange/regulator/multilateral republication: IMF DSBB/NSDP, EU-mandated CRA sovereign review calendars, OPEC, IFES ElectionGuide | same artifact requirement |
| `narrative` | local/regional press, wire, CB social post — already flowing through `feeds_em.json` | must carry `doc_id/chunk_id/quote` |
| `model` | pattern inference, empirical-lag estimate | must carry `model_ref`; **can never outrank an observed row** |

### 3.1 Cross-country sources worth wiring once, that pay everywhere

These are the highest-leverage items because a single adapter covers dozens of countries:

- **IMF SDDS / e-GDDS Advance Release Calendars** (Dissemination Standards Bulletin Board,
  National Summary Data Pages). Subscribing countries commit to publishing forward release
  dates for CPI/GDP/reserves/fiscal. Large African/frontier coverage on a common template.
  *This is the single biggest under-exploited free source for EM release dates.*
- **EU CRA Regulation sovereign review calendars.** S&P/Moody's/Fitch must publish, each
  December, the following year's sovereign review dates for EU-registered entities —
  including EM sovereigns — with actions released after EU market close. Free, exact,
  ex-ante, and routinely under-used.
- **IMF programme milestones** — Executive Board calendar + press releases; staff-level
  agreement → board date is a well-behaved empirical lag (a legitimate `model`-tier row).
- **OPEC/OPEC+ meeting calendar** — published, exact.
- **IFES ElectionGuide + national electoral commissions** — election dates and changes.
- **Central-bank annual MPC calendars** — near-universal now (SARB, CBE, TCMB, CBK, BoG,
  CBN, BoZ, SBP…), *and* frequently amended, which is precisely the signal in §2.2.
- **DMO / treasury issuance calendars** — quarterly or monthly forward auction schedules
  (Nigeria DMO, Ghana MoF, Kenya CBK weekly notices, Egypt MoF, Turkey Treasury's
  three-month domestic borrowing strategy).

> Every URL above must be **live-verified at onboarding**, not trusted from this document.
> The `ics` mode in `scripts/formal_calendar.py` already proves the pattern: it was
> live-verified against BLS/BEA and needed a real User-Agent.

### 3.2 Existing asset to exploit immediately

The narrative extractor **already emits** `eg.event` rows with `scheduled BOOLEAN` and
future `event_time` (`lake/0002_facts.sql`), and `feeds_em.json` (em-v3, 59 feeds) already
scrapes the regional press. A thin bridge routing *scheduled-future* narrative events
through `series_alias` into `calendar_observation` at `tier='narrative'` seeds an EM
calendar for near-zero marginal cost — the local press announces MPC dates and auction
calendars constantly.

**Gap to close:** the current pack is Asia/LatAm-heavy — `sea` 10, `latam` 9+2+1,
`india` 6, versus `africa` 5, `africa_za` 2, `mena` 3. For a sovereign-macro book weighted
to Africa and MENA that distribution is inverted. Africa/MENA feed expansion is a
prerequisite, not a nice-to-have.

---

## 4. The country onboarding runbook (the "robust process")

Onboarding is a **manifest, not a script**. One JSON per venue, mirroring the
`config/feeds_em.json` style, at `config/venues/<iso2>.json`:

```jsonc
{
  "iso2": "NG", "venue": "NG",
  "tz": "Africa/Lagos",              // IANA, never a fixed offset
  "trading_days": ["Mon","Tue","Wed","Thu","Fri"],
  "market_close_local": "14:30",     // per instrument class where they differ
  "holiday_source": { "tier": "primary", "ref": "<CBN/NGX holiday circular>" },
  "calendar_system": "gregorian",    // + "hijri" for moveable religious holidays
  "instruments": { "fx": "USDNGN", "curve": "NGN-OMO", "credit": "NGERIA-EUR" },
  "series": [
    { "series_id": "NG-CPI", "authority": "NBS", "class": "cpi",
      "schedule_regime": "arc",       // arc | fixed_calendar | when_ready | rolling
      "expected_lag_days": 15,
      "sources": [
        { "tier": "primary",   "kind": "pdf",  "ref": "<NBS advance release calendar>" },
        { "tier": "secondary", "kind": "nsdp", "ref": "<IMF e-GDDS NSDP page>" },
        { "tier": "narrative", "kind": "feed", "ref": "feeds_em.json:<name>" }
      ] }
  ]
}
```

**The eight steps, in order, per country:**

1. **Venue mechanics first.** IANA tz, trading days, close time, holiday authority,
   settlement convention. Sun–Thu venues (Egypt EGX, Saudi Tadawul, Qatar) and the Gulf
   working-week changes of recent years are a config problem, not a code problem — but they
   silently corrupt every event study if wrong.
2. **Calendar systems.** Flag Hijri (Gulf religious holidays, moon-sighting dependent →
   genuinely *uncertain* dates, model them as `time_precision='date'` with a ±1d estimate),
   Persian (Iran), Ethiopian (Ethiopia's fiscal year). Never write the conversion math —
   ICU-class library only.
3. **Enumerate the event classes that move the book**, not everything that exists:
   MPC · CPI · GDP · reserves/BoP · T-bill & bond auctions *and results* · FX
   auctions/interventions/devaluations · IMF reviews & board dates · rating reviews ·
   eurobond coupons/maturities · budget presentation · elections.
4. **Classify each series' schedule regime** — `arc` (advance release calendar exists),
   `fixed_calendar` (annual MPC list), `when_ready` (no ex-ante date at all), `rolling`
   (weekly auction pattern). *The regime determines the source strategy and what "beating
   the vendor" even means.*
5. **Build the source ladder per series**, top tier first, and live-verify every endpoint.
6. **Register `event_series` dims in Postgres** with `venue_tz`, `venue_calendar`,
   `calendar_system`, `authority_entity`, `typical_lag_days`.
7. **Turn on the snapshotter.** Hourly poll, content-addressed raw artifact retention.
   *This starts the revision clock — do it before extraction quality is good.*
8. **Backfill outcomes only, never belief history.** Actuals from authoritative archives
   where they exist; revision history starts at t₀ = now. Pretend-backfill makes
   `knowable_at` a fiction and poisons every PIT backtest downstream.

**Definition of done for a country:** every book-moving series has ≥1 `primary`-tier source
or an explicit `when_ready` classification with an empirical-lag estimate; the venue
calendar reproduces the last 12 months of local trading days; and `v_calendar_asof` answers
correctly for a hand-checked sample of ten past events.

---

## 5. Ingest for hostile sources

Reuse the `src/scrape/` activity-workflow shape exactly — idempotent retryable activities
(`fetch_feed`/`resolve_url`/`fetch_and_extract`/`write_partitions`), rayon runner, Temporal
documented as the durable drop-in. Add acquisition activities per source class, under one
hard rule:

> **Every fetch stores the raw artifact content-addressed** (PDF bytes, ICS text, HTML,
> social JSON). Silent upstream mutation *is the phenomenon we are capturing*, so the raw
> snapshot is not hygiene — it is the product. `method='pdf:<sha256>'` makes every claim
> re-verifiable forever.

Extraction is two-stage with a strict boundary:

- **Deterministic first** — ICS (parser exists), agency HTML tables, auction-result formats
  (stable per agency), API endpoints. Emits `tier='primary'` directly.
- **LLM as transcriber, never as judge.** For PDFs, gazettes, local-language press and CB
  social posts, the contract is the one already validated on the Bloomberg batch: verbatim,
  chunk-anchored extraction with a grounding gate — *"this document states the MPC meets on
  ⟨literal string⟩"* — plus the quote span. **The LLM must emit the local literal.**
  Deterministic code does Hijri/Persian/Ethiopian conversion, tz resolution and DST.
- **Designed-out LLM liabilities:** silent date/tz normalisation (confidently wrong),
  status inference, resolving "next Tuesday". A deterministic **verifier** recomputes
  `event_time` from `event_time_local`; anything unverifiable degrades to
  `time_precision='date'` or worse. **Degrade precision, never guess.**

---

## 6. Self-validation without circularity

A derived table scores whether the market reacted where we said the event was:

```sql
eg.calendar_validation (series_id, period_ref, claimed_session, source, source_tier,
                        react_z, vol_z, volume_z, neighbor_max_z,
                        event_density, liquidity_ok, verdict)
```

For each occurrence: does the mapped instrument's abnormal move/vol/volume **concentrate**
in the claimed first-tradeable session versus the ±3 neighbours? Aggregate per
`(source, series)` into a reliability score feeding tier-internal precedence.

Four architectural anti-circularity rules — architectural, not procedural:

1. **Reliability never moves a date.** It reweights *source precedence within a tier*. If
   the market reacted Tuesday and the source said Wednesday, we flag the source; we do not
   relabel the event. The moment realised reaction can edit the calendar, every downstream
   event study is fit-to-target.
2. **PIT scoring.** Backtests read `v_calendar_asof(D)` with reliability computed only from
   occurrences resolved before D; reliability snapshots are append-only with `scored_asof`.
3. **Condition on observability, and abstain rather than fail.** Three confounds this repo
   has already been burned by: stale/administered prices (the 17%-stale-ADR trap — worse in
   EM, where official FX fixes and pegs are *designed* not to move), coverage sparsity
   (always condition on docs/day), and event density (OPEC and a Fed meeting on one day).
   `liquidity_ok=false` or `event_density>1` → **abstain**, never "wrong".
4. **Reaction on the schedule-change announcement day is not a validation failure** — it is
   signal (§2.2) confirming itself. Separate window, separate table.

### Benchmarking against the vendors

Direct scraping of Investing.com/ForexFactory stays out (the existing doctrine in
`formal_calendar.py` is right on both timestamp reliability and ToS). So the primary
benchmark is **vendor-independent**:

| metric | definition |
|---|---|
| **Lead time** | days between our first `knowable_at` and the event — the ex-ante usefulness measure |
| **Coverage** | % of book-moving series with a forward-dated occurrence at horizon H |
| **Accuracy** | \|final claimed − occurred\| in sessions |
| **Revision capture** | reschedules detected / reschedules that occurred |
| **Session-alignment correctness** | reaction concentration in the claimed session (§6) |

A small manual head-to-head audit against publicly visible vendor dates (N≈50, spot check,
not a pipeline) is legitimate for calibration. It is a sanity check, not the benchmark.

---

## 7. What NOT to build

- **A datetime-distribution engine.** Empirical lag quantiles per series; stop.
- **Consensus scraping.** Keep the existing doctrine. Accept the consequence already
  conceded in `formal_calendar.py`: `expected` stays mostly NULL and surprise stays a
  random-walk baseline. The better EM baseline is *market-implied* (NDF/FRA repricing into
  the event) — realised-plane work, noted not built.
- **In-house non-Gregorian calendar math.**
- **Real-time/streaming social ingestion.** These are *scheduled* events; hourly polling
  suffices. Latency is a different product.
- **Deep historical backfill of belief history.** Mostly unrecoverable; pretend-backfill
  poisons PIT. Backfill outcomes only.
- **Eurobond restructuring milestones as structured rows.** Contested, lawyer-driven,
  narrative-shaped — leave in the narrative plane until a milestone is officially dated.
- **A vendor-style calendar product/UI.** This is a research data asset; productisation is
  a later, separate decision.

---

## 8. Research this unlocks inside eventgraph

Each is a testable hypothesis that our calendar can answer and a vendor calendar cannot:

0. ✅ **TESTED — are the under-exploited events material at all?**
   `scripts/em_jump_hazard.py`, full write-up in `docs/FINDINGS-em-calendar-2026-07.md`.
   IMF programme approvals carry a **4.07× jump hazard** in frontier FX against each
   event's own ±90-day local baseline (p = 0.002), stable across two disjoint regimes
   (4.20× / 3.97×) and spread over six countries. **But direction is 60/40 (p = 0.181)** —
   materiality is proven, a directional edge is not. This is a risk-timing layer, not
   alpha, which is what the architecture review predicted.

1. **Publication delay as sovereign-stress signal.** Does a stats office slipping its ARC
   date predict a bad print, a rating action, or spread widening? *Only answerable with
   revision history* — accrues forward, cannot be backfilled. The flagship question.
2. **EM event-conditioned de-diversification.** `scripts/event_regime_correlation.py`
   found rate-cluster × FOMC strong (Δ+0.13, 98% OOS) where the broad universe was null.
   The EM analogue — CBE decision day × Egyptian bank/EGP cluster; CBN MPC × NGN complex —
   is the same measurement on a mechanism where the fund actually has positions.
3. **Session-alignment generalisation.** The 2.2×→3.2× result should replicate wherever
   local prints land after the local close or into a Gulf weekend. Re-run
   `calendar_window_profile.py` per venue; it is the cheapest validation of the whole layer.
4. **Auction-cycle risk timing.** Weekly T-bill auctions are the densest exogenous, exactly
   dated event set in frontier markets — a far better test bed than sparse MPCs.
5. **Coverage-conditioned honesty throughout.** The docs/day lesson applies unchanged: on
   deep-coverage days 27% of >4σ moves were explained vs 2.3% pooled. Never report a pooled
   EM number without conditioning on coverage.

---

## 9. Build order

Sequenced by *what is destroyed by waiting*, not by what is most interesting.

| phase | content | why here |
|---|---|---|
| **0 — start the clock** ✅ **SHIPPED 2026-07-26** | `lake/0012_calendar_observation.sql` + `scripts/em_calendar.py` + `config/em_calendar_sources.json`; 23 sources live-probed, 20 fetching, raw content-addressed | Revision history is the only unrecoverable asset. Ship this minimally *this month*; extraction quality can be bad. |
| **1 — make joins honest** | `event_session` + venue manifests + precision-aware joins; retire the ±36h constant and per-script `next_session()` | Unblocks every measurement; low risk |
| **2 — hostile sources** | PDF/gazette/local-language lane, LLM-transcriber + deterministic verifier; narrative→`calendar_observation` bridge; Africa/MENA feed expansion | Where frontier coverage is actually won |
| **3 — flywheel** | `calendar_validation` + source reliability + PIT snapshots | Needs 1–2 months of accrued observations to be meaningful |
| **4 — research** | §8 hypotheses, delay-as-stress first | Needs the revision history phase 0 started accruing |

## 9a. Phase 0 as built (2026-07-26)

`config/em_calendar_sources.json` (23 sources, every one live-probed) →
`scripts/em_calendar.py` (`snapshot` / `parse` / `report` / `revisions`) →
`lake/0012_calendar_observation.sql`.

First run: **20/23 sources fetched**, 23 raw artifacts content-addressed (3.0 MB),
**180 observations across 180 occurrences** from two parsers — KNBS's Advance
Release Calendar (142) and IFES ElectionGuide (38). Verified in DuckDB: the
as-of selector returns **0 rows** for 2026-07-01, i.e. before our first fetch
nothing was knowable — point-in-time integrity holds by construction rather than
by convention.

Three things the build itself established:

1. **A local-date column is not optional.** Storing only the UTC instant moved
   every date-precision claim back a day for venues east of Greenwich — KNBS's
   `31-Jul-2025` rendered as `2025-07-30`, Zambia's 13 Aug election as 12 Aug.
   Caught in our own first output; `event_date_local` is now authoritative for
   `precision='date'`, and the coverage view keys off it.
2. **Reference month is not an occurrence key.** Two distinct KNBS publications
   (Leading Economic Indicators and Economic Survey 2026) deliver the same
   product for the same reference month; keying on the month alone collided and
   silently dropped one release from current belief. `period_ref` now carries a
   publication slug.
3. **17 of 23 sources are snapshot-only, and that is the expected shape.** The
   hostile-source problem is real and concentrated: Sitecore/JS SPAs (CBE Egypt,
   TurkStat, Fitch, Moody's, Bank of Zambia), WAF/403s (S&P, IMF board calendar,
   OPEC intermittently), and PDF-only publication (Nigeria DMO issuance
   calendar). They are registered with the blocker documented and their raw
   bytes captured every run, so the revision clock is already running on them
   while parsing catches up.

**First substantive finding, from the data rather than the design.** Kenya's
newest published Advance Release Calendar is the **FY2025-26** edition, ending
30 June 2026 — re-uploaded in July 2026 but not extended. As of 2026-07-26 the
country has *no forward advance release calendar at all*, so every Kenyan
release date on a vendor calendar right now is inferred, not published. The
`report` command flags this automatically (a series whose newest entry is in the
past is not a calendar), which is precisely the kind of gap a vendor's
confirmed/estimated flag papers over.

## 10. Decisions needing an ADR

Convention: `docs/adr/NNNN-title.md`, `Status: accepted · date` (see
`../docs/adr/0001-market-color-app.md`). `eventgraph/docs/adr/` does not exist yet.

1. **Occurrence identity** — `(series_id, period_ref)` as the key; `event_time` demoted to
   a belief attribute. Breaking change to 0009/0010 join semantics; everything hangs on it.
2. **`calendar_observation` as the formal-plane primitive** — bitemporal triple, `status` ×
   `source_tier` as orthogonal axes, `calendar_event` demoted to a view, "never UPDATE a
   fact" re-grounded as "the fact is the observation".
3. **Session assignment as a first-class derived mapping** — retires `±36h` and the
   per-script session logic.
4. **The authoritative-vs-inferred boundary** — tier precedence, content-addressed artifact
   retention, LLM-as-transcriber contract (verbatim local literal + verifier + precision
   degradation).
5. **Self-validation without circularity** — reliability reweights source precedence only,
   never dates; PIT reliability snapshots; abstain on unobservable instruments.
