-- 0012 — the calendar OBSERVATION store: the formal plane's bitemporal primitive.
--
-- WHY THIS EXISTS. 0007's supersession rule ("latest ingested_at per
-- (series_id, event_time)") cannot represent the one thing that matters most in
-- EM: a RESCHEDULE. When Nigeria CPI moves 15 Jul -> 22 Jul the two claims land
-- under two different keys, no supersession fires, and the abandoned date
-- survives as "current" -- so v_event_surprise can join a narrative event to a
-- date that never happened.
--
-- The fix is to move identity onto the OCCURRENCE and demote event_time from key
-- to a belief attribute:
--
--     occurrence  = (series_id, period_ref)     -- 'NG-CPI'/'2026-06', 'EG-MPC'/'2026-08'
--     observation = what some source asserted about it, at some time
--
-- This STRENGTHENS "we never UPDATE a fact row" rather than breaking it: an
-- observation ("on 2026-07-10 the CBN site said the MPC meets 2026-07-22") stays
-- true forever, even after the meeting moves. The reschedule is then a queryable
-- fact (v_schedule_revision) instead of something supersession destroys.
--
-- THREE TIME AXES, deliberately distinct (0007 conflated the last two):
--   observed_at  when the source asserted it (document publication date)
--   knowable_at  earliest instant the information was publicly available -- THE
--                POINT-IN-TIME AXIS. For a live snapshot this equals fetch time
--                (we cannot honestly claim to have known it earlier); for an
--                archival source it is the archive's own publication date, the
--                same discipline ALFRED's realtime_start already gives us.
--   ingested_at  when we fetched -- the reproducibility axis.
--
-- STATUS x TIER ARE ORTHOGONAL. status is event LIFECYCLE (scheduled/confirmed/
-- postponed/cancelled/occurred); source_tier is WHO SAYS SO (primary/secondary/
-- narrative/model). A pattern-inferred date is a 'scheduled' claim at tier
-- 'model' -- never a status of its own. Tier precedence is a TOTAL ORDER that
-- reliability weights may reorder only WITHIN a tier, never across it.
--
-- WHY event_date_local EXISTS AND WHY IT IS NOT REDUNDANT. Most EM sources
-- publish a DATE with no time ("31-Jul-2025"). Placing that on the timeline as
-- local midnight and storing the UTC instant silently moves the calendar date
-- backwards for every venue east of Greenwich: Nairobi midnight on 31 Jul is
-- 30 Jul 21:00Z, and Zambia's 13 Aug election becomes a 12 Aug UTC row. Any
-- date-keyed join then lands one session early -- the exact class of error that
-- understated the 8-K event-window effect 2.2x -> 3.2x. So for precision='date'
-- the LOCAL DATE is the authoritative field and event_time is a convenience
-- ordering key; only precision='exact' makes event_time authoritative.
--
-- Additive: eg.calendar_event (0003) is untouched and still loaded by the
-- existing paths. Demoting it to a view over this table is ADR-2's job.

CREATE TABLE IF NOT EXISTS eg.calendar_observation (
  -- occurrence identity ------------------------------------------------------
  series_id        VARCHAR,   -- canonical, -> Postgres event_series
  period_ref       VARCHAR,   -- '2026-06' | 'MPC-2026-04' | auction id | election id

  -- the claim ----------------------------------------------------------------
  event_time       TIMESTAMP, -- UTC instant; only AUTHORITATIVE when precision='exact'
  event_date_local DATE,      -- the venue-local calendar date the source published
  event_time_local VARCHAR,   -- VERBATIM local literal, exactly as published
  time_precision   VARCHAR,   -- exact | date | week | month | quarter | unknown
  status           VARCHAR,   -- scheduled | confirmed | postponed | cancelled | occurred
  venue            VARCHAR,   -- ISO2 / venue key -> config/em_calendar_sources.json
  event_class      VARCHAR,   -- cpi | mpc | tbill_auction | election | rating_review | ...
  title            VARCHAR,   -- what the source called it
  expected         DOUBLE,
  actual           DOUBLE,
  prior            DOUBLE,
  unit             VARCHAR,

  -- epistemics ---------------------------------------------------------------
  source           VARCHAR,   -- source_id from the registry
  source_ref       VARCHAR,   -- exact origin (URL / API route / file)
  source_tier      VARCHAR,   -- primary | secondary | narrative | model
  method           VARCHAR,   -- html_table:<parser> | ics | pdf:<sha256> | llm:<model> | pattern:<v>
  artifact_sha256  VARCHAR,   -- content-addressed raw snapshot this claim was read from
  observed_at      TIMESTAMP,
  knowable_at      TIMESTAMP,
  ingested_at      TIMESTAMP,

  -- narrative bridge (rides eg.v_provenance from 0008 unchanged) --------------
  doc_id           VARCHAR,
  chunk_id         VARCHAR,
  quote            VARCHAR,

  -- source-specific extras as JSON (release frequency, ElectionGuide's
  -- definitive flag, ...). Kept out of the column list so a new parser never
  -- widens the table -- the schema is the contract, not the union of sources.
  attrs            VARCHAR
);

-- Raw-fetch ledger. Every fetch is logged whether or not anything parsed --
-- silent upstream mutation IS the phenomenon we are capturing, so the raw
-- snapshot is the product, not hygiene. `changed` flags a content hash that
-- differs from this source's previous fetch: the cheap change-detection tripwire
-- that works even for sources we cannot parse yet.
CREATE TABLE IF NOT EXISTS eg.source_snapshot (
  source_id     VARCHAR,
  url           VARCHAR,
  final_url     VARCHAR,
  status_code   INTEGER,
  content_type  VARCHAR,
  bytes         BIGINT,
  sha256        VARCHAR,
  artifact_path VARCHAR,
  changed       BOOLEAN,
  error         VARCHAR,
  fetched_at    TIMESTAMP
);

-- Tier precedence as data, so the ordering is auditable rather than buried in a
-- CASE expression in three different scripts.
CREATE OR REPLACE VIEW eg.v_tier_rank AS
SELECT * FROM (VALUES ('primary', 1), ('secondary', 2), ('narrative', 3), ('model', 4))
  AS t(source_tier, tier_rank);

-- Observations with tier precedence resolved. Both selectors below read this.
CREATE OR REPLACE VIEW eg.v_calendar_ranked AS
SELECT o.*, COALESCE(r.tier_rank, 9) AS tier_rank
FROM eg.calendar_observation o
LEFT JOIN eg.v_tier_rank r USING (source_tier);

-- CURRENT BELIEF: one row per occurrence, best claim wins.
-- Ordering: tier precedence, then how recently it became knowable, then fetch.
CREATE OR REPLACE VIEW eg.v_calendar_current AS
SELECT * EXCLUDE (tier_rank)
FROM eg.v_calendar_ranked
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY series_id, period_ref
  ORDER BY tier_rank ASC,
           knowable_at DESC NULLS LAST,
           ingested_at DESC NULLS LAST) = 1;

-- POINT-IN-TIME BELIEF: what we would have said on `asof`, using only what was
-- knowable by then. This is the view every backtest must read -- reading
-- v_calendar_current in a backtest is lookahead.
-- Expressed as "no better claim was knowable by `asof`" rather than a windowed
-- rank because DuckDB (1.5.2) will not parse a window function inside a table
-- macro body. Same ordering as v_calendar_current: tier first, then recency.
-- The parameter is as_of, not asof: ASOF is a reserved keyword (ASOF JOIN) and
-- using it silently turns into a parse error at the following AND.
-- (o.* carries tier_rank through; EXCLUDE is another construct the macro parser
-- rejects, and an extra integer column is a cheaper price than losing the macro.)
CREATE OR REPLACE MACRO eg.v_calendar_asof(as_of) AS TABLE
SELECT o.*
FROM eg.v_calendar_ranked o
WHERE o.knowable_at <= as_of
  AND NOT EXISTS (
    SELECT 1 FROM eg.v_calendar_ranked b
    WHERE b.series_id = o.series_id
      AND b.period_ref = o.period_ref
      AND b.knowable_at <= as_of
      AND (b.tier_rank < o.tier_rank
           OR (b.tier_rank = o.tier_rank
               AND (b.knowable_at > o.knowable_at
                    OR (b.knowable_at = o.knowable_at AND b.ingested_at > o.ingested_at)))));

-- SCHEDULE REVISIONS: consecutive observations of the same occurrence where the
-- claimed time or lifecycle status moved. This is the proprietary asset -- it
-- exists only because we snapshot, and it cannot be backfilled from vendors who
-- overwrite in place. `days_moved` > 0 = slipped later, < 0 = pulled earlier;
-- `lead_days` = how far ahead of the (then-)claimed date the revision landed.
CREATE OR REPLACE VIEW eg.v_schedule_revision AS
WITH seq AS (
  SELECT series_id, period_ref, venue, event_class, source, source_tier,
         event_time, status, knowable_at,
         LAG(event_time)  OVER w AS prev_event_time,
         LAG(status)      OVER w AS prev_status,
         LAG(source)      OVER w AS prev_source,
         LAG(knowable_at) OVER w AS prev_knowable_at
  FROM eg.calendar_observation
  WINDOW w AS (PARTITION BY series_id, period_ref ORDER BY knowable_at, ingested_at)
)
SELECT series_id, period_ref, venue, event_class,
       prev_source AS from_source, source AS to_source, source_tier,
       prev_event_time AS from_time, event_time AS to_time,
       prev_status     AS from_status, status   AS to_status,
       prev_knowable_at AS revised_from, knowable_at AS revised_at,
       DATE_DIFF('day', prev_event_time, event_time) AS days_moved,
       DATE_DIFF('day', knowable_at, prev_event_time) AS lead_days
FROM seq
WHERE prev_knowable_at IS NOT NULL
  AND (event_time IS DISTINCT FROM prev_event_time
       OR status  IS DISTINCT FROM prev_status);

-- Coverage/health of the calendar itself, per venue x class: how many
-- occurrences we hold, how far forward we can see, and how much of it rests on
-- primary sources rather than press or inference.
CREATE OR REPLACE VIEW eg.v_calendar_coverage AS
-- Coverage keys off event_date_local, not event_time: a UTC instant answers the
-- wrong question for a date-precision claim (see the note at the top).
SELECT venue, event_class,
       COUNT(*) AS occurrences,
       COUNT(*) FILTER (WHERE event_date_local > current_date) AS forward,
       COUNT(*) FILTER (WHERE source_tier = 'primary') AS primary_sourced,
       COUNT(*) FILTER (WHERE status = 'confirmed')    AS confirmed,
       MIN(event_date_local) AS first_event,
       MAX(event_date_local) AS last_event,
       MAX(DATE_DIFF('day', current_date, event_date_local)) AS horizon_days
FROM eg.v_calendar_current
GROUP BY 1, 2;
