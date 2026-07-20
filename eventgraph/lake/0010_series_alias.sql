-- eventgraph :: series-slug normalisation (downstream mapping, NO re-run).
-- The extractor emits free-form event.series_id slugs ('us_cpi_report',
-- 'nonfarm_payrolls', 'december_jobs_report'); the formal calendar uses
-- canonical ids (US-CPI, US-NFP, {TICKER}-8-K). Instead of rewriting fact rows
-- (never UPDATE a fact) or re-extracting, aliases live in a mapping table --
-- same doctrine as entity_alias -- populated by scripts/resolve_series.py.
CREATE TABLE IF NOT EXISTS eg.series_alias (
  alias       VARCHAR,               -- extractor slug, as written
  series_id   VARCHAR,               -- canonical calendar id
  method      VARCHAR,               -- 'rule:<name>' | 'earnings-symbol' | 'manual'
  ingested_at TIMESTAMP
);

-- v_event_surprise now resolves narrative slugs through the alias table
-- (latest mapping wins; unmapped slugs fall through unchanged, so events whose
-- extractor slug already matches keep joining exactly as before).
CREATE OR REPLACE VIEW eg.v_event_surprise AS
WITH alias AS (
  SELECT alias, series_id FROM eg.series_alias
  QUALIFY row_number() OVER (PARTITION BY alias ORDER BY ingested_at DESC) = 1
), ev AS (
  SELECT e.*, coalesce(a.series_id, e.series_id) AS canon_series
  FROM eg.event e LEFT JOIN alias a ON a.alias = e.series_id
), cal AS (
  SELECT * FROM eg.calendar_event
  QUALIFY row_number() OVER (PARTITION BY series_id, event_time
                             ORDER BY ingested_at DESC) = 1
)
SELECT e.event_id, e.canon_series AS series_id, e.series_id AS narrative_slug,
       e.doc_id, e.chunk_id,
       e.event_time  AS narrative_time,
       c.event_time  AS formal_time,
       c.actual, coalesce(c.expected, c.prior) AS baseline,
       c.actual - coalesce(c.expected, c.prior)      AS surprise,
       (c.actual - coalesce(c.expected, c.prior))
         / nullif(abs(coalesce(c.expected, c.prior)), 0) AS surprise_pct,
       e.actual - c.actual                            AS narrative_err,
       c.source AS formal_source, c.source_ref AS formal_source_ref
FROM ev e
JOIN cal c
  ON c.series_id = e.canon_series
 AND abs(datediff('hour', c.event_time, e.event_time)) <= 36
QUALIFY row_number() OVER (PARTITION BY e.event_id
                           ORDER BY abs(datediff('hour', c.event_time, e.event_time))) = 1;
