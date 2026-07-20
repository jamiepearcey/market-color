-- eventgraph :: calendar SURPRISE join (narrative event <-> formal calendar).
-- Query-time only, as promised in 0002. OLD-SHAPE SAFE: the narrative side uses
-- 0002 base columns only (v2 extras and 0007 lineage not required), so graphs
-- extracted before these migrations -- including runs in flight -- load and
-- join unchanged; pre-0007 calendar rows (NULL ingested_at) sort as the OLDEST
-- fetch (DESC = NULLS LAST) and are superseded by any stamped re-fetch.
--
--   surprise           = formal actual - (formal expected, else formal prior)
--   narrative_err      = what the ARTICLE claimed the print was, minus truth
--                        (mis-reporting / stale-figure detector)
--
-- Join: same series_id, calendar event within +-36h of the narrative
-- event_time (articles date events at day precision; releases are ~08:30 ET).
CREATE OR REPLACE VIEW eg.v_event_surprise AS
WITH cal AS (  -- current formal knowledge: latest fetch per (series, time)
  SELECT * FROM eg.calendar_event
  QUALIFY row_number() OVER (PARTITION BY series_id, event_time
                             ORDER BY ingested_at DESC) = 1
)
SELECT e.event_id, e.series_id, e.doc_id, e.chunk_id,
       e.event_time  AS narrative_time,
       c.event_time  AS formal_time,
       c.actual, coalesce(c.expected, c.prior) AS baseline,
       c.actual - coalesce(c.expected, c.prior)      AS surprise,
       (c.actual - coalesce(c.expected, c.prior))
         / nullif(abs(coalesce(c.expected, c.prior)), 0) AS surprise_pct,
       e.actual - c.actual                            AS narrative_err,
       c.source AS formal_source, c.source_ref AS formal_source_ref
FROM eg.event e
JOIN cal c
  ON c.series_id = e.series_id
 AND abs(datediff('hour', c.event_time, e.event_time)) <= 36
QUALIFY row_number() OVER (PARTITION BY e.event_id
                           ORDER BY abs(datediff('hour', c.event_time, e.event_time))) = 1;
