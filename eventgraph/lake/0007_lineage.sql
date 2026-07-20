-- eventgraph :: formal-plane lineage columns (additive).
-- The narrative plane already carries run_id on every fact (extraction_run);
-- the formal plane gets the equivalent: WHEN we fetched (ingested_at) and
-- EXACTLY WHERE FROM (source_ref: URL / API series id / HF revision). Schedules
-- mutate upstream (ICS calendars are re-published, release dates shift) -- with
-- ingested_at, "the schedule as currently known" is simply the latest fetch:
--   QUALIFY row_number() OVER (PARTITION BY series_id, event_time
--                              ORDER BY ingested_at DESC) = 1
ALTER TABLE eg.calendar_event  ADD COLUMN IF NOT EXISTS source_ref  VARCHAR;
ALTER TABLE eg.calendar_event  ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP;
ALTER TABLE eg.option_iv       ADD COLUMN IF NOT EXISTS source_ref  VARCHAR;
ALTER TABLE eg.option_iv       ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP;
ALTER TABLE eg.option_surface  ADD COLUMN IF NOT EXISTS source_ref  VARCHAR;
ALTER TABLE eg.option_surface  ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMP;
