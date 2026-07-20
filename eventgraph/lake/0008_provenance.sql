-- eventgraph :: ONE provenance vocabulary across planes (no re-run needed --
-- pure metadata plumbing over columns the facts already carry).
--
-- Narrative facts: fact -> doc_id -> document.run_id -> extraction_run.
-- Formal facts:    row  -> source / source_ref / ingested_at (0007).
-- Unified vocab:   plane, fact_table, fact_key, doc_id, chunk_id,
--                  source     (who produced it: 'extraction:<model>' | feed kind)
--                  source_ref (exact origin: run_id | fred id/ICS url/accession/HF sha)
--                  known_at   (when WE first knew: run created_at | ingested_at)

-- Lake mirror of the Postgres extraction_run dim (tiny; keeps v_provenance
-- lake-resolvable). Backfill one-off from PG, e.g.:
--   ATTACH 'dbname=eventgraph' AS pg (TYPE postgres, READ_ONLY);
--   INSERT INTO eg.extraction_run SELECT run_id,model,prompt_version,schema_version,created_at
--     FROM pg.eventgraph.extraction_run;
CREATE TABLE IF NOT EXISTS eg.extraction_run (
  run_id         VARCHAR,
  model          VARCHAR,
  prompt_version VARCHAR,
  schema_version VARCHAR,
  created_at     TIMESTAMP
);

CREATE OR REPLACE VIEW eg.v_provenance AS
WITH doc_run AS (
  SELECT d.doc_id, d.run_id, r.model, r.created_at
  FROM eg.document d LEFT JOIN eg.extraction_run r USING (run_id)
)
-- narrative plane: one branch per fact table, all through doc_run
SELECT 'narrative' AS plane, 'causal_event_edge' AS fact_table, f.edge_key AS fact_key,
       f.doc_id, f.chunk_id, 'extraction:' || coalesce(dr.model,'?') AS source,
       dr.run_id AS source_ref, dr.created_at AS known_at
FROM eg.causal_event_edge f LEFT JOIN doc_run dr USING (doc_id)
UNION ALL
SELECT 'narrative', 'sensitivity_edge', f.sens_key, f.doc_id, f.chunk_id,
       'extraction:' || coalesce(dr.model,'?'), dr.run_id, dr.created_at
FROM eg.sensitivity_edge f LEFT JOIN doc_run dr USING (doc_id)
UNION ALL
SELECT 'narrative', 'event', f.event_id, f.doc_id, f.chunk_id,
       'extraction:' || coalesce(dr.model,'?'), dr.run_id, dr.created_at
FROM eg.event f LEFT JOIN doc_run dr USING (doc_id)
UNION ALL
SELECT 'narrative', 'sentiment_annotation', f.sent_key, f.doc_id, f.chunk_id,
       'extraction:' || coalesce(dr.model,'?'), dr.run_id, dr.created_at
FROM eg.sentiment_annotation f LEFT JOIN doc_run dr USING (doc_id)
UNION ALL
SELECT 'narrative', 'probability_annotation', f.prob_key, f.doc_id, f.chunk_id,
       'extraction:' || coalesce(dr.model,'?'), dr.run_id, dr.created_at
FROM eg.probability_annotation f LEFT JOIN doc_run dr USING (doc_id)
UNION ALL
SELECT 'narrative', 'doc_figure', f.figure_key, f.doc_id, f.chunk_id,
       'extraction:' || coalesce(dr.model,'?'), dr.run_id, dr.created_at
FROM eg.doc_figure f LEFT JOIN doc_run dr USING (doc_id)
UNION ALL
-- formal plane: natural key rendered as '<series|symbol>@<time>'
SELECT 'formal', 'calendar_event', f.series_id || '@' || strftime(f.event_time, '%Y-%m-%dT%H:%M:%SZ'),
       NULL, NULL, f.source, f.source_ref, f.ingested_at
FROM eg.calendar_event f
UNION ALL
SELECT 'formal', 'option_surface', f.underlier || '@' || f.date,
       NULL, NULL, f.source, f.source_ref, f.ingested_at
FROM eg.option_surface f
UNION ALL
SELECT 'formal', 'option_iv', f.symbol || '@' || f.date,
       NULL, NULL, f.source, f.source_ref, f.ingested_at
FROM eg.option_iv f;
