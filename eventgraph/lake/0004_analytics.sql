-- eventgraph :: DUCKLAKE analytics views (pure graph structure, no LLM).
-- These are the hub-detection + month-concentration signals, computed off the
-- lake facts. Assumes a published snapshot of the Postgres `entity` dimension
-- into eg.entity_dim (a periodic COPY; entities are small).

-- Per-entity hub signal: cause in-degree = # distinct docs blaming it.
CREATE OR REPLACE VIEW eg.v_entity_hub AS
SELECT cause_entity AS entity_id,
       count(DISTINCT doc_id)  AS cause_in_degree,
       count(*)                AS cause_edges
FROM eg.causal_event_edge
WHERE cause_entity IS NOT NULL
GROUP BY cause_entity;

-- Per-month HUB-DOMINATION INDEX. High herfindahl / top-share => attribution
-- concentrates on a few entities => the transport operator will underperform.
CREATE OR REPLACE VIEW eg.v_month_concentration AS
WITH deg AS (
  SELECT strftime(d.published_at, '%Y-%m') AS ym,
         ce.cause_entity,
         count(DISTINCT ce.doc_id) AS indeg
  FROM eg.causal_event_edge ce
  JOIN eg.document d USING (doc_id)
  WHERE ce.cause_entity IS NOT NULL
  GROUP BY 1, 2
), tot AS (
  SELECT ym, sum(indeg) AS total, count(*) AS n_entities FROM deg GROUP BY ym
)
SELECT t.ym, t.n_entities, t.total,
       sum(power(d.indeg::DOUBLE / t.total, 2)) AS herfindahl
FROM deg d JOIN tot t USING (ym)
GROUP BY t.ym, t.n_entities, t.total;

-- Narrative-vs-market probability divergence (needs realised.market_implied_prob).
CREATE OR REPLACE VIEW eg.v_prob_divergence AS
SELECT pa.proposition_id, pa.as_of AS narrative_as_of, pa.probability AS narrative_prob,
       mip.as_of AS market_as_of, mip.implied_prob AS market_prob,
       pa.probability - mip.implied_prob AS divergence
FROM eg.probability_annotation pa
LEFT JOIN eg.market_implied_prob mip USING (proposition_id);
