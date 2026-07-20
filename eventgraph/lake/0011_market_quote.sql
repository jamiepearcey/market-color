-- eventgraph :: implied-probability feed (formal plane). Populates the loop that
-- 0004's v_prob_divergence was built for: narrative probability_annotation vs
-- market-implied prob, per proposition.
--
-- Two tables + a resolver, same shape as the calendar/surprise path:
--   market_quote        raw external fact -- one prediction-market/derivative
--                       quote, keyed by the MARKET's own natural key (Kalshi
--                       ticker, Polymarket conditionId). Append-only; a feed
--                       never needs a proposition to exist. Full lineage.
--   proposition_contract mapping proposition_id <-> contract_ref (the market
--                       natural key). Same alias-table doctrine as series_alias
--                       / entity_alias: facts never rewritten. Backfilled from
--                       Postgres proposition.contract_ref, or by a text-match
--                       pass (scripts/resolve_series-style; the remaining lever).
--   v_market_implied    market_quote resolved through proposition_contract into
--                       the exact market_implied_prob shape -> feeds
--                       v_prob_divergence directly, or INSERT ... SELECT into
--                       the materialized eg.market_implied_prob.
CREATE TABLE IF NOT EXISTS eg.market_quote (
  market_ref   VARCHAR,             -- Kalshi ticker | Polymarket conditionId
  title        VARCHAR,             -- market question (for matching / audit)
  as_of        TIMESTAMP,           -- quote time (feed's, else fetch time)
  implied_prob DOUBLE,              -- 0..1 (YES outcome)
  instrument   VARCHAR,             -- kalshi | polymarket | fed_funds_future | cds | option
  source       VARCHAR,
  source_ref   VARCHAR,             -- exact endpoint/id
  ingested_at  TIMESTAMP,
  close_time   TIMESTAMP            -- market resolution/expiry (context)
);

CREATE TABLE IF NOT EXISTS eg.proposition_contract (
  proposition_id VARCHAR,
  contract_ref   VARCHAR,           -- -> market_quote.market_ref
  method         VARCHAR,           -- 'pg:contract_ref' | 'match:<name>' | 'manual'
  ingested_at    TIMESTAMP
);

-- latest quote per (market, proposition) resolved to the market_implied_prob shape.
CREATE OR REPLACE VIEW eg.v_market_implied AS
WITH pc AS (
  SELECT proposition_id, contract_ref FROM eg.proposition_contract
  QUALIFY row_number() OVER (PARTITION BY proposition_id, contract_ref
                             ORDER BY ingested_at DESC) = 1
)
SELECT pc.proposition_id, q.as_of, q.implied_prob, q.instrument, q.source
FROM eg.market_quote q
JOIN pc ON pc.contract_ref = q.market_ref;
