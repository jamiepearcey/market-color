-- eventgraph :: DUCKLAKE realised + formal planes (append-only, derived).
-- We NEVER update a narrative fact row. Realised metrics and authoritative
-- calendar/market values are SEPARATE append-only tables joined by natural key.
-- Filled by downstream jobs (price regressions, calendar feed, derivative-implied
-- probs) -- integration, later.

-- Did the effect entity actually move after the causal edge? (event study)
CREATE TABLE IF NOT EXISTS eg.realised_edge_move (
  edge_key       VARCHAR,          -- -> causal_event_edge.edge_key
  est_window         VARCHAR,          -- '1h','1d','3d'
  realised_return DOUBLE,
  realised_z     DOUBLE,
  computed_at    TIMESTAMP
);

-- Realised beta of asset returns on the factor's proxy series.
CREATE TABLE IF NOT EXISTS eg.realised_sensitivity (
  sens_key      VARCHAR,           -- -> sensitivity_edge.sens_key
  est_window        VARCHAR,
  realised_beta DOUBLE,
  realised_r2   DOUBLE,
  computed_at   TIMESTAMP
);

-- Forward return / realised vol on a sentiment target (back-test the reading).
CREATE TABLE IF NOT EXISTS eg.realised_sentiment (
  sent_key      VARCHAR,           -- -> sentiment_annotation.sent_key
  est_window        VARCHAR,
  fwd_return    DOUBLE,
  realised_vol  DOUBLE,
  computed_at   TIMESTAMP
);

-- Authoritative scheduled-event values from the formal calendar feed. Join to
-- eg.event on (series_id, event_time) to compute surprise = actual - expected.
CREATE TABLE IF NOT EXISTS eg.calendar_event (
  series_id   VARCHAR,
  event_time  TIMESTAMP,
  expected    DOUBLE,
  actual      DOUBLE,
  prior       DOUBLE,
  unit        VARCHAR,
  source      VARCHAR              -- 'trading_economics','bloomberg_eco',...
);

-- Market-implied probability time series for a proposition (from a prediction
-- market or a derivative). Compare vs narrative probability_annotation to get
-- the narrative-vs-market divergence signal.
CREATE TABLE IF NOT EXISTS eg.market_implied_prob (
  proposition_id VARCHAR,          -- -> Postgres proposition.proposition_id
  as_of          TIMESTAMP,
  implied_prob   DOUBLE,
  instrument     VARCHAR,          -- fed_funds_future|cds|option|kalshi|polymarket
  source         VARCHAR
);
