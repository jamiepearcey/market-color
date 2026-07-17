-- eventgraph :: DUCKLAKE facts (append-only OLAP). Run after 0001_attach.sql.
-- Controlled-vocab columns are VARCHAR here (validated in the Rust normalize
-- layer); entity_id / factor_id / series_id / proposition_id reference the
-- Postgres dimensions by convention (no cross-store FK).

CREATE TABLE IF NOT EXISTS eg.document (
  doc_id            VARCHAR,       -- key
  source            VARCHAR,
  url               VARCHAR,
  headline          VARCHAR,
  published_at      TIMESTAMP,
  doc_type          VARCHAR,       -- news|analysis|opinion|data_release|...
  sentiment_overall DOUBLE,
  run_id            VARCHAR,
  ingested_at       TIMESTAMP
);

CREATE TABLE IF NOT EXISTS eg.chunk (
  chunk_id  VARCHAR,               -- '<doc_id>:<seq>'
  doc_id    VARCHAR,
  seq       BIGINT,
  epoch     BIGINT,
  text      VARCHAR
);

CREATE TABLE IF NOT EXISTS eg.doc_figure (
  figure_key VARCHAR,
  doc_id     VARCHAR,
  chunk_id   VARCHAR,
  entity_id  VARCHAR,
  kind       VARCHAR,              -- price|level|pct|bps|volume|estimate
  value      DOUBLE,
  unit       VARCHAR,
  quote      VARCHAR
);

-- EVENT: what the NEWS asserted about an event (append-only). Authoritative
-- consensus/actual come from the formal calendar (0003 calendar_event); surprise
-- is computed at query time by joining on series_id + event_time.
CREATE TABLE IF NOT EXISTS eg.event (
  event_id      VARCHAR,
  event_type    VARCHAR,
  series_id     VARCHAR,
  issuer_entity VARCHAR,
  region        VARCHAR,
  scheduled     BOOLEAN,
  event_time    TIMESTAMP,
  expected      DOUBLE,
  actual        DOUBLE,
  prior         DOUBLE,
  unit          VARCHAR,
  doc_id        VARCHAR,
  chunk_id      VARCHAR
);

-- EPISODIC cause->effect edges.
CREATE TABLE IF NOT EXISTS eg.causal_event_edge (
  edge_key        VARCHAR,
  cause_entity    VARCHAR,
  effect_entity   VARCHAR,
  mechanism       VARCHAR,
  effect_dir      VARCHAR,
  magnitude_value DOUBLE,
  magnitude_unit  VARCHAR,
  modality        VARCHAR,         -- happened|ongoing|forecast|hypothetical|denied
  attribution     VARCHAR,
  lag             VARCHAR,
  confidence      VARCHAR,
  event_id        VARCHAR,
  doc_id          VARCHAR,
  chunk_id        VARCHAR,
  quote           VARCHAR
);

-- STRUCTURAL asset->factor sensitivity edges.
CREATE TABLE IF NOT EXISTS eg.sensitivity_edge (
  sens_key        VARCHAR,
  asset_entity    VARCHAR,
  factor_id       VARCHAR,
  factor_entity   VARCHAR,
  sign            BIGINT,
  magnitude_qual  VARCHAR,
  magnitude_value DOUBLE,
  magnitude_unit  VARCHAR,
  basis           VARCHAR,
  doc_id          VARCHAR,
  chunk_id        VARCHAR,
  quote           VARCHAR
);

-- Per-target sentiment annotations.
CREATE TABLE IF NOT EXISTS eg.sentiment_annotation (
  sent_key      VARCHAR,
  target_entity VARCHAR,
  polarity      DOUBLE,
  intensity     VARCHAR,
  type          VARCHAR,           -- directional|risk|credit|surprise
  source        VARCHAR,
  horizon       VARCHAR,
  as_of         TIMESTAMP,
  doc_id        VARCHAR,
  chunk_id      VARCHAR,
  quote         VARCHAR
);

-- Narrative probability annotations (one source's stated/implied likelihood).
CREATE TABLE IF NOT EXISTS eg.probability_annotation (
  prob_key         VARCHAR,
  proposition_id   VARCHAR,
  probability      DOUBLE,
  source           VARCHAR,
  source_instrument VARCHAR,
  as_of            TIMESTAMP,
  doc_id           VARCHAR,
  chunk_id         VARCHAR
);
