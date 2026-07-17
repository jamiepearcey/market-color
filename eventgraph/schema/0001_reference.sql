-- eventgraph :: POSTGRES tier (thin) -- reference dimensions + resolution state.
-- Postgres holds ONLY what mutates, needs uniqueness, or is hand-curated. The
-- high-volume append-only facts live in DuckLake (see ../lake/). One Postgres
-- instance can also BE the DuckLake catalog (see ../lake/0001_attach.sql).

CREATE SCHEMA IF NOT EXISTS eventgraph;
SET search_path = eventgraph, public;

-- controlled vocabs used by the dimension/lifecycle columns below.
-- (Fact-only vocabs -- mechanism, modality, sentiment_type, ... -- live as plain
--  strings in the lake and are validated in the Rust normalize layer.)
CREATE TYPE entity_type AS ENUM (
  'company','bank','central_bank','sovereign','supranational','regulator',
  'government_agency','commodity','currency','equity_index','rate_or_bond',
  'sector','person','exchange','economic_indicator','asset_class','market','other');

CREATE TYPE event_type AS ENUM (
  'rate_decision','cb_minutes','cb_speech','cpi','ppi','employment','gdp','pmi',
  'retail_sales','trade_balance','earnings','guidance','dividend','buyback',
  'mergers_acquisitions','ipo','debt_auction','rating_review','election',
  'referendum','budget','opec_meeting','eia_inventory','usda_report','other');

CREATE TYPE factor_type AS ENUM (
  'rates','credit_spread','oil','commodity','usd','fx','equity_beta',
  'inflation','specific','other');

-- extraction versioning: every lake fact carries run_id so the graph is rebuildable.
CREATE TABLE extraction_run (
  run_id         text PRIMARY KEY,
  model          text NOT NULL,
  prompt_version text NOT NULL,
  schema_version text NOT NULL,
  created_at     timestamptz NOT NULL DEFAULT now(),
  notes          text
);

-- ENTITY registry (canonical nodes). The one place with genuine mutable-relational
-- needs: resolution merges surface forms into ONE entity_id; ingestion upserts
-- here to get ids before writing lake facts. `identifier` (ticker/ISO) bridges to
-- the realised plane and is a strong merge key.
CREATE TABLE entity (
  entity_id      text PRIMARY KEY,               -- slug(canonical_name)+'__'+type
  canonical_name text NOT NULL,
  type           entity_type NOT NULL,
  identifier     text,
  sector         text,
  country        text,
  first_seen     date,
  last_seen      date,
  UNIQUE (canonical_name, type)
);
CREATE INDEX entity_identifier_idx ON entity (identifier) WHERE identifier IS NOT NULL;

CREATE TABLE entity_alias (
  alias      text NOT NULL,           -- surface form as written
  entity_id  text NOT NULL REFERENCES entity(entity_id),
  PRIMARY KEY (alias, entity_id)
);

-- FACTOR registry (realised-sensitivity plane). proxy_symbol = the tradeable
-- series a realised beta is regressed against (oil->'CL', rates->'US2Y').
CREATE TABLE factor (
  factor_id     text PRIMARY KEY,
  name          text NOT NULL,
  type          factor_type NOT NULL,
  proxy_symbol  text,
  proxy_entity  text REFERENCES entity(entity_id)
);

-- EVENT SERIES registry (formal-calendar plane). series_id is the recurring key
-- (FOMC, US-CPI, AAPL-EARN); calendar_ref is the external calendar id to join.
CREATE TABLE event_series (
  series_id     text PRIMARY KEY,
  event_type    event_type NOT NULL,
  issuer_entity text REFERENCES entity(entity_id),
  region        text,
  cadence       text,
  calendar_ref  text
);
