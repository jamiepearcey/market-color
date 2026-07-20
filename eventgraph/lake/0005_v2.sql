-- eventgraph :: DUCKLAKE v2 (schema_version 0.2.0 / prompt eg-rich-v2).
-- Idempotent + ADDITIVE: ALTER ... ADD COLUMN IF NOT EXISTS so it upgrades an
-- existing catalog AND is a no-op on a fresh one (which already has the base
-- columns from 0002_facts.sql). Nothing here drops or renames -> no data loss.
-- Run after 0002/0003/0004 (see lake.rs load_script order).

-- EVENT: temporal-hint triple + reference-period capture.
ALTER TABLE eg.event ADD COLUMN IF NOT EXISTS event_time_text VARCHAR;
ALTER TABLE eg.event ADD COLUMN IF NOT EXISTS event_time_kind VARCHAR;  -- absolute|relative|period|deictic|none
ALTER TABLE eg.event ADD COLUMN IF NOT EXISTS period_text     VARCHAR;
ALTER TABLE eg.event ADD COLUMN IF NOT EXISTS period_hint     VARCHAR;

-- CAUSAL EDGE: direction-audit anchor, cause/effect kinds, verbatim magnitude, timing.
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS cause_kind      VARCHAR; -- entity|factor|event|other
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS effect_kind     VARCHAR;
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS effect_verbatim VARCHAR;
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS magnitude_text  VARCHAR;
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS timing_text     VARCHAR;
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS timing_kind     VARCHAR;
ALTER TABLE eg.causal_event_edge ADD COLUMN IF NOT EXISTS direction_audit VARCHAR; -- ok|conflict|unchecked

-- SENSITIVITY EDGE: asset-class disambiguation + verbatim magnitude.
ALTER TABLE eg.sensitivity_edge ADD COLUMN IF NOT EXISTS asset_class    VARCHAR; -- equity|credit|rates|fx|commodity|vol
ALTER TABLE eg.sensitivity_edge ADD COLUMN IF NOT EXISTS magnitude_text VARCHAR;

-- FIGURE: currency + as-of.
ALTER TABLE eg.doc_figure ADD COLUMN IF NOT EXISTS currency   VARCHAR; -- ISO 4217
ALTER TABLE eg.doc_figure ADD COLUMN IF NOT EXISTS as_of_text VARCHAR;

-- NEW: entity<->entity relation edges (M&A roles, parent/brand, officer_of, supply chain).
-- Grounded like a causal edge (verbatim quote must locate in source).
CREATE TABLE IF NOT EXISTS eg.relation_edge (
  rel_key       VARCHAR,
  source_entity VARCHAR,
  target_entity VARCHAR,
  relation      VARCHAR,   -- acquires|merges_with|parent_of|brand_of|stake_in|supplies|customer_of|competes_with|officer_of|regulates|other
  deal_value    DOUBLE,
  deal_currency VARCHAR,
  status_hint   VARCHAR,   -- rumored|proposed|agreed|completed|blocked
  doc_id        VARCHAR,
  chunk_id      VARCHAR,
  quote         VARCHAR
);
