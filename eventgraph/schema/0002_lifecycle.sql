-- eventgraph :: POSTGRES tier -- PROPOSITION lifecycle.
-- Propositions are low-volume and MUTATE (open -> resolved, outcome filled), and
-- need dedup across mentions -> Postgres, not the lake. The many time-stamped
-- probability *annotations* (narrative + market-implied) are append-only facts
-- and live in the lake (../lake/0002_facts.sql, ../lake/0003_realised.sql).
SET search_path = eventgraph, public;

CREATE TYPE proposition_status AS ENUM ('open','resolved','void');
CREATE TYPE resolution_outcome AS ENUM ('yes','no','partial','unknown');

CREATE TABLE proposition (
  proposition_id      text PRIMARY KEY,          -- hash of normalized text
  text                text NOT NULL,
  subject_entity      text REFERENCES entity(entity_id),
  resolution_date     date,
  resolution_criteria text,
  status              proposition_status NOT NULL DEFAULT 'open',
  outcome             resolution_outcome,

  -- market linkage (integration hooks)
  contract_ref        text,                      -- Kalshi/Polymarket contract id
  implied_instrument  text,                      -- derivative giving implied prob

  first_doc_id        text,                      -- where first seen (lake document)
  created_at          timestamptz NOT NULL DEFAULT now(),
  resolved_at         timestamptz
);
CREATE INDEX proposition_open_idx ON proposition (resolution_date) WHERE status = 'open';
