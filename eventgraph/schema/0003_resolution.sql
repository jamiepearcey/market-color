-- eventgraph :: POSTGRES tier -- v2 entity IDENTITY resolution (schema_version 0.2.0).
-- Makes the suggested-vs-verified split first-class. ADDITIVE + idempotent; the
-- existing `identifier` column BECOMES the verified slot (resolver-written), and
-- the 8B's raw guess is quarantined in `suggested_ticker`. Nothing dropped.
SET search_path = eventgraph, public;

-- the 8B's raw ticker guess -- NEVER a merge key; kept for audit + resolver input.
ALTER TABLE entity ADD COLUMN IF NOT EXISTS suggested_ticker  text;

-- resolution lifecycle. `identifier` (verified symbol) stays NULL until a gate confirms it.
ALTER TABLE entity ADD COLUMN IF NOT EXISTS resolution_status text NOT NULL DEFAULT 'unresolved';
  -- unresolved | seen_in_source | verified_yahoo | verified_llm_yahoo
  -- | unlisted_private | unlisted_gov | unlisted_index | unlisted_person | resolution_failed
ALTER TABLE entity ADD COLUMN IF NOT EXISTS resolution_source text;   -- yahoo_exact | llm_yahoo | sec | openfigi | ...
ALTER TABLE entity ADD COLUMN IF NOT EXISTS resolved_at       timestamptz;
ALTER TABLE entity ADD COLUMN IF NOT EXISTS resolver_version  text;

-- query the resolution backlog (what resolve_yahoo/resolve_llm should attempt next).
CREATE INDEX IF NOT EXISTS entity_unresolved_idx
  ON entity (resolution_status)
  WHERE identifier IS NULL AND resolution_status = 'unresolved';
