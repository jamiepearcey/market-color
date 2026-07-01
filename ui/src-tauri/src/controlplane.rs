//! Control-plane store (ADR-0021) — the suite's local system-of-record over an
//! embedded pglite-oxide Postgres via sqlx. Prototyped here, then ported into
//! `quant-fabric-app/src-tauri/src/controlplane.rs` verbatim.
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::{postgres::PgPoolOptions, PgPool, Row};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Target {
    pub id: String,
    pub name: String,
    pub kind: String, // embedded | remote
    pub url: String,
    pub token: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Job {
    pub id: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    pub definition: Value, // steps + params + schedule
    #[serde(default = "yes")]
    pub enabled: bool,
}
fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRun {
    pub id: String,
    pub job_id: String,
    pub target_id: Option<String>,
    pub status: String,
    #[serde(default)]
    pub trigger: String,
    pub started_at: String,
    pub finished_at: Option<String>,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackSource {
    pub id: String,
    pub repo_url: String,
    pub owner: String,
    pub repo: String,
    pub ref_name: String,
    #[serde(default)]
    pub pack_path: String,
    #[serde(default = "default_monitor")]
    pub monitor: String,
    pub auth_ref: Option<String>,
    pub last_checked_at: Option<String>,
    pub last_commit: Option<String>,
    pub status: String,
    pub message: Option<String>,
}

fn default_monitor() -> String {
    "manual".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackVersion {
    pub id: String,
    pub source_id: String,
    pub pack_id: String,
    pub pack_name: String,
    pub manifest_version: String,
    pub commit_sha: String,
    pub ref_name: String,
    pub content_hash: String,
    pub cache_dir: String,
    pub discovered_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackSyncEvent {
    pub id: i64,
    pub source_id: String,
    pub status: String,
    pub commit_sha: Option<String>,
    pub message: String,
    pub created_at: String,
}

/// A configured connector instance: *how to reach/authenticate* a system
/// (ADR-0032). Promoted from the legacy `connectors:instances` settings blob to
/// a first-class control-plane entity so it has a stable id and an independent
/// lifecycle from the bindings that pull from it. `params` are connection-scoped
/// (`scope: connection` in the Celeritas contract, ADR-0031); `job_param_defaults`
/// seed a binding's per-run params.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ConnectorInstanceRecord {
    pub id: String,
    pub driver: String,
    #[serde(default)]
    pub name: String,
    #[serde(default)]
    pub params: Value,
    #[serde(default)]
    pub job_param_defaults: Value,
}

/// A first-class binding: *what you take* from a connector instance for a task
/// (ADR-0032) — a named, reusable relation (query / path / prefix), optionally
/// mapped to a domain contract. `run_params` are per-run (`scope: run`).
/// `contract_ref` (e.g. `Position.v1`) and `mapping` are the **optional** typed
/// overlay; a binding with neither is a plain named relation and stays
/// first-class. Typing never gates the raw path (invariant, ADR-0032).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Binding {
    pub id: String,
    pub name: String,
    pub instance_id: String,
    #[serde(default)]
    pub run_params: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contract_ref: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping: Option<Value>,
    /// The relation's discovered column schema, cached when the binding is
    /// authored (ADR-0032 32-C1, option c). The app validates `contract_ref`
    /// conformance against this without re-running discovery on every check.
    /// `None` until a discovery/preview populates it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schema: Option<Value>,
}

/// Legacy settings key the connector instances were persisted under as a JSON
/// array, before they became a first-class table. Read once to backfill
/// `connector_instances` (see [`ControlStore::migrate_connector_instances_from_blob`]).
pub const LEGACY_CONNECTOR_INSTANCES_SETTING: &str = "connectors:instances";

const MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS targets (
  id text PRIMARY KEY,
  name text NOT NULL,
  kind text NOT NULL,
  url text NOT NULL,
  token text,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS jobs (
  id text PRIMARY KEY,
  name text NOT NULL,
  description text NOT NULL DEFAULT '',
  definition jsonb NOT NULL,
  enabled boolean NOT NULL DEFAULT true,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS job_runs (
  id text PRIMARY KEY,
  job_id text NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
  target_id text,
  status text NOT NULL,
  trigger text NOT NULL DEFAULT 'manual',
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  metadata jsonb NOT NULL DEFAULT '{}'::jsonb
);
ALTER TABLE job_runs ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
CREATE TABLE IF NOT EXISTS job_run_steps (
  id bigserial PRIMARY KEY,
  run_id text NOT NULL REFERENCES job_runs(id) ON DELETE CASCADE,
  step_idx int NOT NULL,
  template text NOT NULL,
  args jsonb NOT NULL DEFAULT '{}'::jsonb,
  status text NOT NULL,
  row_count int,
  metrics jsonb,
  log text
);
CREATE TABLE IF NOT EXISTS settings (
  key text PRIMARY KEY,
  value text NOT NULL
);
-- Single-row lease for scheduler leader election (P4c): only the holder of an
-- unexpired lease fires jobs, so multiple `qf-scheduler` instances against one
-- central store never double-fire.
CREATE TABLE IF NOT EXISTS scheduler_lease (
  id text PRIMARY KEY,
  holder text NOT NULL,
  expires_at bigint NOT NULL
);
CREATE TABLE IF NOT EXISTS pack_sources (
  id text PRIMARY KEY,
  repo_url text NOT NULL,
  owner text NOT NULL,
  repo text NOT NULL,
  ref_name text NOT NULL,
  pack_path text NOT NULL DEFAULT '',
  monitor text NOT NULL DEFAULT 'manual',
  auth_ref text,
  last_checked_at timestamptz,
  last_commit text,
  status text NOT NULL DEFAULT 'linked',
  message text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS pack_versions (
  id text PRIMARY KEY,
  source_id text NOT NULL REFERENCES pack_sources(id) ON DELETE CASCADE,
  pack_id text NOT NULL,
  pack_name text NOT NULL,
  manifest_version text NOT NULL DEFAULT '',
  commit_sha text NOT NULL,
  ref_name text NOT NULL,
  content_hash text NOT NULL,
  cache_dir text NOT NULL,
  discovered_at timestamptz NOT NULL DEFAULT now(),
  UNIQUE(source_id, commit_sha, pack_id)
);
CREATE TABLE IF NOT EXISTS pack_sync_events (
  id bigserial PRIMARY KEY,
  source_id text NOT NULL REFERENCES pack_sources(id) ON DELETE CASCADE,
  status text NOT NULL,
  commit_sha text,
  message text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);
-- Connector instances (ADR-0032): how to reach/auth a system. Promoted from the
-- `connectors:instances` settings blob; connection-scoped params live here.
CREATE TABLE IF NOT EXISTS connector_instances (
  id text PRIMARY KEY,
  driver text NOT NULL,
  name text NOT NULL DEFAULT '',
  params jsonb NOT NULL DEFAULT '{}'::jsonb,
  job_param_defaults jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
-- Bindings (ADR-0032): a named, reusable relation over an instance. `run_params`
-- are per-run; `contract_ref`/`mapping` are the optional typed overlay. Deleting
-- the instance cascades to its bindings.
CREATE TABLE IF NOT EXISTS bindings (
  id text PRIMARY KEY,
  name text NOT NULL,
  instance_id text NOT NULL REFERENCES connector_instances(id) ON DELETE CASCADE,
  run_params jsonb NOT NULL DEFAULT '{}'::jsonb,
  contract_ref text,
  mapping jsonb,
  schema jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
ALTER TABLE bindings ADD COLUMN IF NOT EXISTS schema jsonb;
-- Staged relations (ADR-0033): the addressable, immutable scratchpad. The full
-- `StagedRelationManifest` (quant_fabric::staging) is stored as `manifest`; the
-- queryable columns are projected from it. Bytes live in the ArtifactStore, keyed
-- by content_hash; this table is the mutable control-plane record over them.
CREATE TABLE IF NOT EXISTS staged_relations (
  relation_id text PRIMARY KEY,
  name text,
  content_hash text NOT NULL,
  retention text NOT NULL DEFAULT 'ephemeral',
  manifest jsonb NOT NULL,
  created_ms bigint NOT NULL DEFAULT 0,
  last_used_ms bigint,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);
-- Read model of a coordinator's consensus-committed metadata log (ADR-0021 P2b):
-- one row per committed ConsensusCommand, keyed by (coordinator, log index) so
-- projection is idempotent and resumable from MAX(log_index). Committed log
-- entries are immutable, so rows are insert-only (ON CONFLICT DO NOTHING).
CREATE TABLE IF NOT EXISTS cluster_events (
  coordinator_url text NOT NULL,
  log_index bigint NOT NULL,
  term bigint NOT NULL,
  committed_ms bigint NOT NULL,
  kind text NOT NULL,
  command jsonb NOT NULL,
  PRIMARY KEY (coordinator_url, log_index)
);
"#;

/// Owns the sqlx pool (1 connection — pglite is a single backend). The
/// `PgliteServer` is kept alive by the caller (the app leaks it for process
/// lifetime), so this struct is `Send + Sync` and safe as Tauri state.
#[derive(Clone)]
pub struct ControlStore {
    pool: PgPool,
}

impl ControlStore {
    pub async fn connect(database_url: &str) -> Result<Self> {
        let pool = PgPoolOptions::new()
            .max_connections(1)
            .connect(database_url)
            .await?;
        sqlx::raw_sql(MIGRATION).execute(&pool).await?;
        let store = Self { pool };
        // One-time backfill of the first-class connector_instances table from the
        // legacy settings blob (ADR-0032). Idempotent: a no-op once migrated.
        store.migrate_connector_instances_from_blob().await?;
        Ok(store)
    }

    pub async fn list_targets(&self) -> Result<Vec<Target>> {
        let rows =
            sqlx::query("SELECT id, name, kind, url, token FROM targets ORDER BY created_at")
                .fetch_all(&self.pool)
                .await?;
        Ok(rows
            .into_iter()
            .map(|r| Target {
                id: r.get("id"),
                name: r.get("name"),
                kind: r.get("kind"),
                url: r.get("url"),
                token: r.get("token"),
            })
            .collect())
    }

    pub async fn upsert_target(&self, t: &Target) -> Result<()> {
        sqlx::query(
            "INSERT INTO targets (id, name, kind, url, token) VALUES ($1,$2,$3,$4,$5)
             ON CONFLICT (id) DO UPDATE SET name=$2, kind=$3, url=$4, token=$5",
        )
        .bind(&t.id)
        .bind(&t.name)
        .bind(&t.kind)
        .bind(&t.url)
        .bind(&t.token)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_target(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM targets WHERE id=$1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn list_jobs(&self) -> Result<Vec<Job>> {
        let rows = sqlx::query(
            "SELECT id, name, description, definition, enabled FROM jobs ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| Job {
                id: r.get("id"),
                name: r.get("name"),
                description: r.get("description"),
                definition: r.get("definition"),
                enabled: r.get("enabled"),
            })
            .collect())
    }

    pub async fn upsert_job(&self, j: &Job) -> Result<()> {
        sqlx::query(
            "INSERT INTO jobs (id, name, description, definition, enabled)
             VALUES ($1,$2,$3,$4,$5)
             ON CONFLICT (id) DO UPDATE
               SET name=$2, description=$3, definition=$4, enabled=$5, updated_at=now()",
        )
        .bind(&j.id)
        .bind(&j.name)
        .bind(&j.description)
        .bind(&j.definition)
        .bind(j.enabled)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_job(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM jobs WHERE id=$1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    pub async fn record_run(&self, r: &JobRun) -> Result<()> {
        let metadata = if r.metadata.is_null() {
            json!({})
        } else {
            r.metadata.clone()
        };
        sqlx::query(
            "INSERT INTO job_runs (id, job_id, target_id, status, trigger, finished_at, metadata)
             VALUES ($1,$2,$3,$4,$5, CASE WHEN $6 THEN now() ELSE NULL END, $7)
             ON CONFLICT (id) DO UPDATE
               SET status=$4,
                   finished_at=CASE WHEN $6 THEN now() ELSE job_runs.finished_at END,
                   metadata=$7",
        )
        .bind(&r.id)
        .bind(&r.job_id)
        .bind(&r.target_id)
        .bind(&r.status)
        .bind(&r.trigger)
        .bind(matches!(r.status.as_str(), "complete" | "failed"))
        .bind(metadata)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_runs(&self, job_id: Option<&str>) -> Result<Vec<JobRun>> {
        let rows = sqlx::query(
            "SELECT id, job_id, target_id, status, trigger, metadata,
                    started_at::text AS started_at, finished_at::text AS finished_at
             FROM job_runs
             WHERE ($1::text IS NULL OR job_id = $1)
             ORDER BY started_at DESC LIMIT 200",
        )
        .bind(job_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| JobRun {
                id: r.get("id"),
                job_id: r.get("job_id"),
                target_id: r.get("target_id"),
                status: r.get("status"),
                trigger: r.get("trigger"),
                started_at: r.get("started_at"),
                finished_at: r.get("finished_at"),
                metadata: r.get("metadata"),
            })
            .collect())
    }

    pub async fn get_job(&self, id: &str) -> Result<Option<Job>> {
        Ok(self.list_jobs().await?.into_iter().find(|j| j.id == id))
    }

    pub async fn get_target(&self, id: &str) -> Result<Option<Target>> {
        Ok(self.list_targets().await?.into_iter().find(|t| t.id == id))
    }

    pub async fn record_step(&self, step: &JobRunStep) -> Result<()> {
        sqlx::query(
            "INSERT INTO job_run_steps (run_id, step_idx, template, args, status, row_count, metrics, log)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8)",
        )
        .bind(&step.run_id)
        .bind(step.step_idx)
        .bind(&step.template)
        .bind(&step.args)
        .bind(&step.status)
        .bind(step.row_count)
        .bind(&step.metrics)
        .bind(&step.log)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    /// Read a key from the `settings` table (None if absent). Used by the P4
    /// scheduler to persist its watermark across app restarts (catch-up).
    pub async fn get_setting(&self, key: &str) -> Result<Option<String>> {
        let row = sqlx::query("SELECT value FROM settings WHERE key=$1")
            .bind(key)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get::<String, _>("value")))
    }

    /// Try to acquire/renew the singleton scheduler lease for `holder` until
    /// `now + ttl` (P4c leader election). Atomic: the lease is granted only if it
    /// is currently vacant, expired, or already held by `holder`. Returns true if
    /// this caller now holds the lease.
    pub async fn try_acquire_lease(&self, holder: &str, now: i64, ttl: i64) -> Result<bool> {
        let row = sqlx::query(
            "INSERT INTO scheduler_lease (id, holder, expires_at)
             VALUES ('singleton', $1, $2)
             ON CONFLICT (id) DO UPDATE
               SET holder = excluded.holder, expires_at = excluded.expires_at
               WHERE scheduler_lease.expires_at < $3 OR scheduler_lease.holder = $1
             RETURNING holder",
        )
        .bind(holder)
        .bind(now + ttl)
        .bind(now)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.is_some())
    }

    /// Upsert a key in the `settings` table.
    pub async fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        sqlx::query(
            "INSERT INTO settings (key, value) VALUES ($1,$2)
             ON CONFLICT (key) DO UPDATE SET value=excluded.value",
        )
        .bind(key)
        .bind(value)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    // --- Connector instances (ADR-0032) ------------------------------------

    pub async fn list_connector_instances(&self) -> Result<Vec<ConnectorInstanceRecord>> {
        let rows = sqlx::query(
            "SELECT id, driver, name, params, job_param_defaults
             FROM connector_instances ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| ConnectorInstanceRecord {
                id: r.get("id"),
                driver: r.get("driver"),
                name: r.get("name"),
                params: r.get("params"),
                job_param_defaults: r.get("job_param_defaults"),
            })
            .collect())
    }

    pub async fn upsert_connector_instance(&self, i: &ConnectorInstanceRecord) -> Result<()> {
        sqlx::query(
            "INSERT INTO connector_instances (id, driver, name, params, job_param_defaults)
             VALUES ($1,$2,$3,$4,$5)
             ON CONFLICT (id) DO UPDATE
               SET driver=$2, name=$3, params=$4, job_param_defaults=$5, updated_at=now()",
        )
        .bind(&i.id)
        .bind(&i.driver)
        .bind(&i.name)
        .bind(&i.params)
        .bind(&i.job_param_defaults)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_connector_instance(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM connector_instances WHERE id=$1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // --- Bindings (ADR-0032) ------------------------------------------------

    pub async fn list_bindings(&self) -> Result<Vec<Binding>> {
        let rows = sqlx::query(
            "SELECT id, name, instance_id, run_params, contract_ref, mapping, schema
             FROM bindings ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows.into_iter().map(row_to_binding).collect())
    }

    pub async fn get_binding(&self, id: &str) -> Result<Option<Binding>> {
        let row = sqlx::query(
            "SELECT id, name, instance_id, run_params, contract_ref, mapping, schema
             FROM bindings WHERE id=$1",
        )
        .bind(id)
        .fetch_optional(&self.pool)
        .await?;
        Ok(row.map(row_to_binding))
    }

    pub async fn upsert_binding(&self, b: &Binding) -> Result<()> {
        sqlx::query(
            "INSERT INTO bindings (id, name, instance_id, run_params, contract_ref, mapping, schema)
             VALUES ($1,$2,$3,$4,$5,$6,$7)
             ON CONFLICT (id) DO UPDATE
               SET name=$2, instance_id=$3, run_params=$4, contract_ref=$5,
                   mapping=$6, schema=$7, updated_at=now()",
        )
        .bind(&b.id)
        .bind(&b.name)
        .bind(&b.instance_id)
        .bind(&b.run_params)
        .bind(&b.contract_ref)
        .bind(&b.mapping)
        .bind(&b.schema)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn delete_binding(&self, id: &str) -> Result<()> {
        sqlx::query("DELETE FROM bindings WHERE id=$1")
            .bind(id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    // --- Staged relations (ADR-0033) ----------------------------------------
    //
    // The store works in terms of the serialized `StagedRelationManifest`
    // (`serde_json::Value`); the queryable columns are projected from it. This
    // keeps the control-plane module free of engine types — the caller
    // (`quant_fabric::staging`) owns the manifest shape.

    pub async fn upsert_staged_relation(&self, manifest: &Value) -> Result<()> {
        let relation_id = manifest
            .get("relationId")
            .and_then(Value::as_str)
            .ok_or_else(|| anyhow::anyhow!("staged relation manifest missing `relationId`"))?;
        let content_hash = manifest
            .get("contentHash")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let retention = manifest
            .get("retention")
            .and_then(Value::as_str)
            .unwrap_or("ephemeral");
        let name = manifest.get("name").and_then(Value::as_str);
        let created_ms = manifest
            .get("createdMs")
            .and_then(Value::as_i64)
            .unwrap_or(0);
        sqlx::query(
            "INSERT INTO staged_relations
               (relation_id, name, content_hash, retention, manifest, created_ms)
             VALUES ($1,$2,$3,$4,$5,$6)
             ON CONFLICT (relation_id) DO UPDATE
               SET name=$2, content_hash=$3, retention=$4, manifest=$5,
                   created_ms=$6, updated_at=now()",
        )
        .bind(relation_id)
        .bind(name)
        .bind(content_hash)
        .bind(retention)
        .bind(manifest)
        .bind(created_ms)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_staged_relations(&self) -> Result<Vec<Value>> {
        let rows = sqlx::query(
            "SELECT manifest FROM staged_relations ORDER BY created_ms DESC, updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| r.get::<Value, _>("manifest"))
            .collect())
    }

    pub async fn get_staged_relation(&self, relation_id: &str) -> Result<Option<Value>> {
        let row = sqlx::query("SELECT manifest FROM staged_relations WHERE relation_id=$1")
            .bind(relation_id)
            .fetch_optional(&self.pool)
            .await?;
        Ok(row.map(|r| r.get::<Value, _>("manifest")))
    }

    /// Change a staged relation's retention (e.g. pinning a `session` result to
    /// `pinned`). Reads/modifies/rewrites the manifest so the column and the
    /// embedded `retention` stay consistent.
    pub async fn set_staged_relation_retention(
        &self,
        relation_id: &str,
        retention: &str,
    ) -> Result<()> {
        let Some(mut manifest) = self.get_staged_relation(relation_id).await? else {
            anyhow::bail!("unknown staged relation `{relation_id}`");
        };
        if let Some(obj) = manifest.as_object_mut() {
            obj.insert(
                "retention".to_string(),
                Value::String(retention.to_string()),
            );
        }
        self.upsert_staged_relation(&manifest).await
    }

    /// Forget a staged relation (drop its control-plane record). The immutable
    /// bytes in the ArtifactStore are GC'd separately by retention sweep.
    pub async fn delete_staged_relation(&self, relation_id: &str) -> Result<()> {
        sqlx::query("DELETE FROM staged_relations WHERE relation_id=$1")
            .bind(relation_id)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Sweep all staged relations of a given retention tier (e.g. clearing
    /// `ephemeral` scratch). Returns the number removed. Retention-tier GC
    /// (ADR-0033): `pinned`/`session` are untouched unless explicitly named.
    pub async fn delete_staged_relations_by_retention(&self, retention: &str) -> Result<u64> {
        let result = sqlx::query("DELETE FROM staged_relations WHERE retention=$1")
            .bind(retention)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }

    // ---- Cluster read model (ADR-0021 P2b): engine metadata log → local store ----

    /// The projection watermark for a coordinator: the highest consensus log
    /// index already applied to the read model (0 when none). Sync fetches the
    /// log and applies strictly above this.
    pub async fn cluster_read_model_watermark(&self, coordinator_url: &str) -> Result<i64> {
        let row = sqlx::query(
            "SELECT COALESCE(MAX(log_index), 0) AS hi FROM cluster_events WHERE coordinator_url=$1",
        )
        .bind(coordinator_url)
        .fetch_one(&self.pool)
        .await?;
        Ok(row.get::<i64, _>("hi"))
    }

    /// Project committed metadata-log entries into the read model. Insert-only +
    /// keyed by (coordinator, index), so replaying the same entries is a no-op
    /// (committed Raft entries are immutable). Returns the number newly applied.
    pub async fn apply_cluster_events(
        &self,
        coordinator_url: &str,
        entries: &[(i64, i64, i64, String, Value)],
    ) -> Result<u64> {
        let mut applied = 0;
        for (log_index, term, committed_ms, kind, command) in entries {
            let result = sqlx::query(
                "INSERT INTO cluster_events
                   (coordinator_url, log_index, term, committed_ms, kind, command)
                 VALUES ($1,$2,$3,$4,$5,$6)
                 ON CONFLICT (coordinator_url, log_index) DO NOTHING",
            )
            .bind(coordinator_url)
            .bind(log_index)
            .bind(term)
            .bind(committed_ms)
            .bind(kind)
            .bind(command)
            .execute(&self.pool)
            .await?;
            applied += result.rows_affected();
        }
        Ok(applied)
    }

    /// Recent read-model events for a coordinator, newest first.
    pub async fn list_cluster_events(
        &self,
        coordinator_url: &str,
        limit: i64,
    ) -> Result<Vec<Value>> {
        let rows = sqlx::query(
            "SELECT log_index, term, committed_ms, kind, command
             FROM cluster_events WHERE coordinator_url=$1
             ORDER BY log_index DESC LIMIT $2",
        )
        .bind(coordinator_url)
        .bind(limit)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| {
                json!({
                    "logIndex": r.get::<i64, _>("log_index"),
                    "term": r.get::<i64, _>("term"),
                    "committedMs": r.get::<i64, _>("committed_ms"),
                    "kind": r.get::<String, _>("kind"),
                    "command": r.get::<Value, _>("command"),
                })
            })
            .collect())
    }

    /// Backfill `connector_instances` from the legacy `connectors:instances`
    /// settings blob (a JSON array of `{id, driver, params, jobParamDefaults}`).
    /// Runs only when the table is empty and the blob exists; tolerant of a
    /// missing/invalid blob (yields nothing). Returns the number migrated.
    pub async fn migrate_connector_instances_from_blob(&self) -> Result<usize> {
        let existing: i64 = sqlx::query("SELECT count(*) AS n FROM connector_instances")
            .fetch_one(&self.pool)
            .await?
            .get("n");
        if existing > 0 {
            return Ok(0);
        }
        let Some(blob) = self.get_setting(LEGACY_CONNECTOR_INSTANCES_SETTING).await? else {
            return Ok(0);
        };
        let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&blob) else {
            return Ok(0);
        };
        let mut migrated = 0;
        for v in arr {
            let Some(id) = v.get("id").and_then(Value::as_str) else {
                continue;
            };
            let record = ConnectorInstanceRecord {
                id: id.to_string(),
                driver: v
                    .get("driver")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: v
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                params: v.get("params").cloned().unwrap_or_else(|| json!({})),
                job_param_defaults: v
                    .get("jobParamDefaults")
                    .cloned()
                    .unwrap_or_else(|| json!({})),
            };
            self.upsert_connector_instance(&record).await?;
            migrated += 1;
        }
        Ok(migrated)
    }

    pub async fn list_run_steps(&self, run_id: &str) -> Result<Vec<JobRunStep>> {
        let rows = sqlx::query(
            "SELECT run_id, step_idx, template, args, status, row_count, metrics, log
             FROM job_run_steps WHERE run_id=$1 ORDER BY step_idx",
        )
        .bind(run_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| JobRunStep {
                run_id: r.get("run_id"),
                step_idx: r.get("step_idx"),
                template: r.get("template"),
                args: r.get("args"),
                status: r.get("status"),
                row_count: r.get("row_count"),
                metrics: r.get("metrics"),
                log: r.get("log"),
            })
            .collect())
    }

    pub async fn list_pack_sources(&self) -> Result<Vec<PackSource>> {
        let rows = sqlx::query(
            "SELECT id, repo_url, owner, repo, ref_name, pack_path, monitor, auth_ref,
                    last_checked_at::text AS last_checked_at, last_commit, status, message
             FROM pack_sources ORDER BY updated_at DESC",
        )
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| PackSource {
                id: r.get("id"),
                repo_url: r.get("repo_url"),
                owner: r.get("owner"),
                repo: r.get("repo"),
                ref_name: r.get("ref_name"),
                pack_path: r.get("pack_path"),
                monitor: r.get("monitor"),
                auth_ref: r.get("auth_ref"),
                last_checked_at: r.get("last_checked_at"),
                last_commit: r.get("last_commit"),
                status: r.get("status"),
                message: r.get("message"),
            })
            .collect())
    }

    pub async fn get_pack_source(&self, id: &str) -> Result<Option<PackSource>> {
        Ok(self
            .list_pack_sources()
            .await?
            .into_iter()
            .find(|s| s.id == id))
    }

    pub async fn upsert_pack_source(&self, s: &PackSource) -> Result<()> {
        sqlx::query(
            "INSERT INTO pack_sources
              (id, repo_url, owner, repo, ref_name, pack_path, monitor, auth_ref, last_commit, status, message)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11)
             ON CONFLICT (id) DO UPDATE
               SET repo_url=$2, owner=$3, repo=$4, ref_name=$5, pack_path=$6, monitor=$7,
                   auth_ref=$8, last_commit=COALESCE($9, pack_sources.last_commit),
                   status=$10, message=$11, updated_at=now()",
        )
        .bind(&s.id)
        .bind(&s.repo_url)
        .bind(&s.owner)
        .bind(&s.repo)
        .bind(&s.ref_name)
        .bind(&s.pack_path)
        .bind(&s.monitor)
        .bind(&s.auth_ref)
        .bind(&s.last_commit)
        .bind(&s.status)
        .bind(&s.message)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn update_pack_source_monitor(&self, source_id: &str, monitor: &str) -> Result<()> {
        let result = sqlx::query(
            "UPDATE pack_sources
             SET monitor=$2, updated_at=now()
             WHERE id=$1",
        )
        .bind(source_id)
        .bind(monitor)
        .execute(&self.pool)
        .await?;
        if result.rows_affected() == 0 {
            anyhow::bail!("unknown pack source `{source_id}`");
        }
        Ok(())
    }

    pub async fn mark_pack_source_checked(
        &self,
        source_id: &str,
        status: &str,
        commit_sha: Option<&str>,
        message: Option<&str>,
    ) -> Result<()> {
        sqlx::query(
            "UPDATE pack_sources
             SET last_checked_at=now(), last_commit=COALESCE($2, last_commit),
                 status=$3, message=$4, updated_at=now()
             WHERE id=$1",
        )
        .bind(source_id)
        .bind(commit_sha)
        .bind(status)
        .bind(message)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn upsert_pack_version(&self, v: &PackVersion) -> Result<()> {
        sqlx::query(
            "INSERT INTO pack_versions
              (id, source_id, pack_id, pack_name, manifest_version, commit_sha, ref_name, content_hash, cache_dir)
             VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
             ON CONFLICT (source_id, commit_sha, pack_id) DO UPDATE
               SET pack_name=$4, manifest_version=$5, ref_name=$7, content_hash=$8, cache_dir=$9",
        )
        .bind(&v.id)
        .bind(&v.source_id)
        .bind(&v.pack_id)
        .bind(&v.pack_name)
        .bind(&v.manifest_version)
        .bind(&v.commit_sha)
        .bind(&v.ref_name)
        .bind(&v.content_hash)
        .bind(&v.cache_dir)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_pack_versions(&self, source_id: Option<&str>) -> Result<Vec<PackVersion>> {
        let rows = sqlx::query(
            "SELECT id, source_id, pack_id, pack_name, manifest_version, commit_sha,
                    ref_name, content_hash, cache_dir, discovered_at::text AS discovered_at
             FROM pack_versions
             WHERE ($1::text IS NULL OR source_id=$1)
             ORDER BY discovered_at DESC",
        )
        .bind(source_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| PackVersion {
                id: r.get("id"),
                source_id: r.get("source_id"),
                pack_id: r.get("pack_id"),
                pack_name: r.get("pack_name"),
                manifest_version: r.get("manifest_version"),
                commit_sha: r.get("commit_sha"),
                ref_name: r.get("ref_name"),
                content_hash: r.get("content_hash"),
                cache_dir: r.get("cache_dir"),
                discovered_at: r.get("discovered_at"),
            })
            .collect())
    }

    pub async fn record_pack_sync_event(
        &self,
        source_id: &str,
        status: &str,
        commit_sha: Option<&str>,
        message: &str,
    ) -> Result<()> {
        sqlx::query(
            "INSERT INTO pack_sync_events (source_id, status, commit_sha, message)
             VALUES ($1,$2,$3,$4)",
        )
        .bind(source_id)
        .bind(status)
        .bind(commit_sha)
        .bind(message)
        .execute(&self.pool)
        .await?;
        Ok(())
    }

    pub async fn list_pack_sync_events(
        &self,
        source_id: Option<&str>,
    ) -> Result<Vec<PackSyncEvent>> {
        let rows = sqlx::query(
            "SELECT id, source_id, status, commit_sha, message, created_at::text AS created_at
             FROM pack_sync_events
             WHERE ($1::text IS NULL OR source_id=$1)
             ORDER BY created_at DESC LIMIT 200",
        )
        .bind(source_id)
        .fetch_all(&self.pool)
        .await?;
        Ok(rows
            .into_iter()
            .map(|r| PackSyncEvent {
                id: r.get("id"),
                source_id: r.get("source_id"),
                status: r.get("status"),
                commit_sha: r.get("commit_sha"),
                message: r.get("message"),
                created_at: r.get("created_at"),
            })
            .collect())
    }
}

fn row_to_binding(r: sqlx::postgres::PgRow) -> Binding {
    Binding {
        id: r.get("id"),
        name: r.get("name"),
        instance_id: r.get("instance_id"),
        run_params: r.get("run_params"),
        contract_ref: r.get("contract_ref"),
        mapping: r.get("mapping"),
        schema: r.get("schema"),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct JobRunStep {
    pub run_id: String,
    pub step_idx: i32,
    pub template: String,
    pub args: Value,
    pub status: String,
    pub row_count: Option<i32>,
    pub metrics: Value,
    pub log: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn temp_store() -> (ControlStore, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let server = pglite_oxide::PgliteServer::builder()
            .path(dir.path().join("pg").to_string_lossy().into_owned())
            .start()
            .expect("start pglite");
        let server: &'static pglite_oxide::PgliteServer = Box::leak(Box::new(server));
        let store = ControlStore::connect(&server.database_url())
            .await
            .expect("connect store");
        (store, dir)
    }

    fn pack_source(id: &str) -> PackSource {
        PackSource {
            id: id.to_string(),
            repo_url: "https://github.com/acme/packs".to_string(),
            owner: "acme".to_string(),
            repo: "packs".to_string(),
            ref_name: "main".to_string(),
            pack_path: String::new(),
            monitor: "manual".to_string(),
            auth_ref: None,
            last_checked_at: None,
            last_commit: None,
            status: "linked".to_string(),
            message: None,
        }
    }

    fn instance(id: &str, driver: &str) -> ConnectorInstanceRecord {
        ConnectorInstanceRecord {
            id: id.to_string(),
            driver: driver.to_string(),
            name: format!("{id}-name"),
            params: json!({ "host": "db", "port": "5432" }),
            job_param_defaults: json!({ "query": "SELECT 1" }),
        }
    }

    #[tokio::test]
    async fn connector_instances_round_trip_and_delete() {
        let (store, _dir) = temp_store().await;
        store
            .upsert_connector_instance(&instance("inst-1", "postgres"))
            .await
            .expect("insert instance");

        let got = store.list_connector_instances().await.expect("list");
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].driver, "postgres");
        assert_eq!(got[0].params.get("host").unwrap(), "db");

        store
            .delete_connector_instance("inst-1")
            .await
            .expect("delete");
        assert!(store
            .list_connector_instances()
            .await
            .expect("list")
            .is_empty());
    }

    #[tokio::test]
    async fn bindings_round_trip_with_optional_typed_overlay() {
        let (store, _dir) = temp_store().await;
        store
            .upsert_connector_instance(&instance("inst-1", "postgres"))
            .await
            .expect("instance");

        // A raw (untyped) binding: no contract_ref / mapping — still first-class.
        let raw = Binding {
            id: "b-raw".to_string(),
            name: "raw positions".to_string(),
            instance_id: "inst-1".to_string(),
            run_params: json!({ "query": "SELECT * FROM positions" }),
            contract_ref: None,
            mapping: None,
            schema: None,
        };
        // A typed binding: optional contract + column mapping overlay.
        let typed = Binding {
            id: "b-typed".to_string(),
            name: "positions".to_string(),
            instance_id: "inst-1".to_string(),
            run_params: json!({ "query": "SELECT * FROM positions" }),
            contract_ref: Some("Position.v1".to_string()),
            mapping: Some(json!({ "qty": "quantity", "symbol": "instrument" })),
            schema: Some(json!([
                { "name": "quantity", "kind": "number" },
                { "name": "instrument", "kind": "string" }
            ])),
        };
        store.upsert_binding(&raw).await.expect("raw");
        store.upsert_binding(&typed).await.expect("typed");

        let got = store
            .get_binding("b-typed")
            .await
            .expect("get")
            .expect("some");
        assert_eq!(got.contract_ref.as_deref(), Some("Position.v1"));
        assert_eq!(got.mapping.unwrap().get("qty").unwrap(), "quantity");
        // The discovered schema (32-C1 option c) round-trips for conformance.
        assert_eq!(got.schema.unwrap().as_array().unwrap().len(), 2);

        let raw_back = store
            .get_binding("b-raw")
            .await
            .expect("get")
            .expect("some");
        assert!(raw_back.contract_ref.is_none() && raw_back.mapping.is_none());

        assert_eq!(store.list_bindings().await.expect("list").len(), 2);

        // Deleting the instance cascades to its bindings (FK ON DELETE CASCADE).
        store
            .delete_connector_instance("inst-1")
            .await
            .expect("delete instance");
        assert!(store.list_bindings().await.expect("list").is_empty());
    }

    #[tokio::test]
    async fn backfills_connector_instances_from_legacy_blob() {
        let (store, _dir) = temp_store().await;
        // Seed the legacy settings blob, then run the backfill explicitly (the
        // table is already empty post-connect, so the connect-time run was a no-op).
        store
            .set_setting(
                LEGACY_CONNECTOR_INSTANCES_SETTING,
                &json!([
                    { "id": "fs-src", "driver": "filesystem.csv",
                      "params": { "root_path": "/data" },
                      "jobParamDefaults": { "pattern": "*.csv" } },
                    { "id": "no-driver" }
                ])
                .to_string(),
            )
            .await
            .expect("seed blob");

        let migrated = store
            .migrate_connector_instances_from_blob()
            .await
            .expect("migrate");
        assert_eq!(migrated, 2);
        let got = store.list_connector_instances().await.expect("list");
        assert_eq!(got.len(), 2);
        let fs = got.iter().find(|i| i.id == "fs-src").unwrap();
        assert_eq!(fs.driver, "filesystem.csv");
        assert_eq!(fs.params.get("root_path").unwrap(), "/data");

        // Idempotent: a second run is a no-op because the table is non-empty.
        assert_eq!(
            store
                .migrate_connector_instances_from_blob()
                .await
                .expect("migrate again"),
            0
        );
    }

    #[tokio::test]
    async fn staged_relations_round_trip_pin_and_forget() {
        let (store, _dir) = temp_store().await;
        let manifest = json!({
            "relationId": "rel-1",
            "name": "downside_by_book",
            "contentHash": "abc123",
            "contentType": "application/vnd.apache.arrow.stream",
            "sizeBytes": 10,
            "rowCount": 2,
            "columns": ["book", "pnl"],
            "producer": { "operation": "group_sum", "operationVersion": "1.0.0" },
            "dependsOn": [],
            "retention": "session",
            "createdMs": 1234
        });
        store
            .upsert_staged_relation(&manifest)
            .await
            .expect("upsert");

        let listed = store.list_staged_relations().await.expect("list");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].get("relationId").unwrap(), "rel-1");

        // Pin: the stored manifest's retention flips to pinned.
        store
            .set_staged_relation_retention("rel-1", "pinned")
            .await
            .expect("pin");
        let got = store
            .get_staged_relation("rel-1")
            .await
            .expect("get")
            .expect("some");
        assert_eq!(got.get("retention").unwrap(), "pinned");

        // Pinning an unknown relation is an error.
        assert!(store
            .set_staged_relation_retention("nope", "pinned")
            .await
            .is_err());

        // Forget drops the record.
        store.delete_staged_relation("rel-1").await.expect("forget");
        assert!(store
            .list_staged_relations()
            .await
            .expect("list")
            .is_empty());
    }

    #[tokio::test]
    async fn gc_sweeps_only_the_named_retention_tier() {
        let (store, _dir) = temp_store().await;
        let mk = |id: &str, retention: &str| {
            json!({
                "relationId": id, "contentHash": "h", "contentType": "x",
                "sizeBytes": 1, "rowCount": 0, "columns": [],
                "producer": {}, "dependsOn": [], "retention": retention, "createdMs": 0
            })
        };
        store
            .upsert_staged_relation(&mk("e1", "ephemeral"))
            .await
            .unwrap();
        store
            .upsert_staged_relation(&mk("e2", "ephemeral"))
            .await
            .unwrap();
        store
            .upsert_staged_relation(&mk("p1", "pinned"))
            .await
            .unwrap();

        let removed = store
            .delete_staged_relations_by_retention("ephemeral")
            .await
            .unwrap();
        assert_eq!(removed, 2);
        let left = store.list_staged_relations().await.unwrap();
        assert_eq!(left.len(), 1);
        assert_eq!(left[0].get("relationId").unwrap(), "p1");
    }

    #[tokio::test]
    async fn update_pack_source_monitor_updates_existing_source_and_rejects_unknown() {
        let (store, _dir) = temp_store().await;
        store
            .upsert_pack_source(&pack_source("source-1"))
            .await
            .expect("insert source");

        store
            .update_pack_source_monitor("source-1", "startup")
            .await
            .expect("update monitor");
        let source = store
            .get_pack_source("source-1")
            .await
            .expect("load source")
            .expect("source exists");
        assert_eq!(source.monitor, "startup");

        let error = store
            .update_pack_source_monitor("missing-source", "startup")
            .await
            .expect_err("unknown source should fail");
        assert!(
            error.to_string().contains("unknown pack source"),
            "unexpected error: {error}"
        );
    }
}
