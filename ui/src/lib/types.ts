export type InputKind =
  | "Table"
  | "TypedRelation"
  | "Column"
  | "Columns"
  | "Number"
  | "Date";
export type RelationInputKind = "Table" | "TypedRelation";
export type ScalarInputKind = Exclude<InputKind, RelationInputKind>;
export type TemplateKind = "aggregate" | "projection";

export interface TemplateInput {
  name: string;
  kind: InputKind;
  contract?: string | null;
}

export interface Template {
  name: string;
  displayName: string;
  shortName: string;
  description: string;
  kind: TemplateKind;
  inputs: TemplateInput[];
  requiredAggregateKinds: string[];
  category: string;
  path: string | null;
  source: string | null;
}

// Matches `quant_fabric::studio::RunOutput` (camelCase serde).
export interface RunResult {
  status: string;
  rows: readonly Record<string, unknown>[];
  rowCount?: number | undefined;
  elapsedMs?: number | undefined;
  error?: string | null | undefined;
}

// ---- Explore notebook (ADR-0029) ----
export type CellKind = "sql" | "markdown";

/** A typed `{{placeholder}}` shared across a notebook's SQL cells. `Table` is
 *  excluded — the data source is the notebook-level `source`. */
export interface NotebookParam {
  name: string;
  /** Column | Columns | Number | Date. */
  kind: ScalarInputKind;
  /** Test value substituted into placeholders for preview runs. */
  value: string;
}

export interface NotebookCell {
  id: string;
  kind: CellKind;
  source: string;
}

export interface Notebook {
  id: string;
  name: string;
  /** Data-source file path materialized as the `input` table for SQL cells. */
  source: string;
  params: NotebookParam[];
  cells: NotebookCell[];
}

/** A saved notebook file in the library (mirrors the Rust `NotebookFile`). The
 *  `dir` matches a configured notebook path so the sidebar can group by folder. */
export interface NotebookFile {
  path: string;
  dir: string;
  name: string;
  cellCount: number;
  modifiedMs: number;
}

/** A node in the Explore data tree (mirrors the Rust `DataTreeNode`, ADR-0010).
 *  The tree surfaces all file-backed state under `$QUANT_FABRIC_HOME`; every
 *  file carries a `queryable` flag — queryable ones (`format` set) open a SQL
 *  cell that reads the absolute `path` directly, the rest are browsable. */
export interface DataTreeNode {
  name: string;
  path: string;
  isDir: boolean;
  queryable: boolean;
  format?: "parquet" | "csv" | "tsv" | "json" | null;
  sizeBytes?: number | null;
  children: DataTreeNode[];
}

/** Postgres-wire coordinates of the embedded control-plane server, for
 *  attaching an external notebook (mirrors the Rust `ExternalPg`). */
export interface ExternalEndpoint {
  url: string;
  host: string;
  port: number;
  database: string;
  user: string;
  pinned: boolean;
}

// ---- Cohesive packs (ADR-0027) ----
export interface PackTemplateRef {
  name: string;
  category: string;
}
export interface PackConnectorRef {
  name: string;
  kind: ConnectorKind;
  category: string;
  driver: string;
}
export interface JobTemplateInput {
  name: string;
  kind: string;
}
export interface JobTemplateSource {
  connector?: string | null;
  file?: string | null;
}
export interface JobTemplateStep {
  template: string;
  args: Record<string, string>;
}
export interface JobTemplate {
  name: string;
  description: string;
  inputs: JobTemplateInput[];
  source: JobTemplateSource;
  steps: JobTemplateStep[];
  schedule?: string | null;
}
export interface Pack {
  id: string;
  name: string;
  version: string;
  description: string;
  tags: string[];
  author?: string | null;
  license?: string | null;
  dir: string;
  /** `core` packs are shipped & read-only; `user` packs are editable; `github`
   *  packs are immutable cached snapshots from linked repositories. */
  origin: "core" | "user" | "github";
  templates: PackTemplateRef[];
  connectors: PackConnectorRef[];
  jobTemplates: JobTemplate[];
}
/** One editable text file within a pack (relative path + contents). */
export interface PackFile {
  path: string;
  contents: string;
}

export type PackSourceMonitor = "manual" | "startup";

export interface LinkGithubPackRequest {
  repoUrl: string;
  refName?: string;
  packPath?: string;
  monitor?: PackSourceMonitor;
  authRef?: string | null;
}

export interface PackSource {
  id: string;
  repoUrl: string;
  owner: string;
  repo: string;
  refName: string;
  packPath: string;
  monitor: PackSourceMonitor | string;
  authRef?: string | null;
  lastCheckedAt?: string | null;
  lastCommit?: string | null;
  status: string;
  message?: string | null;
}

export interface PackVersion {
  id: string;
  sourceId: string;
  packId: string;
  packName: string;
  manifestVersion: string;
  commitSha: string;
  refName: string;
  contentHash: string;
  cacheDir: string;
  discoveredAt: string;
}

export interface PackSyncEvent {
  id: number;
  sourceId: string;
  status: string;
  commitSha?: string | null;
  message: string;
  createdAt: string;
}

export interface PackVersionCompare {
  from: PackVersion;
  to: PackVersion;
  status: string;
  changed: boolean;
  notes: string[];
}

export interface JobPackUpdateResolution {
  status: string;
  current?: PackVersion | null;
  latest?: PackVersion | null;
  notes: string[];
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface Target {
  id: string;
  name: string;
  kind: "embedded" | "remote";
  url: string;
  token?: string;
}

/** Embedded in-process engine compute config (mirrors `EmbeddedConfig` in
 *  `src-tauri/src/lib.rs`). `cores` is the configured parallelism (applied on
 *  next launch); `running` is the live worker count; `envOverride` pins it. */
export interface EmbeddedConfig {
  cores: number;
  maxCores: number;
  running: number;
  envOverride: boolean;
}

export interface SourceColumn {
  name: string;
  dataType: string;
}

export interface SourceSchema {
  columns: readonly SourceColumn[];
  preview: readonly Record<string, unknown>[];
}

export interface JobStep {
  kind?: "unit" | "sql" | "markdown" | undefined;
  template: string;
  args: Record<string, string | number>;
  sql?: string | undefined;
  markdown?: string | undefined;
}

export interface JobDefinition {
  source: string;
  steps: JobStep[];
  schedule?: string | undefined;
  /** Pack provenance for jobs scaffolded from a versioned pack. GitHub-linked
   *  packs populate source/version/commit fields for update resolution. */
  packBinding?: JobPackBinding | undefined;
  /** Shared sink base URI for remote inter-step chaining (e.g. `s3://…` or a
   *  shared dir the cluster can write + a later step can read). */
  sink?: string | undefined;
  /** Enabled source-connector instance id that feeds this job's input (ADR-0026). */
  sourceConnectorId?: string | undefined;
  /** First-class source binding id; resolved backend-side to instance + run params. */
  sourceBindingId?: string | undefined;
  /** Per-job inputs bound to the source connector's `job_params`. */
  sourceConnectorParams?: Record<string, string> | undefined;
  /** Enabled target-connector instance id that receives this job's output. */
  targetConnectorId?: string | undefined;
  /** First-class target binding id; resolved backend-side to instance + run params. */
  targetBindingId?: string | undefined;
  /** Per-job inputs bound to the target connector's `job_params`. */
  targetConnectorParams?: Record<string, string> | undefined;
}

export interface JobPackBinding {
  origin: "core" | "user" | "github";
  packId: string;
  packName: string;
  jobTemplate: string;
  sourceId?: string | null;
  versionId?: string | null;
  refName?: string | null;
  commitSha?: string | null;
  contentHash?: string | null;
  updatePolicy: "pinned" | "track";
}

export interface Job {
  id: string;
  name: string;
  description: string;
  definition: JobDefinition;
  enabled: boolean;
}

export interface JobRun {
  id: string;
  jobId: string;
  targetId?: string | null | undefined;
  status: string;
  trigger: string;
  startedAt: string;
  finishedAt?: string | null | undefined;
  metadata?: JobRunMetadata | undefined;
}

export interface JobRunOutputMetadata {
  sinkUri?: string | undefined;
  targetConnectorId?: string | undefined;
  targetBindingId?: string | undefined;
  targetPath?: string | undefined;
  exportDriver?: string | undefined;
  stagedRelationId?: string | undefined;
}

export interface JobRunSummaryMetadata {
  stepCount?: number | undefined;
  successfulSteps?: number | undefined;
  failedSteps?: number | undefined;
  totalRowCount?: number | undefined;
}

export interface JobRunMetadata {
  packBinding?: JobPackBinding | null | undefined;
  executionMode?: "scheduled" | "on_demand" | string | undefined;
  sourceMode?: "binding" | "connector" | "file" | string | undefined;
  syncMode?: string | undefined;
  output?: JobRunOutputMetadata | null | undefined;
  summary?: JobRunSummaryMetadata | null | undefined;
  /** Reproducibility class of the Flow that ran (ADR-0030 P4): a deterministic
   *  job over pinned inputs is reproducible. */
  determinism?: "deterministic" | "nondeterministic";
  /** Future-proof diagnostics payloads without requiring frontend model churn. */
  diagnostics?: Record<string, unknown> | null | undefined;
}

export interface JobRunStep {
  runId: string;
  stepIdx: number;
  template: string;
  args: Record<string, unknown>;
  status: string;
  rowCount?: number | null | undefined;
  metrics?: Record<string, unknown> | null | undefined;
  log?: string | null | undefined;
}

export interface JobRunListResult {
  runs: JobRun[];
  total: number;
  limit: number;
  offset: number;
}

// ---- Connectors (ADR-0026) — mirrors `quant_fabric::connectors`. ----

export type ConnectorKind = "source" | "target";
export type ConnectorParamKind =
  | "string"
  | "path"
  | "url"
  | "number"
  | "bool"
  | "secret"
  | "enum";
export type ConnectorParamInput =
  | "text"
  | "file"
  | "directory"
  | "regex"
  | "glob"
  | "sql"
  | "json"
  | "toml"
  | "yaml"
  | "cron"
  | "date"
  | "datetime"
  | "duration"
  | "identifier"
  | "secret_ref"
  | "multiline"
  | "columns";

export interface ConnectorParam {
  name: string;
  label?: string | null;
  kind: ConnectorParamKind;
  input?: ConnectorParamInput | null;
  group?: string | null;
  advanced?: boolean;
  sensitive?: boolean;
  dependsOn?: string | null;
  min?: number | null;
  max?: number | null;
  unit?: string | null;
  required: boolean;
  default?: string | null;
  placeholder?: string | null;
  help?: string | null;
  options: string[];
}

export interface ConnectorAction {
  name: string;
  label: string;
  /** CLI-style flag passed to the driver (e.g. `--preview`). */
  flag: string;
  description?: string | null;
}

/** A connector *spec* from the TOML library (a type, not a configured instance). */
export interface ConnectorSpec {
  name: string;
  description: string;
  kind: ConnectorKind;
  driver: string;
  icon?: string | null;
  category: string;
  params: ConnectorParam[];
  jobParams: ConnectorParam[];
  actions: ConnectorAction[];
  example: Record<string, string>;
  /** A `path`-typed example value does not resolve on disk → flag as example. */
  exampleUnresolved: boolean;
  /** A real driver for this connector is installed/resolvable. When false the
   *  connector can't actually run — shown as unavailable in the catalog. */
  available: boolean;
  path?: string | null;
  source?: string | null;
}

/** A configured connector *instance* (spec + user params + enabled), persisted. */
export interface ConnectorInstance {
  id: string;
  /** The spec this instance was created from (`ConnectorSpec.name`). */
  spec: string;
  kind: ConnectorKind;
  driver: string;
  /** User-facing instance name. */
  name: string;
  /** Filled connection params (`spec.params`). */
  params: Record<string, string>;
  /** Optional per-job defaults copied when this connector is attached to a job. */
  jobParamDefaults?: Record<string, string>;
  enabled: boolean;
  /** Reproducibility pin state from `celeritas lock --check`. */
  lock?: {
    status: "locked" | "stale" | "missing" | "untracked" | "error";
    message?: string | null;
  } | null;
  /** Result of the last Test, if any. */
  lastTest?: { status: string; message: string; at: string } | null;
}

export interface ConnectorTestResult {
  status: string; // "pass" | "fail" | "example"
  message: string;
  elapsedMs?: number | undefined;
}

export interface ConnectorActionResult {
  status: string; // "complete" | "failed"
  message: string;
  rows: Record<string, unknown>[];
  elapsedMs?: number | undefined;
}

// ---- Connector Store / remote registry (ADR-0016). ----

/** A setting from a package's published manifest (`manifest.settings`). */
export interface RegistrySetting {
  name: string;
  /** string | integer | number | boolean | secret | enum (publication kinds). */
  kind?: string | null;
  /** connection | run scope, when published. */
  scope?: string | null;
  required?: boolean;
  default?: string | number | boolean | null;
  description?: string | null;
  /** Choices for an enum setting, when published. */
  options?: string[] | null;
}

/** A package entry in a remote registry manifest, normalized by the sidecar. */
export interface RegistryPackage {
  name: string;
  title: string;
  summary: string;
  description: string;
  type: "source" | "target";
  version: string;
  author: { name: string; email?: string; url?: string };
  repository: string;
  license: string;
  artifact: { publisher: string; package: string; install_command: string };
  manifest: {
    capabilities?: string[];
    settings?: RegistrySetting[];
  };
  tags: string[];
  category: string;
  /** Already present in the local `hub list`. */
  installed: boolean;
  /** The exact command the sidecar would run to install (provenance/display). */
  installCommand: string;
  /** Who packaged it (= author). */
  packagedBy: { name: string; email?: string; url?: string };
  /** When it was packaged/published (ISO string). */
  packagedAt: string;
}

/** Registry-level metadata that accompanies the package list. */
export interface RegistryInfo {
  name?: string;
  homepage?: string;
  url?: string;
  [k: string]: unknown;
}

export interface RegistryListResult {
  registry: RegistryInfo;
  packages: RegistryPackage[];
}

export interface RegistryInstallResult {
  status: "installed" | "failed";
  message: string;
  installCommand: string;
  log: string;
}

// ---- First-class connector bindings (ADR-0032). ----

export interface ConnectorInstanceRecord {
  id: string;
  driver: string;
  name: string;
  params: Record<string, string>;
  jobParamDefaults: Record<string, string>;
  selection?: string[];
  lock?: {
    status: "locked" | "stale" | "missing" | "untracked" | "error";
    message?: string | null;
  } | null;
}

export interface Binding {
  id: string;
  name: string;
  instanceId: string;
  runParams: Record<string, string>;
  contractRef?: string | null;
  mapping?: Record<string, string> | null;
  schema?: SourceSchema | null;
}

export interface SharedConnectorBinding {
  name: string;
  file: string;
  driver: string;
  status: string;
  landed: boolean;
  description: string;
  researchUse: string;
  auth: string;
  cadence: string;
  sinkStream?: string | null;
  sinkPath?: string | null;
  panelView?: string | null;
  panelValueName?: string | null;
  panelAssetClass?: string | null;
  note?: string | null;
}

export interface FieldMismatch {
  field: string;
  expected: string;
  found: string;
}

export interface Conformance {
  conforms: boolean;
  missing: string[];
  mismatches: FieldMismatch[];
}

export interface BindingConformance {
  binding: Binding;
  conformance: Conformance;
}

export interface ContractField {
  name: string;
  kind: string;
  required: boolean;
  description?: string;
}

// A domain contract surfaced for the binding contract-picker (ADR-0032), from
// the `list_contracts` command. `reference` is what templates declare (`Name.vN`).
export interface ContractInfo {
  reference: string;
  domain: string;
  name: string;
  version: string;
  description: string;
  fields: ContractField[];
}

// ---- Immutable staged relations (ADR-0033). ----

export type StagedRelationRetention = "ephemeral" | "session" | "pinned";
export type StagedRelationFreshness = "fresh" | "stale" | "invalid";
export type StagedRelationDependencyKind = "input" | "operation" | "binding";

export interface StagedRelationDependency {
  kind: StagedRelationDependencyKind;
  reference: string;
  version?: string | null;
}

export interface StagedRelationProducer {
  flowRunId?: string | null;
  stepId?: string | null;
  operation?: string | null;
  operationVersion?: string | null;
}

export interface StagedRelation {
  relationId: string;
  name?: string | null;
  contentHash: string;
  contentType: string;
  sizeBytes: number;
  rowCount?: number | null;
  columns: string[];
  producer: StagedRelationProducer;
  dependsOn: StagedRelationDependency[];
  retention: StagedRelationRetention;
  createdMs: number;
  freshness?: StagedRelationFreshness;
  recomputable?: boolean;
}

// Cluster bootstrap wizard (P6) — mirrors `src-tauri/src/bootstrap.rs`.
export interface WorkerPlan {
  bind: string;
  advertise: string;
  name?: string;
  slots?: number;
}

export interface RaftPlan {
  enabled: boolean;
  dataDir?: string;
  nodeId?: number;
  peers: string[];
  init: boolean;
}

export interface MtlsPlan {
  enabled: boolean;
  caBundle?: string;
  required: boolean;
}

export interface ClusterPlan {
  coordinatorBind: string;
  coordinatorUrl: string;
  workers: WorkerPlan[];
  raft: RaftPlan;
  mtls: MtlsPlan;
}

export interface RaftMember {
  node_id: number;
  base_url: string;
}

export interface RaftMembershipResponse {
  voters: number[];
}

export interface EnvVar {
  key: string;
  value: string;
}

export interface CommandBlock {
  role: "coordinator" | "worker";
  title: string;
  env: EnvVar[];
  command: string;
}

// Loose view of `quant_fabric::model::ClusterStatus` (snake_case serde) — only
// the fields the Clusters view renders.
export interface ClusterStatus {
  generated_ms?: number;
  flight_mode?: string;
  workers?: {
    total: number;
    healthy: number;
    stale: number;
    slots: number;
    in_flight: number;
    flight_capable: number;
    healthy_flight_capable: number;
  };
  queries?: {
    total: number;
    running: number;
    complete: number;
    failed: number;
  };
  tasks?: {
    total: number;
    pending: number;
    leased: number;
    running: number;
    complete: number;
    failed: number;
    retrying: number;
    abandoned: number;
  };
  scheduler?: { health: string; locality_entries?: number };
  consensus?: {
    mode: string;
    is_leader: boolean;
    node_id?: number | null;
    leader_id?: number | null;
    current_term?: number | null;
    last_applied_index?: number | null;
  };
  cost_observations?: number;
}

export interface WorkerStatus {
  worker_id: string;
  name: string;
  base_url: string;
  flight_endpoint?: string | null;
  slots: number;
  in_flight: number;
  healthy: boolean;
  last_seen_ms: number;
  stale: boolean;
  labels: Record<string, string>;
}
