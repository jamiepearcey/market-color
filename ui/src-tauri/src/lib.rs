use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::Mutex as StdMutex;
use std::time::Duration;

use quant_fabric::model::{DataFormat, DataRef};
use quant_fabric::studio::{self, RunOutput, StudioSession, TemplateInfo};
use quant_fabric::contracts::{
    ColumnKind, Conformance, ContractRef, ContractRegistry, RelationColumn,
};
use quant_fabric::{
    demo_requests, execute_demo_query_detailed, execute_demo_suite, fetch_cluster_status,
    fetch_cost_summary, fetch_locality_registry, fetch_workers_list, list_demo_catalog,
    spawn_embedded_cluster_on, DemoCatalogEntry, EmbeddedCluster,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{async_runtime, Manager, State};
use tokio::sync::Mutex;

mod bootstrap;
mod connectors_job;
mod controlplane;
mod github_packs;
mod keychain;
use controlplane::{
    Binding, ConnectorInstanceRecord, ControlStore, Job, JobRun, JobRunStep, PackSource,
    PackSyncEvent, PackVersion, Target,
};

struct AppState {
    cluster: Mutex<Option<EmbeddedCluster>>,
    /// Warm template engine — one DuckDB connection with quant UDFs registered
    /// once, reused across runs (no per-run cold start). See the latency note in
    /// `quant_fabric::studio::StudioSession`.
    studio: StdMutex<StudioSession>,
    /// Operator template directories (recursively scanned).
    template_dirs: Vec<PathBuf>,
    /// Control-plane store (ADR-0021): embedded PGlite (Postgres) via sqlx — the
    /// suite's local system-of-record for targets / jobs / run history, and the
    /// persisted scheduler watermark (catch-up across restarts).
    controlplane: ControlStore,
    /// Live wire-protocol coordinates for the embedded control-plane PGlite,
    /// surfaced to the Explore notebook so an external client (a Jupyter
    /// notebook, `psql`, a BI tool) can attach over the Postgres protocol
    /// (ADR-0029). Loopback + ephemeral port by default; `QUANTFABRIC_PG_TCP`
    /// pins an explicit `host:port` (and can bind a non-loopback address).
    external_pg: ExternalPg,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct SharedConnectorBinding {
    name: String,
    file: String,
    driver: String,
    status: String,
    landed: bool,
    description: String,
    research_use: String,
    auth: String,
    cadence: String,
    sink_stream: Option<String>,
    sink_path: Option<String>,
    panel_view: Option<String>,
    panel_value_name: Option<String>,
    panel_asset_class: Option<String>,
    note: Option<String>,
}

const SHARED_CONNECTOR_RUN_TIMEOUT_SECS: u64 = 120;

/// Postgres-wire connection coordinates for the embedded control-plane server,
/// returned to the UI by [`external_endpoint_info`].
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ExternalPg {
    /// Full `postgresql://…` connection URI (the source of truth).
    url: String,
    host: String,
    port: u16,
    database: String,
    user: String,
    /// `true` when the bind address was pinned via `QUANTFABRIC_PG_TCP` (a
    /// stable URL across restarts); `false` for the default ephemeral port.
    pinned: bool,
}

impl ExternalPg {
    /// Derive the connection coordinates from a started PGlite server. The
    /// server already listens on loopback TCP by default (pglite-oxide binds
    /// `127.0.0.1:0` unless told otherwise), so this is purely reporting.
    fn from_server(server: &pglite_oxide::PgliteServer, pinned: bool) -> Self {
        let url = server.database_url();
        let (host, port) = server
            .tcp_addr()
            .map(|addr| (addr.ip().to_string(), addr.port()))
            .unwrap_or_else(|| ("127.0.0.1".to_string(), 0));
        ExternalPg {
            url,
            host,
            port,
            // The app does not override pglite's startup defaults.
            database: "template1".to_string(),
            user: "postgres".to_string(),
            pinned,
        }
    }

    /// Coordinates for a host that does not own the embedded server (the
    /// headless scheduler, tests). Carries the store URL it connected to; the
    /// Explore endpoint command is GUI-only, so the address fields are nominal.
    fn placeholder(url: &str) -> Self {
        ExternalPg {
            url: url.to_string(),
            host: "127.0.0.1".to_string(),
            port: 0,
            database: "template1".to_string(),
            user: "postgres".to_string(),
            pinned: false,
        }
    }
}

/// The single source-of-truth root for all file-backed state (ADR-0010).
/// `QUANT_FABRIC_HOME` overrides; otherwise an in-repo `.quantfabric/` for dev.
/// A packaged app exports `QUANT_FABRIC_HOME=<os-app-data>/quant-fabric` at
/// startup so the same resolvers point at the OS app-data dir with no code
/// change. The taxonomy under it: `library/ workspace/ sources/ artifacts/
/// cache/ state/` (see ADR-0010).
pub fn home_dir() -> PathBuf {
    if let Ok(val) = std::env::var("QUANT_FABRIC_HOME") {
        if !val.is_empty() {
            return PathBuf::from(val);
        }
    }
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../.quantfabric"))
}

/// Resolve a state subtree (ADR-0010). The `:`-joined `env_override` wins
/// (back-compat for custom layouts); otherwise the canonical
/// `home_dir()/<subpath>`. `subpaths` lets one logical library span tiers
/// (e.g. templates live in both `library/` and `workspace/`).
fn state_dirs(env_override: &str, subpaths: &[&str]) -> Vec<PathBuf> {
    if let Ok(val) = std::env::var(env_override) {
        let dirs: Vec<PathBuf> = val
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if !dirs.is_empty() {
            return dirs;
        }
    }
    let home = home_dir();
    subpaths.iter().map(|s| home.join(s)).collect()
}

/// Resolve where the template library lives (ADR-0010): `QUANT_FABRIC_TEMPLATES`
/// (`:`-joined) overrides; otherwise the shipped `library/templates` plus your
/// `workspace/templates` under `home_dir()`.
pub fn template_dirs() -> Vec<PathBuf> {
    let mut dirs = state_dirs(
        "QUANT_FABRIC_TEMPLATES",
        &["library/templates", "workspace/templates"],
    );
    // Packs (ADR-0027) contribute their `templates/` subdirs through the same loader.
    dirs.extend(quant_fabric::packs::artifact_dirs(
        &pack_dirs(),
        "templates",
    ));
    dirs
}

/// Resolve where the connector/driver defs live (ADR-0009 driver catalogue,
/// ADR-0010 layout): `QUANT_FABRIC_CONNECTORS` overrides; otherwise
/// `library/drivers` under `home_dir()`. Read-only config (ADR-0026).
pub fn connector_dirs() -> Vec<PathBuf> {
    let mut dirs = state_dirs("QUANT_FABRIC_CONNECTORS", &["library/drivers"]);
    dirs.extend(quant_fabric::packs::artifact_dirs(
        &pack_dirs(),
        "connectors",
    ));
    dirs
}

/// Read-only shipped pack library (ADR-0027): `QUANT_FABRIC_CORE_PACKS` or the
/// in-repo default. In a packaged app this points at read-only app resources.
pub fn core_pack_dirs() -> Vec<PathBuf> {
    state_dirs("QUANT_FABRIC_CORE_PACKS", &["library/packs"])
}

/// Writable user packs (editable; import/duplicate land here):
/// `QUANT_FABRIC_USER_PACKS` or the in-repo default. In a packaged app this is a
/// writable app-data dir.
pub fn user_pack_dirs() -> Vec<PathBuf> {
    state_dirs("QUANT_FABRIC_USER_PACKS", &["workspace/packs"])
}

/// Immutable GitHub pack snapshots. Each synced commit is cached as a normal
/// pack directory under this root, so existing pack/template/connector loaders
/// see GitHub packs without a separate execution path.
pub fn github_pack_dirs() -> Vec<PathBuf> {
    github_packs::cache_dirs()
}

fn github_pack_root() -> Result<PathBuf, String> {
    github_packs::cache_root().map_err(|e| e.to_string())
}

/// The first writable user pack root (import/duplicate destination), created if absent.
fn user_pack_root() -> Result<PathBuf, String> {
    let root = user_pack_dirs()
        .into_iter()
        .next()
        .ok_or("no writable user pack directory")?;
    std::fs::create_dir_all(&root).map_err(|e| e.to_string())?;
    Ok(root)
}

/// All pack roots (core + user), for folding artifacts into the catalogs.
pub fn pack_dirs() -> Vec<PathBuf> {
    let mut dirs = core_pack_dirs();
    dirs.extend(user_pack_dirs());
    dirs.extend(github_pack_dirs());
    dirs
}

/// Where the Explore notebook library lives (ADR-0029). The first entry is the
/// writable *default* directory new notebooks land in; users add further read /
/// write directories in the UI (persisted as a pref, passed back to
/// [`list_notebooks`]). Resolution mirrors [`user_pack_dirs`]:
/// `QUANT_FABRIC_NOTEBOOKS` (`:`-joined) or the in-repo default next to this app.
pub fn notebook_dirs() -> Vec<PathBuf> {
    state_dirs("QUANT_FABRIC_NOTEBOOKS", &["workspace/notebooks"])
}

/// Root the Explore data tree walks (ADR-0010): the whole `home_dir()`, so the
/// sidebar shows *all* file-backed state, not just one corner. `QUANT_FABRIC_DATA`
/// (`:`-joined) still overrides for a narrower scope.
pub fn data_dirs() -> Vec<PathBuf> {
    if let Ok(val) = std::env::var("QUANT_FABRIC_DATA") {
        let dirs: Vec<PathBuf> = val
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        if !dirs.is_empty() {
            return dirs;
        }
    }
    vec![home_dir()]
}

// ---- Template studio commands (shared lib contract: quant_fabric::studio) ----

#[tauri::command]
fn list_templates(state: State<'_, AppState>) -> Vec<TemplateInfo> {
    studio::catalog(&state.template_dirs)
}

// ---- Quant function catalog (pack editor SQL completions) ----

/// Argument/return/category metadata manifest for the quant scalar functions.
/// This is a best-effort overlay: it carries signatures for the subset of
/// functions documented for the WASM extension. The authoritative *set* of
/// names is `duckdb_quant::QUANT_FUNCTION_NAMES` (what is actually registered on
/// the engine), so every registered function is surfaced even without metadata.
const QUANT_FUNCTION_METADATA_JSON: &str =
    include_str!("../../../duckdb-quant-wasm-extension/functions.json");

/// One custom DuckDB quant scalar function, exposed to the pack editor so it can
/// offer IntelliSense completions inside `source_sql` SELECT bodies.
#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct QuantFunctionInfo {
    /// Registered function name (e.g. `bs_price`).
    name: String,
    /// Positional argument names, when known (empty when no metadata exists).
    args: Vec<String>,
    /// Return type, when known (e.g. `double`, `struct`).
    returns: Option<String>,
    /// Model/category grouping, when known (e.g. `black_scholes`).
    category: Option<String>,
}

fn quant_function_catalog(template_dirs: &[PathBuf]) -> Vec<QuantFunctionInfo> {
    #[derive(Deserialize)]
    struct RawFn {
        name: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        returns: Option<String>,
        #[serde(default)]
        model: Option<String>,
    }
    #[derive(Deserialize)]
    struct RawManifest {
        functions: Vec<RawFn>,
    }
    let meta: BTreeMap<String, RawFn> =
        serde_json::from_str::<RawManifest>(QUANT_FUNCTION_METADATA_JSON)
            .map(|m| {
                m.functions
                    .into_iter()
                    .map(|f| (f.name.clone(), f))
                    .collect()
            })
            .unwrap_or_default();
    let template_meta = quant_function_template_metadata(template_dirs);
    quant_loadable_extension::QUANT_FUNCTION_NAMES
        .iter()
        .map(|&name| match meta.get(name) {
            Some(f) => QuantFunctionInfo {
                name: name.to_string(),
                args: f.args.clone(),
                returns: f.returns.clone(),
                category: f.model.clone(),
            },
            None => QuantFunctionInfo {
                name: name.to_string(),
                args: template_meta
                    .get(name)
                    .map(|m| m.args.clone())
                    .unwrap_or_default(),
                returns: None,
                category: template_meta.get(name).and_then(|m| m.category.clone()),
            },
        })
        .collect()
}

#[derive(Debug, Clone)]
struct TemplateFunctionMeta {
    args: Vec<String>,
    category: Option<String>,
}

fn quant_function_template_metadata(
    template_dirs: &[PathBuf],
) -> BTreeMap<String, TemplateFunctionMeta> {
    let mut out = BTreeMap::new();
    let names = quant_loadable_extension::QUANT_FUNCTION_NAMES;
    for template in studio::catalog(template_dirs) {
        let Some(source) = template.source.as_deref() else {
            continue;
        };
        for sql in template_sql_fragments(source) {
            for &name in names {
                let Some(args) = infer_function_args_from_sql(name, &sql) else {
                    continue;
                };
                if args.is_empty() {
                    continue;
                }
                out.entry(name.to_string())
                    .or_insert_with(|| TemplateFunctionMeta {
                        args,
                        category: Some(template.category.clone()),
                    });
            }
        }
    }
    out
}

fn template_sql_fragments(source: &str) -> Vec<String> {
    #[derive(Deserialize)]
    struct RawTemplateFile {
        #[serde(default)]
        template: Vec<RawTemplate>,
    }
    #[derive(Deserialize)]
    struct RawTemplate {
        #[serde(default)]
        sql: Option<RawSql>,
    }
    #[derive(Deserialize)]
    struct RawSql {
        template: String,
    }
    toml::from_str::<RawTemplateFile>(source)
        .map(|file| {
            file.template
                .into_iter()
                .filter_map(|template| template.sql.map(|sql| sql.template))
                .collect()
        })
        .unwrap_or_default()
}

fn infer_function_args_from_sql(function_name: &str, sql: &str) -> Option<Vec<String>> {
    let mut search_from = 0;
    while let Some(relative_idx) = sql[search_from..].find(function_name) {
        let idx = search_from + relative_idx;
        let name_end = idx + function_name.len();
        let before = idx
            .checked_sub(1)
            .and_then(|i| sql.as_bytes().get(i))
            .copied();
        let after = sql.as_bytes().get(name_end).copied();
        if before.is_some_and(is_identifier_byte) || after.is_some_and(is_identifier_byte) {
            search_from = name_end;
            continue;
        }
        let Some(open_relative) = sql[name_end..].find('(') else {
            return None;
        };
        if !sql[name_end..name_end + open_relative]
            .chars()
            .all(char::is_whitespace)
        {
            search_from = name_end;
            continue;
        }
        let open = name_end + open_relative;
        let Some(close) = matching_close_paren(sql, open) else {
            return None;
        };
        return Some(extract_placeholders(&sql[open + 1..close]));
    }
    None
}

fn is_identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn matching_close_paren(sql: &str, open: usize) -> Option<usize> {
    let mut depth = 0usize;
    let mut in_single_quote = false;
    let mut in_double_quote = false;
    for (idx, ch) in sql.char_indices().skip_while(|(idx, _)| *idx < open) {
        match ch {
            '\'' if !in_double_quote => in_single_quote = !in_single_quote,
            '"' if !in_single_quote => in_double_quote = !in_double_quote,
            '(' if !in_single_quote && !in_double_quote => depth += 1,
            ')' if !in_single_quote && !in_double_quote => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
    }
    None
}

fn extract_placeholders(fragment: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut search_from = 0;
    while let Some(start_relative) = fragment[search_from..].find("{{") {
        let start = search_from + start_relative + 2;
        let Some(end_relative) = fragment[start..].find("}}") else {
            break;
        };
        let end = start + end_relative;
        let name = fragment[start..end].trim();
        if !name.is_empty() && !out.iter().any(|existing| existing == name) {
            out.push(name.to_string());
        }
        search_from = end + 2;
    }
    out
}

/// Full catalog of custom DuckDB quant scalar functions for editor completions.
#[tauri::command]
fn list_quant_functions(state: State<'_, AppState>) -> Vec<QuantFunctionInfo> {
    quant_function_catalog(&state.template_dirs)
}

// ---- Connector commands (ADR-0026). Catalog is real (parsed from the TOML
//      library); test/run are mocked until drivers land. ----

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConnectorTestResult {
    /// `pass` | `fail` | `example` (a required path param points nowhere real).
    status: String,
    message: String,
    elapsed_ms: u128,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ConnectorActionResult {
    /// `complete` | `failed` | `example` (described but not actually executed).
    status: String,
    message: String,
    rows: Vec<quant_fabric::model::Row>,
    elapsed_ms: u128,
}

/// The full connector catalog: the TOML library on disk plus the Celeritas
/// contract discovered from `CELERITAS_BIN_DIR` (ADR-0031).
#[tauri::command]
fn list_connectors() -> Vec<quant_fabric::connectors::ConnectorInfo> {
    quant_fabric::connectors::catalog_with_discovery(&connector_dirs())
}

/// The cohesive packs available on disk (ADR-0027): core (read-only) + user
/// (editable), each tagged with its origin.
#[tauri::command]
fn list_packs() -> Vec<quant_fabric::packs::Pack> {
    use quant_fabric::packs::{discover, PackOrigin};
    let mut out = discover(&core_pack_dirs(), PackOrigin::Core);
    out.extend(discover(&user_pack_dirs(), PackOrigin::User));
    let mut github = discover(&github_pack_dirs(), PackOrigin::Github);
    for pack in &mut github {
        let stable = std::path::Path::new(&pack.dir)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&pack.manifest.id);
        pack.manifest.id = format!("github:{stable}");
    }
    out.extend(github);
    out.sort_by(|a, b| a.manifest.name.cmp(&b.manifest.name));
    out
}

/// Resolve a pack by id to its directory + origin (across both roots).
fn resolve_pack(id: &str) -> Option<(PathBuf, quant_fabric::packs::PackOrigin)> {
    list_packs()
        .into_iter()
        .find(|p| p.manifest.id == id)
        .map(|p| (PathBuf::from(p.dir), p.origin))
}

/// Re-discover the pack now living at `dir` (after import/duplicate).
fn pack_at(dir: &std::path::Path) -> Result<quant_fabric::packs::Pack, String> {
    list_packs()
        .into_iter()
        .find(|p| std::path::Path::new(&p.dir) == dir)
        .ok_or_else(|| "pack written but not re-discovered".to_string())
}

/// IDE-like editing: the editable text files of a pack (read allowed for both
/// core and user packs).
#[tauri::command]
fn read_pack_files(pack_id: String) -> Result<Vec<quant_fabric::packs::PackFile>, String> {
    let (dir, _) = resolve_pack(&pack_id).ok_or_else(|| format!("unknown pack `{pack_id}`"))?;
    Ok(quant_fabric::packs::read_files(&dir))
}

/// Write a file within a pack. Core packs are read-only — callers must duplicate
/// to a user pack to edit. Enforced here (not just in the UI).
#[tauri::command]
fn write_pack_file(pack_id: String, rel_path: String, contents: String) -> Result<(), String> {
    let (dir, origin) =
        resolve_pack(&pack_id).ok_or_else(|| format!("unknown pack `{pack_id}`"))?;
    if origin != quant_fabric::packs::PackOrigin::User {
        return Err("read-only packs must be duplicated to a user pack before editing".to_string());
    }
    quant_fabric::packs::write_file(&dir, &rel_path, &contents).map_err(|e| e.to_string())
}

/// Create a brand-new **user** pack from in-memory files — the "promote a
/// notebook to a pack" flow (ADR-0030 P5). Writes `pack.toml` + each file under
/// the writable user root; fails rather than clobbering an existing pack id.
#[tauri::command]
fn create_pack(
    id: String,
    name: String,
    description: String,
    files: Vec<quant_fabric::packs::PackFile>,
) -> Result<quant_fabric::packs::Pack, String> {
    let root = user_pack_root()?;
    let mut filemap = serde_json::Map::new();
    for f in &files {
        filemap.insert(f.path.clone(), json!(f.contents));
    }
    // Reuse the bundle import path so the manifest is written consistently.
    let bundle = json!({
        "format": "qfpack/1",
        "pack": { "id": id, "name": name, "version": "0.1.0", "description": description, "tags": [] },
        "files": filemap,
    })
    .to_string();
    let dir = quant_fabric::packs::import_from_str(&bundle, &root).map_err(|e| e.to_string())?;
    pack_at(&dir)
}

/// Copy a pack (core or user) to a new id in the writable user root — the
/// "copy & edit" flow for customising a core pack.
#[tauri::command]
fn duplicate_pack(
    pack_id: String,
    new_id: String,
    new_name: String,
) -> Result<quant_fabric::packs::Pack, String> {
    let (src, _) = resolve_pack(&pack_id).ok_or_else(|| format!("unknown pack `{pack_id}`"))?;
    let root = user_pack_root()?;
    let dir = quant_fabric::packs::duplicate(&src, &root, &new_id, &new_name)
        .map_err(|e| e.to_string())?;
    pack_at(&dir)
}

/// Export a pack (core or user) to a single portable `.qfpack` file.
#[tauri::command]
fn export_pack(pack_id: String, dest_path: String) -> Result<(), String> {
    let (dir, _) = resolve_pack(&pack_id).ok_or_else(|| format!("unknown pack `{pack_id}`"))?;
    let bundle = quant_fabric::packs::export_to_string(&dir).map_err(|e| e.to_string())?;
    std::fs::write(&dest_path, bundle).map_err(|e| e.to_string())
}

/// Import a `.qfpack` file into the writable user root (always a user pack).
#[tauri::command]
fn import_pack(src_path: String) -> Result<quant_fabric::packs::Pack, String> {
    let bundle = std::fs::read_to_string(&src_path).map_err(|e| e.to_string())?;
    let root = user_pack_root()?;
    let dir = quant_fabric::packs::import_from_str(&bundle, &root).map_err(|e| e.to_string())?;
    pack_at(&dir)
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct LinkGithubPackRequest {
    repo_url: String,
    #[serde(default = "default_ref_name")]
    ref_name: String,
    #[serde(default)]
    pack_path: String,
    #[serde(default = "default_monitor")]
    monitor: String,
    #[serde(default)]
    auth_ref: Option<String>,
}

fn default_ref_name() -> String {
    "main".to_string()
}

fn default_monitor() -> String {
    "manual".to_string()
}

fn validate_pack_source_monitor(monitor: &str) -> Result<(), String> {
    match monitor {
        "manual" | "startup" => Ok(()),
        other => Err(format!("unsupported pack source monitor `{other}`")),
    }
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct PackVersionCompare {
    from: PackVersion,
    to: PackVersion,
    status: String,
    changed: bool,
    notes: Vec<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct JobPackUpdateResolution {
    status: String,
    current: Option<PackVersion>,
    latest: Option<PackVersion>,
    notes: Vec<String>,
}

fn github_source_id(owner: &str, repo: &str, pack_path: &str, ref_name: &str) -> String {
    let raw = format!("{owner}-{repo}-{pack_path}-{ref_name}");
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|p| !p.is_empty())
        .collect::<Vec<_>>()
        .join("-")
}

/// Link a GitHub repository/ref as a monitored pack source. This records source
/// metadata only; call `sync_pack_source` to fetch the first version.
#[tauri::command]
async fn link_github_pack(
    request: LinkGithubPackRequest,
    state: State<'_, AppState>,
) -> Result<PackSource, String> {
    validate_pack_source_monitor(&request.monitor)?;
    let (owner, repo) =
        github_packs::parse_github_repo(&request.repo_url).map_err(|e| e.to_string())?;
    let source = PackSource {
        id: github_source_id(&owner, &repo, &request.pack_path, &request.ref_name),
        repo_url: request.repo_url,
        owner,
        repo,
        ref_name: request.ref_name,
        pack_path: request.pack_path,
        monitor: request.monitor,
        auth_ref: request.auth_ref,
        last_checked_at: None,
        last_commit: None,
        status: "linked".to_string(),
        message: None,
    };
    state
        .controlplane
        .upsert_pack_source(&source)
        .await
        .map_err(|e| e.to_string())?;
    Ok(source)
}

#[tauri::command]
async fn update_pack_source_monitor(
    source_id: String,
    monitor: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    validate_pack_source_monitor(&monitor)?;
    state
        .controlplane
        .update_pack_source_monitor(&source_id, &monitor)
        .await
        .map_err(|e| e.to_string())
}

/// Fetch the configured GitHub ref, validate the pack, cache an immutable
/// snapshot, and record the resulting pack version.
#[tauri::command]
async fn sync_pack_source(
    source_id: String,
    state: State<'_, AppState>,
) -> Result<PackVersion, String> {
    let root = github_pack_root()?;
    github_packs::sync_and_record_source(&state.controlplane, &source_id, &root)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_pack_sources(state: State<'_, AppState>) -> Result<Vec<PackSource>, String> {
    state
        .controlplane
        .list_pack_sources()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_pack_versions(
    source_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<PackVersion>, String> {
    state
        .controlplane
        .list_pack_versions(source_id.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_pack_sync_events(
    source_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<PackSyncEvent>, String> {
    state
        .controlplane
        .list_pack_sync_events(source_id.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn compare_pack_versions(
    from_version_id: String,
    to_version_id: String,
    state: State<'_, AppState>,
) -> Result<PackVersionCompare, String> {
    let versions = state
        .controlplane
        .list_pack_versions(None)
        .await
        .map_err(|e| e.to_string())?;
    let from = versions
        .iter()
        .find(|v| v.id == from_version_id)
        .cloned()
        .ok_or_else(|| format!("unknown pack version `{from_version_id}`"))?;
    let to = versions
        .iter()
        .find(|v| v.id == to_version_id)
        .cloned()
        .ok_or_else(|| format!("unknown pack version `{to_version_id}`"))?;
    let mut notes = Vec::new();
    if from.pack_id != to.pack_id {
        notes.push("pack id changed; review required".to_string());
    }
    if from.manifest_version != to.manifest_version {
        notes.push(format!(
            "manifest version changed from {} to {}",
            from.manifest_version, to.manifest_version
        ));
    }
    if from.content_hash != to.content_hash {
        notes.push("pack contents changed".to_string());
    }
    let compatible = from.pack_id == to.pack_id;
    Ok(PackVersionCompare {
        from,
        to,
        status: if compatible {
            "compatible".to_string()
        } else {
            "review".to_string()
        },
        changed: !notes.is_empty(),
        notes,
    })
}

/// Resolve whether a job's current pack version can move to the latest cached
/// version for its source. This is conservative: matching pack id is considered
/// compatible, while template/input-level migration checks are left for the UI
/// diff flow and a later semantic validator.
#[tauri::command]
async fn resolve_job_pack_update(
    source_id: String,
    current_version_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<JobPackUpdateResolution, String> {
    let versions = state
        .controlplane
        .list_pack_versions(Some(&source_id))
        .await
        .map_err(|e| e.to_string())?;
    let latest = versions.first().cloned();
    let current = current_version_id
        .as_deref()
        .and_then(|id| versions.iter().find(|v| v.id == id).cloned());
    let mut notes = Vec::new();
    let status = match (&current, &latest) {
        (_, None) => "no-version",
        (None, Some(_)) => "available",
        (Some(c), Some(l)) if c.id == l.id => "current",
        (Some(c), Some(l)) if c.pack_id == l.pack_id => {
            notes.push("newer cached version has the same pack id".to_string());
            "compatible"
        }
        (Some(_), Some(_)) => {
            notes.push("pack id changed; review required".to_string());
            "review"
        }
    };
    Ok(JobPackUpdateResolution {
        status: status.to_string(),
        current,
        latest,
        notes,
    })
}

/// Delete a user pack's folder. Core packs cannot be deleted; the target must
/// live inside a configured user root (defence against a bad id resolving
/// somewhere unexpected).
#[tauri::command]
fn delete_pack(pack_id: String) -> Result<(), String> {
    let (dir, origin) =
        resolve_pack(&pack_id).ok_or_else(|| format!("unknown pack `{pack_id}`"))?;
    if origin != quant_fabric::packs::PackOrigin::User {
        return Err("read-only packs cannot be deleted".to_string());
    }
    if !user_pack_dirs().iter().any(|root| dir.starts_with(root)) {
        return Err("refusing to delete a pack outside the user pack directory".to_string());
    }
    quant_fabric::packs::delete(&dir).map_err(|e| e.to_string())
}

/// Native open dialog for picking a `.qfpack` bundle to import.
#[tauri::command]
async fn pick_pack_bundle(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .add_filter("pack", &["qfpack", "json"])
        .pick_file(move |path| {
            let _ = tx.send(path);
        });
    rx.await
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Native save dialog (non-blocking, mirrors [`pick_data_file`]) for exporting.
#[tauri::command]
async fn pick_save_path(app: tauri::AppHandle, default_name: String) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .set_file_name(&default_name)
        .add_filter("pack", &["qfpack", "json"])
        .save_file(move |path| {
            let _ = tx.send(path);
        });
    rx.await
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Mock connectivity check. Until real drivers land this validates that required
/// params are filled and that any `path`-typed value resolves on disk, returning
/// `example` when a path is missing so the UI can flag a non-functional preset.
#[tauri::command]
async fn test_connector(
    kind: String,
    driver: String,
    mut params: BTreeMap<String, String>,
    state: State<'_, AppState>,
) -> Result<ConnectorTestResult, String> {
    let started = std::time::Instant::now();

    // Resolve any `secrets-keeper://…` references before probing with the real
    // driver, so a connection secret can be referenced instead of stored inline.
    resolve_params_refs(&mut params, &state).await?;

    // Whether a real driver binary for this connector resolves at all.
    let available = quant_fabric::celeritas::celeritas_binary(&kind, &driver)
        .and_then(|b| quant_fabric::celeritas::resolve_binary(&b))
        .is_some();

    // Real probe (ADR-0028): for a source whose celeritas extension binary
    // resolves, run `--discover` — it actually connects and reports the schema.
    let has_binary = kind == "source" && available;
    if has_binary {
        let d = driver.clone();
        let p = params.clone();
        let probe = async_runtime::spawn_blocking(move || {
            quant_fabric::celeritas::discover("source", &d, &p, &BTreeMap::new())
        })
        .await
        .map_err(|e| e.to_string())?;
        return Ok(match probe {
            Ok(disc) => {
                let cols = disc
                    .field_count()
                    .map(|n| format!(" · {n} columns"))
                    .unwrap_or_default();
                ConnectorTestResult {
                    status: "pass".to_string(),
                    message: format!("Connected via `{driver}`{cols}."),
                    elapsed_ms: started.elapsed().as_millis(),
                }
            }
            Err(e) => ConnectorTestResult {
                status: "fail".to_string(),
                message: e.to_string(),
                elapsed_ms: started.elapsed().as_millis(),
            },
        });
    }

    // No executable probe ran. We must NOT report a green "pass" here — that
    // previously made every untested connector look connected. Report "example"
    // (untested) instead, with a message that says exactly why.
    let unresolved = params.values().any(|v| {
        let looks_pathy = v.starts_with("./") || v.starts_with('/') || v.starts_with("~/");
        looks_pathy && !std::path::Path::new(v).exists()
    });
    let message = if unresolved {
        format!(
            "Driver `{driver}`: a configured path does not resolve on disk — this is still an example connector, not a verified connection."
        )
    } else if available {
        // A real (e.g. sink) binary exists, but there is no non-destructive test
        // probe for it yet, so connectivity remains unverified.
        format!(
            "Driver `{driver}` is installed, but its connectivity can't be verified without running a job — treat this as untested."
        )
    } else {
        format!(
            "No `{driver}` driver is installed, so connectivity can't be verified. This is an example connector — build or point the celeritas `{driver}` extension at it to test for real."
        )
    };
    Ok(ConnectorTestResult {
        status: "example".to_string(),
        message,
        elapsed_ms: started.elapsed().as_millis(),
    })
}

/// Number of rows a `--preview` connector action imports for the UI.
const CONNECTOR_PREVIEW_ROWS: usize = 100;

/// Connector custom-action run. The `--preview` action on a source whose
/// celeritas extension binary resolves runs the **real** import (ADR-0028 data
/// plane): it spawns `source-<driver>-ipc`, reads its Arrow batches over the
/// shared-memory transport, and returns the first rows. Other actions, or
/// drivers with no built extension, fall back to a descriptive echo so the UI
/// keeps working without celeritas present.
#[tauri::command]
async fn run_connector_action(
    driver: String,
    flag: String,
    mut params: BTreeMap<String, String>,
    mut job_params: BTreeMap<String, String>,
    state: State<'_, AppState>,
) -> Result<ConnectorActionResult, String> {
    let started = std::time::Instant::now();

    // Resolve `secrets-keeper://…` references in both connection and job params
    // before the real import runs.
    resolve_params_refs(&mut params, &state).await?;
    resolve_params_refs(&mut job_params, &state).await?;

    let source_binary = quant_fabric::celeritas::celeritas_binary("source", &driver)
        .and_then(|b| quant_fabric::celeritas::resolve_binary(&b))
        .is_some();
    if flag == "--preview" && source_binary {
        let d = driver.clone();
        let p = params.clone();
        let jp = job_params.clone();
        let preview = async_runtime::spawn_blocking(move || {
            quant_fabric::celeritas::run_source_rows(&d, &p, &jp, &[], CONNECTOR_PREVIEW_ROWS)
        })
        .await
        .map_err(|e| e.to_string())?;
        return Ok(match preview {
            Ok(rows) => ConnectorActionResult {
                status: "complete".to_string(),
                message: format!(
                    "Previewed {} row{} via `{driver}`.",
                    rows.len(),
                    if rows.len() == 1 { "" } else { "s" }
                ),
                rows,
                elapsed_ms: started.elapsed().as_millis(),
            },
            Err(e) => ConnectorActionResult {
                status: "fail".to_string(),
                message: e.to_string(),
                rows: Vec::new(),
                elapsed_ms: started.elapsed().as_millis(),
            },
        });
    }

    // No celeritas data plane for this action/driver — describe the invocation
    // but report it as an example (not a real "complete" run), so the UI doesn't
    // present an un-executed action as a success.
    let message = format!(
        "{driver} {flag} — {} connection params, {} job params. This action did not run: build or point the celeritas `{driver}` extension at it to execute for real.",
        params.len(),
        job_params.len()
    );
    Ok(ConnectorActionResult {
        status: "example".to_string(),
        message,
        rows: Vec::new(),
        elapsed_ms: started.elapsed().as_millis(),
    })
}

async fn connector_runtime_refs(
    state: &AppState,
) -> Result<
    (
        Vec<connectors_job::ConnectorInstance>,
        Vec<connectors_job::BindingRef>,
    ),
    String,
> {
    let mut connector_instances = connectors_job::parse_instances(
        state
            .controlplane
            .get_setting(connectors_job::CONNECTOR_INSTANCES_KEY)
            .await
            .ok()
            .flatten(),
    );
    for rec in state
        .controlplane
        .list_connector_instances()
        .await
        .unwrap_or_default()
    {
        let inst = connectors_job::instance_from_parts(
            &rec.id,
            &rec.driver,
            Some(&rec.params),
            Some(&rec.job_param_defaults),
        );
        match connector_instances.iter_mut().find(|i| i.id == inst.id) {
            Some(existing) => *existing = inst,
            None => connector_instances.push(inst),
        }
    }
    let mut bindings: Vec<connectors_job::BindingRef> = state
        .controlplane
        .list_bindings()
        .await
        .unwrap_or_default()
        .iter()
        .map(|b| connectors_job::binding_ref_from_parts(&b.id, &b.instance_id, Some(&b.run_params)))
        .collect();
    // Resolve any `secrets-keeper://…` references at use-time so a connection
    // secret never has to be stored in the connector instance or binding. A
    // failure here (e.g. the backend is sealed/unreachable) is loud, not silent.
    for inst in connector_instances.iter_mut() {
        resolve_params_refs(&mut inst.params, state).await?;
    }
    for bref in bindings.iter_mut() {
        resolve_params_refs(&mut bref.run_params, state).await?;
    }
    Ok((connector_instances, bindings))
}

async fn materialize_studio_binding(
    state: &AppState,
    binding_id: &str,
    contract_ref: Option<&str>,
) -> Result<String, String> {
    let binding = state
        .controlplane
        .get_binding(binding_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown binding `{binding_id}`"))?;
    if let Some(contract_ref) = contract_ref {
        let schema = binding.schema.as_ref().ok_or_else(|| {
            format!("binding `{binding_id}` has no cached schema for `{contract_ref}` validation")
        })?;
        let reference = ContractRef::parse(contract_ref).map_err(|e| e.to_string())?;
        let registry = ContractRegistry::load_default().map_err(|e| e.to_string())?;
        let relation = relation_columns_from_schema(schema)?;
        let mapping = binding_mapping(binding.mapping.as_ref())?;
        let conformance = registry
            .validate(&reference, &relation, mapping.as_ref())
            .map_err(|e| e.to_string())?;
        if !conformance.conforms {
            return Err(format!(
                "binding `{binding_id}` does not conform to `{contract_ref}`"
            ));
        }
    }
    let (instances, bindings) = connector_runtime_refs(state).await?;
    let definition = json!({ "sourceBindingId": binding_id });
    let source = connectors_job::resolve_job_source(&definition, &instances, &bindings)?;
    if source.trim().is_empty() {
        return Err(format!("binding `{binding_id}` could not be materialized"));
    }
    Ok(source)
}

fn studio_binding_arg(raw: &str) -> Option<&str> {
    raw.trim()
        .strip_prefix("binding:")
        .or_else(|| raw.trim().strip_prefix("qf-binding:"))
        .map(str::trim)
}

async fn resolve_studio_binding_args(
    name: &str,
    args: &BTreeMap<String, String>,
    state: &AppState,
) -> Result<BTreeMap<String, String>, String> {
    let Some(template) = quant_fabric::templates::find_template(name) else {
        return Ok(args.clone());
    };
    let mut out = args.clone();
    for input in template.inputs {
        let quant_fabric::templates::TemplateInputKind::TypedRelation(contract_ref) = input.kind
        else {
            continue;
        };
        let Some(raw) = args.get(input.name).map(|v| v.trim()) else {
            continue;
        };
        let Some(binding_id) = studio_binding_arg(raw) else {
            continue;
        };
        if binding_id.is_empty() {
            return Err(format!("input `{}` has an empty binding reference", input.name));
        }
        let source = materialize_studio_binding(state, binding_id, Some(contract_ref)).await?;
        out.insert(input.name.to_string(), source);
    }
    Ok(out)
}

#[tauri::command]
async fn run_template(
    name: String,
    args: BTreeMap<String, String>,
    state: State<'_, AppState>,
) -> Result<RunOutput, String> {
    let args = resolve_studio_binding_args(&name, &args, state.inner()).await?;
    let session = state.studio.lock().map_err(|e| e.to_string())?;
    Ok(session.run(&name, &args))
}

#[tauri::command]
fn run_batch(
    data: Vec<String>,
    runs: Vec<(String, BTreeMap<String, String>)>,
    state: State<'_, AppState>,
) -> Result<Vec<RunOutput>, String> {
    let refs: Vec<DataRef> = data
        .into_iter()
        .map(|uri| DataRef {
            uri,
            format: DataFormat::Auto,
        })
        .collect();
    let session = state.studio.lock().map_err(|e| e.to_string())?;
    session.run_batch(&refs, &runs).map_err(|e| e.to_string())
}

/// Explore notebook (ADR-0029): run an arbitrary DuckDB SQL cell on the warm
/// `StudioSession`. When `source` is a data file it is materialized once as the
/// `input` table (reusing the studio's file-identity cache), so every cell run
/// over the same dataset is warm. Quant UDFs are already registered, so the SQL
/// can call them directly. Failures come back as a `failed` `RunOutput` (with
/// the engine error), not a thrown command error, so the cell renders them.
#[tauri::command]
fn run_query(
    sql: String,
    source: Option<String>,
    state: State<'_, AppState>,
) -> Result<RunOutput, String> {
    let session = state.studio.lock().map_err(|e| e.to_string())?;
    Ok(session.run_sql(&sql, source.as_deref()))
}

/// Explore notebook (ADR-0029): the embedded control-plane PGlite's live
/// Postgres-wire coordinates, so an external notebook / `psql` / BI tool can
/// attach. The server already listens on loopback TCP; `QUANTFABRIC_PG_TCP`
/// pins a stable (and optionally non-loopback) bind address.
#[tauri::command]
fn external_endpoint_info(state: State<'_, AppState>) -> ExternalPg {
    state.external_pg.clone()
}

#[tauri::command]
fn save_template(
    category: String,
    name: String,
    source: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    let base = state
        .template_dirs
        .first()
        .ok_or_else(|| "no templates directory configured".to_string())?;
    let dir = base.join(&category);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{name}.toml")), &source).map_err(|e| e.to_string())?;
    // Register immediately so it is runnable without a restart (best-effort).
    let _ = quant_fabric::templates::load_templates_from_toml(&source);
    Ok(())
}

// ---- Explore notebook library (ADR-0029) — file-backed notebooks ----

/// A saved notebook file discovered in one of the notebook directories.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct NotebookFile {
    /// Absolute path to the `.json` file.
    path: String,
    /// The directory it lives in (matches a configured notebook path).
    dir: String,
    /// Display name (the notebook's `name`, falling back to the file stem).
    name: String,
    /// Number of cells, for the sidebar badge.
    cell_count: usize,
    /// Last-modified time in epoch milliseconds (newest-first ordering).
    modified_ms: u64,
}

/// A human name for a notebook file when its contents can't be parsed.
fn notebook_stem(path: &std::path::Path) -> String {
    path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("notebook")
        .to_string()
}

/// The writable default notebook directory (first entry of [`notebook_dirs`]),
/// shown in the UI and used as the save target when a notebook has no file yet.
#[tauri::command]
fn default_notebook_dir() -> String {
    notebook_dirs()
        .into_iter()
        .next()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// List the saved notebooks (`*.json`) across the default directory plus any
/// extra `dirs` the user has added. Each entry carries the name + cell count
/// parsed from the file so the sidebar needn't read every notebook up front.
#[tauri::command]
fn list_notebooks(dirs: Vec<String>) -> Vec<NotebookFile> {
    let mut roots = notebook_dirs();
    for d in dirs {
        let p = PathBuf::from(&d);
        if !roots.contains(&p) {
            roots.push(p);
        }
    }
    let mut out: Vec<NotebookFile> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for root in roots {
        let dir_str = root.to_string_lossy().into_owned();
        let entries = match std::fs::read_dir(&root) {
            Ok(e) => e,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("json") {
                continue;
            }
            let path_str = path.to_string_lossy().into_owned();
            if !seen.insert(path_str.clone()) {
                continue;
            }
            let contents = std::fs::read_to_string(&path).unwrap_or_default();
            let (name, cell_count) = match serde_json::from_str::<Value>(&contents) {
                Ok(v) => (
                    v.get("name")
                        .and_then(|n| n.as_str())
                        .map(|s| s.trim().to_string())
                        .filter(|s| !s.is_empty())
                        .unwrap_or_else(|| notebook_stem(&path)),
                    v.get("cells")
                        .and_then(|c| c.as_array())
                        .map(|a| a.len())
                        .unwrap_or(0),
                ),
                Err(_) => (notebook_stem(&path), 0),
            };
            let modified_ms = entry
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            out.push(NotebookFile {
                path: path_str,
                dir: dir_str.clone(),
                name,
                cell_count,
                modified_ms,
            });
        }
    }
    out.sort_by(|a, b| b.modified_ms.cmp(&a.modified_ms));
    out
}

/// Read a notebook file's raw JSON contents.
#[tauri::command]
fn read_notebook(path: String) -> Result<String, String> {
    std::fs::read_to_string(&path).map_err(|e| e.to_string())
}

/// Write a notebook to `<dir>/<name>.json` (creating the directory if needed),
/// returning the absolute path it was written to.
#[tauri::command]
fn save_notebook(dir: String, name: String, contents: String) -> Result<String, String> {
    let dir = PathBuf::from(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let file = dir.join(format!("{name}.json"));
    std::fs::write(&file, &contents).map_err(|e| e.to_string())?;
    Ok(file.to_string_lossy().into_owned())
}

/// Delete a saved notebook file.
#[tauri::command]
fn delete_notebook(path: String) -> Result<(), String> {
    std::fs::remove_file(&path).map_err(|e| e.to_string())
}

/// Native open dialog for picking a directory to add to the notebook library.
#[tauri::command]
async fn pick_directory(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog().file().pick_folder(move |path| {
        let _ = tx.send(path);
    });
    rx.await
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Cluster topology/health for any control target — embedded (the in-process
/// cluster's URL) or remote (a coordinator URL). Both are just a URL, so one
/// command serves both via the shared `fetch_cluster_status` lib contract.
#[tauri::command]
async fn cluster_status_at(url: String) -> Result<Value, String> {
    let status = fetch_cluster_status(&url)
        .await
        .map_err(|e| e.to_string())?;
    serde_json::to_value(status).map_err(|e| e.to_string())
}

/// ADR-0021 P2b — Raft → read-model sync. Pull the coordinator's consensus-
/// committed metadata log and project every entry above our watermark into the
/// local `cluster_events` read model. Idempotent + resumable: entries are keyed
/// by (coordinator, log index) and committed Raft entries are immutable, so
/// replay is a no-op and any suite instance pointed at the same cluster
/// converges on the same projection — no central Postgres required. Returns the
/// number of newly applied entries.
async fn sync_cluster_read_model_inner(
    state: &AppState,
    coordinator_url: &str,
) -> Result<u64, String> {
    let watermark = state
        .controlplane
        .cluster_read_model_watermark(coordinator_url)
        .await
        .map_err(|e| e.to_string())?;
    let log = quant_fabric::fetch_metadata_log(coordinator_url)
        .await
        .map_err(|e| e.to_string())?;
    let entries: Vec<(i64, i64, i64, String, Value)> = log
        .into_iter()
        .filter(|e| (e.index as i64) > watermark)
        .map(|e| {
            let command = serde_json::to_value(&e.command).unwrap_or(Value::Null);
            // ConsensusCommand is externally tagged: {"RegisterWorker": {...}}.
            let kind = command
                .as_object()
                .and_then(|o| o.keys().next().cloned())
                .or_else(|| command.as_str().map(String::from))
                .unwrap_or_else(|| "unknown".to_string());
            (
                e.index as i64,
                e.term as i64,
                e.committed_ms as i64,
                kind,
                command,
            )
        })
        .collect();
    state
        .controlplane
        .apply_cluster_events(coordinator_url, &entries)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn sync_cluster_read_model(url: String, state: State<'_, AppState>) -> Result<u64, String> {
    sync_cluster_read_model_inner(&state, &url).await
}

/// Recent read-model events for a coordinator (newest first), from the local
/// projection — readable offline once synced.
#[tauri::command]
async fn list_cluster_events(
    url: String,
    limit: Option<i64>,
    state: State<'_, AppState>,
) -> Result<Vec<Value>, String> {
    state
        .controlplane
        .list_cluster_events(&url, limit.unwrap_or(100).clamp(1, 1000))
        .await
        .map_err(|e| e.to_string())
}

// ---- Control-plane store commands (ADR-0021) ----

/// Read a UI preference (e.g. template favorites / recents) from the persisted
/// `settings` table. Returns null when unset.
#[tauri::command]
async fn get_pref(key: String, state: State<'_, AppState>) -> Result<Option<String>, String> {
    state
        .controlplane
        .get_setting(&key)
        .await
        .map_err(|e| e.to_string())
}

/// Write a UI preference to the persisted `settings` table.
#[tauri::command]
async fn set_pref(key: String, value: String, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .set_setting(&key, &value)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_targets(state: State<'_, AppState>) -> Result<Vec<Target>, String> {
    state
        .controlplane
        .list_targets()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn save_target(target: Target, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .upsert_target(&target)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_target(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .delete_target(&id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_jobs(state: State<'_, AppState>) -> Result<Vec<Job>, String> {
    state
        .controlplane
        .list_jobs()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn save_job(job: Job, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .upsert_job(&job)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_job(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .delete_job(&id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn record_run(run: JobRun, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .record_run(&run)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_runs(
    job_id: Option<String>,
    state: State<'_, AppState>,
) -> Result<Vec<JobRun>, String> {
    state
        .controlplane
        .list_runs(job_id.as_deref())
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_run_steps(
    run_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<JobRunStep>, String> {
    state
        .controlplane
        .list_run_steps(&run_id)
        .await
        .map_err(|e| e.to_string())
}

// --- Connector instances + bindings (ADR-0032) --------------------------------

#[tauri::command]
async fn list_connector_instances(
    state: State<'_, AppState>,
) -> Result<Vec<ConnectorInstanceRecord>, String> {
    state
        .controlplane
        .list_connector_instances()
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn save_connector_instance(
    instance: ConnectorInstanceRecord,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .controlplane
        .upsert_connector_instance(&instance)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_connector_instance(
    id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .controlplane
        .delete_connector_instance(&id)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn list_bindings(state: State<'_, AppState>) -> Result<Vec<Binding>, String> {
    state
        .controlplane
        .list_bindings()
        .await
        .map_err(|e| e.to_string())
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .to_path_buf()
}

#[tauri::command]
async fn list_shared_connector_bindings() -> Result<Vec<SharedConnectorBinding>, String> {
    let repo = repo_root();
    let registry_path = repo.join("connectors/registry.json");
    let registry_raw = std::fs::read_to_string(&registry_path)
        .map_err(|e| format!("read {}: {e}", registry_path.display()))?;
    let registry: Value = serde_json::from_str(&registry_raw)
        .map_err(|e| format!("parse {}: {e}", registry_path.display()))?;
    let Some(bindings) = registry.get("bindings").and_then(Value::as_array) else {
        return Ok(Vec::new());
    };

    let mut out = Vec::with_capacity(bindings.len());
    for entry in bindings {
        let name = entry
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let file = entry
            .get("file")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        if file.is_empty() {
            continue;
        }
        let binding_path = repo.join(&file);
        let binding_raw = std::fs::read_to_string(&binding_path)
            .map_err(|e| format!("read {}: {e}", binding_path.display()))?;
        let binding: Value = serde_json::from_str(&binding_raw)
            .map_err(|e| format!("parse {}: {e}", binding_path.display()))?;
        let sink = binding.get("sink").unwrap_or(&Value::Null);
        let panel_mapping = binding.get("panel_mapping").unwrap_or(&Value::Null);
        let panel_view = if panel_mapping.is_object() {
            Some(format!("connector_{}_panel", safe_connector_view_name(&name)))
        } else {
            None
        };
        out.push(SharedConnectorBinding {
            name,
            file,
            driver: binding
                .get("driver")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            status: entry
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            landed: entry
                .get("landed")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            description: binding
                .get("description")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            research_use: binding
                .get("research_use")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            auth: binding
                .get("auth")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            cadence: binding
                .get("cadence")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            sink_stream: sink
                .get("stream")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            sink_path: sink
                .get("path")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            panel_view,
            panel_value_name: panel_mapping
                .get("value_name")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            panel_asset_class: panel_mapping
                .get("asset_class")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            note: entry
                .get("note")
                .and_then(Value::as_str)
                .map(ToString::to_string),
        });
    }
    Ok(out)
}

fn safe_connector_view_name(name: &str) -> String {
    let mut safe = String::new();
    let mut last_was_underscore = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            safe.push(c.to_ascii_lowercase());
            last_was_underscore = false;
        } else if !last_was_underscore {
            safe.push('_');
            last_was_underscore = true;
        }
    }
    let safe = safe.trim_matches('_').to_string();
    if safe.is_empty() || safe.as_bytes()[0].is_ascii_digit() {
        format!("_{safe}")
    } else {
        safe
    }
}

#[tauri::command]
async fn run_shared_connector_binding(name: String) -> Result<ConnectorActionResult, String> {
    let started = std::time::Instant::now();
    if name.is_empty()
        || !name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("binding name must contain only ASCII letters, digits, '_' or '-'".to_string());
    }
    let repo = repo_root();
    let output = async_runtime::spawn_blocking(move || {
        let timeout = Duration::from_secs(SHARED_CONNECTOR_RUN_TIMEOUT_SECS);
        let mut child = Command::new(repo.join("scripts/connectors/run.sh"))
            .arg("run")
            .arg(&name)
            .current_dir(&repo)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let deadline = std::time::Instant::now() + timeout;
        loop {
            if child.try_wait()?.is_some() {
                let output = child.wait_with_output()?;
                return Ok((name, output, false));
            }
            if std::time::Instant::now() >= deadline {
                let _ = child.kill();
                let output = child.wait_with_output()?;
                return Ok((name, output, true));
            }
            std::thread::sleep(Duration::from_millis(200));
        }
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| e.to_string())?;
    let (name, output, timed_out): (String, Output, bool) = output;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let message = if timed_out {
        format!(
            "Connector `{name}` timed out after {SHARED_CONNECTOR_RUN_TIMEOUT_SECS}s.{}{}",
            if stdout.is_empty() { "" } else { "\nstdout:\n" },
            stdout
        )
    } else if output.status.success() {
        stdout
            .lines()
            .find(|line| line.trim_start().starts_with("landed: "))
            .map(|line| line.trim().to_string())
            .unwrap_or_else(|| format!("Connector `{name}` completed."))
    } else {
        format!(
            "Connector `{name}` failed with status {:?}.{}{}",
            output.status.code(),
            if stdout.is_empty() { "" } else { "\nstdout:\n" },
            stdout
        )
    };
    let message = if stderr.is_empty() {
        message
    } else {
        format!("{message}\nstderr:\n{stderr}")
    };
    Ok(ConnectorActionResult {
        status: if !timed_out && output.status.success() {
            "complete".to_string()
        } else {
            "failed".to_string()
        },
        message,
        rows: Vec::new(),
        elapsed_ms: started.elapsed().as_millis(),
    })
}

#[tauri::command]
async fn save_binding(binding: Binding, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .upsert_binding(&binding)
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn delete_binding(id: String, state: State<'_, AppState>) -> Result<(), String> {
    state
        .controlplane
        .delete_binding(&id)
        .await
        .map_err(|e| e.to_string())
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BindingConformance {
    binding: Binding,
    conformance: Conformance,
}

fn binding_conformance_for(
    reference: &ContractRef,
    registry: &ContractRegistry,
    binding: Binding,
) -> Result<BindingConformance, String> {
    let conformance = match binding.schema.as_ref() {
        Some(schema) => {
            let relation = relation_columns_from_schema(schema)?;
            let mapping = binding_mapping(binding.mapping.as_ref())?;
            registry
                .validate(reference, &relation, mapping.as_ref())
                .map_err(|e| e.to_string())?
        }
        None => Conformance {
            conforms: false,
            missing: vec!["cached schema".to_string()],
            mismatches: Vec::new(),
        },
    };
    Ok(BindingConformance {
        binding,
        conformance,
    })
}

#[tauri::command]
async fn list_conforming_bindings(
    contract_ref: String,
    state: State<'_, AppState>,
) -> Result<Vec<BindingConformance>, String> {
    Ok(list_binding_conformance(contract_ref, state)
        .await?
        .into_iter()
        .filter(|item| item.conformance.conforms)
        .collect())
}

#[tauri::command]
async fn list_binding_conformance(
    contract_ref: String,
    state: State<'_, AppState>,
) -> Result<Vec<BindingConformance>, String> {
    let reference = ContractRef::parse(&contract_ref).map_err(|e| e.to_string())?;
    let registry = ContractRegistry::load_default().map_err(|e| e.to_string())?;
    let bindings = state
        .controlplane
        .list_bindings()
        .await
        .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    for binding in bindings {
        out.push(binding_conformance_for(&reference, &registry, binding)?);
    }
    Ok(out)
}

/// A domain contract surfaced to the UI: its reference (as templates declare it,
/// `Name.vN`), grouping/description, and the fields a conforming relation needs.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct ContractInfo {
    reference: String,
    domain: String,
    name: String,
    version: String,
    description: String,
    fields: Vec<quant_fabric::contracts::ContractField>,
}

/// List the available domain contracts (ADR-0032) so the UI can offer a contract
/// picker and render required fields — replacing the front-end's hardcoded
/// `CONTRACT_FIELDS` fallback with the real registry.
#[tauri::command]
async fn list_contracts() -> Result<Vec<ContractInfo>, String> {
    let registry = ContractRegistry::load_default().map_err(|e| e.to_string())?;
    Ok(registry
        .contracts()
        .iter()
        .map(|c| ContractInfo {
            reference: format!("{}.{}", c.name, c.version),
            domain: c.domain.clone(),
            name: c.name.clone(),
            version: c.version.clone(),
            description: c.description.clone(),
            fields: c.fields.clone(),
        })
        .collect())
}

fn binding_mapping(value: Option<&Value>) -> Result<Option<BTreeMap<String, String>>, String> {
    let Some(value) = value else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| "binding mapping must be a JSON object".to_string())?;
    let mut out = BTreeMap::new();
    for (field, source) in object {
        let Some(source) = source.as_str() else {
            return Err(format!("binding mapping value for `{field}` must be a string"));
        };
        if !source.trim().is_empty() {
            out.insert(field.clone(), source.to_string());
        }
    }
    Ok(Some(out))
}

fn relation_columns_from_schema(schema: &Value) -> Result<Vec<RelationColumn>, String> {
    let columns = schema
        .get("columns")
        .and_then(Value::as_array)
        .or_else(|| schema.as_array())
        .ok_or_else(|| "binding schema must be an array or `{ columns: [...] }`".to_string())?;
    let mut out = Vec::new();
    for column in columns {
        let object = column
            .as_object()
            .ok_or_else(|| "binding schema columns must be objects".to_string())?;
        let name = object
            .get("name")
            .and_then(Value::as_str)
            .ok_or_else(|| "binding schema column is missing `name`".to_string())?
            .trim();
        if name.is_empty() {
            continue;
        }
        let kind = object
            .get("kind")
            .or_else(|| object.get("dataType"))
            .or_else(|| object.get("type"))
            .and_then(Value::as_str)
            .and_then(column_kind_from_str);
        out.push(RelationColumn {
            name: name.to_string(),
            kind,
        });
    }
    Ok(out)
}

fn column_kind_from_str(raw: &str) -> Option<ColumnKind> {
    let s = raw.trim().to_ascii_lowercase();
    if s.is_empty() || s == "unknown" {
        return None;
    }
    if s.contains("int") {
        return Some(ColumnKind::Integer);
    }
    if matches!(s.as_str(), "number" | "numeric" | "decimal" | "double" | "float" | "real")
        || s.contains("decimal")
        || s.contains("double")
        || s.contains("float")
    {
        return Some(ColumnKind::Number);
    }
    if matches!(s.as_str(), "string" | "text" | "varchar" | "char") || s.contains("string") {
        return Some(ColumnKind::String);
    }
    if matches!(s.as_str(), "bool" | "boolean") {
        return Some(ColumnKind::Boolean);
    }
    if s.contains("date") || s.contains("time") {
        return Some(ColumnKind::Date);
    }
    None
}

struct StepResult {
    idx: i32,
    template: String,
    args: Value,
    status: String,
    row_count: i32,
    duration_ms: Option<u64>,
    log: Option<String>,
    /// Output column names of this step (from its first row) — used to populate
    /// the next step's column pickers in the Jobs editor.
    columns: Vec<String>,
}

fn classify_error(log: Option<&str>) -> &'static str {
    let lower = log.unwrap_or("").to_ascii_lowercase();
    if lower.is_empty() {
        "none"
    } else if lower.contains("timeout") || lower.contains("timed out") {
        "timeout"
    } else if lower.contains("auth") || lower.contains("unauthorized") || lower.contains("forbidden") {
        "auth"
    } else if lower.contains("schema") || lower.contains("column") || lower.contains("type mismatch") {
        "schema"
    } else if lower.contains("network") || lower.contains("connection") || lower.contains("dns") {
        "network"
    } else if lower.contains("file") || lower.contains("path") || lower.contains("not found") {
        "filesystem"
    } else {
        "other"
    }
}

fn step_metrics(step: &StepResult) -> Value {
    let mut metrics = serde_json::Map::from_iter([
        ("columnCount".to_string(), json!(step.columns.len())),
        ("outputColumns".to_string(), json!(step.columns)),
        (
            "hasLog".to_string(),
            json!(step
                .log
                .as_ref()
                .map(|s| !s.trim().is_empty())
                .unwrap_or(false)),
        ),
        (
            "errorClass".to_string(),
            json!(classify_error(step.log.as_deref())),
        ),
    ]);
    if let Some(duration_ms) = step.duration_ms {
        metrics.insert("durationMs".to_string(), json!(duration_ms));
    }
    Value::Object(metrics)
}

/// Parse one job step into `(template, raw-args-json, bound-args)`, binding the
/// template's table input to `source`. Used by the remote step path; the
/// embedded path now runs through the shared Flow runner (ADR-0030).
fn build_step_args(step: &Value, source: &str) -> (String, Value, BTreeMap<String, String>) {
    let template = step
        .get("template")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let args_json = step.get("args").cloned().unwrap_or_else(|| json!({}));
    let mut args: BTreeMap<String, String> = BTreeMap::new();
    if let Some(obj) = args_json.as_object() {
        for (k, v) in obj {
            args.insert(
                k.clone(),
                v.as_str()
                    .map(String::from)
                    .unwrap_or_else(|| v.to_string()),
            );
        }
    }
    if let Some(t) = quant_fabric::templates::find_template(&template).and_then(|tpl| {
        tpl.inputs
            .iter()
            .find(|i| i.kind == quant_fabric::templates::TemplateInputKind::Table)
    }) {
        args.insert(t.name.to_string(), source.to_string());
    }
    (template, args_json, args)
}

fn run_metadata(definition: &Value) -> Value {
    let mut meta = serde_json::Map::new();
    if let Some(binding) = definition.get("packBinding") {
        meta.insert("packBinding".to_string(), binding.clone());
    }
    let execution_mode = if definition
        .get("schedule")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
    {
        "scheduled"
    } else {
        "on_demand"
    };
    meta.insert("executionMode".to_string(), json!(execution_mode));
    meta.insert("sourceMode".to_string(), json!(source_mode(definition)));
    meta.insert("syncMode".to_string(), json!(sync_mode(definition)));
    if let Some(output) = output_metadata_from_definition(definition) {
        meta.insert("output".to_string(), Value::Object(output));
    }
    // Record the Flow's reproducibility class on every run (ADR-0030 P4): a
    // deterministic job (SQL + deterministic quant kernels) over pinned inputs
    // is reproducible; a nondeterministic step downgrades it. Surfaced so run
    // history can badge it and (later) gate result caching / backfill.
    let determinism =
        match quant_fabric::flow::Flow::from_job_definition("", "", "", definition).determinism() {
            quant_fabric::flow::Determinism::Deterministic => "deterministic",
            quant_fabric::flow::Determinism::Nondeterministic => "nondeterministic",
        };
    meta.insert("determinism".to_string(), json!(determinism));
    Value::Object(meta)
}

fn source_mode(definition: &Value) -> &'static str {
    if definition
        .get("sourceBindingId")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
    {
        "binding"
    } else if definition
        .get("sourceConnectorId")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false)
    {
        "connector"
    } else {
        "file"
    }
}

fn sync_mode(definition: &Value) -> String {
    if definition
        .get("fullRefresh")
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        return "full_refresh".to_string();
    }
    if let Some(mode) = definition
        .get("targetConnectorParams")
        .and_then(|v| v.get("mode"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        return mode.to_string();
    }
    match source_mode(definition) {
        "binding" | "connector" => "incremental".to_string(),
        _ => "ad_hoc".to_string(),
    }
}

fn output_metadata_from_definition(
    definition: &Value,
) -> Option<serde_json::Map<String, Value>> {
    let mut output = serde_json::Map::new();
    if let Some(sink_uri) = definition
        .get("sink")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        output.insert("sinkUri".to_string(), json!(sink_uri));
    }
    if let Some(target_connector_id) = definition
        .get("targetConnectorId")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        output.insert("targetConnectorId".to_string(), json!(target_connector_id));
    }
    if let Some(target_binding_id) = definition
        .get("targetBindingId")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        output.insert("targetBindingId".to_string(), json!(target_binding_id));
    }
    if let Some(target_path) = definition
        .get("targetConnectorParams")
        .and_then(|v| v.get("output_path"))
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty())
    {
        output.insert("targetPath".to_string(), json!(target_path));
    }
    if output.is_empty() {
        None
    } else {
        Some(output)
    }
}

fn execute_steps(
    session: &StudioSession,
    source: &str,
    steps: &[Value],
) -> (Vec<StepResult>, RunOutput) {
    // Delegate to the shared single Flow runner (ADR-0030 P2): the notebook
    // preview path and this job path now run on one code path. The job's
    // template-ref steps adapt to unit steps; the runner chains each step's
    // output into the next exactly as before. `template`/`args` for history are
    // recovered from the original step JSON by index (order is preserved, and
    // the runner stops at the first failure just like the old loop).
    let def = json!({ "source": source, "steps": steps });
    let flow = quant_fabric::flow::Flow::from_job_definition("", "", "", &def);
    let run = flow.run_on_session(session, Some(source), &BTreeMap::new());
    let results = run
        .steps
        .iter()
        .enumerate()
        .map(|(i, sr)| {
            let template = steps
                .get(i)
                .and_then(|s| s.get("template"))
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            let args = steps
                .get(i)
                .and_then(|s| s.get("args"))
                .cloned()
                .unwrap_or_else(|| json!({}));
            StepResult {
                idx: i as i32,
                template,
                args,
                status: sr.status.clone(),
                row_count: sr.row_count as i32,
                duration_ms: None,
                log: sr.error.clone(),
                columns: sr.columns.clone(),
            }
        })
        .collect();
    (results, run.output)
}

/// Run a job's steps against a REMOTE coordinator (HTTP submit per step). The
/// cluster's workers must be able to reach the job `source` (shared path / object
/// store).
///
/// If `sink` is set, the steps **chain**: each non-final step asks the cluster to
/// materialize its output to `{sink}/{run_id}-step{i}.parquet` (a shared sink the
/// workers can write and the next step can read), and the next step's table input
/// is bound to that URI. Without a `sink`, steps run independently over `source`
/// (chaining host-local intermediates wouldn't be reachable by remote workers).
async fn execute_steps_remote(
    target: &Target,
    source: &str,
    steps: &[Value],
    sink: Option<&str>,
    run_id: &str,
) -> (Vec<StepResult>, Option<RunOutput>) {
    let mut results: Vec<StepResult> = Vec::new();
    let mut current = source.to_string();
    let mut final_output: Option<RunOutput> = None;
    let n = steps.len();
    for (i, step) in steps.iter().enumerate() {
        let (template, args_json, args) = build_step_args(step, &current);
        let step_started = std::time::Instant::now();
        // Sink the output for the next step (only when chaining and not the last).
        let want_uri = match (sink, i + 1 < n) {
            // CSV by default — the engine only writes Parquet artifacts when
            // built with `--features parquet`. The next step reads it via Auto.
            (Some(base), true) => Some(format!(
                "{}/{run_id}-step{i}.csv",
                base.trim_end_matches('/')
            )),
            _ => None,
        };
        let (out, written) = studio::run_remote_with_sink(
            &target.url,
            target.token.as_deref(),
            &template,
            &args,
            want_uri.as_deref(),
        )
        .await;
        let status = if out.status == "complete" {
            "complete"
        } else {
            "failed"
        };
        let failed = status == "failed";
        // Feed the materialized result into the next step.
        if !failed {
            if let Some(uri) = written.or(want_uri) {
                current = uri;
            }
        }
        results.push(StepResult {
            idx: i as i32,
            template,
            args: args_json,
            status: status.into(),
            row_count: out.row_count as i32,
            duration_ms: Some(step_started.elapsed().as_millis() as u64),
            log: out.error.clone(),
            columns: out
                .rows
                .first()
                .map(|r| r.keys().cloned().collect())
                .unwrap_or_default(),
        });
        if failed {
            break;
        }
        // Keep the latest successful step's output (rows the coordinator returned
        // in-process), so a remote run can stage its final result (ADR-0033).
        final_output = Some(out);
    }
    (results, final_output)
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// True when `expr` (standard 5-field cron, evaluated in UTC) has a scheduled
/// instant in the half-open window `(prev, now]` — i.e. the job became due since
/// the previous scheduler tick. Unparseable expressions never fire.
fn cron_due(expr: &str, prev: i64, now: i64) -> bool {
    use chrono::{TimeZone, Utc};
    use std::str::FromStr;
    let Ok(cron) = croner::Cron::from_str(expr) else {
        return false;
    };
    let Some(prev_dt) = Utc.timestamp_opt(prev, 0).single() else {
        return false;
    };
    match cron.find_next_occurrence(&prev_dt, false) {
        Ok(next) => next.timestamp() <= now,
        Err(_) => false,
    }
}

/// Select `(job_id, target_id)` for enabled jobs whose `definition.schedule` cron
/// fires in `(prev, now]`. The target is `definition.target` (a target id) or the
/// embedded engine by default. Pure — unit-tested in `tests`.
fn due_jobs(jobs: &[Job], prev: i64, now: i64) -> Vec<(String, String)> {
    jobs.iter()
        .filter(|j| j.enabled)
        .filter_map(|j| {
            let sched = j.definition.get("schedule").and_then(|v| v.as_str())?;
            let sched = sched.trim();
            if sched.is_empty() || !cron_due(sched, prev, now) {
                return None;
            }
            let target = j
                .definition
                .get("target")
                .and_then(|v| v.as_str())
                .unwrap_or("embedded")
                .to_string();
            Some((j.id.clone(), target))
        })
        .collect()
}

/// Settings key holding the scheduler's last-tick watermark (epoch secs).
const SCHED_WATERMARK_KEY: &str = "scheduler:watermark";

/// Scheduler lease TTL (secs). Must exceed the tick interval so a single slow/
/// missed tick doesn't drop leadership; short enough that a dead leader is
/// replaced promptly.
const SCHED_LEASE_TTL_SECS: i64 = 90;

/// A per-process scheduler identity for leader election.
fn instance_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("qf-{}-{nanos}", std::process::id())
}

/// Lease-gated scheduler pass (P4c): acquire/renew the singleton lease for
/// `holder`; only the leader runs [`scheduler_tick`]. Returns `(is_leader,
/// fired)`. Standbys do nothing, so N instances against one store never
/// double-fire; if the leader dies, its lease expires and a standby takes over.
async fn scheduler_tick_leased(
    state: &AppState,
    now: i64,
    holder: &str,
    ttl: i64,
) -> (bool, usize) {
    let leader = match state.controlplane.try_acquire_lease(holder, now, ttl).await {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[scheduler] lease error: {e}");
            return (false, 0);
        }
    };
    if !leader {
        return (false, 0);
    }
    (true, scheduler_tick(state, now).await)
}

/// One scheduler pass at simulated time `now` (epoch secs): fire every job due
/// since the **persisted** watermark, then advance it. Returns the number fired.
///
/// Pulling the pass out of the loop (and taking `now` as a parameter) is what
/// makes the scheduler verifiable without waiting: a test drives a single tick
/// with a clock it controls and asserts a real run was recorded — no real cron
/// interval ever elapses. The watermark living in the store (not memory) gives
/// catch-up across restarts: a cron instant that passed while the app was closed
/// fires once on the next tick.
async fn scheduler_tick(state: &AppState, now: i64) -> usize {
    // Watermark: persisted previous tick, or `now` on first ever launch (a fresh
    // install doesn't retro-fire history it never saw).
    let prev = match state.controlplane.get_setting(SCHED_WATERMARK_KEY).await {
        Ok(Some(v)) => v.parse::<i64>().unwrap_or(now),
        Ok(None) => now,
        Err(e) => {
            eprintln!("[scheduler] read watermark failed: {e}");
            return 0;
        }
    };
    let mut fired = 0;
    match state.controlplane.list_jobs().await {
        Ok(jobs) => {
            for (job_id, target_id) in due_jobs(&jobs, prev, now) {
                eprintln!("[scheduler] firing job {job_id} -> {target_id}");
                match execute_job(state, &job_id, &target_id, "schedule").await {
                    Ok(_) => fired += 1,
                    Err(e) => eprintln!("[scheduler] job {job_id} failed: {e}"),
                }
            }
        }
        Err(e) => eprintln!("[scheduler] list_jobs failed: {e}"),
    }
    // Advance only after processing, so a crash mid-tick re-processes the window
    // rather than skipping it.
    if let Err(e) = state
        .controlplane
        .set_setting(SCHED_WATERMARK_KEY, &now.to_string())
        .await
    {
        eprintln!("[scheduler] persist watermark failed: {e}");
    }
    fired
}

/// P4 scheduler: tick every 30s for the app's lifetime. Lease-gated (P4c) so it
/// coordinates with any other instance (e.g. a headless `qf-scheduler`) sharing
/// the same control-plane store. (App-side; the headless [`SchedulerService`] is
/// the always-on, GUI-free counterpart — see ADR-0021.)
fn spawn_scheduler(app: tauri::AppHandle) {
    async_runtime::spawn(async move {
        let tick = std::time::Duration::from_secs(30);
        let holder = instance_id();
        loop {
            if let Some(state) = app.try_state::<AppState>() {
                scheduler_tick_leased(&state, now_secs(), &holder, SCHED_LEASE_TTL_SECS).await;
            }
            tokio::time::sleep(tick).await;
        }
    });
}

/// P4b headless scheduler — the GUI-free, always-on counterpart of
/// [`spawn_scheduler`]. It connects to a **control-plane store URL** (a central
/// Postgres for a deployment where the desktop app may be closed, or a PGlite
/// path) and runs the *same* [`scheduler_tick`] / [`execute_job`] the GUI uses,
/// so behaviour cannot drift between the two. Embedded-target jobs run on this
/// process's warm engine (their source must be reachable here); remote-target
/// jobs dispatch to their coordinator. Drive it from the `qf-scheduler` binary.
pub struct SchedulerService {
    state: AppState,
    tick: std::time::Duration,
}

impl SchedulerService {
    /// Connect to the control-plane store at `store_url`, register the operator
    /// templates under `template_dirs`, and prepare a warm engine. `tick_secs` is
    /// the loop cadence for [`run`](Self::run).
    pub async fn connect(
        store_url: &str,
        template_dirs: Vec<PathBuf>,
        tick_secs: u64,
    ) -> anyhow::Result<Self> {
        let controlplane = ControlStore::connect(store_url).await?;
        // Register operator templates so scheduled jobs resolve by name.
        let _ = studio::catalog(&template_dirs);
        let external_pg = ExternalPg::placeholder(store_url);
        let state = AppState {
            cluster: Mutex::new(None),
            studio: StdMutex::new(StudioSession::new()?),
            template_dirs,
            controlplane,
            external_pg,
        };
        Ok(Self {
            state,
            tick: std::time::Duration::from_secs(tick_secs.max(1)),
        })
    }

    /// Run one pass at simulated `now` (epoch secs); returns the number fired.
    /// Exposed so the service is verifiable with a controlled clock — no waiting.
    pub async fn tick_once(&self, now: i64) -> usize {
        scheduler_tick(&self.state, now).await
    }

    /// The control-plane store this service is bound to (for seeding in tests /
    /// administrative use).
    pub fn store(&self) -> &ControlStore {
        &self.state.controlplane
    }

    /// Run the scheduler loop forever, ticking at the configured cadence.
    /// Lease-gated (P4c): run several instances against one central store for HA —
    /// exactly one is leader and fires; the rest stand by until it dies.
    pub async fn run(self) {
        let holder = instance_id();
        loop {
            let (leader, _fired) =
                scheduler_tick_leased(&self.state, now_secs(), &holder, SCHED_LEASE_TTL_SECS).await;
            if !leader {
                eprintln!("[scheduler] standby (another instance holds the lease)");
            }
            tokio::time::sleep(self.tick).await;
        }
    }
}

/// Core job execution shared by the `run_job` command (trigger `"manual"`) and
/// the scheduler (trigger `"schedule"`): resolve job + target, record a `running`
/// run, execute the steps (embedded chained / remote dispatched), record each
/// step + the final run status. Returns the run id.
/// The local artifact-store root for staged relations (ADR-0033). Overridable
/// via `QUANT_FABRIC_STAGING_DIR`; defaults to a stable temp subdir. (A config-
/// driven, app-data-dir-anchored root is a later refinement.)
fn staging_root() -> std::path::PathBuf {
    std::env::var("QUANT_FABRIC_STAGING_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("quant-fabric-staging"))
}

/// Select the staging artifact store at runtime (ADR-0033 L0 — the single
/// content-addressed bucket): the shared S3/MinIO/Garage bucket when
/// `QUANT_FABRIC_STAGING_BUCKET` is set, else the local store at
/// `staging_root()`. Falls back to local if the S3 client can't be built. (This
/// is the single shared bucket, NOT a replicating cluster — that stays deferred.)
/// The per-node L1 read-cache directory (ADR-0033 33-C), distinct from the local
/// store root. Overridable via `QUANT_FABRIC_STAGING_CACHE_DIR`.
fn staging_cache_dir() -> std::path::PathBuf {
    std::env::var("QUANT_FABRIC_STAGING_CACHE_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("quant-fabric-staging-cache"))
}

fn staging_store() -> Result<Box<dyn quant_fabric::staging::ArtifactStore>, String> {
    match std::env::var("QUANT_FABRIC_STAGING_BUCKET") {
        Ok(bucket) if !bucket.trim().is_empty() => {
            let store = quant_fabric::staging::S3ArtifactStore::from_env(bucket.trim())
                .map_err(|e| e.to_string())?;
            Ok(Box::new(store))
        }
        _ => Ok(Box::new(quant_fabric::staging::LocalArtifactStore::new(
            staging_root(),
        ))),
    }
}

/// The current version of every template/operation, by name: the content hash of
/// its source (operator templates) or a name-stable token (built-ins). Recorded
/// as `Operation` dependencies on staged outputs so an edit to a template's logic
/// marks results produced by it `Invalid` (ADR-0033 33-B).
fn template_versions(
    template_dirs: &[std::path::PathBuf],
) -> std::collections::BTreeMap<String, String> {
    quant_fabric::studio::catalog(template_dirs)
        .into_iter()
        .map(|t| {
            let version = t
                .source
                .as_deref()
                .map(|s| quant_fabric::staging::content_hash(s.as_bytes()))
                .unwrap_or_else(|| quant_fabric::staging::content_hash(t.name.as_bytes()));
            (t.name, version)
        })
        .collect()
}

fn file_input_reference(path: &str) -> String {
    format!("file:{path}")
}

fn file_input_version(path: &str) -> Option<String> {
    let bytes = std::fs::read(path).ok()?;
    Some(quant_fabric::staging::content_hash(&bytes))
}

fn current_input_version(reference: &str) -> Option<String> {
    reference
        .strip_prefix("file:")
        .and_then(file_input_version)
}

fn add_non_binding_input_dependencies(
    definition: &serde_json::Value,
    depends_on: &mut Vec<quant_fabric::staging::Dependency>,
) {
    let has_source_binding = definition
        .get("sourceBindingId")
        .and_then(|v| v.as_str())
        .map(|s| !s.trim().is_empty())
        .unwrap_or(false);
    if has_source_binding {
        return;
    }
    let Some(source) = definition.get("source").and_then(|v| v.as_str()) else {
        return;
    };
    let source = source.trim();
    if source.is_empty() {
        return;
    }
    if let Some(version) = file_input_version(source) {
        depends_on.push(quant_fabric::staging::Dependency::input(
            file_input_reference(source),
            version,
        ));
    }
}

/// Stage a completed job's final output as an addressable, immutable staged
/// relation (ADR-0033) and record it in the control-plane registry. Best-effort:
/// a staging failure is logged, never fails the job. Returns the relation id.
async fn stage_final_output(
    state: &AppState,
    run_id: &str,
    job_name: &str,
    definition: &serde_json::Value,
    rows: &[quant_fabric::model::Row],
) -> Option<String> {
    use quant_fabric::staging::{stage_relation, Producer, RetentionPolicy};
    let columns: Vec<String> = rows
        .first()
        .map(|r| r.keys().cloned().collect())
        .unwrap_or_default();
    // Retention is a property of the relation (ADR-0033); a job may request it,
    // else the conservative default (lives only as long as this run).
    let retention = match definition.get("retention").and_then(serde_json::Value::as_str) {
        Some("session") => RetentionPolicy::Session,
        Some("pinned") => RetentionPolicy::Pinned,
        _ => RetentionPolicy::Ephemeral,
    };
    let producer = Producer {
        flow_run_id: Some(run_id.to_string()),
        ..Producer::default()
    };
    // Dependencies (ADR-0033) pin the current versions of what produced this
    // relation, so a later upstream change flips its freshness:
    //  - the source binding (query/path change ⇒ `Stale`);
    //  - plain file sources (content-hash change ⇒ `Stale`);
    //  - each operation/template used in the steps (logic change ⇒ `Invalid`).
    let mut depends_on = Vec::new();
    if let Some(binding_id) = definition.get("sourceBindingId").and_then(|v| v.as_str()) {
        if let Ok(Some(binding)) = state.controlplane.get_binding(binding_id).await {
            depends_on.push(quant_fabric::staging::Dependency::binding(
                binding_id,
                binding_version(&binding),
            ));
        }
    }
    add_non_binding_input_dependencies(definition, &mut depends_on);
    let versions = template_versions(&state.template_dirs);
    let mut seen_ops = std::collections::BTreeSet::new();
    if let Some(steps) = definition.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            if let Some(name) = step.get("template").and_then(|v| v.as_str()) {
                if seen_ops.insert(name.to_string()) {
                    if let Some(version) = versions.get(name) {
                        depends_on.push(quant_fabric::staging::Dependency::operation(
                            name,
                            version.clone(),
                        ));
                    }
                }
            }
        }
    }
    let artifact_store = match staging_store() {
        Ok(store) => store,
        Err(e) => {
            eprintln!("[job staging] store: {e}");
            return None;
        }
    };
    let manifest = match stage_relation(
        artifact_store.as_ref(),
        run_id, // relation id — unique per run
        Some(job_name.to_string()),
        rows,
        columns,
        producer,
        depends_on,
        retention,
    ) {
        Ok(manifest) => manifest,
        Err(e) => {
            eprintln!("[job staging] {e}");
            return None;
        }
    };
    let value = serde_json::to_value(&manifest).ok()?;
    if let Err(e) = state.controlplane.upsert_staged_relation(&value).await {
        eprintln!("[job staging] record: {e}");
        return None;
    }
    Some(manifest.relation_id)
}

// --- Staged relations: list (freshness-enriched) / pin / forget (ADR-0033) ---

#[tauri::command]
async fn list_staged_relations(
    state: State<'_, AppState>,
) -> Result<Vec<serde_json::Value>, String> {
    use quant_fabric::staging::{
        compute_freshness, is_recomputable, DependencyKind, StagedRelationManifest,
    };
    let manifests = state
        .controlplane
        .list_staged_relations()
        .await
        .map_err(|e| e.to_string())?;
    // Current versions of everything a staged relation can depend on:
    //  - every staged relation, by id → its content hash (input staleness);
    //  - every binding, by id → its current version (binding staleness, so a
    //    changed query/path marks dependents `Stale`).
    //  - file-backed source inputs, by `file:<path>` → their current content hash.
    // Current operation/template versions are folded in below, so editing a
    // template invalidates staged relations produced by the old template logic.
    let mut current: std::collections::BTreeMap<String, String> = manifests
        .iter()
        .filter_map(|m| {
            Some((
                m.get("relationId")?.as_str()?.to_string(),
                m.get("contentHash")?.as_str()?.to_string(),
            ))
        })
        .collect();
    if let Ok(bindings) = state.controlplane.list_bindings().await {
        for b in &bindings {
            current.insert(b.id.clone(), binding_version(b));
        }
    }
    for manifest in &manifests {
        if let Ok(parsed) = serde_json::from_value::<StagedRelationManifest>(manifest.clone()) {
            for dep in parsed.depends_on {
                if dep.kind == DependencyKind::Input {
                    if let Some(version) = current_input_version(&dep.reference) {
                        current.insert(dep.reference, version);
                    }
                }
            }
        }
    }
    // Current operation/template versions, so an edited template marks results
    // produced by it `Invalid`.
    let ops = template_versions(&state.template_dirs);
    let enriched = manifests
        .into_iter()
        .map(|m| {
            let mut out = m.clone();
            if let Ok(parsed) = serde_json::from_value::<StagedRelationManifest>(m) {
                let freshness = compute_freshness(&parsed.depends_on, &current, &ops);
                if let Some(obj) = out.as_object_mut() {
                    obj.insert("freshness".to_string(), json!(freshness));
                    obj.insert(
                        "recomputable".to_string(),
                        json!(is_recomputable(&parsed.producer)),
                    );
                }
            }
            out
        })
        .collect();
    Ok(enriched)
}

#[tauri::command]
async fn pin_staged_relation(
    relation_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .controlplane
        .set_staged_relation_retention(&relation_id, "pinned")
        .await
        .map_err(|e| e.to_string())
}

#[tauri::command]
async fn forget_staged_relation(
    relation_id: String,
    state: State<'_, AppState>,
) -> Result<(), String> {
    state
        .controlplane
        .delete_staged_relation(&relation_id)
        .await
        .map_err(|e| e.to_string())
}

/// Clear ephemeral (job-run-scoped) staged scratch (ADR-0033 retention GC).
/// Returns the number removed; `session`/`pinned` relations are untouched.
#[tauri::command]
async fn gc_staged_relations(state: State<'_, AppState>) -> Result<u64, String> {
    state
        .controlplane
        .delete_staged_relations_by_retention("ephemeral")
        .await
        .map_err(|e| e.to_string())
}

/// Read a staged relation's rows **through the per-node L1 cache** (ADR-0033) —
/// the live "build on the last result" read path. Bytes are fetched from the
/// selected store (local or the shared S3 bucket), cached locally for locality,
/// and verified against the manifest's content hash before decoding. This closes
/// the write-global / read-through-cache loop.
#[tauri::command]
async fn read_staged_relation(
    relation_id: String,
    state: State<'_, AppState>,
) -> Result<Vec<quant_fabric::model::Row>, String> {
    use quant_fabric::staging::{
        load_staged_rows_via_cache, LocalReadThroughCache, StagedRelationManifest,
    };
    let manifest_value = state
        .controlplane
        .get_staged_relation(&relation_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown staged relation `{relation_id}`"))?;
    let manifest: StagedRelationManifest =
        serde_json::from_value(manifest_value).map_err(|e| e.to_string())?;
    let cache = LocalReadThroughCache::new(staging_cache_dir(), staging_store()?);
    load_staged_rows_via_cache(&cache, &manifest).map_err(|e| e.to_string())
}

/// A stable version token for a binding (ADR-0033 dependency tracking): the
/// content hash of its per-run shape (`run_params` + contract). When a binding's
/// query/path changes this changes, so staged relations that depend on it become
/// `Stale`. `run_params` serializes deterministically (serde_json sorts object
/// keys), so the token is stable for an unchanged binding.
fn binding_version(binding: &Binding) -> String {
    let basis = format!(
        "{}\u{1}{}",
        binding.run_params,
        binding.contract_ref.as_deref().unwrap_or("")
    );
    quant_fabric::staging::content_hash(basis.as_bytes())
}

/// Convert a binding's discovered `schema` (a JSON array of `{name, kind}`) into
/// engine `RelationColumn`s for conformance checking (ADR-0032 32-C1). A column
/// with an unknown/absent kind is presence-only (still validated for presence).
fn binding_schema_to_columns(
    schema: Option<&serde_json::Value>,
) -> Vec<quant_fabric::contracts::RelationColumn> {
    schema
        .and_then(|schema| relation_columns_from_schema(schema).ok())
        .unwrap_or_default()
}

/// Flatten a JSON object of string values to a `String`→`String` map (a binding's
/// contract-field → source-column mapping).
fn value_to_string_map(v: &serde_json::Value) -> std::collections::BTreeMap<String, String> {
    v.as_object()
        .map(|o| {
            o.iter()
                .filter_map(|(k, val)| Some((k.clone(), val.as_str()?.to_string())))
                .collect()
        })
        .unwrap_or_default()
}

/// Validate a binding against a domain contract (ADR-0032 32-C1) using its cached
/// discovered schema + optional column mapping — the backend gate the UI calls to
/// filter a `TypedRelation` picker to conforming bindings (GPT 302). Returns the
/// `Conformance { conforms, missing, mismatches }`.
#[tauri::command]
async fn validate_binding_conformance(
    binding_id: String,
    contract_ref: String,
    state: State<'_, AppState>,
) -> Result<serde_json::Value, String> {
    use quant_fabric::contracts::{ContractRef, ContractRegistry};
    let binding = state
        .controlplane
        .get_binding(&binding_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown binding `{binding_id}`"))?;
    let reference = ContractRef::parse(&contract_ref).map_err(|e| e.to_string())?;
    let registry = ContractRegistry::load_default().map_err(|e| e.to_string())?;
    let columns = binding_schema_to_columns(binding.schema.as_ref());
    let mapping = binding.mapping.as_ref().map(value_to_string_map);
    let conformance = registry
        .validate(&reference, &columns, mapping.as_ref())
        .map_err(|e| e.to_string())?;
    serde_json::to_value(conformance).map_err(|e| e.to_string())
}

async fn execute_job(
    state: &AppState,
    job_id: &str,
    target_id: &str,
    trigger: &str,
) -> Result<String, String> {
    let mut job = state
        .controlplane
        .get_job(job_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown job {job_id}"))?;
    let target = state
        .controlplane
        .get_target(target_id)
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| format!("unknown target {target_id}"))?;
    if job
        .definition
        .get("packBinding")
        .and_then(|v| v.get("updatePolicy"))
        .and_then(|v| v.as_str())
        == Some("track")
    {
        let root = github_pack_root()?;
        github_packs::apply_tracked_update(&state.controlplane, &mut job, &root)
            .await
            .map_err(|e| e.to_string())?;
    }

    let run_id = format!(
        "run-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0)
    );
    let mut metadata = run_metadata(&job.definition);
    state
        .controlplane
        .record_run(&JobRun {
            id: run_id.clone(),
            job_id: job_id.to_string(),
            target_id: Some(target_id.to_string()),
            status: "running".into(),
            trigger: trigger.to_string(),
            started_at: String::new(),
            finished_at: None,
            metadata: metadata.clone(),
        })
        .await
        .map_err(|e| e.to_string())?;

    // Configured connector instances + first-class bindings (ADR-0032). The
    // helper unions the promoted table with the legacy `connectors:instances`
    // blob, builds the binding refs the run path consumes, and resolves any
    // `secrets-keeper://…` connection secrets. A resolution failure is recorded
    // as a failed run, like a source-resolution failure below.
    let (connector_instances, bindings) = match connector_runtime_refs(state).await {
        Ok(v) => v,
        Err(e) => {
            state
                .controlplane
                .record_run(&JobRun {
                    id: run_id.clone(),
                    job_id: job_id.to_string(),
                    target_id: Some(target_id.to_string()),
                    status: "failed".into(),
                    trigger: trigger.to_string(),
                    started_at: String::new(),
                    finished_at: None,
                    metadata: json!({ "error": e }),
                })
                .await
                .map_err(|e| e.to_string())?;
            return Ok(run_id);
        }
    };

    // Resolve the source: a plain path passes through; a connector binding runs
    // the real celeritas import (ADR-0028) and materialises it to a temp CSV.
    let source = match connectors_job::resolve_job_source(
        &job.definition,
        &connector_instances,
        &bindings,
    ) {
        Ok(s) => s,
        Err(e) => {
            state
                .controlplane
                .record_run(&JobRun {
                    id: run_id.clone(),
                    job_id: job_id.to_string(),
                    target_id: Some(target_id.to_string()),
                    status: "failed".into(),
                    trigger: trigger.to_string(),
                    started_at: String::new(),
                    finished_at: None,
                    metadata: json!({ "error": e }),
                })
                .await
                .map_err(|e| e.to_string())?;
            return Ok(run_id);
        }
    };
    let steps = job
        .definition
        .get("steps")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    // Optional shared sink for remote inter-step chaining (a URI the cluster can
    // write + a later step can read, e.g. a shared dir or `s3://…`).
    let sink = job
        .definition
        .get("sink")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    // Embedded: run all steps chained under one scoped lock (sync — the guard is
    // never held across an await). Remote: dispatch each step to the coordinator,
    // chaining via the shared `sink` when configured.
    // The final step's output rows are available in-process for BOTH paths: the
    // embedded runner returns them, and the remote coordinator returns them via
    // `run_remote_with_sink`. `Some` on success, `None` if a step failed.
    let (results, final_output) = if target.kind == "embedded" {
        let session = state.studio.lock().map_err(|e| e.to_string())?;
        let (r, last) = execute_steps(&session, &source, &steps);
        (r, Some(last))
    } else {
        execute_steps_remote(&target, &source, &steps, sink, &run_id).await
    };
    let mut overall = if results.iter().any(|r| r.status == "failed") {
        "failed"
    } else {
        "complete"
    };

    let total_row_count: i64 = results
        .iter()
        .map(|r| i64::from(r.row_count.max(0)))
        .sum();
    let successful_steps = results.iter().filter(|r| r.status == "complete").count();
    let failed_steps = results.iter().filter(|r| r.status == "failed").count();

    if let Some(meta) = metadata.as_object_mut() {
        meta.insert(
            "summary".to_string(),
            json!({
                "stepCount": results.len(),
                "successfulSteps": successful_steps,
                "failedSteps": failed_steps,
                "totalRowCount": total_row_count,
            }),
        );
    }

    if overall == "complete" {
        if let Some(out) = &final_output {
            // Export the final output to the target connector when one is bound —
            // embedded only (a remote run already delivers via the shared sink URI
            // when chaining).
            if target.kind == "embedded" {
                match connectors_job::export_job_sink(
                    &job.definition,
                    &connector_instances,
                    &bindings,
                    &out.rows,
                ) {
                    Ok(Some(driver)) => {
                        if let Some(meta) = metadata.as_object_mut() {
                            let output = meta
                                .entry("output".to_string())
                                .or_insert_with(|| json!({}));
                            if let Some(output_map) = output.as_object_mut() {
                                output_map
                                    .insert("exportDriver".to_string(), json!(driver.clone()));
                            }
                        }
                        eprintln!(
                            "[job {job_id}] exported {} rows via `{driver}`",
                            out.rows.len()
                        );
                    }
                    Ok(None) => {}
                    Err(e) => {
                        eprintln!("[job {job_id}] {e}");
                        overall = "failed";
                    }
                }
            }
            // Stage the computed result as an addressable, immutable staged
            // relation (ADR-0033) — the scratchpad a later prompt/flow builds on.
            // Works for both embedded and remote runs; best-effort, never fails it.
            if let Some(rel) =
                stage_final_output(state, &run_id, &job.name, &job.definition, &out.rows).await
            {
                if let Some(meta) = metadata.as_object_mut() {
                    let output = meta
                        .entry("output".to_string())
                        .or_insert_with(|| json!({}));
                    if let Some(output_map) = output.as_object_mut() {
                        output_map.insert("stagedRelationId".to_string(), json!(rel.clone()));
                    }
                }
                eprintln!("[job {job_id}] staged final output as `{rel}`");
            }
        }
    }

    for r in &results {
        state
            .controlplane
            .record_step(&JobRunStep {
                run_id: run_id.clone(),
                step_idx: r.idx,
                template: r.template.clone(),
                args: r.args.clone(),
                status: r.status.clone(),
                row_count: Some(r.row_count),
                metrics: step_metrics(r),
                log: r.log.clone(),
            })
            .await
            .map_err(|e| e.to_string())?;
    }
    state
        .controlplane
        .record_run(&JobRun {
            id: run_id.clone(),
            job_id: job_id.to_string(),
            target_id: Some(target_id.to_string()),
            status: overall.into(),
            trigger: trigger.to_string(),
            started_at: String::new(),
            finished_at: None,
            metadata,
        })
        .await
        .map_err(|e| e.to_string())?;
    Ok(run_id)
}

/// Execute a job (P3) — template steps against the target, recording the run +
/// per-step status into the control-plane store. Embedded runs **chained** on the
/// warm `StudioSession`; remote (P3b) dispatches each step to the coordinator.
#[tauri::command]
async fn run_job(
    job_id: String,
    target_id: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    execute_job(&state, &job_id, &target_id, "manual").await
}

/// P6: generate the `quant-fabric` launch commands (coordinator + workers, with
/// Raft HA + mTLS env) for a desired cluster topology. Pure — see `bootstrap`.
#[tauri::command]
fn generate_bootstrap(plan: bootstrap::ClusterPlan) -> Vec<bootstrap::CommandBlock> {
    bootstrap::generate_bootstrap(&plan)
}

/// P6 live node ops: set a coordinator's Raft voter membership (the full desired
/// set). Drives the engine's `/v1/raft/membership` endpoint via the shared lib
/// helper; returns the resulting voter ids.
#[tauri::command]
async fn set_cluster_membership(
    url: String,
    members: Vec<quant_fabric::model::RaftMember>,
) -> Result<quant_fabric::model::RaftMembershipResponse, String> {
    quant_fabric::set_raft_membership(&url, members)
        .await
        .map_err(|e| e.to_string())
}

/// Dry-run a (possibly unsaved) job definition: chained steps over the source,
/// returning the final step's output. Nothing is recorded to history.
#[tauri::command]
async fn dry_run_job(
    source: String,
    steps: Vec<Value>,
    state: State<'_, AppState>,
) -> Result<RunOutput, String> {
    let last = {
        let session = state.studio.lock().map_err(|e| e.to_string())?;
        execute_steps(&session, &source, &steps).1
    };
    Ok(last)
}

/// Describe each step's **output columns** by running the chain on the warm
/// engine (embedded). Returns one column list per step (empty for a step that
/// failed or produced no rows). The Jobs editor uses these so step *i*'s column
/// pickers offer the columns produced by step *i-1* — not just the source's.
#[tauri::command]
async fn describe_steps(
    source: String,
    steps: Vec<Value>,
    state: State<'_, AppState>,
) -> Result<Vec<Vec<String>>, String> {
    let results = {
        let session = state.studio.lock().map_err(|e| e.to_string())?;
        execute_steps(&session, &source, &steps).0
    };
    // Align to the input length: a step that failed early stops the chain, so
    // pad with empties to keep indices stable for the frontend.
    let mut cols: Vec<Vec<String>> = results.into_iter().map(|r| r.columns).collect();
    cols.resize(steps.len(), Vec::new());
    Ok(cols)
}

/// Native file picker. Uses the **non-blocking** callback API (not
/// `blocking_pick_file`): a sync command + a blocking dialog wedges the thread the
/// modal needs, freezing the webview. Here the dialog runs and the result arrives
/// via a oneshot the async command awaits, so the UI stays interactive.
#[tauri::command]
async fn pick_data_file(app: tauri::AppHandle) -> Option<String> {
    use tauri_plugin_dialog::DialogExt;
    let (tx, rx) = tokio::sync::oneshot::channel();
    app.dialog()
        .file()
        .add_filter("data", &["csv", "tsv", "parquet", "arrow", "arrows", "ipc"])
        .pick_file(move |path| {
            let _ = tx.send(path);
        });
    rx.await
        .ok()
        .flatten()
        .and_then(|p| p.into_path().ok())
        .map(|p| p.to_string_lossy().into_owned())
}

/// Inspect a data source's schema (columns + types) + preview, for source-driven
/// column pickers in the job/run editors.
#[tauri::command]
fn inspect_source(
    path: String,
    state: State<'_, AppState>,
) -> Result<studio::SourceSchema, String> {
    let session = state.studio.lock().map_err(|e| e.to_string())?;
    session.describe_source(&path).map_err(|e| e.to_string())
}

/// One node in the Explore data tree: a directory (with `children`) or a single
/// file. ADR-0010: the tree surfaces *all* file-backed state, so every file
/// appears with a `queryable` flag — queryable ones (`format` set) open a SQL
/// cell that reads the absolute `path` directly; the rest are browsable.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct DataTreeNode {
    /// Display label (the file/dir name).
    name: String,
    /// Absolute path on disk.
    path: String,
    /// `true` for directories.
    is_dir: bool,
    /// `true` when the embedded engine can read this file directly (drives the
    /// "add a query cell" affordance). Equivalent to `format.is_some()`.
    queryable: bool,
    /// Query format for queryable files (`parquet` | `csv` | `tsv` | `json`);
    /// absent for directories and non-queryable files.
    format: Option<&'static str>,
    /// File size in bytes (absent for directories).
    size_bytes: Option<u64>,
    /// Child nodes (directories first, then files; each alphabetical). Empty for files.
    children: Vec<DataTreeNode>,
}

/// Query format for a data file by extension, or `None` if it is not a format
/// the embedded engine can read directly. Drives the generated cell SQL.
fn data_file_format(path: &Path) -> Option<&'static str> {
    let ext = path.extension().and_then(|e| e.to_str())?.to_ascii_lowercase();
    match ext.as_str() {
        "parquet" => Some("parquet"),
        "csv" => Some("csv"),
        "tsv" => Some("tsv"),
        "json" | "ndjson" => Some("json"),
        _ => None,
    }
}

/// Recursively collect file-backed state under `dir` (ADR-0010): every file is
/// included with a `queryable` flag, not just data formats. Hidden entries
/// (dotfiles like `.celeritas`) and the opaque `state/` control-plane subtree
/// are skipped; genuinely empty directories are pruned. `depth` bounds the walk.
fn scan_data_dir(dir: &Path, depth: usize) -> Vec<DataTreeNode> {
    if depth == 0 {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut dirs: Vec<DataTreeNode> = Vec::new();
    let mut files: Vec<DataTreeNode> = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        // Dotfiles are scratch; `state/` is the DB (relational truth, not files
        // to browse) — both are excluded from the file-backed view.
        if name.starts_with('.') || name == "state" {
            continue;
        }
        let path = entry.path();
        let Ok(meta) = entry.metadata() else { continue };
        if meta.is_dir() {
            let children = scan_data_dir(&path, depth - 1);
            if children.is_empty() {
                continue;
            }
            dirs.push(DataTreeNode {
                name,
                path: path.to_string_lossy().into_owned(),
                is_dir: true,
                queryable: false,
                format: None,
                size_bytes: None,
                children,
            });
        } else {
            let format = data_file_format(&path);
            files.push(DataTreeNode {
                name,
                path: path.to_string_lossy().into_owned(),
                is_dir: false,
                queryable: format.is_some(),
                format,
                size_bytes: Some(meta.len()),
                children: Vec::new(),
            });
        }
    }
    dirs.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    files.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()));
    dirs.extend(files);
    dirs
}

/// Walk the data roots (default: the whole `home_dir()`, ADR-0010) and return
/// the file-backed-state tree for the Explore sidebar. Each root becomes a
/// top-level node; clicking a queryable file opens a SQL cell that reads it.
#[tauri::command]
fn list_data_tree(roots: Vec<String>) -> Vec<DataTreeNode> {
    // Default: surface the taxonomy tiers (library/ workspace/ sources/ …) as the
    // top level, not a single `$HOME` wrapper node.
    if roots.is_empty() {
        for root in data_dirs() {
            let canonical = root.canonicalize().unwrap_or(root);
            if canonical.is_dir() {
                return scan_data_dir(&canonical, 8);
            }
        }
        return Vec::new();
    }
    // Explicit roots: each becomes its own top-level node.
    let mut out: Vec<DataTreeNode> = Vec::new();
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    for root in roots {
        let root = PathBuf::from(root);
        let canonical = root.canonicalize().unwrap_or(root);
        if !canonical.is_dir() {
            continue;
        }
        let key = canonical.to_string_lossy().into_owned();
        if !seen.insert(key.clone()) {
            continue;
        }
        let children = scan_data_dir(&canonical, 8);
        let name = canonical
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| key.clone());
        out.push(DataTreeNode {
            name,
            path: key,
            is_dir: true,
            queryable: false,
            format: None,
            size_bytes: None,
            children,
        });
    }
    out
}

#[derive(Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
}

/// Persisted-preference key holding the user-configured OpenAI API key (set in
/// the app's Settings page). When present it takes precedence over the
/// `ANTHROPIC_API_KEY` env fallback.
const OPENAI_API_KEY_PREF: &str = "openai.api_key";

// ---- secrets-keeper backend (option B: token in the pref store) ----
/// Base URL of the user's secrets-keeper instance, e.g. `https://127.0.0.1:8200`.
const SECRETSKEEPER_URL_PREF: &str = "secretskeeper.url";
/// Legacy pref key that used to hold the bearer token in plaintext. Retained
/// only for one-time migration into the OS keychain (option A); never written
/// to with a real value any more.
const SECRETSKEEPER_TOKEN_PREF: &str = "secretskeeper.token";
/// Keychain account (under the `quant-fabric` service) holding the bearer token.
const SECRETSKEEPER_TOKEN_ACCOUNT: &str = "secretskeeper.token";
/// Scheme marking a value as a reference to a secrets-keeper KV v2 secret rather
/// than an inline secret: `secrets-keeper://{mount}/{path}#{field}`.
const SECRET_REF_SCHEME: &str = "secrets-keeper://";

/// Parse `secrets-keeper://{mount}/{path}#{field}` into its parts. Returns `None`
/// for any value that is not such a reference (i.e. an ordinary inline secret).
fn parse_secret_ref(value: &str) -> Option<(String, String, String)> {
    let rest = value.trim().strip_prefix(SECRET_REF_SCHEME)?;
    let (location, field) = rest.split_once('#')?;
    let (mount, path) = location.split_once('/')?;
    let (mount, path, field) = (mount.trim(), path.trim(), field.trim());
    if mount.is_empty() || path.is_empty() || field.is_empty() {
        return None;
    }
    Some((mount.to_owned(), path.to_owned(), field.to_owned()))
}

/// Read the secrets-keeper bearer token from the OS keychain (option A custody),
/// migrating it out of the legacy plaintext pref store on first read after
/// upgrade. This is the single spot that knows where the token rests.
async fn secretskeeper_token(state: &AppState) -> Result<Option<String>, String> {
    let from_keychain = async_runtime::spawn_blocking(|| keychain::get(SECRETSKEEPER_TOKEN_ACCOUNT))
        .await
        .map_err(|e| e.to_string())??
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if from_keychain.is_some() {
        return Ok(from_keychain);
    }
    // Legacy: the token used to live in the plaintext pref store. Migrate it into
    // the keychain and clear the plaintext copy so it never lingers on disk.
    let legacy = state
        .controlplane
        .get_setting(SECRETSKEEPER_TOKEN_PREF)
        .await
        .ok()
        .flatten()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    if let Some(token) = legacy {
        let to_store = token.clone();
        let _ = async_runtime::spawn_blocking(move || {
            keychain::set(SECRETSKEEPER_TOKEN_ACCOUNT, &to_store)
        })
        .await;
        let _ = state
            .controlplane
            .set_setting(SECRETSKEEPER_TOKEN_PREF, "")
            .await;
        return Ok(Some(token));
    }
    Ok(None)
}

/// Read the configured secrets backend URL (pref store) + token (OS keychain).
async fn secretskeeper_credentials(state: &AppState) -> Result<(String, String), String> {
    let base = state
        .controlplane
        .get_setting(SECRETSKEEPER_URL_PREF)
        .await
        .ok()
        .flatten()
        .map(|s| s.trim().trim_end_matches('/').to_string())
        .filter(|s| !s.is_empty())
        .ok_or("No secrets backend configured. Set its URL in Settings.")?;
    let token = secretskeeper_token(state)
        .await?
        .ok_or("No secrets backend token configured. Set it in Settings.")?;
    Ok((base, token))
}

/// Store the secrets-keeper bearer token in the OS keychain (option A). An empty
/// value clears it. Any legacy plaintext copy in the pref store is also cleared.
#[tauri::command]
async fn set_secretskeeper_token(token: String, state: State<'_, AppState>) -> Result<(), String> {
    let value = token.trim().to_string();
    async_runtime::spawn_blocking(move || keychain::set(SECRETSKEEPER_TOKEN_ACCOUNT, &value))
        .await
        .map_err(|e| e.to_string())??;
    let _ = state
        .controlplane
        .set_setting(SECRETSKEEPER_TOKEN_PREF, "")
        .await;
    Ok(())
}

/// Whether a secrets-keeper token is configured (keychain, or a legacy pref that
/// gets migrated). The raw token is never returned to the UI — it is write-only.
#[tauri::command]
async fn has_secretskeeper_token(state: State<'_, AppState>) -> Result<bool, String> {
    Ok(secretskeeper_token(&state).await?.is_some())
}

/// Read a single string field from a secrets-keeper KV v2 secret.
async fn secretskeeper_kv_field(
    state: &AppState,
    mount: &str,
    path: &str,
    field: &str,
) -> Result<String, String> {
    let (base, token) = secretskeeper_credentials(state).await?;
    let url = format!("{base}/v1/kv/{mount}/data/{path}");
    let resp = reqwest::Client::new()
        .get(&url)
        .header("authorization", format!("Bearer {token}"))
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !resp.status().is_success() {
        return Err(format!(
            "secrets backend returned {} reading {mount}/{path}",
            resp.status()
        ));
    }
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    v["data"][field]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| format!("secret {mount}/{path} has no string field '{field}'"))
}

/// Resolve a possibly-referenced secret: inline values pass through unchanged,
/// while `secrets-keeper://…` references are fetched from the backend. This is
/// the seam any secret-bearing field (AI key, connector creds) routes through.
async fn resolve_secret_ref(value: &str, state: &AppState) -> Result<String, String> {
    match parse_secret_ref(value) {
        Some((mount, path, field)) => secretskeeper_kv_field(state, &mount, &path, &field).await,
        None => Ok(value.trim().to_string()),
    }
}

/// Resolve every `secrets-keeper://…` reference within a connector param map in
/// place. Inline values are left byte-for-byte untouched (no trimming), so only
/// explicit references hit the backend. Used at connector use-time (test, custom
/// action, job run) so a connection secret never has to live in the instance.
async fn resolve_params_refs(
    params: &mut BTreeMap<String, String>,
    state: &AppState,
) -> Result<(), String> {
    for value in params.values_mut() {
        if parse_secret_ref(value).is_some() {
            *value = resolve_secret_ref(value, state).await?;
        }
    }
    Ok(())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct SecretsKeeperStatus {
    ok: bool,
    initialized: bool,
    sealed: bool,
    token_valid: bool,
    message: String,
}

/// Probe a secrets-keeper instance: reachability + seal status (unauthenticated)
/// and, when a token is available, whether it is accepted (`lookup-self`). An
/// empty `token` falls back to the stored keychain token so the saved
/// configuration can be tested without re-entering the secret.
#[tauri::command]
async fn secretskeeper_test(
    url: String,
    token: String,
    state: State<'_, AppState>,
) -> Result<SecretsKeeperStatus, String> {
    let base = url.trim().trim_end_matches('/').to_string();
    if base.is_empty() {
        return Err("Enter the secrets backend URL.".to_string());
    }
    let token = if token.trim().is_empty() {
        secretskeeper_token(&state).await?.unwrap_or_default()
    } else {
        token
    };
    let client = reqwest::Client::new();
    let seal = client
        .get(format!("{base}/v1/sys/seal-status"))
        .send()
        .await
        .map_err(|e| format!("Backend not reachable: {e}"))?;
    if !seal.status().is_success() {
        return Err(format!(
            "Backend not reachable: {} from /v1/sys/seal-status",
            seal.status()
        ));
    }
    let seal_v: Value = seal.json().await.map_err(|e| e.to_string())?;
    let initialized = seal_v["initialized"].as_bool().unwrap_or(false);
    let sealed = seal_v["sealed"].as_bool().unwrap_or(true);

    let mut token_valid = false;
    let mut message = if sealed {
        "Reachable, but the backend is sealed.".to_string()
    } else {
        "Reachable and unsealed.".to_string()
    };
    let token = token.trim();
    if !token.is_empty() {
        let look = client
            .get(format!("{base}/v1/auth/token/lookup-self"))
            .header("authorization", format!("Bearer {token}"))
            .send()
            .await
            .map_err(|e| e.to_string())?;
        token_valid = look.status().is_success();
        message = format!(
            "{message} {}",
            if token_valid {
                "Token valid."
            } else {
                "Token was rejected."
            }
        );
    }
    Ok(SecretsKeeperStatus {
        ok: !sealed && (token.is_empty() || token_valid),
        initialized,
        sealed,
        token_valid,
        message,
    })
}

#[tauri::command]
async fn ai_chat(
    messages: Vec<ChatMessage>,
    context: String,
    state: State<'_, AppState>,
) -> Result<String, String> {
    let system = format!(
        "You are the QuantFabric template assistant. Help users explain, edit, and \
         create distributable task templates (the [template.plan] / [template.sql] \
         schema with a [template.example] self-test). Be concise and produce valid \
         TOML when asked.\n\nContext:\n{context}"
    );

    // Prefer a user-configured OpenAI key (Settings page); fall back to the
    // ANTHROPIC_API_KEY env var so existing setups keep working.
    let openai_key = state
        .controlplane
        .get_setting(OPENAI_API_KEY_PREF)
        .await
        .ok()
        .flatten()
        .map(|k| k.trim().to_string())
        .filter(|k| !k.is_empty());

    if let Some(key) = openai_key {
        // The stored key may be an inline secret or a `secrets-keeper://…`
        // reference resolved from the secrets backend at use-time.
        let key = resolve_secret_ref(&key, &state).await?;
        return openai_chat(&key, &system, &messages).await;
    }

    let key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
        "Add an OpenAI API key in Settings, or set ANTHROPIC_API_KEY, to enable the AI assistant."
            .to_string()
    })?;
    let body = json!({
        "model": "claude-sonnet-4-6",
        "max_tokens": 1500,
        "system": system,
        "messages": messages.iter().map(|m| json!({"role": m.role, "content": m.content})).collect::<Vec<_>>(),
    });
    let resp = reqwest::Client::new()
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", key)
        .header("anthropic-version", "2023-06-01")
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    Ok(v["content"][0]["text"]
        .as_str()
        .unwrap_or("(no response)")
        .to_string())
}

/// Chat completion via the OpenAI API. The QuantFabric system prompt is sent as
/// a leading `system` message (OpenAI's chat schema), followed by the turn
/// history.
async fn openai_chat(key: &str, system: &str, messages: &[ChatMessage]) -> Result<String, String> {
    let mut payload = vec![json!({"role": "system", "content": system})];
    payload.extend(
        messages
            .iter()
            .map(|m| json!({"role": m.role, "content": m.content})),
    );
    let body = json!({
        "model": "gpt-4o",
        "max_tokens": 1500,
        "messages": payload,
    });
    let resp = reqwest::Client::new()
        .post("https://api.openai.com/v1/chat/completions")
        .header("authorization", format!("Bearer {key}"))
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    let v: Value = resp.json().await.map_err(|e| e.to_string())?;
    if let Some(text) = v["choices"][0]["message"]["content"].as_str() {
        return Ok(text.to_string());
    }
    // Surface the API's error message rather than a silent "(no response)".
    if let Some(err) = v["error"]["message"].as_str() {
        return Err(format!("OpenAI error: {err}"));
    }
    Ok("(no response)".to_string())
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ClusterOverview {
    coordinator_url: String,
    worker_urls: Vec<String>,
    worker_count: usize,
    demos: Vec<DemoCatalogEntry>,
}

async fn coordinator_url(state: &State<'_, AppState>) -> Result<String, String> {
    let guard = state.cluster.lock().await;
    Ok(guard
        .as_ref()
        .ok_or_else(|| "embedded cluster is still starting".to_string())?
        .coordinator_url
        .clone())
}

#[tauri::command]
async fn cluster_overview(state: State<'_, AppState>) -> Result<ClusterOverview, String> {
    let guard = state.cluster.lock().await;
    let cluster = guard
        .as_ref()
        .ok_or_else(|| "embedded cluster is still starting".to_string())?;
    Ok(ClusterOverview {
        coordinator_url: cluster.coordinator_url.clone(),
        worker_urls: cluster.worker_urls.clone(),
        worker_count: cluster.worker_urls.len(),
        demos: list_demo_catalog(cluster.fixtures()),
    })
}

/// Default embedded compute parallelism (worker count) when the user has not
/// pinned one. Conservative so the in-app engine never monopolises the host.
const DEFAULT_EMBEDDED_CORES: usize = 2;

/// Persisted-preference key for the embedded compute parallelism.
const EMBEDDED_CORES_PREF: &str = "embedded.cores";

/// Env override for the embedded compute parallelism — wins over the persisted
/// preference and the default, for headless/power-user control.
const EMBEDDED_CORES_ENV: &str = "QUANTFABRIC_EMBEDDED_WORKERS";

/// Host parallelism ceiling for the embedded engine (logical cores).
fn max_embedded_cores() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

/// Clamp a requested core count into `[1, max]` (with `max` floored at 1).
fn clamp_cores(requested: usize, max: usize) -> usize {
    requested.clamp(1, max.max(1))
}

/// Whether the env override is in effect (pins cores, so the UI control is
/// read-only when set).
fn embedded_cores_env() -> Option<usize> {
    std::env::var(EMBEDDED_CORES_ENV)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}

/// Resolve the embedded worker count to launch: env override wins, else the
/// persisted `embedded.cores` preference, else the default — always clamped to
/// `[1, host cores]`.
async fn resolve_embedded_cores(controlplane: &ControlStore) -> usize {
    let max = max_embedded_cores();
    if let Some(env) = embedded_cores_env() {
        return clamp_cores(env, max);
    }
    let pref = controlplane
        .get_setting(EMBEDDED_CORES_PREF)
        .await
        .ok()
        .flatten()
        .and_then(|v| v.trim().parse::<usize>().ok());
    clamp_cores(pref.unwrap_or(DEFAULT_EMBEDDED_CORES), max)
}

/// Embedded compute configuration for the Clusters view: the configured core
/// count (what the engine launches with), the host ceiling, the currently live
/// worker count, and whether the env override pins it.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct EmbeddedConfig {
    cores: usize,
    max_cores: usize,
    running: usize,
    env_override: bool,
}

/// Report the embedded engine's compute configuration. `running` reflects the
/// live topology; `cores` is the configured value applied on (next) launch.
#[tauri::command]
async fn embedded_config(state: State<'_, AppState>) -> Result<EmbeddedConfig, String> {
    let cores = resolve_embedded_cores(&state.controlplane).await;
    let running = state
        .cluster
        .lock()
        .await
        .as_ref()
        .map(|c| c.worker_urls.len())
        .unwrap_or(0);
    Ok(EmbeddedConfig {
        cores,
        max_cores: max_embedded_cores(),
        running,
        env_override: embedded_cores_env().is_some(),
    })
}

/// Persist a desired embedded core count (clamped to `[1, host cores]`). The
/// env override, when present, pins the value, so this is rejected. The new
/// value applies when the embedded engine next starts (app restart) — kept
/// deliberately simple to avoid tearing down live in-flight compute.
#[tauri::command]
async fn set_embedded_cores(
    cores: usize,
    state: State<'_, AppState>,
) -> Result<EmbeddedConfig, String> {
    if embedded_cores_env().is_some() {
        return Err(format!(
            "embedded cores are pinned by {EMBEDDED_CORES_ENV}; unset it to change here"
        ));
    }
    let clamped = clamp_cores(cores, max_embedded_cores());
    state
        .controlplane
        .set_setting(EMBEDDED_CORES_PREF, &clamped.to_string())
        .await
        .map_err(|e| e.to_string())?;
    embedded_config(state).await
}

#[tauri::command]
async fn cluster_status(state: State<'_, AppState>) -> Result<Value, String> {
    let url = coordinator_url(&state).await?;
    fetch_cluster_status(&url)
        .await
        .map(|status| serde_json::to_value(status).expect("cluster status serializes"))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn cost_summary(state: State<'_, AppState>) -> Result<Value, String> {
    let url = coordinator_url(&state).await?;
    fetch_cost_summary(&url)
        .await
        .map(|summary| serde_json::to_value(summary).expect("cost summary serializes"))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn workers_list(state: State<'_, AppState>) -> Result<Value, String> {
    let url = coordinator_url(&state).await?;
    fetch_workers_list(&url)
        .await
        .map(|workers| serde_json::to_value(workers).expect("workers serializes"))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn locality_registry(state: State<'_, AppState>) -> Result<Value, String> {
    let url = coordinator_url(&state).await?;
    fetch_locality_registry(&url)
        .await
        .map(|registry| serde_json::to_value(registry).expect("locality serializes"))
        .map_err(|error| error.to_string())
}

#[tauri::command]
async fn run_demo(demo_id: String, state: State<'_, AppState>) -> Result<Value, String> {
    let (url, request, catalog_entry) = {
        let guard = state.cluster.lock().await;
        let cluster = guard
            .as_ref()
            .ok_or_else(|| "embedded cluster is still starting".to_string())?;
        let request = demo_requests(cluster.fixtures())
            .remove(&demo_id)
            .ok_or_else(|| format!("unknown demo id: {demo_id}"))?;
        let catalog_entry = list_demo_catalog(cluster.fixtures())
            .into_iter()
            .find(|entry| entry.id == demo_id);
        (cluster.coordinator_url.clone(), request, catalog_entry)
    };
    let started = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or_default();
    let response = execute_demo_query_detailed(&url, &request)
        .await
        .map_err(|error| error.to_string())?;
    Ok(json!({
        "demoId": demo_id,
        "startedMs": started,
        "catalog": catalog_entry,
        "request": request,
        "response": response,
    }))
}

#[tauri::command]
async fn run_all_demos(state: State<'_, AppState>) -> Result<Value, String> {
    let guard = state.cluster.lock().await;
    let cluster = guard
        .as_ref()
        .ok_or_else(|| "embedded cluster is still starting".to_string())?;
    execute_demo_suite(cluster)
        .await
        .map_err(|error| error.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            // Control-plane store: an embedded PGlite under the app data dir,
            // connected via sqlx. The server is leaked for process lifetime (it
            // must outlive every connection); the `ControlStore` (just the pool)
            // is Send+Sync and goes into managed state.
            let data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::env::temp_dir());
            let pg_path = data_dir.join("controlplane").join("pglite");
            std::fs::create_dir_all(&pg_path).ok();
            // Opt-in pinned bind address (`QUANTFABRIC_PG_TCP=host:port`) so the
            // control-plane PGlite is reachable at a stable URL — and, if a
            // non-loopback host is given, by external notebooks across the LAN.
            // Without it, pglite binds an ephemeral loopback TCP port (still a
            // valid external endpoint for same-machine notebooks).
            let pinned_addr = std::env::var("QUANTFABRIC_PG_TCP")
                .ok()
                .and_then(|s| s.trim().parse::<std::net::SocketAddr>().ok());
            let mut pg_builder =
                pglite_oxide::PgliteServer::builder().path(pg_path.to_string_lossy().into_owned());
            if let Some(addr) = pinned_addr {
                pg_builder = pg_builder.tcp(addr);
            }
            let server = pg_builder.start().expect("start control-plane pglite");
            let server: &'static pglite_oxide::PgliteServer = Box::leak(Box::new(server));
            let external_pg = ExternalPg::from_server(server, pinned_addr.is_some());
            let controlplane =
                async_runtime::block_on(ControlStore::connect(&server.database_url()))
                    .expect("connect control-plane store");

            let state = AppState {
                cluster: Mutex::new(None),
                studio: StdMutex::new(StudioSession::new().expect("initialize template engine")),
                template_dirs: template_dirs(),
                controlplane,
                external_pg,
            };
            app.manage(state);

            // P4: fire scheduled jobs on their cron (catch-up across restarts).
            spawn_scheduler(app.handle().clone());

            let handle = app.handle().clone();
            // Opt-in fixed coordinator port (QUANTFABRIC_EMBEDDED_PORT) so the
            // embedded URL is stable across restarts; otherwise ephemeral.
            let embedded_port = std::env::var("QUANTFABRIC_EMBEDDED_PORT")
                .ok()
                .and_then(|s| s.trim().parse::<u16>().ok());
            async_runtime::spawn(async move {
                // Compute parallelism is config-driven (env override → persisted
                // `embedded.cores` pref → default), clamped to host cores.
                let cores = match handle.try_state::<AppState>() {
                    Some(st) => resolve_embedded_cores(&st.controlplane).await,
                    None => DEFAULT_EMBEDDED_CORES,
                };
                match spawn_embedded_cluster_on(cores, embedded_port).await {
                    Ok(cluster) => {
                        let coordinator_url = cluster.coordinator_url.clone();
                        if let Some(app_state) = handle.try_state::<AppState>() {
                            *app_state.cluster.lock().await = Some(cluster);
                        }
                        eprintln!("quant-fabric cluster ready at {coordinator_url}");
                    }
                    Err(error) => {
                        eprintln!("failed to start embedded quant-fabric cluster: {error:#}");
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            cluster_overview,
            embedded_config,
            set_embedded_cores,
            cluster_status,
            cost_summary,
            workers_list,
            locality_registry,
            run_demo,
            run_all_demos,
            list_templates,
            list_quant_functions,
            run_template,
            run_batch,
            run_query,
            external_endpoint_info,
            save_template,
            default_notebook_dir,
            list_notebooks,
            read_notebook,
            save_notebook,
            delete_notebook,
            pick_directory,
            list_connectors,
            list_packs,
            read_pack_files,
            write_pack_file,
            create_pack,
            duplicate_pack,
            export_pack,
            import_pack,
            delete_pack,
            link_github_pack,
            update_pack_source_monitor,
            sync_pack_source,
            list_pack_sources,
            list_pack_versions,
            list_pack_sync_events,
            compare_pack_versions,
            resolve_job_pack_update,
            pick_pack_bundle,
            pick_save_path,
            test_connector,
            run_connector_action,
            pick_data_file,
            list_data_tree,
            ai_chat,
            secretskeeper_test,
            set_secretskeeper_token,
            has_secretskeeper_token,
            cluster_status_at,
            sync_cluster_read_model,
            list_cluster_events,
            list_targets,
            save_target,
            delete_target,
            list_jobs,
            save_job,
            delete_job,
            record_run,
            list_runs,
            list_run_steps,
            list_connector_instances,
            save_connector_instance,
            delete_connector_instance,
            list_bindings,
            list_shared_connector_bindings,
            run_shared_connector_binding,
            save_binding,
            delete_binding,
            validate_binding_conformance,
            list_binding_conformance,
            list_conforming_bindings,
            list_contracts,
            list_staged_relations,
            pin_staged_relation,
            forget_staged_relation,
            gc_staged_relations,
            read_staged_relation,
            run_job,
            dry_run_job,
            describe_steps,
            generate_bootstrap,
            set_cluster_membership,
            get_pref,
            set_pref,
            inspect_source
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

#[cfg(test)]
mod binding_staging_helper_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn binding_schema_to_columns_parses_kinds_and_presence_only() {
        use quant_fabric::contracts::ColumnKind;
        let schema = json!([
            { "name": "quantity", "kind": "number" },
            { "name": "instrument", "kind": "string" },
            { "name": "unknownkind" } // presence-only (no kind)
        ]);
        let cols = binding_schema_to_columns(Some(&schema));
        assert_eq!(cols.len(), 3);
        assert_eq!(cols[0].kind, Some(ColumnKind::Number));
        assert_eq!(cols[1].kind, Some(ColumnKind::String));
        assert_eq!(cols[2].kind, None);
        assert!(binding_schema_to_columns(None).is_empty());
    }

    #[test]
    fn binding_version_is_stable_and_change_sensitive() {
        let mut b = Binding {
            id: "b".into(),
            name: "n".into(),
            instance_id: "i".into(),
            run_params: json!({ "query": "SELECT 1" }),
            contract_ref: Some("Position.v1".into()),
            mapping: None,
            schema: None,
        };
        let v1 = binding_version(&b);
        assert_eq!(v1, binding_version(&b), "stable for an unchanged binding");
        b.run_params = json!({ "query": "SELECT 2" });
        assert_ne!(v1, binding_version(&b), "changes when run_params change");
    }

    #[test]
    fn value_to_string_map_keeps_only_string_values() {
        let m = value_to_string_map(&json!({ "qty": "quantity", "n": 5 }));
        assert_eq!(m.get("qty").unwrap(), "quantity");
        assert!(!m.contains_key("n"), "non-string values are dropped");
    }

    #[test]
    fn file_source_dependencies_track_current_content_hashes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("input.csv");
        std::fs::write(&path, "symbol,qty\nAAA,10\n").unwrap();
        let path_str = path.to_string_lossy().into_owned();
        let mut deps = Vec::new();

        add_non_binding_input_dependencies(&json!({ "source": path_str.clone() }), &mut deps);

        assert_eq!(deps.len(), 1);
        assert_eq!(deps[0].kind, quant_fabric::staging::DependencyKind::Input);
        assert_eq!(deps[0].reference, file_input_reference(&path_str));
        let first = deps[0].version.clone().expect("version");
        assert_eq!(current_input_version(&deps[0].reference), Some(first.clone()));

        std::fs::write(&path, "symbol,qty\nAAA,20\n").unwrap();
        assert_ne!(current_input_version(&deps[0].reference), Some(first));
    }

    #[test]
    fn source_binding_skips_file_dependency_tracking() {
        let mut deps = Vec::new();
        add_non_binding_input_dependencies(
            &json!({ "source": "/tmp/input.csv", "sourceBindingId": "positions" }),
            &mut deps,
        );
        assert!(deps.is_empty());
    }

    #[test]
    fn studio_binding_arg_accepts_explicit_binding_markers() {
        assert_eq!(studio_binding_arg("binding:positions"), Some("positions"));
        assert_eq!(studio_binding_arg(" qf-binding: rates "), Some("rates"));
        assert_eq!(studio_binding_arg("/data/positions.csv"), None);
    }
}

#[cfg(test)]
mod quant_function_catalog_tests {
    use super::*;

    #[test]
    fn infers_template_call_placeholders() {
        let args = infer_function_args_from_sql(
            "bond_convexity",
            "SELECT *, bond_convexity({{face}}, {{coupon_rate}}, {{ytm}}, {{maturity_years}}, {{frequency}}) AS {{output}} FROM input",
        )
        .expect("call found");

        assert_eq!(
            args,
            vec!["face", "coupon_rate", "ytm", "maturity_years", "frequency"]
        );
    }

    #[test]
    fn catalog_backfills_missing_function_args_from_templates() {
        let dir = tempfile::tempdir().unwrap();
        let category = dir.path().join("Factors");
        std::fs::create_dir(&category).unwrap();
        std::fs::write(
            category.join("active_beta.toml"),
            r#"
[[template]]
name = "active_beta"
description = "Compute active beta."

[[template.inputs]]
name = "data"
kind = "table"

[[template.inputs]]
name = "portfolio_beta"
kind = "column"

[[template.inputs]]
name = "benchmark_beta"
kind = "number"

[[template.inputs]]
name = "output"
kind = "column"

[template.sql]
template = "SELECT *, active_beta({{portfolio_beta}}, {{benchmark_beta}}) AS {{output}} FROM input"
"#,
        )
        .unwrap();

        let catalog = quant_function_catalog(&[dir.path().to_path_buf()]);
        let active_beta = catalog
            .iter()
            .find(|function| function.name == "active_beta")
            .expect("active_beta registered");

        assert_eq!(active_beta.args, vec!["portfolio_beta", "benchmark_beta"]);
        assert_eq!(active_beta.category.as_deref(), Some("Factors"));
    }

    #[test]
    fn manifest_metadata_stays_authoritative() {
        let catalog = quant_function_catalog(&[]);
        let accrued_interest = catalog
            .iter()
            .find(|function| function.name == "accrued_interest")
            .expect("accrued_interest registered");

        assert_eq!(
            accrued_interest.args,
            vec![
                "face_value",
                "coupon_rate",
                "days_since_coupon",
                "days_in_coupon_period"
            ]
        );
    }
}

#[cfg(test)]
mod notebook_library_tests {
    use super::*;

    fn notebook_json(name: &str, cells: usize) -> String {
        let cells: Vec<Value> = (0..cells)
            .map(|i| json!({ "id": format!("c{i}"), "kind": "sql", "source": "SELECT 1" }))
            .collect();
        json!({ "id": "nb1", "name": name, "source": "", "params": [], "cells": cells }).to_string()
    }

    #[test]
    fn save_list_read_delete_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let dir_str = dir.path().to_string_lossy().into_owned();

        // Save two notebooks into an explicit (extra) directory.
        let p1 = save_notebook(dir_str.clone(), "alpha".into(), notebook_json("Alpha", 3)).unwrap();
        let p2 = save_notebook(dir_str.clone(), "beta".into(), notebook_json("Beta", 1)).unwrap();
        assert!(std::path::Path::new(&p1).exists());
        assert!(std::path::Path::new(&p2).exists());

        // List the directory: both notebooks surface with parsed name + cell count.
        let listed = list_notebooks(vec![dir_str.clone()]);
        let alpha = listed.iter().find(|n| n.path == p1).expect("alpha listed");
        let beta = listed.iter().find(|n| n.path == p2).expect("beta listed");
        assert_eq!(alpha.name, "Alpha");
        assert_eq!(alpha.cell_count, 3);
        assert_eq!(alpha.dir, dir_str);
        assert_eq!(beta.name, "Beta");
        assert_eq!(beta.cell_count, 1);

        // Read returns the exact bytes we wrote.
        assert_eq!(
            read_notebook(p1.clone()).unwrap(),
            notebook_json("Alpha", 3)
        );

        // Delete removes it; only beta remains.
        delete_notebook(p1.clone()).unwrap();
        assert!(!std::path::Path::new(&p1).exists());
        let after = list_notebooks(vec![dir_str.clone()]);
        assert!(after.iter().all(|n| n.path != p1));
        assert!(after.iter().any(|n| n.path == p2));
    }

    #[test]
    fn unparseable_file_falls_back_to_filename() {
        let dir = tempfile::tempdir().unwrap();
        let dir_str = dir.path().to_string_lossy().into_owned();
        std::fs::write(dir.path().join("broken.json"), "{ not valid json").unwrap();

        let listed = list_notebooks(vec![dir_str.clone()]);
        let entry = listed
            .iter()
            .find(|n| n.name == "broken")
            .expect("falls back to file stem");
        assert_eq!(entry.cell_count, 0);
    }

    #[test]
    fn default_dir_honors_env_override() {
        let dir = tempfile::tempdir().unwrap();
        let dir_str = dir.path().to_string_lossy().into_owned();
        std::env::set_var("QUANT_FABRIC_NOTEBOOKS", &dir_str);
        assert_eq!(default_notebook_dir(), dir_str);
        std::env::remove_var("QUANT_FABRIC_NOTEBOOKS");
    }
}

#[cfg(test)]
mod embedded_cores_tests {
    use super::*;

    #[test]
    fn clamp_keeps_request_within_one_and_max() {
        assert_eq!(clamp_cores(4, 8), 4, "in-range request unchanged");
        assert_eq!(clamp_cores(0, 8), 1, "below floor → 1");
        assert_eq!(clamp_cores(99, 8), 8, "above ceiling → max");
        assert_eq!(clamp_cores(4, 1), 1, "single-core host caps at 1");
        assert_eq!(clamp_cores(4, 0), 1, "degenerate max floored to 1");
    }

    #[test]
    fn host_ceiling_is_at_least_one() {
        assert!(max_embedded_cores() >= 1);
    }
}

#[cfg(test)]
mod scheduler_tests {
    use super::*;

    fn job(id: &str, schedule: Option<&str>, enabled: bool, target: Option<&str>) -> Job {
        let mut def = json!({ "source": "x.csv", "steps": [] });
        if let Some(s) = schedule {
            def["schedule"] = json!(s);
        }
        if let Some(t) = target {
            def["target"] = json!(t);
        }
        Job {
            id: id.to_string(),
            name: id.to_string(),
            description: String::new(),
            definition: def,
            enabled,
        }
    }

    #[test]
    fn cron_due_fires_when_an_instant_falls_in_the_window() {
        // "* * * * *" — every minute. Next after 1970-01-01T00:00:00 (excl) is 00:01:00 (=60).
        assert!(
            cron_due("* * * * *", 0, 60),
            "minute boundary at 60 should fire"
        );
        assert!(!cron_due("* * * * *", 0, 30), "no minute instant in (0,30]");
        // "0 0 * * *" — daily midnight. Next after epoch is 1970-01-02 (=86400).
        assert!(
            !cron_due("0 0 * * *", 0, 30),
            "daily midnight not due 30s in"
        );
        assert!(
            cron_due("0 0 * * *", 0, 86_400),
            "next midnight is due at 86400"
        );
    }

    #[test]
    fn catch_up_fires_a_daily_job_after_a_closed_window() {
        // App was closed across a midnight: watermark `prev` is ~2 days stale, so
        // the daily job has an instant in (prev, now] and fires once on next tick.
        // This is the whole point — no waiting until the schedule is "met" live.
        let jobs = vec![job("nightly", Some("0 0 * * *"), true, None)];
        let prev = 0; // 1970-01-01T00:00:00
        let now = 2 * 86_400 + 100; // ~2 days later
        assert_eq!(
            due_jobs(&jobs, prev, now),
            vec![("nightly".to_string(), "embedded".to_string())]
        );
        // Once caught up (watermark advanced to now), it won't re-fire until the
        // next genuine midnight.
        assert!(due_jobs(&jobs, now, now + 60).is_empty());
    }

    /// End-to-end proof that the *running* scheduler fires a real job — driven by
    /// a single tick at a simulated clock, so it completes in seconds instead of
    /// waiting for a cron interval to elapse. Exercises the whole chain: persisted
    /// watermark → due-check → `execute_job` (embedded, warm engine) → recorded run.
    #[tokio::test]
    async fn scheduler_tick_fires_a_real_job_at_a_simulated_clock() {
        let dir = tempfile::tempdir().unwrap();

        // Real control-plane store on a temp PGlite (same path the app uses).
        let server = pglite_oxide::PgliteServer::builder()
            .path(dir.path().join("pg").to_string_lossy().into_owned())
            .start()
            .expect("start pglite");
        let server: &'static pglite_oxide::PgliteServer = Box::leak(Box::new(server));
        let controlplane = ControlStore::connect(&server.database_url())
            .await
            .expect("connect store");

        // Register templates; write a CSV the job will actually read.
        let tdirs = template_dirs();
        let _ = studio::catalog(&tdirs);
        let csv = dir.path().join("pairs.csv");
        std::fs::write(&csv, "sector,x,y\na,1,2\na,2,4\na,3,6\n").unwrap();

        // Embedded target + a per-minute job.
        controlplane
            .upsert_target(&Target {
                id: "embedded".into(),
                name: "Embedded".into(),
                kind: "embedded".into(),
                url: "in-process".into(),
                token: None,
            })
            .await
            .unwrap();
        controlplane
            .upsert_job(&Job {
                id: "nightly".into(),
                name: "nightly".into(),
                description: String::new(),
                definition: json!({
                    "source": csv.to_string_lossy(),
                    "schedule": "* * * * *",
                    "steps": [{
                        "template": "pair_correlation",
                        "args": { "group_by": "sector", "x": "x", "y": "y" }
                    }]
                }),
                enabled: true,
            })
            .await
            .unwrap();

        // Watermark at epoch 0 → a per-minute instant lies in (0, 60].
        controlplane
            .set_setting(SCHED_WATERMARK_KEY, "0")
            .await
            .unwrap();

        let state = AppState {
            cluster: Mutex::new(None),
            studio: StdMutex::new(StudioSession::new().unwrap()),
            template_dirs: tdirs,
            controlplane,
            external_pg: ExternalPg::placeholder("test"),
        };

        // One simulated tick at t=60s — fires the due job. No real time elapses.
        let fired = scheduler_tick(&state, 60).await;
        assert_eq!(fired, 1, "the due per-minute job should fire on this tick");

        // The run was recorded as a completed, schedule-triggered run.
        let runs = state.controlplane.list_runs(Some("nightly")).await.unwrap();
        let done = runs
            .iter()
            .find(|r| r.status == "complete")
            .expect("a completed run should be recorded");
        assert_eq!(done.trigger, "schedule");

        // Watermark advanced to 60 → a tick at t=70 finds no new minute instant.
        let again = scheduler_tick(&state, 70).await;
        assert_eq!(
            again, 0,
            "watermark advanced; no double-fire within the same minute"
        );
    }

    /// P4b: the headless `SchedulerService`, built from a store URL (not a
    /// hand-assembled `AppState`), fires a due job through the same path the GUI
    /// uses — proven with a simulated clock, no waiting.
    #[tokio::test]
    async fn headless_service_fires_a_due_job_from_a_store_url() {
        let dir = tempfile::tempdir().unwrap();
        let server = pglite_oxide::PgliteServer::builder()
            .path(dir.path().join("pg").to_string_lossy().into_owned())
            .start()
            .expect("start pglite");
        let server: &'static pglite_oxide::PgliteServer = Box::leak(Box::new(server));

        // Construct the service via its public, URL-based constructor.
        let svc = SchedulerService::connect(&server.database_url(), template_dirs(), 30)
            .await
            .expect("connect headless scheduler");

        // Seed through the service's own store (single PGlite backend).
        let csv = dir.path().join("pairs.csv");
        std::fs::write(&csv, "sector,x,y\na,1,2\na,2,4\na,3,6\n").unwrap();
        svc.store()
            .upsert_target(&Target {
                id: "embedded".into(),
                name: "Embedded".into(),
                kind: "embedded".into(),
                url: "in-process".into(),
                token: None,
            })
            .await
            .unwrap();
        svc.store()
            .upsert_job(&Job {
                id: "nightly".into(),
                name: "nightly".into(),
                description: String::new(),
                definition: json!({
                    "source": csv.to_string_lossy(),
                    "schedule": "* * * * *",
                    "steps": [{
                        "template": "pair_correlation",
                        "args": { "group_by": "sector", "x": "x", "y": "y" }
                    }]
                }),
                enabled: true,
            })
            .await
            .unwrap();
        svc.store()
            .set_setting(SCHED_WATERMARK_KEY, "0")
            .await
            .unwrap();

        // One simulated tick — fires the job; assert a recorded scheduled run.
        assert_eq!(
            svc.tick_once(60).await,
            1,
            "headless service should fire the due job"
        );
        let runs = svc.store().list_runs(Some("nightly")).await.unwrap();
        assert!(
            runs.iter()
                .any(|r| r.status == "complete" && r.trigger == "schedule"),
            "expected a completed schedule-triggered run, got {runs:?}"
        );
    }

    /// P4c leader election: against one shared store, only the lease holder fires
    /// (no double-fire), and a standby takes over once the holder's lease expires.
    /// Driven with a simulated clock — no waiting on real lease TTLs.
    #[tokio::test]
    async fn lease_election_prevents_double_fire_and_fails_over() {
        let dir = tempfile::tempdir().unwrap();
        let server = pglite_oxide::PgliteServer::builder()
            .path(dir.path().join("pg").to_string_lossy().into_owned())
            .start()
            .expect("start pglite");
        let server: &'static pglite_oxide::PgliteServer = Box::leak(Box::new(server));
        let controlplane = ControlStore::connect(&server.database_url()).await.unwrap();

        let tdirs = template_dirs();
        let _ = studio::catalog(&tdirs);
        let csv = dir.path().join("pairs.csv");
        std::fs::write(&csv, "sector,x,y\na,1,2\na,2,4\na,3,6\n").unwrap();
        controlplane
            .upsert_target(&Target {
                id: "embedded".into(),
                name: "Embedded".into(),
                kind: "embedded".into(),
                url: "in-process".into(),
                token: None,
            })
            .await
            .unwrap();
        controlplane
            .upsert_job(&Job {
                id: "nightly".into(),
                name: "nightly".into(),
                description: String::new(),
                definition: json!({
                    "source": csv.to_string_lossy(),
                    "schedule": "* * * * *",
                    "steps": [{ "template": "pair_correlation", "args": { "group_by": "sector", "x": "x", "y": "y" } }]
                }),
                enabled: true,
            })
            .await
            .unwrap();
        controlplane
            .set_setting(SCHED_WATERMARK_KEY, "0")
            .await
            .unwrap();

        let state = AppState {
            cluster: Mutex::new(None),
            studio: StdMutex::new(StudioSession::new().unwrap()),
            template_dirs: tdirs,
            controlplane,
            external_pg: ExternalPg::placeholder("test"),
        };

        let ttl = 90;
        // A becomes leader at t=60 and fires the due minute job.
        let (a_leader, a_fired) = scheduler_tick_leased(&state, 60, "A", ttl).await;
        assert!(a_leader, "A should win the vacant lease");
        assert_eq!(a_fired, 1);

        // B at t=61: A's lease is still valid (expires 150) → B stands by, fires 0.
        let (b_leader, b_fired) = scheduler_tick_leased(&state, 61, "B", ttl).await;
        assert!(!b_leader, "B must not lead while A's lease is valid");
        assert_eq!(b_fired, 0, "standby must not fire");

        // A "dies" (stops renewing). After A's lease expires (>150), B takes over
        // at t=200 and fires the instant(s) due since the watermark.
        let (b2_leader, b2_fired) = scheduler_tick_leased(&state, 200, "B", ttl).await;
        assert!(b2_leader, "B should take over after A's lease expired");
        assert!(
            b2_fired >= 1,
            "new leader fires the due job, got {b2_fired}"
        );
    }

    #[test]
    fn cron_due_rejects_unparseable_expressions() {
        assert!(!cron_due("not a cron", 0, 1_000_000));
        assert!(!cron_due("", 0, 1_000_000));
    }

    #[test]
    fn due_jobs_selects_enabled_scheduled_jobs_with_target_default() {
        let jobs = vec![
            job("a", Some("* * * * *"), true, None), // due, default target
            job("b", Some("* * * * *"), false, None), // disabled → skip
            job("c", Some("0 0 * * *"), true, None), // not due in this window
            job("d", None, true, None),              // no schedule → skip
            job("e", Some("* * * * *"), true, Some("remote-1")), // due, explicit target
        ];
        let due = due_jobs(&jobs, 0, 60);
        assert_eq!(
            due,
            vec![
                ("a".to_string(), "embedded".to_string()),
                ("e".to_string(), "remote-1".to_string()),
            ]
        );
    }
}

#[cfg(test)]
mod chain_tests {
    use super::*;

    /// Execution proof that `execute_steps` reports each step's real output
    /// columns — what the Jobs editor needs so step *i*'s pickers offer the
    /// columns produced by step *i-1*, not just the source's.
    #[test]
    fn execute_steps_reports_each_steps_output_columns() {
        let dir = tempfile::tempdir().unwrap();
        let _ = studio::catalog(&template_dirs());
        let csv = dir.path().join("p.csv");
        std::fs::write(
            &csv,
            "sector,x,y\na,1,2\na,2,4\na,3,6\nb,1,1\nb,2,3\nb,3,5\n",
        )
        .unwrap();

        let session = StudioSession::new().unwrap();
        let steps = vec![
            json!({ "template": "pair_correlation", "args": { "group_by": "sector", "x": "x", "y": "y" } }),
            json!({ "template": "group_mean", "args": { "group_by": "sector", "value": "correlation" } }),
        ];
        let (results, _last) = execute_steps(&session, &csv.to_string_lossy(), &steps);

        assert_eq!(results.len(), 2);
        assert_eq!(
            results[0].status, "complete",
            "step0 failed: {:?}",
            results[0].log
        );
        // Step 0's output columns become step 1's available pickers.
        assert!(
            results[0].columns.contains(&"correlation".to_string()),
            "step0 cols: {:?}",
            results[0].columns
        );
        assert!(
            results[0].columns.contains(&"sector".to_string()),
            "step0 cols: {:?}",
            results[0].columns
        );
        assert_eq!(
            results[1].status, "complete",
            "step1 failed: {:?}",
            results[1].log
        );
        assert!(
            results[1].columns.contains(&"mean_value".to_string()),
            "step1 cols: {:?}",
            results[1].columns
        );
    }
}

#[cfg(test)]
mod binding_conformance_tests {
    use super::*;

    #[test]
    fn relation_columns_accept_source_schema_shape() {
        let schema = json!({
            "columns": [
                { "name": "instrument", "dataType": "varchar" },
                { "name": "quantity", "dataType": "integer" },
                { "name": "as_of", "dataType": "timestamp" }
            ],
            "preview": []
        });

        let columns = relation_columns_from_schema(&schema).unwrap();

        assert_eq!(columns.len(), 3);
        assert_eq!(columns[0].name, "instrument");
        assert_eq!(columns[0].kind, Some(ColumnKind::String));
        assert_eq!(columns[1].kind, Some(ColumnKind::Integer));
        assert_eq!(columns[2].kind, Some(ColumnKind::Date));
    }

    #[test]
    fn relation_columns_accept_array_shape_and_unknown_types() {
        let schema = json!([
            { "name": "instrument", "kind": "string" },
            { "name": "quantity", "type": "unknown" }
        ]);

        let columns = relation_columns_from_schema(&schema).unwrap();

        assert_eq!(columns.len(), 2);
        assert_eq!(columns[0].kind, Some(ColumnKind::String));
        assert_eq!(columns[1].kind, None);
    }

    #[test]
    fn binding_mapping_rejects_non_string_values() {
        let err = binding_mapping(Some(&json!({ "quantity": 42 }))).unwrap_err();

        assert!(err.contains("quantity"));
    }

    #[test]
    fn binding_conformance_helper_returns_exclusion_reasons() {
        let registry = ContractRegistry::load_default().unwrap();
        let reference = ContractRef::parse("Position.v1").unwrap();
        let binding = |id: &str, schema: Option<Value>| Binding {
            id: id.into(),
            name: id.into(),
            instance_id: "inst".into(),
            run_params: json!({ "query": "SELECT * FROM positions" }),
            contract_ref: None,
            mapping: None,
            schema,
        };

        let conforming = binding_conformance_for(
            &reference,
            &registry,
            binding(
                "ok",
                Some(json!({
                    "columns": [
                        { "name": "instrument", "kind": "string" },
                        { "name": "quantity", "kind": "number" }
                    ],
                    "preview": []
                })),
            ),
        )
        .unwrap();
        assert!(conforming.conformance.conforms);

        let missing = binding_conformance_for(
            &reference,
            &registry,
            binding(
                "missing",
                Some(json!({
                    "columns": [{ "name": "instrument", "kind": "string" }],
                    "preview": []
                })),
            ),
        )
        .unwrap();
        assert!(!missing.conformance.conforms);
        assert_eq!(missing.conformance.missing, vec!["quantity"]);

        let no_schema = binding_conformance_for(&reference, &registry, binding("raw", None)).unwrap();
        assert!(!no_schema.conformance.conforms);
        assert_eq!(no_schema.conformance.missing, vec!["cached schema"]);
    }
}

#[cfg(test)]
mod read_model_tests {
    use super::*;

    /// ADR-0021 P2b end-to-end: a real embedded cluster commits consensus
    /// commands (worker registrations), and the read-model sync projects them
    /// into a real PGlite store — idempotently (re-sync applies nothing new),
    /// with the watermark advancing. This is the decentralized control-plane
    /// mechanism: same log → same projection on every suite instance.
    #[tokio::test]
    async fn read_model_sync_projects_consensus_log_idempotently() {
        let dir = tempfile::tempdir().unwrap();
        let server = pglite_oxide::PgliteServer::builder()
            .path(dir.path().join("pg").to_string_lossy().into_owned())
            .start()
            .expect("start pglite");
        let server: &'static pglite_oxide::PgliteServer = Box::leak(Box::new(server));
        let controlplane = ControlStore::connect(&server.database_url())
            .await
            .expect("connect store");

        // Real cluster: 2 workers self-register → ≥2 committed RegisterWorker
        // entries in the coordinator's metadata log.
        let cluster = quant_fabric::spawn_embedded_cluster(2)
            .await
            .expect("spawn embedded cluster");
        let url = cluster.coordinator_url.clone();

        let state = AppState {
            cluster: Mutex::new(None),
            studio: StdMutex::new(StudioSession::new().unwrap()),
            template_dirs: template_dirs(),
            controlplane,
            external_pg: ExternalPg::placeholder("test"),
        };

        let applied = sync_cluster_read_model_inner(&state, &url)
            .await
            .expect("sync read model");
        assert!(applied >= 2, "expected ≥2 projected entries, got {applied}");

        let events = state
            .controlplane
            .list_cluster_events(&url, 100)
            .await
            .expect("list events");
        assert_eq!(events.len() as u64, applied);
        assert!(
            events
                .iter()
                .any(|e| e["kind"].as_str() == Some("RegisterWorker")),
            "expected a RegisterWorker projection, got {events:?}"
        );
        let watermark = state
            .controlplane
            .cluster_read_model_watermark(&url)
            .await
            .unwrap();
        assert!(watermark >= 2, "watermark should advance, got {watermark}");

        // Idempotent: replaying the same committed log applies nothing new.
        let again = sync_cluster_read_model_inner(&state, &url)
            .await
            .expect("re-sync");
        assert_eq!(again, 0, "re-sync must be a no-op, applied {again}");
    }
}
