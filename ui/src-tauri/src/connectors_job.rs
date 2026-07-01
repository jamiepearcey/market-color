//! Connector import/export wiring for job execution (ADR-0028 phase 3).
//!
//! A job `definition` (free-form JSON) may bind a celeritas connector as its
//! source and/or its sink:
//!
//! ```json
//! { "source": { "connector": { "driver": "filesystem.csv",
//!                              "params": {…}, "jobParams": {…} } },
//!   "steps": [ … ],
//!   "sinkConnector": { "driver": "postgres", "params": {…}, "jobParams": {…} } }
//! ```
//!
//! Import resolves the source connector to a local CSV the step chain reads
//! (reusing the existing "source is a file path" contract); export streams the
//! final step's rows to the target connector. Both drive the real celeritas
//! extension binaries over the shared-memory data plane (`quant_fabric::
//! celeritas::run_source_rows` / `run_sink_rows`). A plain-string `source` and a
//! missing `sinkConnector` are pass-through no-ops, so existing jobs are
//! unaffected.

use std::collections::BTreeMap;
use std::io::Write;

use quant_fabric::model::Row;
use serde_json::Value;

/// A connector binding parsed from a job definition.
pub struct ConnectorSpec {
    pub driver: String,
    pub params: BTreeMap<String, String>,
    pub job_params: BTreeMap<String, String>,
}

/// Control-plane preference key the app persists configured connector instances
/// under (a JSON array; see the desktop store's `CONNECTORS_KEY`).
pub const CONNECTOR_INSTANCES_KEY: &str = "connectors:instances";

/// A configured connector instance, as persisted in the `connectors:instances`
/// preference blob. Only the fields the run path needs are kept.
pub struct ConnectorInstance {
    pub id: String,
    pub driver: String,
    /// Connection-level params filled when the instance was configured.
    pub params: BTreeMap<String, String>,
    /// Per-job defaults copied when the connector is attached to a job.
    pub job_param_defaults: BTreeMap<String, String>,
}

/// Build a run-path [`ConnectorInstance`] from already-typed parts — e.g. a
/// control-plane `connector_instances` row (ADR-0032). `params` /
/// `job_param_defaults` are flattened to the string maps `celeritas::run_*`
/// expects.
pub fn instance_from_parts(
    id: impl Into<String>,
    driver: impl Into<String>,
    params: Option<&Value>,
    job_param_defaults: Option<&Value>,
) -> ConnectorInstance {
    ConnectorInstance {
        id: id.into(),
        driver: driver.into(),
        params: string_map(params),
        job_param_defaults: string_map(job_param_defaults),
    }
}

/// A first-class binding (ADR-0032) resolved for the run path: which instance it
/// pulls from, and its per-run params (the "what you take"). Built from a
/// control-plane `bindings` row via [`binding_ref_from_parts`].
pub struct BindingRef {
    pub id: String,
    pub instance_id: String,
    pub run_params: BTreeMap<String, String>,
}

/// Build a [`BindingRef`] from already-typed parts (a control-plane `bindings`
/// row). `run_params` is flattened to a string map.
pub fn binding_ref_from_parts(
    id: impl Into<String>,
    instance_id: impl Into<String>,
    run_params: Option<&Value>,
) -> BindingRef {
    BindingRef {
        id: id.into(),
        instance_id: instance_id.into(),
        run_params: string_map(run_params),
    }
}

/// Parse the `connectors:instances` preference blob into instances. Tolerant: a
/// missing/invalid blob yields an empty list (no connector resolution, prior
/// behaviour).
pub fn parse_instances(blob: Option<String>) -> Vec<ConnectorInstance> {
    let Some(blob) = blob else {
        return Vec::new();
    };
    let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(&blob) else {
        return Vec::new();
    };
    arr.into_iter()
        .filter_map(|v| {
            let id = v.get("id").and_then(Value::as_str)?.to_string();
            Some(ConnectorInstance {
                id,
                driver: v
                    .get("driver")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                params: string_map(v.get("params")),
                job_param_defaults: string_map(v.get("jobParamDefaults")),
            })
        })
        .collect()
}

/// Resolve a connector binding for one side of a job. Precedence (highest
/// first):
///
/// 1. **Inline spec** (`inline_key` → `{driver, params, jobParams}`) — the data
///    plane's original encoding; always wins for back-compat.
/// 2. **First-class binding** (`binding_id_key` → `<bindingId>`, ADR-0032) —
///    resolves the named binding to its instance ⊕ the binding's `run_params`,
///    overridden by any job-level values under `params_key`. A binding id that
///    is present but does not resolve is an authoritative miss (no silent
///    fall-through to the instance form).
/// 3. **Instance reference** (`id_key` → `<instanceId>`, `params_key` → per-job
///    params) — the editor/pack form bridged via the configured instances
///    (ADR-0028/0031).
fn resolve_binding(
    definition: &Value,
    inline_key: &str,
    binding_id_key: &str,
    id_key: &str,
    params_key: &str,
    instances: &[ConnectorInstance],
    bindings: &[BindingRef],
) -> Option<ConnectorSpec> {
    // 1. Inline spec (back-compat: `source.connector` / `sinkConnector`).
    if let Some(spec) = definition.get(inline_key).and_then(parse_connector_spec) {
        return Some(spec);
    }
    // 2. First-class binding reference (`sourceBindingId` / `targetBindingId`).
    if let Some(binding_id) = definition.get(binding_id_key).and_then(Value::as_str) {
        let binding_id = binding_id.trim();
        if !binding_id.is_empty() {
            let spec = bindings
                .iter()
                .find(|b| b.id == binding_id)
                .and_then(|binding| {
                    let inst = instances.iter().find(|i| i.id == binding.instance_id)?;
                    if inst.driver.trim().is_empty() {
                        return None;
                    }
                    // Per-run params: the binding's run_params, overridden by any
                    // job-level values under `params_key`.
                    let mut job_params = binding.run_params.clone();
                    job_params.extend(string_map(definition.get(params_key)));
                    Some(ConnectorSpec {
                        driver: inst.driver.clone(),
                        params: inst.params.clone(),
                        job_params,
                    })
                });
            // A binding id was specified: it is authoritative whether or not it
            // resolved — do not fall through to the instance-id form.
            return spec;
        }
    }
    // 3. Instance reference (`sourceConnectorId` + `sourceConnectorParams`, etc.).
    let id = definition.get(id_key).and_then(Value::as_str)?.trim();
    if id.is_empty() {
        return None;
    }
    let inst = instances.iter().find(|i| i.id == id)?;
    if inst.driver.trim().is_empty() {
        return None;
    }
    // Per-run params: the instance's defaults, overridden by the job's values.
    let mut job_params = inst.job_param_defaults.clone();
    job_params.extend(string_map(definition.get(params_key)));
    Some(ConnectorSpec {
        driver: inst.driver.clone(),
        params: inst.params.clone(),
        job_params,
    })
}

/// Flatten a JSON object of scalar values into a `String`→`String` map (the
/// shape `celeritas::run_*` expects). Non-string scalars are stringified.
fn string_map(v: Option<&Value>) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    if let Some(Value::Object(o)) = v {
        for (k, val) in o {
            let s = val
                .as_str()
                .map(String::from)
                .unwrap_or_else(|| val.to_string());
            m.insert(k.clone(), s);
        }
    }
    m
}

/// Parse a connector binding: either `{ "connector": { "driver": … } }` or the
/// bare `{ "driver": …, "params": …, "jobParams": … }` object. Returns `None`
/// for a plain string or any value without a non-empty `driver`.
pub fn parse_connector_spec(v: &Value) -> Option<ConnectorSpec> {
    let obj = v.get("connector").unwrap_or(v);
    let driver = obj
        .get("driver")
        .and_then(Value::as_str)?
        .trim()
        .to_string();
    if driver.is_empty() {
        return None;
    }
    Some(ConnectorSpec {
        driver,
        params: string_map(obj.get("params")),
        job_params: string_map(obj.get("jobParams")),
    })
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

fn cell(v: Option<&Value>) -> String {
    match v {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(s)) => s.clone(),
        Some(other) => other.to_string(),
    }
}

/// Materialise rows to a fresh temp CSV and return its path. Columns come from
/// the first row's keys (the step chain reads this back as the job input table).
/// Materialize connector-imported rows as an immutable, **content-addressed CSV**
/// in the staging tier (ADR-0033 33-D1) — the job's source artifact. Same
/// immutability / dedup / addressability as the flow step intermediates
/// (`flow::stage_intermediate_csv`): identical imports collapse to one artifact,
/// the bytes are immutable, and they live under the staging root rather than
/// `/tmp`. CSV because the first step reads it back through DuckDB.
fn stage_connector_source_csv(rows: &[Row]) -> Result<String, String> {
    use std::io::Write as _;
    let cols: Vec<String> = rows
        .first()
        .and_then(|r| serde_json::to_value(r).ok())
        .and_then(|v| v.as_object().map(|o| o.keys().cloned().collect()))
        .unwrap_or_default();
    let mut buf = cols
        .iter()
        .map(|c| csv_escape(c))
        .collect::<Vec<_>>()
        .join(",");
    buf.push('\n');
    for r in rows {
        let v = serde_json::to_value(r).map_err(|e| e.to_string())?;
        let obj = v.as_object();
        buf.push_str(
            &cols
                .iter()
                .map(|c| csv_escape(&cell(obj.and_then(|o| o.get(c)))))
                .collect::<Vec<_>>()
                .join(","),
        );
        buf.push('\n');
    }
    let bytes = buf.into_bytes();
    let hash = quant_fabric::staging::content_hash(&bytes);
    let root = std::env::var("QUANT_FABRIC_STAGING_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir().join("quant-fabric-staging"));
    let path = root.join(quant_fabric::staging::content_address_key(&hash, "csv"));
    if !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_file_name(format!(".{hash}.{}.tmp", std::process::id()));
        let mut f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
        f.write_all(&bytes).map_err(|e| e.to_string())?;
        f.sync_all().map_err(|e| e.to_string())?;
        std::fs::rename(&tmp, &path).map_err(|e| e.to_string())?;
    }
    Ok(path.display().to_string())
}

/// Resolve a job's `source` to a path the step chain can read. When `source` is
/// a connector binding, run the real import and materialise it to a temp CSV;
/// when it is a plain string, return it unchanged; otherwise an empty string
/// (matching the prior `source.as_str().unwrap_or("")` behaviour).
pub fn resolve_job_source(
    definition: &Value,
    instances: &[ConnectorInstance],
    bindings: &[BindingRef],
) -> Result<String, String> {
    // A non-empty string `source` is a file path — pass it through unchanged.
    if let Some(s) = definition.get("source").and_then(Value::as_str) {
        if !s.is_empty() {
            return Ok(s.to_string());
        }
    }
    // A connector source — inline spec, first-class binding, or instance ref.
    match resolve_binding(
        definition,
        "source",
        "sourceBindingId",
        "sourceConnectorId",
        "sourceConnectorParams",
        instances,
        bindings,
    ) {
        Some(spec) => {
            let rows = quant_fabric::celeritas::run_source_rows(
                &spec.driver,
                &spec.params,
                &spec.job_params,
                &[],
                0,
            )
            .map_err(|e| format!("connector import via `{}`: {e}", spec.driver))?;
            stage_connector_source_csv(&rows)
        }
        // No source binding: empty (matches the prior absent/empty behaviour).
        None => Ok(String::new()),
    }
}

/// Export the final step's rows to the job's sink connector, if configured —
/// accepting either the inline `sinkConnector` spec or the editor/pack
/// `targetConnectorId` instance reference. Returns the driver name on success,
/// `None` when no sink connector is bound.
pub fn export_job_sink(
    definition: &Value,
    instances: &[ConnectorInstance],
    bindings: &[BindingRef],
    rows: &[Row],
) -> Result<Option<String>, String> {
    let Some(spec) = resolve_binding(
        definition,
        "sinkConnector",
        "targetBindingId",
        "targetConnectorId",
        "targetConnectorParams",
        instances,
        bindings,
    ) else {
        return Ok(None);
    };
    quant_fabric::celeritas::run_sink_rows(&spec.driver, &spec.params, &spec.job_params, &[], rows)
        .map_err(|e| format!("connector export via `{}`: {e}", spec.driver))?;
    Ok(Some(spec.driver))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_wrapped_and_bare_specs() {
        let wrapped = json!({
            "connector": { "driver": "filesystem.csv",
                           "params": { "root_path": "/data" },
                           "jobParams": { "pattern": "*.csv" } }
        });
        let spec = parse_connector_spec(&wrapped).expect("wrapped spec");
        assert_eq!(spec.driver, "filesystem.csv");
        assert_eq!(spec.params.get("root_path").unwrap(), "/data");
        assert_eq!(spec.job_params.get("pattern").unwrap(), "*.csv");

        let bare = json!({ "driver": "postgres", "params": { "port": 5432 } });
        let spec = parse_connector_spec(&bare).expect("bare spec");
        assert_eq!(spec.driver, "postgres");
        // Non-string scalars are stringified for the config layer.
        assert_eq!(spec.params.get("port").unwrap(), "5432");
    }

    #[test]
    fn plain_string_and_missing_driver_are_not_connectors() {
        assert!(parse_connector_spec(&json!("/data/in.csv")).is_none());
        assert!(parse_connector_spec(&json!({ "params": {} })).is_none());
        assert!(parse_connector_spec(&json!({ "driver": "  " })).is_none());
    }

    #[test]
    fn resolve_passes_through_string_source_without_touching_a_binary() {
        let def = json!({ "source": "/data/in.csv", "steps": [] });
        assert_eq!(resolve_job_source(&def, &[], &[]).unwrap(), "/data/in.csv");
        // Absent source resolves to empty (prior behaviour), no connector run.
        assert_eq!(resolve_job_source(&json!({}), &[], &[]).unwrap(), "");
    }

    #[test]
    fn export_is_noop_without_sink_connector() {
        let def = json!({ "steps": [] });
        assert_eq!(export_job_sink(&def, &[], &[], &[]).unwrap(), None);
    }

    #[test]
    fn resolve_binding_resolves_a_first_class_binding_reference() {
        // A first-class binding (ADR-0032): the job references it by id; the
        // binding names an instance and carries the per-run params.
        let instances = parse_instances(Some(
            json!([{ "id": "pg-1", "driver": "postgres",
                     "params": { "host": "db" },
                     "jobParamDefaults": {} }])
            .to_string(),
        ));
        let bindings = vec![binding_ref_from_parts(
            "positions",
            "pg-1",
            Some(&json!({ "query": "SELECT * FROM positions" })),
        )];
        let def = json!({ "sourceBindingId": "positions", "steps": [] });
        let spec = resolve_binding(
            &def,
            "source",
            "sourceBindingId",
            "sourceConnectorId",
            "sourceConnectorParams",
            &instances,
            &bindings,
        )
        .expect("binding resolves to a spec");
        assert_eq!(spec.driver, "postgres");
        assert_eq!(spec.params.get("host").unwrap(), "db");
        assert_eq!(
            spec.job_params.get("query").unwrap(),
            "SELECT * FROM positions"
        );

        // A job-level override beats the binding's stored run_params.
        let def = json!({
            "sourceBindingId": "positions",
            "sourceConnectorParams": { "query": "SELECT 1" }
        });
        let spec = resolve_binding(
            &def,
            "source",
            "sourceBindingId",
            "sourceConnectorId",
            "sourceConnectorParams",
            &instances,
            &bindings,
        )
        .unwrap();
        assert_eq!(spec.job_params.get("query").unwrap(), "SELECT 1");

        // A binding id that doesn't resolve is an authoritative miss — it does
        // NOT silently fall through to a `sourceConnectorId` instance form.
        let def = json!({ "sourceBindingId": "nope", "sourceConnectorId": "pg-1" });
        assert!(resolve_binding(
            &def,
            "source",
            "sourceBindingId",
            "sourceConnectorId",
            "sourceConnectorParams",
            &instances,
            &bindings,
        )
        .is_none());
    }

    /// End-to-end: a job that binds a source connector BY ID (the editor/pack
    /// form) actually runs the real celeritas `filesystem.csv` import and
    /// materialises the rows to the temp CSV the step chain reads. Skips when the
    /// extension binary isn't resolvable (no `CELERITAS_BIN_DIR`/`PATH`), like
    /// the engine's data-plane tests.
    #[test]
    fn end_to_end_import_resolves_instance_ref_and_imports() {
        use std::io::Write;
        let resolvable = quant_fabric::celeritas::celeritas_binary("source", "filesystem.csv")
            .and_then(|b| quant_fabric::celeritas::resolve_binary(&b))
            .is_some();
        if !resolvable {
            eprintln!("skip: source-filesystem-csv-ipc not on CELERITAS_BIN_DIR/PATH");
            return;
        }
        let dir = std::env::temp_dir().join(format!("qf-bridge-e2e-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let csv = dir.join("trades.csv");
        let mut f = std::fs::File::create(&csv).unwrap();
        writeln!(f, "symbol,qty").unwrap();
        writeln!(f, "AAA,10").unwrap();
        writeln!(f, "BBB,20").unwrap();
        drop(f);

        // Configured instance + a job referencing it by id (no inline driver).
        let instances = parse_instances(Some(
            json!([{
                "id": "fs-src",
                "driver": "filesystem.csv",
                "params": { "root_path": dir.to_string_lossy(), "has_header": "true" },
                "jobParamDefaults": { "pattern": "*.csv" }
            }])
            .to_string(),
        ));
        let def = json!({
            "sourceConnectorId": "fs-src",
            "sourceConnectorParams": { "pattern": "trades.csv" },
            "steps": []
        });

        let out = resolve_job_source(&def, &instances, &[]).expect("import resolves");
        assert!(!out.is_empty(), "expected a materialised temp CSV path");
        let body = std::fs::read_to_string(&out).unwrap_or_default();
        assert!(
            body.contains("AAA") && body.contains("BBB"),
            "imported rows should reach the step chain's CSV, got:\n{body}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn parse_instances_reads_the_pref_blob() {
        let blob = json!([
            { "id": "src-1", "driver": "postgres",
              "params": { "host": "db", "port": 5432 },
              "jobParamDefaults": { "query": "SELECT 1" } },
            { "id": "no-driver" }
        ])
        .to_string();
        let got = parse_instances(Some(blob));
        assert_eq!(got.len(), 2);
        let pg = got.iter().find(|i| i.id == "src-1").unwrap();
        assert_eq!(pg.driver, "postgres");
        assert_eq!(pg.params.get("host").unwrap(), "db");
        assert_eq!(pg.params.get("port").unwrap(), "5432"); // stringified
        assert_eq!(pg.job_param_defaults.get("query").unwrap(), "SELECT 1");
    }

    #[test]
    fn resolve_binding_bridges_an_instance_reference() {
        // The editor/pack form: a `sourceConnectorId` + per-job params, with the
        // driver/connection params living on the configured instance.
        let instances = parse_instances(Some(
            json!([{ "id": "src-1", "driver": "filesystem.csv",
                     "params": { "root_path": "/data" },
                     "jobParamDefaults": { "pattern": "*.csv" } }])
            .to_string(),
        ));
        let def = json!({
            "sourceConnectorId": "src-1",
            "sourceConnectorParams": { "pattern": "trades-*.csv" },
            "steps": []
        });
        let spec = resolve_binding(
            &def,
            "source",
            "sourceBindingId",
            "sourceConnectorId",
            "sourceConnectorParams",
            &instances,
            &[],
        )
        .expect("instance reference resolves to a spec");
        assert_eq!(spec.driver, "filesystem.csv");
        assert_eq!(spec.params.get("root_path").unwrap(), "/data");
        // job override beats the instance default
        assert_eq!(spec.job_params.get("pattern").unwrap(), "trades-*.csv");

        // Unknown id (no matching instance) → no binding.
        let missing = json!({ "sourceConnectorId": "nope" });
        assert!(resolve_binding(
            &missing,
            "source",
            "sourceBindingId",
            "sourceConnectorId",
            "sourceConnectorParams",
            &instances,
            &[],
        )
        .is_none());
        // The inline form still wins when present.
        let inline = json!({ "source": { "connector": { "driver": "s3.parquet" } },
                             "sourceConnectorId": "src-1" });
        let spec = resolve_binding(
            &inline,
            "source",
            "sourceBindingId",
            "sourceConnectorId",
            "sourceConnectorParams",
            &instances,
            &[],
        )
        .unwrap();
        assert_eq!(spec.driver, "s3.parquet");
    }
}
