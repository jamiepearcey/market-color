use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;

use quant_fabric::studio::{self, StudioSession};
use serde_json::{json, Value};

use crate::controlplane::{ControlStore, Job, JobRun, JobRunStep, Target};

pub struct SchedulerState {
    pub studio: StdMutex<StudioSession>,
    pub controlplane: ControlStore,
}

/// Resolve where the template library lives: `QUANT_FABRIC_TEMPLATES`
/// (`:`-joined) or the in-repo default next to this app.
pub fn template_dirs() -> Vec<PathBuf> {
    if let Ok(val) = std::env::var("QUANT_FABRIC_TEMPLATES") {
        return val
            .split(':')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
    }
    vec![PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../quant-fabric/templates"
    ))]
}

struct StepResult {
    idx: i32,
    template: String,
    args: Value,
    status: String,
    row_count: i32,
    duration_ms: Option<u64>,
    log: Option<String>,
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
        ("columnCount".to_string(), json!(0)),
        ("outputColumns".to_string(), json!([])),
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

fn execute_steps(
    session: &StudioSession,
    source: &str,
    steps: &[Value],
) -> (Vec<StepResult>, Vec<quant_fabric::model::Row>) {
    // Delegate to the shared single Flow runner (ADR-0030 P2), the same path the
    // desktop app uses — so the headless daemon and the GUI never drift. The
    // final step's rows are returned for sink-connector export.
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
            }
        })
        .collect();
    (results, run.output.rows)
}

async fn execute_steps_remote(
    target: &Target,
    source: &str,
    steps: &[Value],
    sink: Option<&str>,
    run_id: &str,
) -> Vec<StepResult> {
    let mut results: Vec<StepResult> = Vec::new();
    let mut current = source.to_string();
    let n = steps.len();
    for (i, step) in steps.iter().enumerate() {
        let (template, args_json, args) = build_step_args(step, &current);
        let step_started = std::time::Instant::now();
        let want_uri = match (sink, i + 1 < n) {
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
            log: out.error,
        });
        if failed {
            break;
        }
    }
    results
}

fn run_metadata(definition: &Value) -> Value {
    definition
        .get("packBinding")
        .cloned()
        .map(|binding| json!({ "packBinding": binding }))
        .unwrap_or_else(|| json!({}))
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

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

const SCHED_WATERMARK_KEY: &str = "scheduler:watermark";
const SCHED_LEASE_TTL_SECS: i64 = 90;

fn instance_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("qf-{}-{nanos}", std::process::id())
}

pub async fn scheduler_tick_leased(
    state: &SchedulerState,
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

pub async fn scheduler_tick(state: &SchedulerState, now: i64) -> usize {
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
    if let Err(e) = state
        .controlplane
        .set_setting(SCHED_WATERMARK_KEY, &now.to_string())
        .await
    {
        eprintln!("[scheduler] persist watermark failed: {e}");
    }
    fired
}

pub struct SchedulerService {
    state: Arc<SchedulerState>,
    tick: std::time::Duration,
}

impl SchedulerService {
    pub async fn connect(
        store_url: &str,
        template_dirs: Vec<PathBuf>,
        tick_secs: u64,
    ) -> anyhow::Result<Self> {
        let controlplane = ControlStore::connect(store_url).await?;
        let _ = studio::catalog(&template_dirs);
        let state = SchedulerState {
            studio: StdMutex::new(StudioSession::new()?),
            controlplane,
        };
        Ok(Self {
            state: Arc::new(state),
            tick: std::time::Duration::from_secs(tick_secs.max(1)),
        })
    }

    pub async fn tick_once(&self, now: i64) -> usize {
        scheduler_tick(&self.state, now).await
    }

    pub fn store(&self) -> &ControlStore {
        &self.state.controlplane
    }

    pub fn state(&self) -> Arc<SchedulerState> {
        Arc::clone(&self.state)
    }

    pub async fn run(&self) {
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

pub async fn execute_job(
    state: &SchedulerState,
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
        let root = crate::github_packs::cache_root().map_err(|e| e.to_string())?;
        crate::github_packs::apply_tracked_update(&state.controlplane, &mut job, &root)
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
    let metadata = run_metadata(&job.definition);
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

    // Configured connector instances, so a job that binds a connector by id
    // (the editor/pack form) resolves to its driver + params at run time.
    let connector_instances = crate::connectors_job::parse_instances(
        state
            .controlplane
            .get_setting(crate::connectors_job::CONNECTOR_INSTANCES_KEY)
            .await
            .ok()
            .flatten(),
    );

    // A plain-path source passes through; a connector binding runs the real
    // celeritas import (ADR-0028) into a temp CSV the step chain reads.
    let source = match crate::connectors_job::resolve_job_source(&job.definition, &connector_instances) {
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
    let sink = job
        .definition
        .get("sink")
        .and_then(|v| v.as_str())
        .map(str::trim)
        .filter(|s| !s.is_empty());

    let (results, final_rows) = if target.kind == "embedded" {
        let session = state.studio.lock().map_err(|e| e.to_string())?;
        execute_steps(&session, &source, &steps)
    } else {
        (
            execute_steps_remote(&target, &source, &steps, sink, &run_id).await,
            Vec::new(),
        )
    };
    let mut overall = if results.iter().any(|r| r.status == "failed") {
        "failed"
    } else {
        "complete"
    };

    // Export the final step output to the target connector when one is bound
    // (embedded only — remote step output lives at a URI, not in-process rows).
    if overall == "complete" && !final_rows.is_empty() {
        match crate::connectors_job::export_job_sink(&job.definition, &connector_instances, &final_rows) {
            Ok(Some(driver)) => {
                eprintln!(
                    "[scheduler] job {job_id} exported {} rows via `{driver}`",
                    final_rows.len()
                )
            }
            Ok(None) => {}
            Err(e) => {
                eprintln!("[scheduler] job {job_id} export failed: {e}");
                overall = "failed";
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
