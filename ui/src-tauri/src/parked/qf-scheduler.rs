//! Headless QuantFabric job scheduler (ADR-0021 P4b).
//!
//! The always-on, GUI-free counterpart of the desktop app's in-process
//! scheduler: it connects to a control-plane store and fires jobs on their cron
//! using the SAME `scheduler_tick` / `execute_job` the GUI uses. Run it as a
//! service (systemd / launchd) against a **central Postgres** store so scheduled
//! jobs fire even when no desktop app is open.
//!
//! Config (env):
//!   QUANTFABRIC_CONTROLPLANE_URL   required — Postgres URL of the central store
//!   QUANT_FABRIC_TEMPLATES         optional — `:`-joined template dirs
//!   QUANTFABRIC_SCHEDULER_TICK_SECS optional — loop cadence (default 30)
//!   QUANTFABRIC_JOB_API_BIND       optional — opt-in REST API bind, e.g. 127.0.0.1:7700

#[path = "../connectors_job.rs"]
#[allow(dead_code)]
mod connectors_job;
#[path = "../controlplane.rs"]
#[allow(dead_code)]
mod controlplane;
#[path = "../github_packs.rs"]
#[allow(dead_code)]
mod github_packs;
#[path = "../scheduler_core.rs"]
#[allow(dead_code)]
mod scheduler_core;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{delete, get, post, put};
use axum::{Json, Router};
use controlplane::Job;
use scheduler_core::{execute_job, template_dirs, SchedulerService, SchedulerState};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let store_url = std::env::var("QUANTFABRIC_CONTROLPLANE_URL").map_err(|_| {
        anyhow::anyhow!(
            "set QUANTFABRIC_CONTROLPLANE_URL to the control-plane Postgres URL \
             (e.g. postgres://user:pass@host/quantfabric)"
        )
    })?;
    let tick_secs = std::env::var("QUANTFABRIC_SCHEDULER_TICK_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(30);

    eprintln!("qf-scheduler: connecting to control-plane store…");
    let service = SchedulerService::connect(&store_url, template_dirs(), tick_secs).await?;
    if let Some(bind) = std::env::var("QUANTFABRIC_JOB_API_BIND")
        .ok()
        .filter(|s| !s.trim().is_empty())
    {
        let addr = bind.parse::<SocketAddr>()?;
        let state = service.state();
        tokio::spawn(async move {
            if let Err(e) = serve_job_api(addr, state).await {
                eprintln!("qf-scheduler: job API failed: {e}");
            }
        });
        eprintln!("qf-scheduler: job REST API listening on http://{addr}");
    }
    eprintln!("qf-scheduler: running (tick {tick_secs}s) — Ctrl-C to stop");
    service.run().await;
    Ok(())
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunJobRequest {
    target_id: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RunJobResponse {
    run_id: String,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RunsQuery {
    job_id: Option<String>,
    status: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ErrorBody {
    error: String,
}

struct ApiError {
    status: StatusCode,
    message: String,
}

impl ApiError {
    fn internal(error: impl std::fmt::Display) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: error.to_string(),
        }
    }

    fn not_found(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (
            self.status,
            Json(ErrorBody {
                error: self.message,
            }),
        )
            .into_response()
    }
}

async fn serve_job_api(addr: SocketAddr, state: Arc<SchedulerState>) -> anyhow::Result<()> {
    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/openapi.json", get(openapi))
        .route("/docs", get(docs))
        .route("/v1/jobs", get(list_jobs).post(create_job))
        .route(
            "/v1/jobs/:job_id",
            get(get_job).put(update_job).delete(delete_job),
        )
        .route("/v1/jobs/:job_id/run", post(run_job))
        .route("/v1/runs", get(list_runs))
        .route("/v1/runs/:run_id/steps", get(list_run_steps))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

async fn healthz() -> Json<Value> {
    Json(json!({ "status": "ok", "service": "quantfabric-job-api" }))
}

async fn list_jobs(State(state): State<Arc<SchedulerState>>) -> Result<Json<Vec<Job>>, ApiError> {
    state
        .controlplane
        .list_jobs()
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn create_job(
    State(state): State<Arc<SchedulerState>>,
    Json(job): Json<Job>,
) -> Result<(StatusCode, Json<Job>), ApiError> {
    state
        .controlplane
        .upsert_job(&job)
        .await
        .map_err(ApiError::internal)?;
    Ok((StatusCode::CREATED, Json(job)))
}

async fn get_job(
    State(state): State<Arc<SchedulerState>>,
    Path(job_id): Path<String>,
) -> Result<Json<Job>, ApiError> {
    let job = state
        .controlplane
        .get_job(&job_id)
        .await
        .map_err(ApiError::internal)?
        .ok_or_else(|| ApiError::not_found(format!("unknown job {job_id}")))?;
    Ok(Json(job))
}

async fn update_job(
    State(state): State<Arc<SchedulerState>>,
    Path(job_id): Path<String>,
    Json(mut job): Json<Job>,
) -> Result<Json<Job>, ApiError> {
    job.id = job_id;
    state
        .controlplane
        .upsert_job(&job)
        .await
        .map_err(ApiError::internal)?;
    Ok(Json(job))
}

async fn delete_job(
    State(state): State<Arc<SchedulerState>>,
    Path(job_id): Path<String>,
) -> Result<StatusCode, ApiError> {
    state
        .controlplane
        .delete_job(&job_id)
        .await
        .map_err(ApiError::internal)?;
    Ok(StatusCode::NO_CONTENT)
}

async fn run_job(
    State(state): State<Arc<SchedulerState>>,
    Path(job_id): Path<String>,
    Json(request): Json<RunJobRequest>,
) -> Result<Json<RunJobResponse>, ApiError> {
    let run_id = execute_job(&state, &job_id, &request.target_id, "api")
        .await
        .map_err(|message| ApiError {
            status: StatusCode::BAD_REQUEST,
            message,
        })?;
    Ok(Json(RunJobResponse { run_id }))
}

async fn list_runs(
    State(state): State<Arc<SchedulerState>>,
    Query(query): Query<RunsQuery>,
) -> Result<Json<Vec<controlplane::JobRun>>, ApiError> {
    let mut runs = state
        .controlplane
        .list_runs(query.job_id.as_deref())
        .await
        .map_err(ApiError::internal)?;
    if let Some(status) = query.status {
        runs.retain(|run| run.status == status);
    }
    Ok(Json(runs))
}

async fn list_run_steps(
    State(state): State<Arc<SchedulerState>>,
    Path(run_id): Path<String>,
) -> Result<Json<Vec<controlplane::JobRunStep>>, ApiError> {
    state
        .controlplane
        .list_run_steps(&run_id)
        .await
        .map(Json)
        .map_err(ApiError::internal)
}

async fn openapi() -> Json<Value> {
    Json(openapi_document())
}

async fn docs() -> Html<&'static str> {
    Html(JOB_API_DOCS_HTML)
}

fn openapi_document() -> Value {
    json!({
      "openapi": "3.0.3",
      "info": {
        "title": "QuantFabric Job API",
        "version": "0.1.0",
        "description": "Opt-in REST API exposed by qf-scheduler on a port separate from the coordinator."
      },
      "paths": {
        "/healthz": { "get": { "summary": "Health check", "responses": { "200": { "description": "API is reachable" } } } },
        "/v1/jobs": {
          "get": { "summary": "List jobs", "responses": { "200": { "description": "Jobs" } } },
          "post": {
            "summary": "Create or replace a job",
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Job" } } } },
            "responses": { "201": { "description": "Created job" } }
          }
        },
        "/v1/jobs/{jobId}": {
          "get": {
            "summary": "Get one job",
            "parameters": [{ "name": "jobId", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Job" }, "404": { "description": "Not found" } }
          },
          "put": {
            "summary": "Update one job",
            "parameters": [{ "name": "jobId", "in": "path", "required": true, "schema": { "type": "string" } }],
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Job" } } } },
            "responses": { "200": { "description": "Updated job" } }
          },
          "delete": {
            "summary": "Delete one job",
            "parameters": [{ "name": "jobId", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "204": { "description": "Deleted" } }
          }
        },
        "/v1/jobs/{jobId}/run": {
          "post": {
            "summary": "Trigger a job run",
            "parameters": [{ "name": "jobId", "in": "path", "required": true, "schema": { "type": "string" } }],
            "requestBody": { "required": true, "content": { "application/json": { "schema": { "$ref": "#/components/schemas/RunJobRequest" } } } },
            "responses": { "200": { "description": "Run id" } }
          }
        },
        "/v1/runs": {
          "get": {
            "summary": "List runs",
            "parameters": [
              { "name": "jobId", "in": "query", "required": false, "schema": { "type": "string" } },
              { "name": "status", "in": "query", "required": false, "schema": { "type": "string", "enum": ["queued", "running", "complete", "failed"] } }
            ],
            "responses": { "200": { "description": "Runs" } }
          }
        },
        "/v1/runs/{runId}/steps": {
          "get": {
            "summary": "List run steps",
            "parameters": [{ "name": "runId", "in": "path", "required": true, "schema": { "type": "string" } }],
            "responses": { "200": { "description": "Run steps" } }
          }
        }
      },
      "components": {
        "schemas": {
          "Job": {
            "type": "object",
            "required": ["id", "name", "definition", "enabled"],
            "properties": {
              "id": { "type": "string", "example": "job-nightly-risk" },
              "name": { "type": "string", "example": "Nightly risk rollup" },
              "description": { "type": "string" },
              "enabled": { "type": "boolean" },
              "definition": { "$ref": "#/components/schemas/JobDefinition" }
            }
          },
          "JobDefinition": {
            "type": "object",
            "required": ["source", "steps"],
            "properties": {
              "source": { "type": "string", "example": "/data/returns.csv" },
              "steps": {
                "type": "array",
                "items": { "$ref": "#/components/schemas/JobStep" }
              },
              "schedule": { "type": "string", "example": "0 2 * * *" },
              "sink": { "type": "string", "example": "s3://bucket/qf-intermediates" }
            }
          },
          "JobStep": {
            "type": "object",
            "required": ["template", "args"],
            "properties": {
              "template": { "type": "string", "example": "mean_return" },
              "args": { "type": "object", "additionalProperties": true }
            }
          },
          "RunJobRequest": {
            "type": "object",
            "required": ["targetId"],
            "properties": { "targetId": { "type": "string", "example": "embedded" } }
          }
        }
      }
    })
}

const JOB_API_DOCS_HTML: &str = r#"<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>QuantFabric Job API</title>
  <style>
    :root { color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #0e1116; color: #d7dde8; }
    body { margin: 0; }
    main { max-width: 1100px; margin: 0 auto; padding: 28px; }
    h1 { margin: 0 0 4px; font-size: 22px; }
    p { margin: 0 0 20px; color: #8d98a8; }
    .op { border: 1px solid #293241; background: #151a22; border-radius: 8px; margin: 10px 0; overflow: hidden; }
    .head { display: flex; align-items: center; gap: 10px; padding: 11px 13px; border-bottom: 1px solid #293241; }
    .method { min-width: 64px; text-align: center; border-radius: 5px; padding: 3px 7px; font: 700 12px ui-monospace, SFMono-Regular, Menlo, monospace; }
    .get { background: #123f63; color: #94d5ff; }
    .post { background: #174c35; color: #9cf2bc; }
    .put { background: #4b3d16; color: #ffe08a; }
    .delete { background: #572025; color: #ff9da7; }
    code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: #f4f7fb; }
    .body { padding: 11px 13px; color: #9ba6b5; font-size: 13px; }
    a { color: #88c7ff; }
  </style>
</head>
<body>
<main>
  <h1>QuantFabric Job API</h1>
  <p>OpenAPI: <a href="/openapi.json">/openapi.json</a></p>
  <section class="op"><div class="head"><span class="method get">GET</span><code>/healthz</code></div><div class="body">Check the API process.</div></section>
  <section class="op"><div class="head"><span class="method get">GET</span><code>/v1/jobs</code></div><div class="body">List saved jobs.</div></section>
  <section class="op"><div class="head"><span class="method post">POST</span><code>/v1/jobs</code></div><div class="body">Create or replace a job using the Job JSON schema.</div></section>
  <section class="op"><div class="head"><span class="method get">GET</span><code>/v1/jobs/{jobId}</code></div><div class="body">Read one job.</div></section>
  <section class="op"><div class="head"><span class="method put">PUT</span><code>/v1/jobs/{jobId}</code></div><div class="body">Update one job. The path id is authoritative.</div></section>
  <section class="op"><div class="head"><span class="method delete">DELETE</span><code>/v1/jobs/{jobId}</code></div><div class="body">Delete one job and its run history.</div></section>
  <section class="op"><div class="head"><span class="method post">POST</span><code>/v1/jobs/{jobId}/run</code></div><div class="body">Trigger a run with <code>{"targetId":"embedded"}</code>. Runs are recorded with trigger <code>api</code>.</div></section>
  <section class="op"><div class="head"><span class="method get">GET</span><code>/v1/runs?jobId=&amp;status=running</code></div><div class="body">List recent runs, optionally filtered by job id and status.</div></section>
  <section class="op"><div class="head"><span class="method get">GET</span><code>/v1/runs/{runId}/steps</code></div><div class="body">List recorded step results for a run.</div></section>
</main>
</body>
</html>"#;
