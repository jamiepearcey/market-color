//! Feed-scraping job: a durable workflow over idempotent activities that scrapes
//! every feed in `../config/feeds*.json` and writes date-partitioned Parquet in
//! the exact shape the eventgraph pipeline consumes.
//!
//! This is a Rust port of `../crawl_corpus.py` (feed discovery + gnews decode +
//! HTML->text extraction + content-hash/canonical-URL dedup + date-partitioned
//! merge-write). It is structured as ACTIVITIES (`activities.rs`) orchestrated by
//! a WORKFLOW (`workflow.rs`); `temporal.rs` documents the Temporal drop-in.
//!
//! The standalone rayon runner is the shipped orchestrator -- no async runtime,
//! no Temporal server. See `temporal.rs` for how each activity/workflow maps onto
//! a Temporal worker when durable orchestration is wanted.

pub mod activities;
pub mod model;
pub mod temporal;
pub mod urls;
pub mod workflow;

pub use model::Row;
pub use workflow::{ScrapeConfig, ScrapeReport};

use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

/// A feed spec from feeds.json / feeds_em.json. Optional fields (`region`,
/// `lang`) cover the EM pack; everything else matches both files.
#[derive(Debug, Clone, Deserialize)]
pub struct FeedSpec {
    pub name: String,
    pub url: String,
    #[serde(default = "default_scope")]
    pub scope: String,
    #[serde(default)]
    pub tier: i64,
    #[serde(default = "default_access")]
    pub access: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub desks: Vec<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub lang: Option<String>,
}

fn default_scope() -> String {
    "unknown".into()
}
fn default_access() -> String {
    "open".into()
}
fn default_method() -> String {
    "rss".into()
}

#[derive(Deserialize)]
struct FeedsFile {
    feeds: Vec<FeedSpec>,
}

/// Load one feeds file (accepts `{"feeds":[...]}` or a bare array).
pub fn load_feeds_file(path: &Path) -> Result<Vec<FeedSpec>> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read feeds {path:?}"))?;
    // Try the wrapped form first, then a bare array.
    if let Ok(f) = serde_json::from_str::<FeedsFile>(&text) {
        return Ok(f.feeds);
    }
    let arr: Vec<FeedSpec> =
        serde_json::from_str(&text).with_context(|| format!("parse feeds {path:?}"))?;
    Ok(arr)
}

/// Load + concatenate multiple feeds files.
pub fn load_all_feeds(paths: &[std::path::PathBuf]) -> Result<Vec<FeedSpec>> {
    let mut out = Vec::new();
    for p in paths {
        out.extend(load_feeds_file(p)?);
    }
    Ok(out)
}

/// One discovered feed entry (RSS/Atom item), pre-resolution.
#[derive(Debug, Clone)]
pub struct FeedItem {
    pub link: String,
    pub published: Option<chrono::DateTime<chrono::Utc>>,
    /// RSS summary/content, used as the extraction fallback when the page body
    /// comes back thin (mirrors the Python graceful degradation).
    pub summary: Option<String>,
    pub title: Option<String>,
}

/// A fully-scraped article (Python `Article`); converts 1:1 to `model::Row`.
#[derive(Debug, Clone)]
pub struct Article {
    pub row: Row,
}
