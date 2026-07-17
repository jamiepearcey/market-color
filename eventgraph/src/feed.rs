//! Feeds: a source of raw documents. Default reader is raw-doc JSONL; a
//! Bloomberg/news Parquet feed (arrow/parquet) is a drop-in follow-on.

use crate::util::fnv1a;
use anyhow::{Context, Result};
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct RawDoc {
    #[serde(default)]
    pub doc_id: String,
    #[serde(default = "unknown")]
    pub source: String,
    #[serde(default)]
    pub url: Option<String>,
    #[serde(default, alias = "title")]
    pub headline: String,
    #[serde(default, alias = "published_utc", alias = "date")]
    pub published_at: Option<String>,
    #[serde(default, alias = "body_text", alias = "text")]
    pub article: String,
}

fn unknown() -> String {
    "unknown".into()
}

impl RawDoc {
    /// Ensure a stable doc_id (hash of url|headline) if the feed didn't supply one.
    pub fn with_id(mut self) -> Self {
        if self.doc_id.is_empty() {
            let basis = self.url.clone().unwrap_or_else(|| self.headline.clone());
            self.doc_id = format!("eg_{}", fnv1a(&basis));
        }
        self
    }
}

pub trait Feed {
    fn load(&self) -> Result<Vec<RawDoc>>;
}

/// One JSON object per line. Recognizes market-color / bloomberg field aliases.
pub struct JsonlFeed {
    pub path: std::path::PathBuf,
    pub limit: Option<usize>,
}

impl Feed for JsonlFeed {
    fn load(&self) -> Result<Vec<RawDoc>> {
        let text = std::fs::read_to_string(&self.path)
            .with_context(|| format!("read feed {:?}", self.path))?;
        let mut out = Vec::new();
        for (i, line) in text.lines().enumerate() {
            if line.trim().is_empty() {
                continue;
            }
            let d: RawDoc = serde_json::from_str(line)
                .with_context(|| format!("parse feed line {}", i + 1))?;
            out.push(d.with_id());
            if self.limit.is_some_and(|l| out.len() >= l) {
                break;
            }
        }
        Ok(out)
    }
}

impl JsonlFeed {
    pub fn new(path: impl AsRef<Path>, limit: Option<usize>) -> Self {
        Self { path: path.as_ref().to_path_buf(), limit }
    }
}

/// Parquet feed drop-in: reads the date-partitioned news corpus written by
/// `eventgraph scrape` (columns doc_id/source_name/title/body_text/... — see
/// `scrape::model`) back into `RawDoc` via the aliases above. Accepts either a
/// single `part-000.parquet` file or a corpus root containing `dt=*/part-*.parquet`.
pub struct ParquetFeed {
    pub path: std::path::PathBuf,
    pub limit: Option<usize>,
}

impl ParquetFeed {
    pub fn new(path: impl AsRef<Path>, limit: Option<usize>) -> Self {
        Self { path: path.as_ref().to_path_buf(), limit }
    }

    fn part_files(&self) -> Vec<std::path::PathBuf> {
        if self.path.is_file() {
            return vec![self.path.clone()];
        }
        let mut parts = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.path) {
            let mut dirs: Vec<_> = entries
                .flatten()
                .map(|e| e.path())
                .filter(|p| {
                    p.is_dir()
                        && p.file_name()
                            .and_then(|n| n.to_str())
                            .is_some_and(|n| n.starts_with("dt="))
                })
                .collect();
            dirs.sort();
            for d in dirs {
                let part = d.join("part-000.parquet");
                if part.exists() {
                    parts.push(part);
                }
            }
        }
        parts
    }
}

impl Feed for ParquetFeed {
    fn load(&self) -> Result<Vec<RawDoc>> {
        let mut out = Vec::new();
        for part in self.part_files() {
            let rows = crate::scrape::model::read_parquet(&part)
                .with_context(|| format!("read parquet feed {part:?}"))?;
            for r in rows {
                let doc = RawDoc {
                    doc_id: r.doc_id,
                    source: r.source_name,
                    url: Some(r.url),
                    headline: r.title,
                    published_at: r.published_utc,
                    article: r.body_text.unwrap_or_default(),
                }
                .with_id();
                out.push(doc);
                if self.limit.is_some_and(|l| out.len() >= l) {
                    return Ok(out);
                }
            }
        }
        Ok(out)
    }
}
