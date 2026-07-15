//! Load the rich per-fact export (facts.npy + facts.jsonl) produced by
//! scripts/export_facts.py. One row per atomic fact, carrying every free label
//! that already sits in facts.parquet. Shared substrate for the atlas / impact /
//! latency / source / consensus / regime receptors.

use anyhow::{Context, Result};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Fact {
    pub fact_id: String,
    pub doc_id: String,
    pub published_epoch: i64,
    #[serde(default)]
    pub date: String,
    #[serde(default)]
    pub claim: String,
    #[serde(default = "other")]
    pub predicate: String,
    #[serde(default)]
    pub direction: f32, // +1 up / -1 down / 0 else
    #[serde(default)]
    pub confidence: f32,
    #[serde(default)]
    pub magnitude: Option<f32>,
    #[serde(default)]
    pub source_name: String,
    #[serde(default = "other")]
    pub desk: String,
    #[serde(default)]
    pub desks: Vec<String>,
    #[serde(default)]
    pub entities: Vec<String>,
    #[serde(default)]
    pub cause_entities: Vec<String>,
}

fn other() -> String {
    "other".into()
}

pub struct FactSet {
    /// (N, D) L2-normalised claim embeddings; row i <-> facts[i].
    pub emb: Array2<f32>,
    pub facts: Vec<Fact>,
    #[allow(dead_code)]
    pub id2idx: HashMap<String, usize>,
}

impl FactSet {
    pub fn load(dir: &Path) -> Result<Self> {
        let emb: Array2<f32> = read_npy(dir.join("facts.npy")).context(
            "read facts.npy (run: uv run scripts/export_facts.py)",
        )?;
        let text =
            std::fs::read_to_string(dir.join("facts.jsonl")).context("read facts.jsonl")?;
        let facts: Vec<Fact> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Fact>(l).context("parse fact line"))
            .collect::<Result<_>>()?;
        anyhow::ensure!(
            emb.nrows() == facts.len(),
            "row mismatch: facts.npy {} vs facts.jsonl {}",
            emb.nrows(),
            facts.len()
        );
        let id2idx = facts
            .iter()
            .enumerate()
            .map(|(i, f)| (f.fact_id.clone(), i))
            .collect();
        Ok(Self { emb, facts, id2idx })
    }

    pub fn n(&self) -> usize {
        self.facts.len()
    }
    pub fn d(&self) -> usize {
        self.emb.ncols()
    }

    /// Deterministic temporal cutoff epoch: the value at `frac` through the
    /// sorted publish epochs. Facts at or after it are the held-out test set.
    pub fn temporal_cutoff(&self, frac: f32) -> i64 {
        let mut ep: Vec<i64> = self.facts.iter().map(|f| f.published_epoch).collect();
        ep.sort_unstable();
        if ep.is_empty() {
            return i64::MAX;
        }
        ep[((ep.len() as f32 * frac) as usize).min(ep.len() - 1)]
    }
}
