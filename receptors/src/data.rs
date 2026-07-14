//! Load the exported embedding matrix + per-doc metadata.

use anyhow::{Context, Result};
use ndarray::Array2;
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize, Clone)]
pub struct Doc {
    pub doc_id: String,
    pub published_epoch: i64,
    #[serde(default)]
    pub date: String,
    pub entities: Vec<String>,
    #[serde(default)]
    pub cause_entities: Vec<String>,
}

pub struct Dataset {
    /// (N, D) L2-normalised embeddings; row i <-> docs[i].
    pub emb: Array2<f32>,
    pub docs: Vec<Doc>,
    pub id2idx: HashMap<String, usize>,
}

impl Dataset {
    pub fn load(dir: &Path) -> Result<Self> {
        let emb: Array2<f32> =
            read_npy(dir.join("embeddings.npy")).context("read embeddings.npy")?;
        let text = std::fs::read_to_string(dir.join("docs.jsonl")).context("read docs.jsonl")?;
        let docs: Vec<Doc> = text
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| serde_json::from_str::<Doc>(l).context("parse doc line"))
            .collect::<Result<_>>()?;

        anyhow::ensure!(
            emb.nrows() == docs.len(),
            "row mismatch: embeddings {} vs docs {}",
            emb.nrows(),
            docs.len()
        );

        let id2idx = docs
            .iter()
            .enumerate()
            .map(|(i, d)| (d.doc_id.clone(), i))
            .collect();

        Ok(Self { emb, docs, id2idx })
    }

    pub fn n(&self) -> usize {
        self.docs.len()
    }
    pub fn d(&self) -> usize {
        self.emb.ncols()
    }
}
