//! DuckLake tier writer: serializes each fact table to JSONL and emits a load
//! script (ATTACH + create facts + `INSERT ... BY NAME SELECT * FROM
//! read_json_auto`). Applied via the `duckdb` CLI -- no native libduckdb dep.

use crate::model::GraphBatch;
use anyhow::{Context, Result};
use serde::Serialize;
use std::io::Write;
use std::path::{Path, PathBuf};

fn write_jsonl<T: Serialize>(dir: &Path, name: &str, rows: &[T]) -> Result<(String, usize)> {
    let path = dir.join(format!("{name}.jsonl"));
    let mut f = std::fs::File::create(&path).with_context(|| format!("create {path:?}"))?;
    for r in rows {
        writeln!(f, "{}", serde_json::to_string(r)?)?;
    }
    Ok((path.to_string_lossy().into_owned(), rows.len()))
}

/// Write all fact tables. Returns (lake_table, file, row_count) for non-empty tables.
pub fn write_lake(dir: &Path, b: &GraphBatch) -> Result<Vec<(&'static str, String, usize)>> {
    std::fs::create_dir_all(dir)?;
    let mut out = Vec::new();
    macro_rules! emit {
        ($tbl:literal, $rows:expr) => {{
            let (file, n) = write_jsonl(dir, $tbl, $rows)?;
            if n > 0 { out.push(($tbl, file, n)); }
        }};
    }
    emit!("document", &b.documents);
    emit!("chunk", &b.chunks);
    emit!("doc_figure", &b.figures);
    emit!("event", &b.events);
    emit!("causal_event_edge", &b.causal_edges);
    emit!("sensitivity_edge", &b.sensitivities);
    emit!("sentiment_annotation", &b.sentiments);
    emit!("probability_annotation", &b.probabilities);
    emit!("relation_edge", &b.relations);
    Ok(out)
}

/// Full DuckLake load script: attach, ensure fact DDL, then load each JSONL.
pub fn load_script(
    tables: &[(&'static str, String, usize)],
    catalog: &str,
    data_path: &str,
    ddl_dir: &Path,
) -> String {
    let mut s = String::new();
    s.push_str("INSTALL ducklake; LOAD ducklake;\n");
    s.push_str(&format!("ATTACH '{catalog}' AS eg (DATA_PATH '{data_path}');\nUSE eg;\n"));
    // ensure the fact + realised + analytics + formal-plane DDL exists (idempotent,
    // numeric order = dependency order: views ref tables created in earlier files)
    for f in [
        "0002_facts.sql", "0003_realised.sql", "0004_analytics.sql", "0005_v2.sql",
        "0006_options.sql", "0007_lineage.sql", "0008_provenance.sql",
        "0009_surprise.sql", "0010_series_alias.sql", "0011_market_quote.sql",
    ] {
        s.push_str(&format!(".read {}\n", ddl_dir.join(f).to_string_lossy()));
    }
    for (tbl, file, _) in tables {
        s.push_str(&format!(
            "INSERT INTO eg.{tbl} BY NAME SELECT * FROM read_json_auto('{file}');\n"
        ));
    }
    s
}

/// Apply a script through the duckdb CLI (stdin).
pub fn apply(script: &str) -> Result<()> {
    let mut child = std::process::Command::new("duckdb")
        .arg(":memory:")
        .stdin(std::process::Stdio::piped())
        .spawn()
        .context("spawn duckdb (is the CLI installed?)")?;
    child
        .stdin
        .take()
        .context("duckdb stdin")?
        .write_all(script.as_bytes())?;
    let status = child.wait()?;
    anyhow::ensure!(status.success(), "duckdb load failed");
    Ok(())
}

pub fn write_script(dir: &Path, script: &str) -> Result<PathBuf> {
    let p = dir.join("ducklake_load.sql");
    std::fs::write(&p, script)?;
    Ok(p)
}
