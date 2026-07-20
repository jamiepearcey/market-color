//! Orchestration: feed -> extract -> normalize -> one merged GraphBatch.
//! The caller routes the batch: dims/lifecycle -> Postgres, facts -> DuckLake.

use crate::extract::Extractor;
use crate::feed::Feed;
use crate::model::{DocExtraction, GraphBatch};
use crate::normalize::{build_batch, EntityResolver, GateStats};
use crate::util::chunk_text;
use anyhow::Result;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Clone)]
pub struct RunMeta {
    pub run_id: String,
    pub model: String,
    pub prompt_version: String,
    pub schema_version: String,
    pub created_at: String,
}

pub fn run(
    feed: &dyn Feed,
    extractor: &dyn Extractor,
    meta: &RunMeta,
    concurrency: usize,
    checkpoint: Option<&Path>,
) -> Result<GraphBatch> {
    let docs = feed.load()?;
    let n = docs.len();
    let conc = concurrency.clamp(1, 64);

    // Resume: load any prior extractions (doc_id -> DocExtraction) so an
    // interrupted run continues instead of re-paying for finished docs.
    let mut cached: HashMap<String, DocExtraction> = HashMap::new();
    if let Some(cp) = checkpoint {
        if cp.exists() {
            for line in BufReader::new(std::fs::File::open(cp)?).lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(&line) {
                    if let (Some(id), Some(ex)) = (v.get("doc_id").and_then(|x| x.as_str()), v.get("ex")) {
                        // Parse leniently (element-by-element) so a checkpointed
                        // extraction with one malformed array item keeps its good
                        // facts instead of failing the whole doc to a mock. Only an
                        // object ex is a real extraction (a null = failed doc).
                        if ex.is_object() {
                            let parsed = crate::extract::lenient_extraction(ex.clone());
                            cached.insert(id.to_string(), parsed);
                        }
                    }
                }
            }
        }
    }
    let todo: Vec<usize> = (0..n).filter(|i| !cached.contains_key(&docs[*i].doc_id)).collect();
    eprintln!("extracting {} of {n} docs ({} resumed from checkpoint, {conc} workers)", todo.len(), cached.len());

    // Thread-safe append-checkpoint: each completed extraction is persisted
    // immediately, so a kill/interrupt never loses finished work.
    let writer = match checkpoint {
        Some(cp) => Some(Mutex::new(BufWriter::new(
            OpenOptions::new().create(true).append(true).open(cp)?,
        ))),
        None => None,
    };

    // Parallelize the network-bound extraction (work-stealing over an atomic cursor).
    let cursor = AtomicUsize::new(0);
    let mut new_results: Vec<(usize, Result<DocExtraction, String>)> = Vec::with_capacity(todo.len());
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..conc)
            .map(|_| {
                let (cursor, docs, extractor, todo, writer) =
                    (&cursor, &docs, extractor, &todo, &writer);
                s.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let k = cursor.fetch_add(1, Ordering::Relaxed);
                        if k >= todo.len() {
                            break;
                        }
                        let i = todo[k];
                        let d = &docs[i];
                        let chunks = chunk_text(&d.headline, &d.article);
                        let r = extractor.extract(d, &chunks).map_err(|e| e.to_string());
                        if let (Ok(ex), Some(w)) = (&r, writer.as_ref()) {
                            if let Ok(line) =
                                serde_json::to_string(&serde_json::json!({"doc_id": d.doc_id, "ex": ex}))
                            {
                                let mut g = w.lock().unwrap();
                                let _ = writeln!(g, "{line}");
                                let _ = g.flush();
                            }
                        }
                        out.push((i, r));
                    }
                    out
                })
            })
            .collect();
        for h in handles {
            new_results.extend(h.join().unwrap());
        }
    });
    let mut new_map: HashMap<usize, Result<DocExtraction, String>> = new_results.into_iter().collect();

    // Resolve ONE canonical entity type per name across the whole run (over every
    // extraction, cached + newly-extracted) so a name the 8B typed inconsistently
    // collapses to a single node instead of splitting a hub by type.
    let canonical_types = crate::normalize::canonical_entity_types(
        cached.values().chain(new_map.values().filter_map(|r| r.as_ref().ok())),
    );

    // Normalize (sequential -> deterministic entity resolution) from cached + new.
    let mut resolver = EntityResolver::with_canonical_types(canonical_types);
    let mut gates = GateStats::default();
    let mut batch = GraphBatch::default();
    let (mut ok, mut err) = (0usize, 0usize);
    let mut sample_errs: Vec<String> = Vec::new();
    for i in 0..n {
        let ex = if let Some(ex) = cached.get(&docs[i].doc_id) {
            Some(ex.clone())
        } else {
            match new_map.remove(&i) {
                Some(Ok(ex)) => Some(ex),
                Some(Err(e)) => {
                    err += 1;
                    if sample_errs.len() < 5 {
                        sample_errs.push(e);
                    }
                    None
                }
                None => None,
            }
        };
        if let Some(ex) = ex {
            batch.merge(build_batch(&mut resolver, &mut gates, &docs[i], &ex, &meta.run_id));
            ok += 1;
        }
    }
    eprintln!("pipeline: {ok} extracted, {err} failed ({n} docs)");
    if !sample_errs.is_empty() {
        eprintln!("  sample failures: {sample_errs:?}");
    }
    eprintln!("{}", gates.report());
    Ok(batch)
}
