//! Receptor B2 — CROSS-ASSET SPILLOVER (contagion).
//!
//! Unlike `price.rs` (which binds an article to THE one symbol that moved),
//! this receptor learns a multi-output operator W: (D, S) that maps a document
//! embedding to a signed score over the WHOLE basket of S symbols. The training
//! target for a doc is a row over all symbols carrying the signed z-score for
//! every symbol that moved after that doc (0 elsewhere), so a single doc's
//! target may light up several symbols at once — that co-movement is the
//! contagion signal. We then ask whether the receptor recovers not just the
//! PRIMARY (largest-|z|) moved symbol but the SECONDARY spillover symbols, and
//! whether it beats a naive cosine(doc, symbol-name) ranking at doing so.

use crate::data::Dataset;
use crate::linalg;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
struct Symbol {
    symbol: String,
    #[allow(dead_code)]
    name: String,
    #[allow(dead_code)]
    #[serde(default)]
    desks: Vec<String>,
}

#[derive(Deserialize)]
struct PricePair {
    doc_id: String,
    symbol: String,
    #[serde(default)]
    sign: i32,
    zscore: f32,
    move_date: String,
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(p: &Path) -> Result<Vec<T>> {
    let s = std::fs::read_to_string(p).with_context(|| format!("read {p:?}"))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<T>(l).context("parse jsonl"))
        .collect()
}

/// One moved symbol for a doc: its symbol index and signed magnitude.
#[derive(Clone)]
struct Move {
    si: usize,
    signed_z: f32, // sign * |zscore|
    date: String,
}

/// A doc with >=1 moved symbol.
struct DocTargets {
    ai: usize,             // row index into ds.emb
    doc_id: String,
    moves: Vec<Move>,      // all moved symbols (>=1)
    earliest_date: String, // earliest move_date -> temporal split key
}

pub fn run(dir: &Path, lambda: f32) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);

    let mut symvec: Array2<f32> = read_npy(dir.join("symbols.npy")).context("symbols.npy")?;
    linalg::normalize_rows(&mut symvec);
    let symbols: Vec<Symbol> = load_jsonl(&dir.join("symbols.jsonl"))?;
    let s_count = symbols.len();
    anyhow::ensure!(
        symvec.nrows() == s_count,
        "symbols.npy rows {} vs symbols.jsonl {}",
        symvec.nrows(),
        s_count
    );
    let sym2idx: HashMap<&str, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.symbol.as_str(), i))
        .collect();

    let pairs: Vec<PricePair> = load_jsonl(&dir.join("price_pairs.jsonl"))?;

    // Group moves per document (only docs present in the embedding index and
    // whose symbol is known). Each group becomes one multi-output target row.
    let mut per_doc: HashMap<String, DocTargets> = HashMap::new();
    let mut skipped = 0usize;
    for p in pairs.iter() {
        let (Some(&ai), Some(&si)) = (ds.id2idx.get(&p.doc_id), sym2idx.get(p.symbol.as_str()))
        else {
            skipped += 1;
            continue;
        };
        // signed z: prefer explicit sign, fall back to zscore's own sign.
        let mag = p.zscore.abs();
        let signed_z = if p.sign != 0 {
            p.sign.signum() as f32 * mag
        } else {
            p.zscore
        };
        let entry = per_doc.entry(p.doc_id.clone()).or_insert_with(|| DocTargets {
            ai,
            doc_id: p.doc_id.clone(),
            moves: Vec::new(),
            earliest_date: p.move_date.clone(),
        });
        if p.move_date < entry.earliest_date {
            entry.earliest_date = p.move_date.clone();
        }
        entry.moves.push(Move {
            si,
            signed_z,
            date: p.move_date.clone(),
        });
    }
    let mut docs: Vec<DocTargets> = per_doc.into_values().filter(|d| !d.moves.is_empty()).collect();
    // stable order for reproducible examples
    docs.sort_by(|a, b| a.doc_id.cmp(&b.doc_id));

    // Temporal split. We split each DOC by its EARLIEST move_date (simple &
    // deterministic): a doc is test iff its earliest move is on/after the
    // cutoff. Cutoff = the date at the 0.6 quantile of all distinct move-dates.
    let mut dates: Vec<&str> = pairs.iter().map(|p| p.move_date.as_str()).collect();
    dates.sort_unstable();
    dates.dedup();
    anyhow::ensure!(!dates.is_empty(), "no move-dates");
    let cutoff = dates[((dates.len() as f32 * 0.6) as usize).min(dates.len() - 1)].to_string();

    println!(
        "spillover receptor: {} symbols, {} pairs, {} docs w/ moves, {} move-dates (test cutoff {}, split by earliest move_date)",
        s_count,
        pairs.len(),
        docs.len(),
        dates.len(),
        cutoff
    );
    let multi = docs.iter().filter(|d| d.moves.len() >= 2).count();
    println!(
        "  {} docs move >=2 symbols (the contagion-bearing docs; {} pairs skipped: unknown doc/symbol)",
        multi, skipped
    );

    // Build train matrices: A = (n_train, D) doc embeddings, B = (n_train, S)
    // signed-zscore target rows. One row per TRAIN doc.
    let d = ds.d();
    let mut a_rows: Vec<f32> = Vec::new();
    let mut b_rows: Vec<f32> = Vec::new();
    let mut n_train = 0usize;
    let mut n_test = 0usize;
    for doc in docs.iter() {
        if doc.earliest_date >= cutoff {
            n_test += 1;
            continue;
        }
        a_rows.extend(ds.emb.row(doc.ai).iter().copied());
        let mut target = vec![0.0f32; s_count];
        for m in doc.moves.iter() {
            // keep the largest-magnitude entry if a symbol repeats for a doc
            if m.signed_z.abs() > target[m.si].abs() {
                target[m.si] = m.signed_z;
            }
        }
        b_rows.extend(target);
        n_train += 1;
    }
    anyhow::ensure!(n_train > 20, "not enough train docs ({n_train})");
    anyhow::ensure!(n_test > 0, "no test docs");
    let a_mat = Array2::from_shape_vec((n_train, d), a_rows)?;
    let b_mat = Array2::from_shape_vec((n_train, s_count), b_rows)?;
    // Multi-output ridge: W = (AᵀA + λI)⁻¹ Aᵀ B, shape (D, S).
    let w_spill = linalg::fit_transport(&a_mat, &b_mat, lambda)?;
    println!("fit W_spill (D={d} -> S={s_count}) on {n_train} train docs; {n_test} test docs\n");

    // ---- eval on held-out docs ----
    // For each test doc: receptor scores = emb·W (S,), rank symbols descending.
    // Naive baseline: cosine(doc_emb, symbol_emb) = symvec·emb (S,).
    // For every actually-moved symbol of that doc, check top-3 membership.
    // Split moved symbols into PRIMARY (largest |z| for the doc) vs SECONDARY.
    let top3 = |scores: &Array1<f32>, target: usize| -> bool {
        let s = scores[target];
        scores.iter().filter(|&&v| v > s).count() < 3
    };

    let (mut prim_hits_r, mut prim_hits_n, mut prim_tot) = (0usize, 0usize, 0usize);
    let (mut sec_hits_r, mut sec_hits_n, mut sec_tot) = (0usize, 0usize, 0usize);
    let (mut all_hits_r, mut all_hits_n, mut all_tot) = (0usize, 0usize, 0usize);

    // per-moved-symbol 0/1 hit indicators for confidence intervals
    let mut prim_naive_hits: Vec<f32> = Vec::new();
    let mut prim_recp_hits: Vec<f32> = Vec::new();
    let mut sec_naive_hits: Vec<f32> = Vec::new();
    let mut sec_recp_hits: Vec<f32> = Vec::new();

    // collect a few illustrative test docs (prefer multi-symbol ones)
    struct Example {
        doc_id: String,
        actual: Vec<(String, f32)>, // symbol, signed_z
        recp_top: Vec<String>,
    }
    let mut examples: Vec<Example> = Vec::new();

    for doc in docs.iter().filter(|d| d.earliest_date >= cutoff) {
        let emb = ds.emb.row(doc.ai).to_owned();
        let recp: Array1<f32> = emb.dot(&w_spill); // (S,) receptor scores
        let naive: Array1<f32> = symvec.dot(&emb); // (S,) cosine to symbol names

        // dedup moved symbols for this doc, keeping max |z|; find primary.
        let mut best: HashMap<usize, f32> = HashMap::new();
        for m in doc.moves.iter() {
            let e = best.entry(m.si).or_insert(0.0);
            if m.signed_z.abs() > e.abs() {
                *e = m.signed_z;
            }
        }
        let primary = best
            .iter()
            .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap())
            .map(|(&si, _)| si);

        for (&si, &z) in best.iter() {
            let hr = top3(&recp, si);
            let hn = top3(&naive, si);
            all_hits_r += hr as usize;
            all_hits_n += hn as usize;
            all_tot += 1;
            if Some(si) == primary {
                prim_hits_r += hr as usize;
                prim_hits_n += hn as usize;
                prim_tot += 1;
                prim_recp_hits.push(hr as u8 as f32);
                prim_naive_hits.push(hn as u8 as f32);
            } else {
                sec_hits_r += hr as usize;
                sec_hits_n += hn as usize;
                sec_tot += 1;
                sec_recp_hits.push(hr as u8 as f32);
                sec_naive_hits.push(hn as u8 as f32);
            }
            let _ = z;
        }

        // gather an example if it has spillover (>=2 symbols) and we want ~6
        if best.len() >= 2 && examples.len() < 6 {
            let mut actual: Vec<(String, f32)> = best
                .iter()
                .map(|(&si, &z)| (symbols[si].symbol.clone(), z))
                .collect();
            actual.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
            let mut idx: Vec<usize> = (0..s_count).collect();
            idx.sort_by(|&x, &y| recp[y].partial_cmp(&recp[x]).unwrap());
            let recp_top: Vec<String> =
                idx.iter().take(4).map(|&j| symbols[j].symbol.clone()).collect();
            examples.push(Example {
                doc_id: doc.doc_id.clone(),
                actual,
                recp_top,
            });
        }
    }

    let pct = |num: usize, den: usize| if den == 0 { 0.0 } else { num as f32 / den as f32 };
    println!("== SPILLOVER EVAL: is each actually-moved symbol in the top-3? (Recall@3) ==");
    println!(
        "  {:<28} {:>10} {:>10} {:>7}",
        "cohort", "naive cos", "receptor", "n"
    );
    println!(
        "  {:<28} {:>10.3} {:>10.3} {:>7}",
        "PRIMARY (largest |z|)",
        pct(prim_hits_n, prim_tot),
        pct(prim_hits_r, prim_tot),
        prim_tot
    );
    println!(
        "  {:<28} {:>10.3} {:>10.3} {:>7}   <- real contagion test",
        "SECONDARY (spillover)",
        pct(sec_hits_n, sec_tot),
        pct(sec_hits_r, sec_tot),
        sec_tot
    );
    println!(
        "  {:<28} {:>10.3} {:>10.3} {:>7}",
        "ALL moved symbols",
        pct(all_hits_n, all_tot),
        pct(all_hits_r, all_tot),
        all_tot
    );

    println!("\n== example test docs: actually-moved symbols  vs  receptor top-4 ==");
    for ex in examples.iter() {
        let actual: Vec<String> = ex
            .actual
            .iter()
            .map(|(s, z)| format!("{s}({z:+.1})"))
            .collect();
        let id = &ex.doc_id[..ex.doc_id.len().min(20)];
        println!("  {:<22} moved: {}", id, actual.join(", "));
        println!("  {:<22}  recp: {}", "", ex.recp_top.join(", "));
    }

    // ---- honest note ----
    let sec_r = pct(sec_hits_r, sec_tot);
    let sec_n = pct(sec_hits_n, sec_tot);
    println!("\n== honest note ==");
    if sec_tot == 0 {
        println!(
            "  No secondary (spillover) symbols in the held-out docs, so the contagion claim\n\
             \x20 cannot be tested on this split. Only primary movers were evaluable."
        );
    } else if sec_r > sec_n + 1e-4 {
        println!(
            "  On SECONDARY (spillover) symbols the receptor Recall@3 {sec_r:.3} beats naive\n\
             \x20 cosine {sec_n:.3} (+{:.3}). The learned doc->basket operator recovers contagion\n\
             \x20 symbols that cosine-to-symbol-name cannot — the co-movement signal only exists\n\
             \x20 in the news<->price join, not in any single document.",
            sec_r - sec_n
        );
    } else if (sec_r - sec_n).abs() <= 1e-4 {
        println!(
            "  The receptor ties naive cosine {sec_r:.3} on spillover symbols: no evidence it\n\
             \x20 recovers contagion beyond topical cosine on this split."
        );
    } else {
        println!(
            "  The receptor UNDERPERFORMS naive cosine on spillover symbols ({sec_r:.3} < {sec_n:.3}).\n\
             \x20 On this split it does NOT recover contagion better than cosine-to-symbol-name;\n\
             \x20 secondary movers may be too sparse/noisy for the ridge operator to fit."
        );
    }

    crate::report::save(
        dir,
        "spillover",
        &json!({
            "primary": {
                "naive": pct(prim_hits_n, prim_tot),
                "receptor": pct(prim_hits_r, prim_tot),
                "naive_hits": prim_naive_hits,
                "recp_hits": prim_recp_hits,
            },
            "secondary": {
                "naive": pct(sec_hits_n, sec_tot),
                "receptor": pct(sec_hits_r, sec_tot),
                "naive_hits": sec_naive_hits,
                "recp_hits": sec_recp_hits,
            },
            "all": {
                "naive": pct(all_hits_n, all_tot),
                "receptor": pct(all_hits_r, all_tot),
            },
            "n_symbols": s_count,
            "n_test_docs": n_test,
        }),
    )?;

    Ok(())
}
