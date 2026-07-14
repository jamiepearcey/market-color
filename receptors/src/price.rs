//! Price-grounded receptor. Learns a bilinear operator W_price such that
//! article·W_price binds to the embedding of the SYMBOL that moved after it.
//! The target relation ("article A preceded a large move in symbol S") is not
//! present in any single document — it exists only by joining news to prices —
//! so this receptor carries signal naive RAG structurally cannot retrieve.

use crate::data::Dataset;
use crate::linalg;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
struct Symbol {
    symbol: String,
    name: String,
    #[allow(dead_code)]
    #[serde(default)]
    desks: Vec<String>,
}
#[derive(Deserialize)]
struct PricePair {
    doc_id: String,
    symbol: String,
    #[allow(dead_code)]
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

pub fn run(dir: &Path, lambda: f32) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);

    let mut symvec: Array2<f32> = read_npy(dir.join("symbols.npy")).context("symbols.npy")?;
    linalg::normalize_rows(&mut symvec);
    let symbols: Vec<Symbol> = load_jsonl(&dir.join("symbols.jsonl"))?;
    let sym2idx: HashMap<&str, usize> = symbols
        .iter()
        .enumerate()
        .map(|(i, s)| (s.symbol.as_str(), i))
        .collect();

    let pairs: Vec<PricePair> = load_jsonl(&dir.join("price_pairs.jsonl"))?;

    // temporal split: last ~30% of move-dates are test
    let mut dates: Vec<&str> = pairs.iter().map(|p| p.move_date.as_str()).collect();
    dates.sort_unstable();
    dates.dedup();
    let cutoff = dates[(dates.len() as f32 * 0.6) as usize].to_string();
    println!(
        "price receptor: {} symbols, {} pairs, {} move-dates (test cutoff {})",
        symbols.len(),
        pairs.len(),
        dates.len(),
        cutoff
    );

    // build train rows (article -> its moved symbol), skipping unknown docs/symbols
    let d = ds.d();
    let mut a_rows: Vec<f32> = Vec::new();
    let mut b_rows: Vec<f32> = Vec::new();
    let mut n_train = 0usize;
    let mut skipped = 0usize;
    for p in pairs.iter().filter(|p| p.move_date < cutoff) {
        let (Some(&ai), Some(&si)) = (ds.id2idx.get(&p.doc_id), sym2idx.get(p.symbol.as_str()))
        else {
            skipped += 1;
            continue;
        };
        let w = p.zscore.abs().sqrt(); // weight by move magnitude
        a_rows.extend(ds.emb.row(ai).iter().map(|x| x * w));
        b_rows.extend(symvec.row(si).iter().map(|x| x * w));
        n_train += 1;
    }
    anyhow::ensure!(n_train > 20, "not enough train pairs ({n_train})");
    let a_mat = Array2::from_shape_vec((n_train, d), a_rows)?;
    let b_mat = Array2::from_shape_vec((n_train, d), b_rows)?;
    let w_price = linalg::fit_transport(&a_mat, &b_mat, lambda)?;
    println!("fit W_price on {n_train} train pairs ({skipped} skipped, unknown doc/symbol)\n");

    // ---- micro-eval on held-out (later) move-dates ----
    // For each test pair, rank all symbols; is the true moved symbol ranked
    // higher via the receptor (cos(a·W, sym)) than naive cosine (cos(a, sym))?
    let rank_of = |scores: &Array1<f32>, target: usize| -> usize {
        let s = scores[target];
        1 + scores.iter().filter(|&&v| v > s).count()
    };
    let (mut n_eval, mut naive_mrr, mut recp_mrr) = (0usize, 0.0f32, 0.0f32);
    let (mut naive_r3, mut recp_r3) = (0usize, 0usize);
    for p in pairs.iter().filter(|p| p.move_date >= cutoff) {
        let (Some(&ai), Some(&si)) = (ds.id2idx.get(&p.doc_id), sym2idx.get(p.symbol.as_str()))
        else {
            continue;
        };
        let a = ds.emb.row(ai).to_owned();
        let mut aw = a.dot(&w_price); // predicted-symbol vector
        let n = aw.dot(&aw).sqrt();
        if n > 1e-9 {
            aw.mapv_inplace(|x| x / n);
        }
        let naive: Array1<f32> = symvec.dot(&a); // cos(article, symbol name)
        let recp: Array1<f32> = symvec.dot(&aw); // cos(article·W, symbol)
        let (rn, rr) = (rank_of(&naive, si), rank_of(&recp, si));
        naive_mrr += 1.0 / rn as f32;
        recp_mrr += 1.0 / rr as f32;
        naive_r3 += (rn <= 3) as usize;
        recp_r3 += (rr <= 3) as usize;
        n_eval += 1;
    }
    let nf = n_eval as f32;
    println!("== MICRO-EVAL: rank the actually-moved symbol ({n_eval} held-out pairs) ==");
    println!("  {:<26} {:>7} {:>9}", "scorer", "MRR", "Recall@3");
    println!("  {:<26} {:>7.3} {:>9.3}", "naive cos(article,symbol)", naive_mrr / nf, naive_r3 as f32 / nf);
    println!("  {:<26} {:>7.3} {:>9.3}", "price receptor cos(a·W,s)", recp_mrr / nf, recp_r3 as f32 / nf);

    // ---- example bindings for the EM questions ----
    if let (Ok(q), Ok(txt)) = (
        read_npy::<_, Array2<f32>>(dir.join("questions_custom.npy")),
        std::fs::read_to_string(dir.join("questions_custom.txt")),
    ) {
        let qtexts: Vec<&str> = txt.lines().filter(|l| !l.trim().is_empty()).collect();
        println!("\n== price receptor bindings: query -> symbols it predicts will move ==");
        for (i, qt) in qtexts.iter().enumerate().take(10) {
            let qr = q.row(i).to_owned();
            let mut aw = qr.dot(&w_price);
            let n = aw.dot(&aw).sqrt();
            if n > 1e-9 {
                aw.mapv_inplace(|x| x / n);
            }
            let sc: Array1<f32> = symvec.dot(&aw);
            let mut idx: Vec<usize> = (0..symbols.len()).collect();
            idx.sort_by(|&x, &y| sc[y].partial_cmp(&sc[x]).unwrap());
            let top: Vec<&str> = idx.iter().take(3).map(|&j| symbols[j].symbol.as_str()).collect();
            println!("  {:<60}  ->  {}", &qt[..qt.len().min(58)], top.join(", "));
        }
    }
    Ok(())
}
