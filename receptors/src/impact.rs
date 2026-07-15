//! A3 — MARKET-IMPACT receptor. "How market-moving is a fact?"
//!
//! We label each fact by the largest realized price move (|zscore|) on the day
//! its source document landed, then fit a ridge linear probe from the 384-d claim
//! embedding to that magnitude. On a temporal held-out set we ask: does the
//! embedding rank facts by how much the market actually moved (Spearman), does it
//! concentrate the big moves in its top decile (top-decile lift), and does it beat
//! the trivial `fact.confidence` predictor? Impact is a property of *what a fact
//! says*, not just how confidently it's asserted — so this is the real test.

use crate::facts::FactSet;
use crate::linalg;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct PricePair {
    doc_id: String,
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    sign: i32,
    zscore: f32,
    #[allow(dead_code)]
    #[serde(default)]
    move_date: String,
}

/// Ordinal ranks (1-based) in ascending order of value. Ties are broken by the
/// stable sort (i.e. simple ordinal ranks, not average ranks) — acceptable for a
/// continuous |zscore| target where exact ties are rare; noted in the output.
fn ranks(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut r = vec![0.0f32; v.len()];
    for (rank, &i) in idx.iter().enumerate() {
        r[i] = (rank + 1) as f32;
    }
    r
}

/// Spearman rank correlation = Pearson correlation on the ordinal ranks.
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f32;
    if n < 2.0 {
        return 0.0;
    }
    let (ma, mb) = (ra.iter().sum::<f32>() / n, rb.iter().sum::<f32>() / n);
    let mut num = 0.0f32;
    let (mut da, mut db) = (0.0f32, 0.0f32);
    for i in 0..ra.len() {
        let (x, y) = (ra[i] - ma, rb[i] - mb);
        num += x * y;
        da += x * x;
        db += y * y;
    }
    let den = (da * db).sqrt();
    if den < 1e-9 {
        0.0
    } else {
        num / den
    }
}

pub fn run(dir: &Path, lambda: f32) -> Result<()> {
    let mut fs = FactSet::load(dir)?;
    linalg::normalize_rows(&mut fs.emb);

    // ---- load price moves, keep max |zscore| per doc ----
    let text = std::fs::read_to_string(dir.join("price_pairs.jsonl"))
        .context("read price_pairs.jsonl")?;
    let mut doc_move: HashMap<String, f32> = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let p: PricePair = serde_json::from_str(line).context("parse price_pair line")?;
        let m = p.zscore.abs();
        let e = doc_move.entry(p.doc_id).or_insert(0.0);
        if m > *e {
            *e = m;
        }
    }

    println!("market-impact receptor (A3): {} facts x {} dims", fs.n(), fs.d());
    println!("  {} docs with a realized price move", doc_move.len());

    // ---- supervised set: facts whose doc has a move; target y = max |zscore| ----
    let sup: Vec<usize> = (0..fs.n())
        .filter(|&i| doc_move.contains_key(&fs.facts[i].doc_id))
        .collect();
    anyhow::ensure!(!sup.is_empty(), "no facts matched a priced document");
    let y_of = |i: usize| doc_move[&fs.facts[i].doc_id];

    // ---- temporal split (published_epoch>0), fall back to hash split if thin ----
    let cutoff = fs.temporal_cutoff(0.7);
    let dated: Vec<usize> = sup
        .iter()
        .copied()
        .filter(|&i| fs.facts[i].published_epoch > 0)
        .collect();
    let mut train: Vec<usize> = dated.iter().copied().filter(|&i| fs.facts[i].published_epoch < cutoff).collect();
    let mut test: Vec<usize> = dated.iter().copied().filter(|&i| fs.facts[i].published_epoch >= cutoff).collect();
    let split_kind;
    if test.len() < 30 {
        // deterministic 70/30 hash split on fact index
        let is_test = |i: usize| (i.wrapping_mul(2654435761) % 100) < 30;
        train = sup.iter().copied().filter(|&i| !is_test(i)).collect();
        test = sup.iter().copied().filter(|&i| is_test(i)).collect();
        split_kind = "hash 70/30 (temporal split left <30 test)";
    } else {
        split_kind = "temporal (published_epoch, 0.7 cutoff)";
    }
    println!(
        "  supervised={}  train={}  test={}  split={}",
        sup.len(),
        train.len(),
        test.len(),
        split_kind
    );
    anyhow::ensure!(train.len() >= 5 && test.len() >= 5, "too few labelled facts to fit/evaluate");

    // ---- fit ridge probe on train: w = (XᵀX + λI)⁻¹ Xᵀy ----
    let d = fs.d();
    let mut xtr = Array2::<f32>::zeros((train.len(), d));
    let mut ytr = Array1::<f32>::zeros(train.len());
    for (r, &i) in train.iter().enumerate() {
        xtr.row_mut(r).assign(&fs.emb.row(i));
        ytr[r] = y_of(i);
    }
    let w = linalg::ridge_probe(&xtr, &ytr, lambda)?;

    // ---- predict on test ----
    let pred: Vec<f32> = test.iter().map(|&i| fs.emb.row(i).dot(&w)).collect();
    let actual: Vec<f32> = test.iter().map(|&i| y_of(i)).collect();
    let conf: Vec<f32> = test.iter().map(|&i| fs.facts[i].confidence).collect();

    // ---- metrics ----
    let sp_emb = spearman(&pred, &actual);
    let sp_conf = spearman(&conf, &actual);

    // top-decile lift = mean(actual among top-10% predicted) / mean(actual all)
    let base_mean = actual.iter().sum::<f32>() / actual.len() as f32;
    let mut order: Vec<usize> = (0..test.len()).collect();
    order.sort_by(|&a, &b| pred[b].partial_cmp(&pred[a]).unwrap());
    let k = ((test.len() as f32 * 0.10).ceil() as usize).max(1);
    let top_mean = order.iter().take(k).map(|&j| actual[j]).sum::<f32>() / k as f32;
    let lift = if base_mean > 1e-9 { top_mean / base_mean } else { f32::NAN };

    println!("\n== impact metrics (TEST) ==  (Spearman = Pearson on ordinal ranks; simple ties)");
    println!("  {:<38} {:>10}", "metric", "value");
    println!("  {:<38} {:>10.3}", "Spearman(embedding-probe, |z|)", sp_emb);
    println!("  {:<38} {:>10.3}", "Spearman(fact.confidence, |z|)  [baseline]", sp_conf);
    println!("  {:<38} {:>10.2}x", "top-decile lift (embedding)", lift);
    println!(
        "  {:<38} {:>10} / {:.3}",
        "top-decile n / base mean |z|", k, base_mean
    );

    // ---- top-8 highest predicted-impact test facts ----
    println!("\n== top-8 predicted-impact TEST facts ==");
    println!("  {:>7} {:>7}  claim", "pred", "|z|");
    for &j in order.iter().take(8) {
        let i = test[j];
        println!(
            "  {:>7.3} {:>7.2}  {}",
            pred[j],
            actual[j],
            clip(&fs.facts[i].claim)
        );
    }

    // ---- honest interpretation ----
    let verdict = if sp_emb.abs() < 0.05 {
        "essentially no signal — the embedding does not predict realized market impact here"
    } else if sp_emb < 0.15 {
        "weak: a faint but unreliable ranking of impact from the embedding"
    } else if sp_emb < 0.35 {
        "modest: the embedding ranks impact meaningfully above chance"
    } else {
        "strong: the embedding is a genuine predictor of realized impact"
    };
    let vs_conf = if sp_emb > sp_conf + 0.02 {
        "and it beats the confidence baseline"
    } else if sp_conf > sp_emb + 0.02 {
        "but plain fact.confidence ranks impact at least as well — the embedding adds little"
    } else {
        "on par with the confidence baseline"
    };
    println!("\nnote: {verdict}; {vs_conf}.");

    crate::report::save(dir, "impact", &json!({
        "spearman_embed": sp_emb,
        "spearman_conf": sp_conf,
        "topdecile_lift": lift,
        "n_test": pred.len(),
        "pred": pred,
        "actual": actual,
    }))?;
    Ok(())
}

fn clip(s: &str) -> String {
    let s = s.trim();
    s.chars().take(96).collect()
}
