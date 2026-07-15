//! B4 — SOURCE / PROVENANCE & CORROBORATION.
//!
//! There is no ground-truth reliability label in this corpus. So we use
//! CORROBORATION as a proxy signal: a fact is more trustworthy, roughly, if
//! independent facts near it in time say the same directional thing about the
//! same entities. We then ask two questions.
//!
//!   Q1: does the 384-d claim embedding carry any signal about whether a fact
//!       is corroborated? We fit a ridge probe (embedding -> corroboration
//!       count) on a train split and score the held-out test split with
//!       Spearman + Mann-Whitney AUC, against a confidence-as-predictor
//!       baseline.
//!   Q2 (provenance): for the busiest source_name values, how confident,
//!       how corroborated, and what fraction of their facts are corroborated?
//!       This is the honesty axis — a source that is prolific but rarely
//!       corroborated is a single-source firehose, not a second witness.
//!
//! Honest caveat: corroboration is a proxy, not verified truth. Echoed facts
//! (the same claim copied across wires) inflate it; genuinely scoop-worthy
//! true facts deflate it. Read the numbers as "agreement structure", not
//! "correctness".

use crate::facts::FactSet;
use crate::linalg;
use anyhow::Result;
use serde_json::json;
use ndarray::{Array1, Array2};
use std::collections::HashMap;
use std::path::Path;

/// Deterministic 70/30 split by fact index (matches atlas.rs style).
fn is_test(i: usize) -> bool {
    (i.wrapping_mul(2654435761) % 100) < 30
}

fn sign(x: f32) -> i8 {
    if x > 0.5 {
        1
    } else if x < -0.5 {
        -1
    } else {
        0
    }
}

/// Mann-Whitney AUC = P(pos scored > neg), from average ranks (ties handled).
/// Mirrors atlas.rs::auc.
fn auc(pos: &[f32], neg: &[f32]) -> f32 {
    let mut all: Vec<(f32, u8)> = pos
        .iter()
        .map(|&s| (s, 1u8))
        .chain(neg.iter().map(|&s| (s, 0u8)))
        .collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 2) as f64 / 2.0;
        for k in i..=j {
            if all[k].1 == 1 {
                rank_sum_pos += avg_rank;
            }
        }
        i = j + 1;
    }
    let (np, nn) = (pos.len() as f64, neg.len() as f64);
    if np == 0.0 || nn == 0.0 {
        return 0.5;
    }
    ((rank_sum_pos - np * (np + 1.0) / 2.0) / (np * nn)) as f32
}

/// Fractional (tie-averaged) 1-based ranks of `v`.
fn ranks(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut r = vec![0.0f32; v.len()];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 2) as f32 / 2.0; // mean of 1-based ranks i+1..=j+1
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation between two equal-length vectors.
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    let ma: f32 = ra.iter().sum::<f32>() / n as f32;
    let mb: f32 = rb.iter().sum::<f32>() / n as f32;
    let mut num = 0.0f32;
    let mut da = 0.0f32;
    let mut db = 0.0f32;
    for i in 0..n {
        let x = ra[i] - ma;
        let y = rb[i] - mb;
        num += x * y;
        da += x * x;
        db += y * y;
    }
    if da <= 0.0 || db <= 0.0 {
        return 0.0;
    }
    num / (da.sqrt() * db.sqrt())
}

pub fn run(dir: &Path, lambda: f32) -> Result<()> {
    let mut fs = FactSet::load(dir)?;
    linalg::normalize_rows(&mut fs.emb);
    let n = fs.n();
    println!(
        "source / provenance receptor: {} facts x {} dims\n",
        n,
        fs.d()
    );

    // ---- corroboration via an entity->fact_ids inverted index ----------
    // For each fact i, count OTHER facts j sharing >=1 entity, within a 3-day
    // window (both epochs > 0), with the same direction sign. We only ever
    // compare i against the union of candidates that share an entity, which
    // keeps this well below the naive O(n^2).
    const WIN: i64 = 3 * 86_400;

    let mut inv: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, f) in fs.facts.iter().enumerate() {
        for e in &f.entities {
            inv.entry(e.as_str()).or_default().push(i);
        }
    }

    let mut corrob = vec![0u32; n];
    let mut seen = vec![u32::MAX; n]; // per-i scratch to dedupe candidate j's
    for i in 0..n {
        let fi = &fs.facts[i];
        let ei = fi.published_epoch;
        if ei <= 0 || fi.entities.is_empty() {
            continue; // cannot be corroborated without a valid epoch + entity
        }
        let si = sign(fi.direction);
        let mut c = 0u32;
        for e in &fi.entities {
            if let Some(cands) = inv.get(e.as_str()) {
                for &j in cands {
                    if j == i || seen[j] == i as u32 {
                        continue;
                    }
                    seen[j] = i as u32; // count each shared-entity j once
                    let fj = &fs.facts[j];
                    let ej = fj.published_epoch;
                    if ej <= 0 {
                        continue;
                    }
                    if (ei - ej).abs() <= WIN && sign(fj.direction) == si {
                        c += 1;
                    }
                }
            }
        }
        corrob[i] = c;
    }

    let corroborated: Vec<bool> = corrob.iter().map(|&c| c >= 2).collect();
    let n_corr = corroborated.iter().filter(|&&b| b).count();
    println!("corroboration (>=2 same-sign, same-entity facts within 3 days):");
    println!(
        "  {}/{} facts corroborated ({:.1}%), {} single-source; mean corrob = {:.2}\n",
        n_corr,
        n,
        100.0 * n_corr as f32 / n.max(1) as f32,
        n - n_corr,
        corrob.iter().map(|&c| c as f64).sum::<f64>() / n.max(1) as f64,
    );

    // ---- Q1: can the embedding predict corroboration? ------------------
    let train_idx: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_idx: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();

    let d = fs.d();
    let mut xtr = Array2::<f32>::zeros((train_idx.len(), d));
    let mut ytr = Array1::<f32>::zeros(train_idx.len());
    for (r, &i) in train_idx.iter().enumerate() {
        xtr.row_mut(r).assign(&fs.emb.row(i));
        ytr[r] = corrob[i] as f32;
    }
    let w = linalg::ridge_probe(&xtr, &ytr, lambda)?;

    // test-set probe scores + targets
    let pred: Vec<f32> = test_idx.iter().map(|&i| fs.emb.row(i).dot(&w)).collect();
    let ytest: Vec<f32> = test_idx.iter().map(|&i| corrob[i] as f32).collect();
    let conf_test: Vec<f32> = test_idx.iter().map(|&i| fs.facts[i].confidence).collect();

    let rho = spearman(&pred, &ytest);

    // aligned probe scores + binary corroborated labels for the test split
    let probe_scores: Vec<f32> = pred.clone();
    let labels: Vec<u8> = test_idx
        .iter()
        .map(|&i| if corroborated[i] { 1u8 } else { 0u8 })
        .collect();

    // AUC: corroborated (label true) vs single-source on the test split
    let (mut probe_pos, mut probe_neg) = (Vec::new(), Vec::new());
    let (mut conf_pos, mut conf_neg) = (Vec::new(), Vec::new());
    for (k, &i) in test_idx.iter().enumerate() {
        if corroborated[i] {
            probe_pos.push(pred[k]);
            conf_pos.push(conf_test[k]);
        } else {
            probe_neg.push(pred[k]);
            conf_neg.push(conf_test[k]);
        }
    }
    let probe_auc = auc(&probe_pos, &probe_neg);
    let conf_auc = auc(&conf_pos, &conf_neg);

    println!("== Q1: does the embedding predict corroboration? (70/30 index split) ==");
    println!("  train n={}  test n={}", train_idx.len(), test_idx.len());
    println!(
        "  test corroborated={} / single-source={}",
        probe_pos.len(),
        probe_neg.len()
    );
    println!(
        "  embedding ridge probe   Spearman(pred, corrob) = {:+.3}",
        rho
    );
    println!(
        "  embedding ridge probe   AUC(corrob vs single)  =  {:.3}",
        probe_auc
    );
    println!(
        "  baseline confidence     AUC(corrob vs single)  =  {:.3}",
        conf_auc
    );
    println!(
        "  -> embedding {} confidence by {:+.3} AUC\n",
        if probe_auc >= conf_auc { "beats" } else { "trails" },
        probe_auc - conf_auc
    );

    // ---- Q2: provenance table by source_name ---------------------------
    struct Agg {
        n: usize,
        conf: f64,
        corr: f64,
        n_corr: usize,
    }
    let mut by_src: HashMap<&str, Agg> = HashMap::new();
    for (i, f) in fs.facts.iter().enumerate() {
        let name = if f.source_name.is_empty() {
            "(unknown)"
        } else {
            f.source_name.as_str()
        };
        let a = by_src.entry(name).or_insert(Agg {
            n: 0,
            conf: 0.0,
            corr: 0.0,
            n_corr: 0,
        });
        a.n += 1;
        a.conf += f.confidence as f64;
        a.corr += corrob[i] as f64;
        if corroborated[i] {
            a.n_corr += 1;
        }
    }
    let mut rows: Vec<(&str, &Agg)> = by_src.iter().map(|(k, v)| (*k, v)).collect();
    rows.sort_by(|a, b| b.1.n.cmp(&a.1.n)); // busiest first
    rows.truncate(12);
    // present sorted by % corroborated (the honesty axis)
    rows.sort_by(|a, b| {
        let pa = a.1.n_corr as f64 / a.1.n as f64;
        let pb = b.1.n_corr as f64 / b.1.n as f64;
        pb.partial_cmp(&pa).unwrap()
    });

    println!(
        "== Q2: provenance table (top {} sources by fact count, sorted by %corroborated) ==",
        rows.len()
    );
    println!(
        "  {:<26} {:>6} {:>9} {:>10} {:>9}",
        "source_name", "n", "mean-conf", "mean-corr", "%corrob"
    );
    for (name, a) in &rows {
        let disp: String = name.chars().take(26).collect();
        println!(
            "  {:<26} {:>6} {:>9.3} {:>10.2} {:>8.1}%",
            disp,
            a.n,
            a.conf / a.n as f64,
            a.corr / a.n as f64,
            100.0 * a.n_corr as f64 / a.n as f64,
        );
    }

    println!(
        "\n  note: corroboration is a PROXY, not verified truth — echoed wire copy\n  inflates it and genuine scoops deflate it. Read the table as agreement\n  structure across sources, not a correctness ranking. A high-n, low-%corrob\n  source is a single-source firehose; a high-%corrob source tends to report\n  the same directional events others independently confirm."
    );

    let provenance: Vec<serde_json::Value> = rows
        .iter()
        .map(|(name, a)| {
            json!({
                "source": name.to_string(),
                "n": a.n,
                "mean_conf": (a.conf / a.n as f64) as f32,
                "mean_corr": (a.corr / a.n as f64) as f32,
                "pct_corrob": (100.0 * a.n_corr as f64 / a.n as f64) as f32,
            })
        })
        .collect();

    crate::report::save(
        dir,
        "source",
        &json!({
            "embed_auc": probe_auc,
            "conf_auc": conf_auc,
            "spearman": rho,
            "n_test": labels.len(),
            "probe_scores": probe_scores,
            "labels": labels,
            "provenance": provenance,
        }),
    )?;

    Ok(())
}
