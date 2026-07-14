//! Modality (epistemic-status) receptor. Multi-class one-vs-rest ridge probe
//! predicting a claim's `predicate` (event / forecast / statement / policy_action
//! / ...). Modality is orthogonal to topic — within one subject you get events,
//! forecasts and statements about the same thing — so we test the probe against a
//! cosine-kNN baseline (classification) and a within-subject ranking test (does
//! adding modality rank same-modality above different-modality candidates?).

use crate::linalg;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

#[derive(Deserialize)]
struct Lab {
    y: usize,
    #[serde(default)]
    subj: String,
}

pub fn run(dir: &Path, lambda: f32) -> Result<()> {
    let mut x: Array2<f32> = read_npy(dir.join("modality.npy")).context("modality.npy")?;
    linalg::normalize_rows(&mut x);
    let labs: Vec<Lab> = std::fs::read_to_string(dir.join("modality_labels.jsonl"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let classes: Vec<String> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("modality_classes.json"))?)?;
    let (n, d, k) = (x.nrows(), x.ncols(), classes.len());
    anyhow::ensure!(n == labs.len(), "row mismatch");
    let y: Vec<usize> = labs.iter().map(|l| l.y).collect();

    let is_test = |i: usize| (i.wrapping_mul(2654435761) % 100) < 30;
    let train: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();

    // one-vs-rest ridge:  W (d x k) = (X^T X + λI)^-1 X^T Y,  Y in {+1,-1}
    let mut xt = Array2::<f32>::zeros((train.len(), d));
    let mut ymat = Array2::<f32>::zeros((train.len(), k));
    for (r, &i) in train.iter().enumerate() {
        xt.row_mut(r).assign(&x.row(i));
        for c in 0..k {
            ymat[[r, c]] = if y[i] == c { 1.0 } else { -1.0 };
        }
    }
    let mut g = xt.t().dot(&xt);
    for j in 0..d {
        g[[j, j]] += lambda;
    }
    let ginv = {
        use nalgebra::DMatrix;
        let m = DMatrix::from_row_iterator(d, d, g.iter().copied());
        let inv = m.try_inverse().context("probe not invertible")?;
        Array2::from_shape_fn((d, d), |(a, b)| inv[(a, b)])
    };
    let wmod = ginv.dot(&xt.t().dot(&ymat)); // (d x k)

    // per-claim modality vector (k-dim logits), row-normalised
    let mut modv = x.dot(&wmod); // (n x k)
    linalg::normalize_rows(&mut modv);

    // ---- classification: argmax probe vs cosine-kNN vs majority ----
    let mut counts = vec![0usize; k];
    for &i in &train {
        counts[y[i]] += 1;
    }
    let majority = *counts.iter().max().unwrap() as f32 / train.len() as f32;
    let knn = 15usize;
    let (mut probe_ok, mut knn_ok) = (0usize, 0usize);
    let mut probe_bal = vec![0f32; k];
    let mut cls_n = vec![0f32; k];
    for &i in &test {
        let logit: Array1<f32> = x.row(i).dot(&wmod);
        let pred = (0..k).max_by(|&a, &b| logit[a].partial_cmp(&logit[b]).unwrap()).unwrap();
        // kNN over train by cosine, majority predicate
        let mut sims: Vec<(f32, usize)> = train.iter().map(|&j| (x.row(i).dot(&x.row(j)), y[j])).collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let mut votes = vec![0usize; k];
        for s in sims.iter().take(knn) {
            votes[s.1] += 1;
        }
        let kp = (0..k).max_by_key(|&c| votes[c]).unwrap();
        cls_n[y[i]] += 1.0;
        if pred == y[i] {
            probe_ok += 1;
            probe_bal[y[i]] += 1.0;
        }
        if kp == y[i] {
            knn_ok += 1;
        }
    }
    let nt = test.len() as f32;
    let bal: f32 = (0..k).map(|c| probe_bal[c] / cls_n[c].max(1.0)).sum::<f32>() / k as f32;
    println!("modality receptor: {n} claims, {k} classes, {} test", test.len());
    println!("\n  == classification ({k}-way) ==");
    println!("  {:<26} {:>8} {:>10}", "predictor", "acc", "bal-acc");
    println!("  {:<26} {:>8.3} {:>10.3}", "majority baseline", majority, 1.0 / k as f32);
    println!("  {:<26} {:>8.3}", "cosine-kNN (topical)", knn_ok as f32 / nt);
    println!("  {:<26} {:>8.3} {:>10.3}", "modality receptor (probe)", probe_ok as f32 / nt, bal);

    // ---- within-subject ranking: same-modality vs different-modality ----
    let subj: Vec<&str> = labs.iter().map(|l| l.subj.as_str()).collect();
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for &i in &test {
        if !subj[i].is_empty() {
            groups.entry(subj[i]).or_default().push(i);
        }
    }
    let betas = [0.0f32, 1.0, 3.0];
    let mut micro = vec![0.0f64; betas.len() + 1];
    let mut macro_s = vec![0.0f64; betas.len() + 1];
    let (mut npairs, mut nsub) = (0u64, 0usize);
    for ids in groups.values() {
        if ids.len() < 2 {
            continue;
        }
        let mut g_ok = vec![0.0f64; betas.len() + 1];
        let mut g_n = 0u64;
        for &qi in ids {
            for &ci in ids {
                for &cj in ids {
                    if ci == qi || cj == qi || ci == cj {
                        continue;
                    }
                    if !(y[ci] == y[qi] && y[cj] != y[qi]) {
                        continue;
                    }
                    let (cos_i, cos_j) = (x.row(qi).dot(&x.row(ci)), x.row(qi).dot(&x.row(cj)));
                    let (m_i, m_j) = (modv.row(qi).dot(&modv.row(ci)), modv.row(qi).dot(&modv.row(cj)));
                    for (b, &beta) in betas.iter().enumerate() {
                        let (si, sj) = (cos_i + beta * m_i, cos_j + beta * m_j);
                        let v = if si > sj { 1.0 } else if si == sj { 0.5 } else { 0.0 };
                        g_ok[b] += v;
                        micro[b] += v;
                    }
                    let v = if m_i > m_j { 1.0 } else if m_i == m_j { 0.5 } else { 0.0 };
                    g_ok[betas.len()] += v;
                    micro[betas.len()] += v;
                    g_n += 1;
                }
            }
        }
        if g_n > 0 {
            for b in 0..=betas.len() {
                macro_s[b] += g_ok[b] / g_n as f64;
            }
            npairs += g_n;
            nsub += 1;
        }
    }
    anyhow::ensure!(npairs > 0, "no same/diff-modality pairs within subjects");
    let (np, ns) = (npairs as f64, nsub as f64);
    println!("\n  == ranking: P(same-modality ranked > different-modality), within-subject ==");
    println!("  {} subjects, {} triple-comparisons", nsub, npairs);
    println!("  {:<24} {:>10} {:>12}", "scorer", "micro-AUC", "macro-AUC");
    for (b, &beta) in betas.iter().enumerate() {
        let tag = if beta == 0.0 { "cosine only".to_string() } else { format!("cosine + {beta}·modality") };
        println!("  {:<24} {:>10.3} {:>12.3}", tag, micro[b] / np, macro_s[b] / ns);
    }
    println!("  {:<24} {:>10.3} {:>12.3}", "modality only", micro[betas.len()] / np, macro_s[betas.len()] / ns);
    Ok(())
}
