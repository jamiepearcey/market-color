//! Signed-impact (polarity) receptor. Learns a single direction w in embedding
//! space such that sign(emb·w) predicts a bullish/bearish market move. Polarity
//! is orthogonal to topic — two opposite-direction claims about the same event
//! are cosine-near — so we test the linear probe against a cosine-kNN baseline:
//! if the probe beats kNN, the receptor captures a signal cosine cannot.

use crate::linalg;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Lab {
    y: f32,
    #[serde(default)]
    subj: String,
}

pub fn run(dir: &Path, lambda: f32) -> Result<()> {
    let mut x: Array2<f32> = read_npy(dir.join("claims.npy")).context("claims.npy")?;
    linalg::normalize_rows(&mut x);
    let labs: Vec<Lab> = std::fs::read_to_string(dir.join("claims_labels.jsonl"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let n = x.nrows();
    anyhow::ensure!(n == labs.len(), "row mismatch");
    let y: Vec<f32> = labs.iter().map(|l| l.y).collect();

    // deterministic 70/30 split
    let is_test = |i: usize| (i.wrapping_mul(2654435761) % 100) < 30;
    let train: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();

    // ridge linear probe:  w = (X^T X + λI)^-1 X^T y   over train rows
    let d = x.ncols();
    let mut xt = Array2::<f32>::zeros((train.len(), d));
    let mut yt = Array1::<f32>::zeros(train.len());
    for (r, &i) in train.iter().enumerate() {
        xt.row_mut(r).assign(&x.row(i));
        yt[r] = y[i];
    }
    let mut g = xt.t().dot(&xt);
    for k in 0..d {
        g[[k, k]] += lambda;
    }
    let rhs = xt.t().dot(&yt); // (d,)
    let ginv = {
        use nalgebra::DMatrix;
        let m = DMatrix::from_row_iterator(d, d, g.iter().copied());
        let inv = m.try_inverse().context("probe not invertible")?;
        Array2::from_shape_fn((d, d), |(a, b)| inv[(a, b)])
    };
    let w: Array1<f32> = ginv.dot(&rhs);

    // baselines + probe on test
    let up = y.iter().filter(|&&v| v > 0.0).count() as f32 / n as f32;
    let majority = up.max(1.0 - up);

    // cosine-kNN (k=15) majority-vote over TRAIN, using topical similarity only
    let k = 15usize;
    let (mut probe_ok, mut knn_ok, mut probe_bal, mut knn_bal) = (0usize, 0usize, [0f32; 2], [0f32; 2]);
    let mut cls_n = [0f32; 2];
    for &i in &test {
        let xi = x.row(i).to_owned();
        // probe
        let p = if xi.dot(&w) >= 0.0 { 1.0 } else { -1.0 };
        // kNN over train by cosine
        let mut sims: Vec<(f32, f32)> = train.iter().map(|&j| (xi.dot(&x.row(j)), y[j])).collect();
        sims.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let vote: f32 = sims.iter().take(k).map(|s| s.1).sum();
        let knn = if vote >= 0.0 { 1.0 } else { -1.0 };

        let ci = if y[i] > 0.0 { 0 } else { 1 };
        cls_n[ci] += 1.0;
        if p == y[i] {
            probe_ok += 1;
            probe_bal[ci] += 1.0;
        }
        if knn == y[i] {
            knn_ok += 1;
            knn_bal[ci] += 1.0;
        }
    }
    let nt = test.len() as f32;
    let bal = |b: [f32; 2]| 0.5 * (b[0] / cls_n[0].max(1.0) + b[1] / cls_n[1].max(1.0));
    println!("polarity receptor: {n} claims ({:.0}% up), {} test", up * 100.0, test.len());
    println!("\n  == classification ==");
    println!("  {:<28} {:>8} {:>10}", "predictor", "acc", "bal-acc");
    println!("  {:<28} {:>8.3} {:>10.3}", "majority baseline", majority, 0.5);
    println!("  {:<28} {:>8.3} {:>10.3}", "cosine-kNN (topical)", knn_ok as f32 / nt, bal(knn_bal));
    println!("  {:<28} {:>8.3} {:>10.3}", "polarity receptor (probe)", probe_ok as f32 / nt, bal(probe_bal));

    // ===================== RANKING VALUE (within-subject) =====================
    // Held-out queries; candidates = other EVAL claims about the SAME subject.
    // Same-direction candidates are "relevant", opposite-direction are hard
    // negatives (topically identical). Does adding polarity rank same-dir above
    // opposite-dir? Pairwise AUC = P(same-dir scored > opp-dir). Cosine alone
    // should be ~0.5 (can't tell direction within a subject).
    use std::collections::HashMap;
    let subj: Vec<&str> = labs.iter().map(|l| l.subj.as_str()).collect();
    // group EVAL (test) claims by subject
    let mut groups: HashMap<&str, Vec<usize>> = HashMap::new();
    for &i in &test {
        if !subj[i].is_empty() {
            groups.entry(subj[i]).or_default().push(i);
        }
    }
    let proj: Vec<f32> = (0..n).map(|i| x.row(i).dot(&w)).collect(); // polarity score per claim
    let scorers: [(&str, f32); 4] = [
        ("cosine only", 0.0),
        ("cosine + 1·polarity", 1.0),
        ("cosine + 3·polarity", 3.0),
        ("polarity only", f32::INFINITY),
    ];
    // micro (per-triple) and macro (per-subject, equal weight) AUC
    let mut micro = vec![0.0f64; scorers.len()];
    let mut macro_sum = vec![0.0f64; scorers.len()];
    let mut npairs = 0u64;
    let mut n_subj_used = 0usize;
    for ids in groups.values() {
        if ids.len() < 2 {
            continue;
        }
        let mut g_ok = vec![0.0f64; scorers.len()];
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
                    let (pol_i, pol_j) = (proj[qi] * proj[ci], proj[qi] * proj[cj]);
                    for (b, &(_, beta)) in scorers.iter().enumerate() {
                        let (si, sj) = if beta.is_infinite() {
                            (pol_i, pol_j)
                        } else {
                            (cos_i + beta * pol_i, cos_j + beta * pol_j)
                        };
                        let v = if si > sj { 1.0 } else if si == sj { 0.5 } else { 0.0 };
                        g_ok[b] += v;
                        micro[b] += v;
                    }
                    g_n += 1;
                }
            }
        }
        if g_n > 0 {
            for b in 0..scorers.len() {
                macro_sum[b] += g_ok[b] / g_n as f64;
            }
            npairs += g_n;
            n_subj_used += 1;
        }
    }
    anyhow::ensure!(npairs > 0, "no same/opposite-direction pairs within subjects");
    let (np, ns) = (npairs as f64, n_subj_used as f64);
    println!(
        "\n  == ranking value: P(same-dir ranked > opp-dir), within-subject ==",
    );
    println!("  {} subjects, {} triple-comparisons (micro is bitcoin/gold-heavy)", n_subj_used, npairs);
    println!("  {:<24} {:>10} {:>12}", "scorer", "micro-AUC", "macro-AUC");
    for (b, &(name, _)) in scorers.iter().enumerate() {
        println!("  {:<24} {:>10.3} {:>12.3}", name, micro[b] / np, macro_sum[b] / ns);
    }
    Ok(())
}
