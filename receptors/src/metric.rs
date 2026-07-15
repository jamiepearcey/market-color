//! B5 — Learned relatedness metric. Every eval says the real bottleneck is
//! first-stage recall: cosine simply doesn't put enough true causes in the top
//! shortlist. Here we learn a low-rank projection L (r x D) so that causally
//! linked (cause, effect) pairs are near under the symmetric metric
//! s(x,y) = (Lx)·(Ly), while random later docs are far. Unlike the asymmetric
//! transport operator this stays a genuine metric — index Lx with any ANN
//! backend — so it can improve *finding*, not just reordering.
//!
//! Trained with a contrastive hinge (positive causal pair beats a random later
//! effect by a margin); every step is a handful of BLAS matmuls. Warm-started
//! from the top-r PCA directions so it begins ~ cosine-in-a-subspace.

use crate::data::Dataset;
use crate::linalg;
use crate::pairs::{self, Pair};
use anyhow::Result;
use ndarray::{Array1, Array2, Axis};
use serde_json::json;
use std::path::Path;

fn stack(emb: &Array2<f32>, pairs: &[&Pair]) -> (Array2<f32>, Array2<f32>) {
    let d = emb.ncols();
    let mut a = Array2::<f32>::zeros((pairs.len(), d));
    let mut b = Array2::<f32>::zeros((pairs.len(), d));
    for (r, p) in pairs.iter().enumerate() {
        a.row_mut(r).assign(&emb.row(p.a));
        b.row_mut(r).assign(&emb.row(p.b));
    }
    (a, b)
}

fn rowdot(p: &Array2<f32>, q: &Array2<f32>) -> Array1<f32> {
    (p * q).sum_axis(Axis(1))
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64, test_frac: u64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let mined = pairs::mine(&ds, specific_frac, test_frac);
    let train: Vec<&Pair> = mined.pairs.iter().filter(|p| !p.is_test).collect();
    anyhow::ensure!(train.len() > 50, "not enough pairs");

    let d = ds.d();
    let r: usize = std::env::var("METRIC_R").ok().and_then(|v| v.parse().ok()).unwrap_or(64);
    let iters: usize = std::env::var("METRIC_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(250);
    let lr: f32 = std::env::var("METRIC_LR").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
    let margin: f32 = 0.15;

    // warm start: L0 = top-r PCA directions transposed (r x D)
    let gist = linalg::gist_subspace(&ds.emb, r)?; // (D, r)
    let mut l = gist.t().to_owned(); // (r, D)

    let (a, b) = stack(&ds.emb, &train);
    let n = a.nrows();
    println!("metric receptor: {} docs, {} train pairs, rank r={}", ds.n(), n, r);

    for it in 0..iters {
        let pa = a.dot(&l.t()); // (n, r)
        let pb = b.dot(&l.t());
        // deterministic derangement -> one random later-effect negative each
        let mut bperm = Array2::<f32>::zeros((n, d));
        let mut perm = vec![0usize; n];
        for i in 0..n {
            let mut j = (i.wrapping_mul(2654435761).wrapping_add(it.wrapping_mul(40503)) + 12345) % n;
            if j == i { j = (j + 1) % n; }
            perm[i] = j;
            bperm.row_mut(i).assign(&b.row(perm[i]));
        }
        let pbp = bperm.dot(&l.t());
        let s_pos = rowdot(&pa, &pb);
        let s_neg = rowdot(&pa, &pbp);
        let mut v = Array1::<f32>::zeros(n);
        for i in 0..n {
            if margin - s_pos[i] + s_neg[i] > 0.0 { v[i] = 1.0; }
        }
        // M = Σ v_i (a_i b_negᵀ + b_neg a_iᵀ - a_i b_iᵀ - b_i a_iᵀ)  (D x D)
        let av = &a * &v.view().insert_axis(Axis(1));
        let bv = &b * &v.view().insert_axis(Axis(1));
        let bpv = &bperm * &v.view().insert_axis(Axis(1));
        let m = av.t().dot(&bperm) + bpv.t().dot(&a) - av.t().dot(&b) - bv.t().dot(&a);
        let g = l.dot(&m); // (r, D)
        let scale = lr / n as f32;
        l = &l - &g.mapv(|x| x * scale) - &l.mapv(|x| x * (lr * lambda * 0.01));
    }

    // Projected, row-normalised corpus in the learned metric.
    let mut proj = ds.emb.dot(&l.t()); // (N, r)
    linalg::normalize_rows(&mut proj);

    // First-stage retrieval: cosine vs the learned metric, recall@k curve.
    let epochs: Vec<i64> = ds.docs.iter().map(|d| d.published_epoch).collect();
    let ks = [5usize, 10, 30, 100];
    let (cos_curve, cos_r10) = multi_recall(&mined.pairs, &epochs, &ks, |i, j| {
        ds.emb.row(i).dot(&ds.emb.row(j))
    });
    let (met_curve, met_r10) = multi_recall(&mined.pairs, &epochs, &ks, |i, j| {
        proj.row(i).dot(&proj.row(j))
    });
    let (cos_pt, cos_lo, cos_hi) = crate::report::ci_mean(&cos_r10);
    let (met_pt, met_lo, met_hi) = crate::report::ci_mean(&met_r10);

    println!("\n== FIRST-STAGE RETRIEVAL: recall@k (cosine vs learned metric) ==");
    print!("  {:<20}", "k");
    for k in ks {
        print!(" {:>8}", k);
    }
    println!();
    print!("  {:<20}", "raw cosine (384-d)");
    for v in &cos_curve {
        print!(" {:>8.3}", v);
    }
    println!();
    print!("  {:<20}", "learned metric L");
    for v in &met_curve {
        print!(" {:>8.3}", v);
    }
    println!();
    println!("  Recall@10  cosine {:.3} [{:.3},{:.3}]  vs  metric {:.3} [{:.3},{:.3}]",
        cos_pt, cos_lo, cos_hi, met_pt, met_lo, met_hi);
    let lift = if cos_pt > 0.0 { 100.0 * (met_pt - cos_pt) / cos_pt } else { 0.0 };
    println!("  metric lift on Recall@10: {:+.1}%  (CI overlap => not significant)", lift);

    crate::report::save(
        dir,
        "metric",
        &json!({
            "r": r, "ks": ks.to_vec(),
            "cosine_curve": cos_curve, "metric_curve": met_curve,
            "cosine_r10": cos_pt, "cosine_r10_ci": [cos_lo, cos_hi],
            "metric_r10": met_pt, "metric_r10_ci": [met_lo, met_hi],
        }),
    )?;
    Ok(())
}

/// Recall@k curve (macro over test causes) + the per-cause Recall@10 array (for CIs).
fn multi_recall<F: Fn(usize, usize) -> f32>(
    pairs: &[Pair],
    epochs: &[i64],
    ks: &[usize],
    scorer: F,
) -> (Vec<f32>, Vec<f32>) {
    use std::collections::{HashMap, HashSet};
    let mut test: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut train: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in pairs {
        if p.is_test {
            test.entry(p.a).or_default().insert(p.b);
        } else {
            train.entry(p.a).or_default().insert(p.b);
        }
    }
    let n = epochs.len();
    let mut sums = vec![0.0f32; ks.len()];
    let mut per_cause_r10: Vec<f32> = Vec::new();
    let empty = HashSet::new();
    let r10_pos = ks.iter().position(|&k| k == 10);
    let mut causes = 0usize;
    for (&a, truth) in &test {
        let ta = epochs[a];
        let excl = train.get(&a).unwrap_or(&empty);
        let mut cands: Vec<(f32, usize)> = Vec::new();
        for j in 0..n {
            if j == a || epochs[j] <= ta || excl.contains(&j) {
                continue;
            }
            cands.push((scorer(a, j), j));
        }
        if cands.is_empty() {
            continue;
        }
        cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());
        for (ki, &k) in ks.iter().enumerate() {
            let hits = cands.iter().take(k).filter(|(_, j)| truth.contains(j)).count();
            let rc = hits as f32 / truth.len().min(k) as f32;
            sums[ki] += rc;
            if Some(ki) == r10_pos {
                per_cause_r10.push(rc);
            }
        }
        causes += 1;
    }
    let curve = sums.iter().map(|s| if causes > 0 { s / causes as f32 } else { 0.0 }).collect();
    (curve, per_cause_r10)
}
