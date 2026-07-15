//! A1 — Mechanism tensor {W_r}: one asymmetric transport operator *per
//! predicate* instead of collapsing all causality into a single W. Every effect
//! doc carries a dominant `predicate` (policy_action, supply_change, sanction,
//! forecast, ...); we type each mined cause->effect pair by the effect's
//! predicate and fit a separate ridge operator for each well-supported type.
//!
//! Two questions:
//!   1. Does typing (route a pair to its mechanism's operator) beat the single
//!      untyped W on direction accuracy?
//!   2. What is the transmission grammar — which mechanisms carry the strongest
//!      directional signal (‖W_r‖, leading channel, own-type accuracy)?
//!
//! Plus a cross-desk transfer probe: fit W on one desk's pairs, test on another,
//! to see whether market causality is a universal geometry or desk-specific.

use crate::data::Dataset;
use crate::linalg;
use crate::pairs::{self, Pair};
use anyhow::Result;
use ndarray::Array2;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::Path;

/// Build (A, B) cause/effect row matrices for a set of pairs over `emb`.
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

/// Direction accuracy on `test` pairs given a transported-cause matrix `ahat`
/// (row i = normalized emb_i·W) scored against raw `emb`.
fn dir_acc(test: &[&Pair], ahat: &Array2<f32>, emb: &Array2<f32>) -> f32 {
    let mut ok = 0usize;
    for p in test {
        let fwd = ahat.row(p.a).dot(&emb.row(p.b));
        let rev = ahat.row(p.b).dot(&emb.row(p.a));
        if fwd > rev {
            ok += 1;
        }
    }
    if test.is_empty() {
        0.0
    } else {
        ok as f32 / test.len() as f32
    }
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64, test_frac: u64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let mined = pairs::mine(&ds, specific_frac, test_frac);
    let train: Vec<&Pair> = mined.pairs.iter().filter(|p| !p.is_test).collect();
    let test: Vec<&Pair> = mined.pairs.iter().filter(|p| p.is_test).collect();
    anyhow::ensure!(train.len() > 50 && test.len() > 20, "not enough pairs");
    println!(
        "mechanism tensor: {} docs, {} train / {} test pairs",
        ds.n(),
        train.len(),
        test.len()
    );

    // ---- baseline: single untyped W ----
    let (a, b) = stack(&ds.emb, &train);
    let w_global = linalg::fit_transport(&a, &b, lambda)?;
    let ahat_global = linalg::transport(&ds.emb, &w_global);

    // ---- per-predicate operators, typed by the EFFECT doc's predicate ----
    let pred_of = |p: &Pair| ds.docs[p.b].predicate.clone();
    let mut by_pred: HashMap<String, Vec<&Pair>> = HashMap::new();
    for p in &train {
        by_pred.entry(pred_of(p)).or_default().push(p);
    }
    const MIN_SUPPORT: usize = 40;

    // Fit W_r and its transported matrix for each well-supported predicate.
    let mut ahat_by_pred: HashMap<String, Array2<f32>> = HashMap::new();
    let mut summary: Vec<(String, usize, f32, f32, f32)> = Vec::new(); // pred, n, own-acc, ‖W‖, σ1
    for (pred, ps) in &by_pred {
        if ps.len() < MIN_SUPPORT {
            continue;
        }
        let (ar, br) = stack(&ds.emb, ps);
        let w_r = linalg::fit_transport(&ar, &br, lambda)?;
        let ahat_r = linalg::transport(&ds.emb, &w_r);
        // own-type test accuracy
        let own_test: Vec<&Pair> = test
            .iter()
            .copied()
            .filter(|p| &pred_of(p) == pred)
            .collect();
        let own_acc = dir_acc(&own_test, &ahat_r, &ds.emb);
        let fro = w_r.iter().map(|x| x * x).sum::<f32>().sqrt();
        let s1 = linalg::top_singular(&w_r, 1)
            .first()
            .map(|c| c.0)
            .unwrap_or(0.0);
        summary.push((pred.clone(), ps.len(), own_acc, fro, s1));
        ahat_by_pred.insert(pred.clone(), ahat_r);
    }

    // ---- typed direction accuracy: route each test pair to its W_r (fallback global) ----
    // Collect per-test-pair correctness for both scorers (bootstrap CIs + McNemar).
    let mut single_correct: Vec<f32> = Vec::with_capacity(test.len());
    let mut typed_correct: Vec<f32> = Vec::with_capacity(test.len());
    for p in &test {
        let sg = ahat_global.row(p.a).dot(&ds.emb.row(p.b))
            > ahat_global.row(p.b).dot(&ds.emb.row(p.a));
        single_correct.push(sg as u8 as f32);
        let ah = ahat_by_pred.get(&pred_of(p)).unwrap_or(&ahat_global);
        let tp = ah.row(p.a).dot(&ds.emb.row(p.b)) > ah.row(p.b).dot(&ds.emb.row(p.a));
        typed_correct.push(tp as u8 as f32);
    }
    let (acc_global, sg_lo, sg_hi) = crate::report::ci_mean(&single_correct);
    let (acc_typed, tp_lo, tp_hi) = crate::report::ci_mean(&typed_correct);
    // McNemar discordant counts (typed right & single wrong vs the reverse).
    let (mut b01, mut b10) = (0u32, 0u32);
    for i in 0..test.len() {
        match (single_correct[i] as u8, typed_correct[i] as u8) {
            (0, 1) => b10 += 1,
            (1, 0) => b01 += 1,
            _ => {}
        }
    }
    // McNemar chi-square (1 df) with continuity correction.
    let mcnemar = if b01 + b10 > 0 {
        let d = (b10 as f32 - b01 as f32).abs() - 1.0;
        (d.max(0.0)).powi(2) / (b01 + b10) as f32
    } else {
        0.0
    };

    println!("\n== DIRECTION: typed mechanism tensor vs single W ==");
    println!("  symmetric cosine (floor)     : 0.500");
    println!("  single untyped W             : {:.3}  [95% CI {:.3}, {:.3}]", acc_global, sg_lo, sg_hi);
    println!("  typed {{W_r}} (route by mech)  : {:.3}  [95% CI {:.3}, {:.3}]", acc_typed, tp_lo, tp_hi);
    println!(
        "  McNemar chi2={:.1} (typed>single {} vs single>typed {}); chi2>10.8 => p<0.001",
        mcnemar, b10, b01
    );

    println!("\n== transmission grammar (per effect-predicate operator) ==");
    println!("  {:<18} {:>6} {:>9} {:>8} {:>8}", "predicate", "n", "dir-acc", "‖W‖_F", "σ1");
    summary.sort_by(|x, y| y.2.partial_cmp(&x.2).unwrap());
    for (pred, n, acc, fro, s1) in &summary {
        println!("  {:<18} {:>6} {:>9.3} {:>8.2} {:>8.2}", pred, n, acc, fro, s1);
    }

    // ---- cross-desk transfer: fit on desk X pairs, test on desk Y ----
    let (xdesks, xmatrix) = cross_desk(&ds, &train, &test, lambda)?;

    // ---- emit metrics bundle for figures + CIs ----
    let per_pred: Vec<Value> = summary
        .iter()
        .map(|(p, n, acc, fro, s1)| json!({"predicate": p, "n": n, "acc": acc, "fro": fro, "sigma1": s1}))
        .collect();
    crate::report::save(
        dir,
        "mechanism",
        &json!({
            "n_test": test.len(),
            "single_acc": acc_global, "single_ci": [sg_lo, sg_hi],
            "typed_acc": acc_typed, "typed_ci": [tp_lo, tp_hi],
            "mcnemar_chi2": mcnemar, "mcnemar_typed_wins": b10, "mcnemar_single_wins": b01,
            "per_predicate": per_pred,
            "cross_desk": {"desks": xdesks, "matrix": xmatrix},
        }),
    )?;
    Ok(())
}

/// Fit W separately on each desk's training pairs; report a direction-accuracy
/// matrix (train desk rows x test desk cols). High off-diagonal => universal
/// causal geometry; strong diagonal only => desk-specific mechanisms.
fn cross_desk(
    ds: &Dataset,
    train: &[&Pair],
    test: &[&Pair],
    lambda: f32,
) -> Result<(Vec<String>, Vec<Vec<f32>>)> {
    let desk_of = |p: &Pair| ds.docs[p.b].desk.clone();
    let mut counts: HashMap<String, usize> = HashMap::new();
    for p in train {
        *counts.entry(desk_of(p)).or_default() += 1;
    }
    let mut desks: Vec<(String, usize)> = counts.into_iter().collect();
    desks.sort_by(|a, b| b.1.cmp(&a.1));
    desks.truncate(4);
    let desks: Vec<String> = desks.into_iter().map(|(d, _)| d).collect();
    if desks.len() < 2 {
        return Ok((desks, vec![]));
    }

    // fit one W per train-desk
    let mut ahat_by_desk: HashMap<String, Array2<f32>> = HashMap::new();
    for d in &desks {
        let ps: Vec<&Pair> = train.iter().copied().filter(|p| &desk_of(p) == d).collect();
        if ps.len() < 40 {
            continue;
        }
        let (a, b) = stack(&ds.emb, &ps);
        let w = linalg::fit_transport(&a, &b, lambda)?;
        ahat_by_desk.insert(d.clone(), linalg::transport(&ds.emb, &w));
    }

    println!("\n== cross-desk transfer (direction acc; row=train desk, col=test desk) ==");
    print!("  {:<12}", "train\\test");
    for td in &desks {
        print!(" {:>10}", td);
    }
    println!();
    let mut matrix: Vec<Vec<f32>> = Vec::new();
    for tr in &desks {
        let Some(ahat) = ahat_by_desk.get(tr) else {
            matrix.push(vec![f32::NAN; desks.len()]);
            continue;
        };
        print!("  {:<12}", tr);
        let mut row = Vec::with_capacity(desks.len());
        for te in &desks {
            let tp: Vec<&Pair> = test.iter().copied().filter(|p| &desk_of(p) == te).collect();
            let acc = dir_acc(&tp, ahat, &ds.emb);
            print!(" {:>10.3}", acc);
            row.push(acc);
        }
        println!();
        matrix.push(row);
    }
    Ok((desks, matrix))
}
