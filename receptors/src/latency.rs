//! B1 — TRANSMISSION LATENCY: given a cause article, predict the lag (in days)
//! until the effect article lands. We mine directed cause->effect pairs for free
//! (shared specific cause-entity + strict time precedence), read the wall-clock
//! gap off the publication epochs, and fit a ridge probe from the CAUSE embedding
//! to lag_days. The question is receptor-shaped: does the *content* of a cause
//! carry information about how fast the market reacts, or is the lag mostly noise?
//!
//! We answer it two ways:
//!   1. Predictive — probe MAE vs a mean-lag baseline, plus Spearman(pred, actual).
//!   2. Descriptive — split test causes into FAST vs SLOW by predicted-lag median
//!      and tally the entities that dominate each bucket (the transmission archetypes).

use crate::data::Dataset;
use crate::linalg;
use crate::pairs::{self, Pair};
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

/// Pearson correlation of two equal-length slices.
fn pearson(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mx = x.iter().sum::<f32>() / n as f32;
    let my = y.iter().sum::<f32>() / n as f32;
    let mut sxy = 0.0f32;
    let mut sxx = 0.0f32;
    let mut syy = 0.0f32;
    for i in 0..n {
        let dx = x[i] - mx;
        let dy = y[i] - my;
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    let denom = (sxx * syy).sqrt();
    if denom > 1e-12 {
        sxy / denom
    } else {
        0.0
    }
}

/// Fractional ranks (average rank for ties) of a slice.
fn ranks(v: &[f32]) -> Vec<f32> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    let mut r = vec![0.0f32; n];
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && (v[idx[j]] - v[idx[i]]).abs() < 1e-9 {
            j += 1;
        }
        // ranks i..j are tied -> average rank (1-based, using mid of the block)
        let avg = ((i + j - 1) as f32) / 2.0 + 1.0;
        for k in i..j {
            r[idx[k]] = avg;
        }
        i = j;
    }
    r
}

/// Spearman rank correlation = Pearson on the ranks.
fn spearman(x: &[f32], y: &[f32]) -> f32 {
    pearson(&ranks(x), &ranks(y))
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64, test_frac: u64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let mined = pairs::mine(&ds, specific_frac, test_frac);

    // Keep only pairs with dated endpoints and a plausible lag in (0, 30] days.
    let mut kept: Vec<(&Pair, f32)> = Vec::new();
    for p in &mined.pairs {
        let ea = ds.docs[p.a].published_epoch;
        let eb = ds.docs[p.b].published_epoch;
        if ea <= 0 || eb <= 0 {
            continue;
        }
        let lag_days = (eb - ea) as f32 / 86400.0;
        if lag_days > 0.0 && lag_days <= 30.0 {
            kept.push((p, lag_days));
        }
    }

    let train: Vec<(&Pair, f32)> = kept.iter().copied().filter(|(p, _)| !p.is_test).collect();
    let test: Vec<(&Pair, f32)> = kept.iter().copied().filter(|(p, _)| p.is_test).collect();
    anyhow::ensure!(
        train.len() > 50 && test.len() > 20,
        "not enough dated pairs (train {}, test {})",
        train.len(),
        test.len()
    );
    println!(
        "transmission latency: {} docs, {} dated pairs in (0,30]d -> {} train / {} test",
        ds.n(),
        kept.len(),
        train.len(),
        test.len()
    );

    // ---- Fit ridge probe: cause embedding -> lag_days ----
    let d = ds.d();
    let mut xtr = Array2::<f32>::zeros((train.len(), d));
    let mut ytr = Array1::<f32>::zeros(train.len());
    for (r, (p, lag)) in train.iter().enumerate() {
        xtr.row_mut(r).assign(&ds.emb.row(p.a));
        ytr[r] = *lag;
    }
    let w = linalg::ridge_probe(&xtr, &ytr, lambda)?;
    let mean_lag = ytr.sum() / train.len() as f32;

    // ---- Predict on test causes ----
    let mut preds = Vec::with_capacity(test.len());
    let mut actuals = Vec::with_capacity(test.len());
    for (p, lag) in &test {
        let pred = ds.emb.row(p.a).dot(&w);
        preds.push(pred);
        actuals.push(*lag);
    }

    // ---- Metrics ----
    let mae_probe: f32 =
        preds.iter().zip(&actuals).map(|(p, a)| (p - a).abs()).sum::<f32>() / test.len() as f32;
    let mae_base: f32 =
        actuals.iter().map(|a| (mean_lag - a).abs()).sum::<f32>() / test.len() as f32;
    let rho = spearman(&preds, &actuals);
    let test_mean = actuals.iter().sum::<f32>() / actuals.len() as f32;

    println!("\n== PREDICTIVE (test) ==");
    println!("  train mean lag        : {:.2} d", mean_lag);
    println!("  test  mean lag        : {:.2} d", test_mean);
    println!("  MAE  mean-baseline    : {:.2} d", mae_base);
    println!("  MAE  content probe    : {:.2} d", mae_probe);
    let lift = mae_base - mae_probe;
    println!(
        "  probe lift vs baseline: {:+.2} d  ({:+.1}%)",
        lift,
        if mae_base > 1e-9 { 100.0 * lift / mae_base } else { 0.0 }
    );
    println!("  Spearman(pred,actual) : {:+.3}", rho);

    // ---- Archetypes: FAST vs SLOW by predicted-lag median ----
    let mut sorted_pred = preds.clone();
    sorted_pred.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted_pred[sorted_pred.len() / 2];

    let mut fast_ent: HashMap<&str, usize> = HashMap::new();
    let mut slow_ent: HashMap<&str, usize> = HashMap::new();
    for (i, (p, _)) in test.iter().enumerate() {
        let bucket = if preds[i] <= median { &mut fast_ent } else { &mut slow_ent };
        for e in &ds.docs[p.a].entities {
            *bucket.entry(e.as_str()).or_insert(0) += 1;
        }
    }

    let top = |m: &HashMap<&str, usize>, k: usize| -> Vec<(String, usize)> {
        let mut v: Vec<(String, usize)> = m.iter().map(|(e, &c)| (e.to_string(), c)).collect();
        v.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        v.truncate(k);
        v
    };

    println!(
        "\n== ARCHETYPES (test causes split at predicted-lag median {:.2}d) ==",
        median
    );
    println!("  FAST bucket — top cause entities:");
    for (e, c) in top(&fast_ent, 8) {
        println!("    {:<28} {:>4}", e, c);
    }
    println!("  SLOW bucket — top cause entities:");
    for (e, c) in top(&slow_ent, 8) {
        println!("    {:<28} {:>4}", e, c);
    }

    // ---- Sample edges: 3 fastest and 3 slowest by ACTUAL lag on test ----
    let mut by_actual: Vec<usize> = (0..test.len()).collect();
    by_actual.sort_by(|&i, &j| actuals[i].partial_cmp(&actuals[j]).unwrap());
    let show = |title: &str, order: &[usize]| {
        println!("  {title}");
        for &i in order.iter().take(3) {
            let (p, lag) = test[i];
            let a = &ds.docs[p.a];
            let b = &ds.docs[p.b];
            let ea: Vec<&str> = a.entities.iter().take(4).map(|s| s.as_str()).collect();
            let cb: Vec<&str> = b.cause_entities.iter().take(4).map(|s| s.as_str()).collect();
            println!(
                "    {:>5.1}d  [{}] {:?}  ->  [{}] cause_of {:?}",
                lag, a.date, ea, b.date, cb
            );
        }
    };
    println!("\n== SAMPLE EDGES (test, by actual lag) ==");
    show("fastest:", &by_actual);
    let rev: Vec<usize> = by_actual.iter().rev().copied().collect();
    show("slowest:", &rev);

    // ---- Honest verdict, read straight off the numbers ----
    println!("\n== VERDICT ==");
    let helps = lift > 0.0;
    let signalful = rho.abs() >= 0.10 && lift > 0.0;
    if signalful {
        println!(
            "  Content carries a real (if modest) transmission-speed signal: the probe\n  \
             beats the mean baseline by {:+.2}d and ranks lags at Spearman {:+.3}. Faster vs\n  \
             slower causes are separable by their entities — see the archetype buckets.",
            lift, rho
        );
    } else if helps {
        println!(
            "  Barely-there signal: probe edges the baseline by {:+.2}d but Spearman is only\n  \
             {:+.3}. Lag is mostly noise here — most of the transmission delay is not written\n  \
             into the cause article's content.",
            lift, rho
        );
    } else {
        println!(
            "  No usable signal: the content probe does NOT beat predicting the mean lag\n  \
             (lift {:+.2}d, Spearman {:+.3}). Transmission latency in this corpus is dominated\n  \
             by noise / exogenous timing, not by anything legible in the cause embedding.",
            lift, rho
        );
    }

    crate::report::save(
        dir,
        "latency",
        &json!({
            "mae_baseline": mae_base,
            "mae_probe": mae_probe,
            "spearman": rho,
            "n_test": preds.len(),
            "pred": preds,
            "actual": actuals,
        }),
    )?;

    Ok(())
}
