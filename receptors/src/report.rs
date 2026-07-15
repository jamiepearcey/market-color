//! Machine-readable metrics emission. Every receptor writes a small JSON bundle
//! (headline scalars + the raw per-item arrays its figure and bootstrap CIs need)
//! to data/metrics/<name>.json. scripts/make_figures.py consumes these to render
//! figures and a master table with 95% bootstrap confidence intervals — so the
//! plots and the stats derive from the exact same numbers the Rust side computed.

use anyhow::Result;
use serde_json::Value;
use std::path::Path;

pub fn save(dir: &Path, name: &str, v: &Value) -> Result<()> {
    let mdir = dir.join("metrics");
    std::fs::create_dir_all(&mdir)?;
    std::fs::write(mdir.join(format!("{name}.json")), serde_json::to_string_pretty(v)?)?;
    eprintln!("[metrics] wrote data/metrics/{name}.json");
    Ok(())
}

/// Deterministic xorshift stream (no rand dep) for reproducible resampling.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Bootstrap 95% CI for the mean of per-item contributions (accuracy, recall,
/// per-item error, ...). Returns (point, lo, hi). 2000 resamples, seed fixed.
pub fn ci_mean(items: &[f32]) -> (f32, f32, f32) {
    let n = items.len();
    if n == 0 {
        return (f32::NAN, f32::NAN, f32::NAN);
    }
    let point = items.iter().sum::<f32>() / n as f32;
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut means = Vec::with_capacity(2000);
    for _ in 0..2000 {
        let mut acc = 0.0f32;
        for _ in 0..n {
            let idx = (xorshift(&mut s) as usize) % n;
            acc += items[idx];
        }
        means.push(acc / n as f32);
    }
    means.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (point, means[50], means[1949]) // 2.5th / 97.5th percentile of 2000
}

/// Bootstrap 95% CI for AUC (Mann-Whitney) from paired (score, label) data,
/// resampling items with replacement. Returns (point, lo, hi).
pub fn ci_auc(scores: &[f32], labels: &[u8]) -> (f32, f32, f32) {
    let auc = |sc: &[f32], la: &[u8]| -> f32 {
        let (mut c, mut n) = (0.0f64, 0.0f64);
        for i in 0..sc.len() {
            if la[i] == 0 {
                continue;
            }
            for j in 0..sc.len() {
                if la[j] == 1 {
                    continue;
                }
                n += 1.0;
                if sc[i] > sc[j] {
                    c += 1.0;
                } else if sc[i] == sc[j] {
                    c += 0.5;
                }
            }
        }
        if n > 0.0 {
            (c / n) as f32
        } else {
            0.5
        }
    };
    let point = auc(scores, labels);
    let n = scores.len();
    let mut s = 0xD1B54A32D192ED03u64;
    let mut vals = Vec::with_capacity(600);
    // 600 resamples (AUC is O(n^2); keep it affordable)
    for _ in 0..600 {
        let (mut sc, mut la) = (Vec::with_capacity(n), Vec::with_capacity(n));
        for _ in 0..n {
            let idx = (xorshift(&mut s) as usize) % n;
            sc.push(scores[idx]);
            la.push(labels[idx]);
        }
        vals.push(auc(&sc, &la));
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (point, vals[15], vals[584]) // ~2.5th / 97.5th of 600
}
