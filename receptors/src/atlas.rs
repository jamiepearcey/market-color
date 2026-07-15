//! A2 — Concept-axis atlas. A battery of interpretable 1-D directions, each a
//! difference-of-means axis between two groups defined by free labels already on
//! every fact (direction, predicate, magnitude). Projecting every fact onto all
//! of them turns the 384-d black box into a handful of readable dials.
//!
//! For each axis we report (a) held-out separation AUC — does the axis actually
//! order unseen pos/neg facts? — and (b) the facts at both extremes, so a human
//! can name and sanity-check it. We also print each desk's mean coordinate on
//! every axis: a qualitative fingerprint of the corpus.

use crate::facts::{Fact, FactSet};
use crate::linalg;
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// An axis: name + a classifier returning +1 (positive pole), -1 (negative
/// pole), 0 (not part of this contrast).
struct Axis {
    name: &'static str,
    pos: &'static str,
    neg: &'static str,
    cls: fn(&Fact) -> i8,
}

fn axes() -> Vec<Axis> {
    fn has(f: &Fact, p: &str) -> bool {
        f.predicate == p
    }
    vec![
        Axis { name: "polarity", pos: "bullish", neg: "bearish",
            cls: |f| if f.direction > 0.5 { 1 } else if f.direction < -0.5 { -1 } else { 0 } },
        Axis { name: "certainty", pos: "hard-fact", neg: "speculative",
            cls: |f| if has(f,"event")||has(f,"statement")||has(f,"price_move") { 1 }
                     else if has(f,"forecast") { -1 } else { 0 } },
        Axis { name: "supply/demand", pos: "supply-side", neg: "demand-side",
            cls: |f| if has(f,"supply_change")||has(f,"production_change") { 1 }
                     else if has(f,"demand_change") { -1 } else { 0 } },
        Axis { name: "official/market", pos: "policy-action", neg: "market-move",
            cls: |f| if has(f,"policy_action")||has(f,"sanction") { 1 }
                     else if has(f,"price_move") { -1 } else { 0 } },
        Axis { name: "horizon", pos: "anticipated", neg: "realized",
            cls: |f| if has(f,"forecast") { 1 }
                     else if has(f,"event")||has(f,"price_move") { -1 } else { 0 } },
        Axis { name: "deal/friction", pos: "deal", neg: "sanction/conflict",
            cls: |f| if has(f,"deal_or_contract") { 1 }
                     else if has(f,"sanction") { -1 } else { 0 } },
    ]
}

/// Deterministic 70/30 split by fact index (stable, avoids the missing-epoch bias).
fn is_test(i: usize) -> bool {
    (i.wrapping_mul(2654435761) % 100) < 30
}

/// Mann-Whitney AUC = P(pos scored > neg), from rank sums.
fn auc(pos: &[f32], neg: &[f32]) -> f32 {
    let mut all: Vec<(f32, u8)> = pos.iter().map(|&s| (s, 1u8)).chain(neg.iter().map(|&s| (s, 0u8))).collect();
    all.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    // average ranks (1-based), handle ties
    let mut rank_sum_pos = 0.0f64;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i;
        while j + 1 < all.len() && all[j + 1].0 == all[i].0 {
            j += 1;
        }
        let avg_rank = (i + j + 2) as f64 / 2.0; // mean of (i+1..=j+1)
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

/// Bootstrap CI for the pos-vs-neg AUC via the shared resampler.
fn auc_ci(pos: &[f32], neg: &[f32]) -> (f32, f32, f32) {
    let mut scores = Vec::with_capacity(pos.len() + neg.len());
    let mut labels = Vec::with_capacity(pos.len() + neg.len());
    for &p in pos {
        scores.push(p);
        labels.push(1u8);
    }
    for &n in neg {
        scores.push(n);
        labels.push(0u8);
    }
    crate::report::ci_auc(&scores, &labels)
}

fn mean_rows(emb: &Array2<f32>, idx: &[usize]) -> Array1<f32> {
    let d = emb.ncols();
    let mut m = Array1::<f32>::zeros(d);
    for &i in idx {
        m += &emb.row(i);
    }
    if !idx.is_empty() {
        m /= idx.len() as f32;
    }
    m
}

pub fn run(dir: &Path, _lambda: f32) -> Result<()> {
    let mut fs = FactSet::load(dir)?;
    linalg::normalize_rows(&mut fs.emb);
    println!("concept-axis atlas: {} facts x {} dims\n", fs.n(), fs.d());

    let axes = axes();
    // learned unit axis per contrast (fit on TRAIN)
    let mut learned: Vec<(usize, Array1<f32>)> = Vec::new();
    let mut emitted: Vec<Value> = Vec::new();

    println!("== axis separation (held-out) & poles ==");
    println!("  {:<16} {:>8} {:>7} {:>7}   poles", "axis", "test-AUC", "n+", "n-");
    for (ai, ax) in axes.iter().enumerate() {
        let (mut ptr, mut ntr): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        let (mut pte, mut nte): (Vec<usize>, Vec<usize>) = (Vec::new(), Vec::new());
        for (i, f) in fs.facts.iter().enumerate() {
            match (ax.cls)(f) {
                1 => (if is_test(i) { &mut pte } else { &mut ptr }).push(i),
                -1 => (if is_test(i) { &mut nte } else { &mut ntr }).push(i),
                _ => {}
            }
        }
        if ptr.len() < 20 || ntr.len() < 20 {
            println!("  {:<16} {:>8} (insufficient support)", ax.name, "-");
            continue;
        }
        let mut u = &mean_rows(&fs.emb, &ptr) - &mean_rows(&fs.emb, &ntr);
        let n = u.dot(&u).sqrt();
        if n > 1e-9 {
            u.mapv_inplace(|x| x / n);
        }
        let ps: Vec<f32> = pte.iter().map(|&i| fs.emb.row(i).dot(&u)).collect();
        let ns: Vec<f32> = nte.iter().map(|&i| fs.emb.row(i).dot(&u)).collect();
        let a = auc(&ps, &ns);
        let (ci_pt, ci_lo, ci_hi) = auc_ci(&ps, &ns);
        println!(
            "  {:<16} {:>8.3} [{:.3},{:.3}] {:>7} {:>7}   +{} / -{}",
            ax.name, a, ci_lo, ci_hi, pte.len(), nte.len(), ax.pos, ax.neg
        );
        emitted.push(json!({
            "axis": ax.name, "pos": ax.pos, "neg": ax.neg,
            "auc": ci_pt, "ci": [ci_lo, ci_hi],
            "n_pos": pte.len(), "n_neg": nte.len(),
        }));
        learned.push((ai, u));
    }

    // ---- extremes for the two most-separating axes (interpretability) ----
    println!("\n== reading the poles (top facts at each extreme) ==");
    for (ai, u) in learned.iter().take(3) {
        let ax = &axes[*ai];
        let proj: Vec<(f32, usize)> = (0..fs.n()).map(|i| (fs.emb.row(i).dot(u), i)).collect();
        let mut sorted = proj.clone();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        println!("\n  [{}]  + {}", ax.name, ax.pos);
        for &(s, i) in sorted.iter().take(3) {
            println!("    {:+.2}  {}", s, clip(&fs.facts[i].claim));
        }
        println!("        - {}", ax.neg);
        for &(s, i) in sorted.iter().rev().take(3) {
            println!("    {:+.2}  {}", s, clip(&fs.facts[i].claim));
        }
    }

    // ---- per-desk fingerprint: mean coordinate on each axis ----
    println!("\n== per-desk fingerprint (mean coordinate per axis; blank = n/a) ==");
    let names: Vec<&str> = learned.iter().map(|(ai, _)| axes[*ai].name).collect();
    print!("  {:<12}", "desk");
    for n in &names {
        print!(" {:>12}", &n[..n.len().min(12)]);
    }
    println!();
    let mut by_desk: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, f) in fs.facts.iter().enumerate() {
        by_desk.entry(f.desk.as_str()).or_default().push(i);
    }
    let mut fp_desks: Vec<String> = Vec::new();
    let mut fp_matrix: Vec<Vec<f32>> = Vec::new();
    for (desk, idx) in by_desk.iter() {
        if idx.len() < 50 {
            continue;
        }
        print!("  {:<12}", &desk[..desk.len().min(12)]);
        let mut row = Vec::with_capacity(learned.len());
        for (_, u) in &learned {
            let m: f32 = idx.iter().map(|&i| fs.emb.row(i).dot(u)).sum::<f32>() / idx.len() as f32;
            print!(" {:>12.3}", m);
            row.push(m);
        }
        println!();
        fp_desks.push(desk.to_string());
        fp_matrix.push(row);
    }

    crate::report::save(
        dir,
        "atlas",
        &json!({
            "axes": emitted,
            "fingerprint": {"desks": fp_desks, "axis_names": names, "matrix": fp_matrix},
        }),
    )?;
    Ok(())
}

fn clip(s: &str) -> String {
    let s = s.trim();
    s.chars().take(96).collect()
}
