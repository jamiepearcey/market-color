//! C2 — MARKET REGIME. A single interpretable risk-on / risk-off axis, learned
//! not from labels on the facts but from the *price tape*: for every trading day
//! we take the mean embedding of that day's atomic facts (the "day-centroid" —
//! what the desks were actually talking about) and regress it onto a precomputed
//! composite market signal (equities up + vol down + dollar down = risk-on).
//!
//! The fitted weight vector `w` is a direction in embedding space; projecting any
//! fact onto it reads off "how risk-on does this claim sound". We hold out the
//! latest 30% of days, report Spearman + sign-agreement there, and print the poles
//! so the axis can be named and sanity-checked. n_days is small — exploratory.

use crate::facts::FactSet;
use crate::linalg;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::json;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Debug, Deserialize)]
struct MarketDay {
    date: String,
    #[serde(default)]
    #[allow(dead_code)]
    spx_ret: f32,
    #[serde(default)]
    #[allow(dead_code)]
    vix_ret: f32,
    #[serde(default)]
    #[allow(dead_code)]
    dxy_ret: f32,
    #[serde(default)]
    #[allow(dead_code)]
    y10_ret: f32,
    #[serde(default)]
    risk_on: f32,
}

/// Spearman rank correlation: Pearson on the (tie-averaged) ranks of each vector.
fn spearman(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return 0.0;
    }
    let ra = ranks(a);
    let rb = ranks(b);
    pearson(&ra, &rb)
}

/// Tie-averaged ranks (1-based).
fn ranks(v: &[f32]) -> Vec<f32> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
    let mut r = vec![0.0f32; v.len()];
    let mut i = 0usize;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j + 2) as f32 / 2.0; // mean of (i+1..=j+1)
        for k in i..=j {
            r[idx[k]] = avg;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    let ma = a.iter().sum::<f32>() / n;
    let mb = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut va = 0.0f32;
    let mut vb = 0.0f32;
    for i in 0..a.len() {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    if va <= 1e-12 || vb <= 1e-12 {
        return 0.0;
    }
    cov / (va.sqrt() * vb.sqrt())
}

fn clip(s: &str) -> String {
    let s = s.trim();
    s.chars().take(90).collect()
}

pub fn run(dir: &Path) -> Result<()> {
    let mut fs = FactSet::load(dir)?;
    linalg::normalize_rows(&mut fs.emb);
    println!(
        "market-regime receptor (C2): {} facts x {} dims\n",
        fs.n(),
        fs.d()
    );

    // ---- load the market tape ----
    let mpath = dir.join("market_daily.jsonl");
    let text = std::fs::read_to_string(&mpath)
        .with_context(|| format!("read {}", mpath.display()))?;
    let mut market: BTreeMap<String, f32> = BTreeMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let md: MarketDay = serde_json::from_str(line).context("parse market_daily line")?;
        market.insert(md.date, md.risk_on);
    }
    println!("loaded {} market days from market_daily.jsonl", market.len());

    // ---- group fact indices by date ----
    let d = fs.d();
    let mut by_date: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    for (i, f) in fs.facts.iter().enumerate() {
        if f.date.is_empty() {
            continue;
        }
        by_date.entry(f.date.clone()).or_default().push(i);
    }

    // ---- day-centroids for dates with >=3 facts AND a market row ----
    // (dates sorted ascending because BTreeMap iterates in key order, and
    //  "YYYY-MM-DD" sorts lexicographically == chronologically)
    let mut dates: Vec<String> = Vec::new();
    let mut rows: Vec<Array1<f32>> = Vec::new();
    let mut y: Vec<f32> = Vec::new();
    for (date, idx) in by_date.iter() {
        if idx.len() < 3 {
            continue;
        }
        let Some(&risk_on) = market.get(date) else {
            continue;
        };
        let mut c = Array1::<f32>::zeros(d);
        for &i in idx {
            c += &fs.emb.row(i);
        }
        c /= idx.len() as f32;
        // row-normalise the centroid
        let nrm = c.dot(&c).sqrt();
        if nrm > 1e-9 {
            c.mapv_inplace(|x| x / nrm);
        }
        dates.push(date.clone());
        rows.push(c);
        y.push(risk_on);
    }

    let n_days = dates.len();
    println!(
        "usable days (>=3 facts AND in market tape): {}\n",
        n_days
    );
    if n_days < 6 {
        println!("too few usable days ({}) to split & fit; aborting cleanly.", n_days);
        return Ok(());
    }

    // stack into X (n_days, D)
    let mut x = Array2::<f32>::zeros((n_days, d));
    for (r, row) in rows.iter().enumerate() {
        x.row_mut(r).assign(row);
    }
    let yv = Array1::from(y.clone());

    // ---- temporal split: earliest 70% train, latest 30% test ----
    let n_train = ((n_days as f32) * 0.70).round() as usize;
    let n_train = n_train.clamp(1, n_days - 1);
    let x_train = x.slice(ndarray::s![..n_train, ..]).to_owned();
    let y_train = yv.slice(ndarray::s![..n_train]).to_owned();

    let w = linalg::ridge_probe(&x_train, &y_train, 0.1)
        .context("ridge_probe on day-centroids")?;

    // ---- evaluate on held-out latest days ----
    let mut pred: Vec<f32> = Vec::new();
    let mut actual: Vec<f32> = Vec::new();
    for r in n_train..n_days {
        pred.push(x.row(r).dot(&w));
        actual.push(y[r]);
    }
    let rho = spearman(&pred, &actual);
    // sign-agreement: does predicted regime direction match actual regime sign?
    let mut agree = 0usize;
    let mut counted = 0usize;
    // centre predictions & actuals at their test-set medians so "direction" is
    // relative to the held-out period, not to an arbitrary zero.
    let pm = median(&pred);
    let am = median(&actual);
    for i in 0..pred.len() {
        let ps = pred[i] - pm;
        let as_ = actual[i] - am;
        if ps.abs() < 1e-9 || as_.abs() < 1e-9 {
            continue;
        }
        counted += 1;
        if ps.signum() == as_.signum() {
            agree += 1;
        }
    }
    let sign_pct = if counted > 0 {
        100.0 * agree as f32 / counted as f32
    } else {
        0.0
    };

    println!("== held-out evaluation (latest {} days) ==", pred.len());
    println!("  train days : {}", n_train);
    println!("  test  days : {}", pred.len());
    println!("  Spearman(pred, risk_on)   : {:+.3}", rho);
    println!(
        "  sign-agreement (dir match): {:.0}%  ({}/{} decisive)",
        sign_pct, agree, counted
    );

    // ---- project ALL facts onto the learned axis -> interpretable poles ----
    let scores: Vec<f32> = (0..fs.n()).map(|i| fs.emb.row(i).dot(&w)).collect();
    let mut order: Vec<usize> = (0..fs.n()).collect();
    order.sort_by(|&i, &j| scores[j].partial_cmp(&scores[i]).unwrap());

    let label = |i: usize| -> String {
        let f = &fs.facts[i];
        let ent = if !f.entities.is_empty() {
            format!("[{}] ", f.entities.join(", "))
        } else {
            String::new()
        };
        format!("{}{}", ent, clip(&f.claim))
    };

    println!("\n== the learned axis (project every fact onto w) ==");
    println!("  RISK-ON pole (highest score):");
    for &i in order.iter().take(8) {
        println!("    {:+.3}  {}", scores[i], label(i));
    }
    println!("  RISK-OFF pole (lowest score):");
    for &i in order.iter().rev().take(8) {
        println!("    {:+.3}  {}", scores[i], label(i));
    }

    // ---- compact timeline over the test dates ----
    println!("\n== test-window timeline (date | actual risk_on | projected day score) ==");
    let show = pred.len().min(12);
    for k in 0..show {
        let r = n_train + k;
        println!(
            "  {}   actual {:+.3}   proj {:+.3}",
            dates[r], actual[k], pred[k]
        );
    }
    if pred.len() > show {
        println!("  ... ({} more test days)", pred.len() - show);
    }

    // ---- honest note ----
    println!("\n== honest read ==");
    let tracks = rho > 0.15 && sign_pct >= 55.0;
    println!(
        "  n_days is small ({} total, {} held-out) -> treat as EXPLORATORY, not a backtest.",
        n_days,
        pred.len()
    );
    if tracks {
        println!(
            "  Verdict: fact content DOES track the market regime on held-out days \
             (Spearman {:+.2}, {:.0}% sign-agreement). The risk-on axis is coherent \
             (see poles above), but the sample is too thin to trade on.",
            rho, sign_pct
        );
    } else {
        println!(
            "  Verdict: fact content does NOT cleanly track the market regime out-of-sample \
             (Spearman {:+.2}, {:.0}% sign-agreement). The in-sample axis is readable, but the \
             day-centroid signal is weak/noisy on held-out days at this corpus size.",
            rho, sign_pct
        );
    }

    // ---- emit metrics JSON: full timeline over ALL usable days (train+test) ----
    // project each day-centroid onto the learned axis w so train days also get a
    // projected score, giving a complete timeline.
    let timeline: Vec<serde_json::Value> = (0..n_days)
        .map(|r| {
            let proj = x.row(r).dot(&w);
            json!({
                "date": dates[r],
                "actual": y[r],
                "proj": proj,
                "is_test": r >= n_train,
            })
        })
        .collect();
    crate::report::save(
        dir,
        "regime",
        &json!({
            "spearman": rho,
            "sign_agreement": sign_pct,
            "n_days": n_days,
            "n_test": pred.len(),
            "timeline": timeline,
        }),
    )?;

    Ok(())
}

fn median(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        0.5 * (s[n / 2 - 1] + s[n / 2])
    }
}
