//! C1 — CONSENSUS vs CONTRARIAN. On any given (date, desk) a cluster of facts
//! forms a local consensus: the centroid of their claim embeddings is the
//! "house view" for that desk that day. A fact's distance from its own group's
//! centroid measures how much it dissents. `contrarian[i] = 1 - cos(emb_i,
//! centroid)`: near 0 = squarely on-message, near 1 = out on a limb.
//!
//! Consensus needs support, so we only score facts in groups of >=5. We report
//! the distribution of dissent, the loudest dissenters (and a few conformists
//! for contrast), then ask the market: do the top-decile contrarian facts sit
//! ahead of bigger subsequent price moves than the rest? The caveat runs
//! through the whole thing — contrarian-by-embedding is dissent in *wording*,
//! not a claim to being contrarian-and-right.

use crate::facts::FactSet;
use crate::linalg;
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde::Deserialize;
use serde_json::json;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

#[derive(Debug, Deserialize)]
struct PricePair {
    doc_id: String,
    #[allow(dead_code)]
    symbol: String,
    #[allow(dead_code)]
    sign: i32,
    zscore: f64,
    #[allow(dead_code)]
    move_date: String,
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

/// Percentile of a pre-sorted (ascending) slice, nearest-rank.
fn pct(sorted: &[f32], p: f32) -> f32 {
    if sorted.is_empty() {
        return f32::NAN;
    }
    let rank = ((p / 100.0) * (sorted.len() as f32 - 1.0)).round() as usize;
    sorted[rank.min(sorted.len() - 1)]
}

fn clip(s: &str) -> String {
    let s = s.trim();
    s.chars().take(88).collect()
}

pub fn run(dir: &Path) -> Result<()> {
    let mut fs = FactSet::load(dir)?;
    linalg::normalize_rows(&mut fs.emb);
    println!(
        "consensus receptor (C1): {} facts x {} dims\n",
        fs.n(),
        fs.d()
    );

    // ---- group by (date, desk) ----
    let mut groups: BTreeMap<(String, String), Vec<usize>> = BTreeMap::new();
    for (i, f) in fs.facts.iter().enumerate() {
        if f.date.is_empty() {
            continue;
        }
        groups
            .entry((f.date.clone(), f.desk.clone()))
            .or_default()
            .push(i);
    }

    const MIN: usize = 5;
    // contrarian score per scored fact; -1 = unscored (group too small / no date)
    let mut score = vec![-1.0f32; fs.n()];
    let mut n_groups_scored = 0usize;
    for idx in groups.values() {
        if idx.len() < MIN {
            continue;
        }
        n_groups_scored += 1;
        // centroid of L2-normalised rows; raw dot with normalised row_i is then
        // cos(emb_i, centroid) up to the (per-fact-identical) centroid norm — we
        // normalise the centroid so it's an honest cosine.
        let mut c = mean_rows(&fs.emb, idx);
        let cn = c.dot(&c).sqrt();
        if cn > 1e-9 {
            c.mapv_inplace(|x| x / cn);
        }
        for &i in idx {
            let cos = fs.emb.row(i).dot(&c);
            score[i] = 1.0 - cos;
        }
    }

    let scored: Vec<usize> = (0..fs.n()).filter(|&i| score[i] >= 0.0).collect();
    let n_scored = scored.len();
    println!(
        "grouped by (date, desk): {} groups with >={} facts scored, {} facts scored ({} skipped, no local consensus)",
        n_groups_scored,
        MIN,
        n_scored,
        fs.n() - n_scored
    );
    if n_scored == 0 {
        println!("\nno group reached the support threshold — nothing to say.");
        return Ok(());
    }

    // ---- distribution of contrarian scores ----
    let mut vals: Vec<f32> = scored.iter().map(|&i| score[i]).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
    println!("\n== contrarian-score distribution (0 = on-message, 1 = orthogonal to house view) ==");
    println!(
        "  mean {:.3}   p10 {:.3}  p25 {:.3}  p50 {:.3}  p75 {:.3}  p90 {:.3}  p99 {:.3}",
        mean,
        pct(&vals, 10.0),
        pct(&vals, 25.0),
        pct(&vals, 50.0),
        pct(&vals, 75.0),
        pct(&vals, 90.0),
        pct(&vals, 99.0),
    );

    // ---- extremes ----
    let mut order = scored.clone();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap());

    println!("\n== top-12 most-contrarian facts (dissent in wording vs their desk/day) ==");
    println!("  {:>6}  {:<8} {:<10}  claim", "score", "desk", "date");
    for &i in order.iter().take(12) {
        let f = &fs.facts[i];
        println!(
            "  {:>6.3}  {:<8} {:<10}  {}",
            score[i],
            &f.desk[..f.desk.len().min(8)],
            &f.date[..f.date.len().min(10)],
            clip(&f.claim)
        );
    }

    println!("\n== 4 most-consensus facts (lowest dissent — squarely on the house view) ==");
    for &i in order.iter().rev().take(4) {
        let f = &fs.facts[i];
        println!(
            "  {:>6.3}  {:<8} {:<10}  {}",
            score[i],
            &f.desk[..f.desk.len().min(8)],
            &f.date[..f.date.len().min(10)],
            clip(&f.claim)
        );
    }

    // ---- market test ----
    let mkt = market_test(dir, &fs, &score, &scored, &mut vals)?;

    // ---- metrics JSON ----
    let contrarian_scores: Vec<f32> = vals.clone();
    let (top_decile_z, rest_z, mean_top, mean_rest, ratio) = match mkt {
        Some(m) => (
            m.top_decile_z,
            m.rest_z,
            m.mean_top,
            m.mean_rest,
            m.ratio,
        ),
        None => (Vec::new(), Vec::new(), f32::NAN, f32::NAN, f32::NAN),
    };
    crate::report::save(
        dir,
        "consensus",
        &json!({
            "contrarian_scores": contrarian_scores,
            "percentiles": {
                "p10": pct(&vals, 10.0),
                "p25": pct(&vals, 25.0),
                "p50": pct(&vals, 50.0),
                "p75": pct(&vals, 75.0),
                "p90": pct(&vals, 90.0),
                "p99": pct(&vals, 99.0),
                "mean": mean,
            },
            "market": {
                "topdecile_z": top_decile_z,
                "rest_z": rest_z,
                "mean_top": mean_top,
                "mean_rest": mean_rest,
                "ratio": ratio,
            },
        }),
    )?;

    println!("\n== reading this ==");
    println!("  The dissent axis is embedding-geometric: it measures how far a fact's *phrasing*");
    println!("  sits from the day's desk centroid, NOT whether the dissenter turned out right.");
    println!("  A high score can be genuine off-consensus insight, an off-topic aside that landed");
    println!("  in the same (date, desk) bin, or a rephrasing quirk. Thin (date, desk) groups are");
    println!("  unscored (no consensus defined). Read the market ratio as suggestive, not causal:");
    println!("  we only see facts whose doc later moved a price, and |zscore| is a coincident,");
    println!("  survivorship-flavoured proxy — contrarian-by-embedding is not contrarian-by-truth.");

    Ok(())
}

struct MarketResult {
    top_decile_z: Vec<f32>,
    rest_z: Vec<f32>,
    mean_top: f32,
    mean_rest: f32,
    ratio: f32,
}

fn market_test(
    dir: &Path,
    fs: &FactSet,
    score: &[f32],
    scored: &[usize],
    sorted_vals: &mut [f32],
) -> Result<Option<MarketResult>> {
    let path = dir.join("price_pairs.jsonl");
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(_) => {
            println!("\n== market test == (skipped: {} not found)", path.display());
            return Ok(None);
        }
    };
    // per doc_id -> max |zscore| of any subsequent move
    let mut move_by_doc: HashMap<String, f64> = HashMap::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let p: PricePair = match serde_json::from_str(line) {
            Ok(p) => p,
            Err(_) => continue,
        };
        let e = move_by_doc.entry(p.doc_id).or_insert(0.0);
        *e = e.max(p.zscore.abs());
    }
    if move_by_doc.is_empty() {
        println!("\n== market test == (skipped: no price pairs parsed)");
        return Ok(None);
    }

    // top-decile contrarian threshold among scored facts
    let thr = pct(sorted_vals, 90.0);

    let (mut top_moves, mut rest_moves): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut n_docmatched = 0usize;
    for &i in scored {
        if let Some(&mv) = move_by_doc.get(&fs.facts[i].doc_id) {
            n_docmatched += 1;
            if score[i] >= thr {
                top_moves.push(mv);
            } else {
                rest_moves.push(mv);
            }
        }
    }

    println!("\n== market test: do contrarian facts precede bigger moves? ==");
    if top_moves.is_empty() || rest_moves.is_empty() {
        println!(
            "  {} scored facts matched a doc with a price move, but one bucket was empty — inconclusive.",
            n_docmatched
        );
        return Ok(None);
    }
    let m = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (mt, mr) = (m(&top_moves), m(&rest_moves));
    println!(
        "  {} scored facts land in docs that later moved a price (top-decile thr = {:.3} dissent)",
        n_docmatched, thr
    );
    println!(
        "  mean |zscore|:  top-decile contrarian = {:.3} (n={})   rest = {:.3} (n={})   ratio = {:.2}x",
        mt,
        top_moves.len(),
        mr,
        rest_moves.len(),
        if mr.abs() > 1e-9 { mt / mr } else { f64::NAN }
    );
    if mt > mr {
        println!("  -> contrarian facts sit ahead of LARGER moves here (ratio > 1).");
    } else {
        println!("  -> contrarian facts do NOT precede larger moves (ratio <= 1); dissent looks like noise.");
    }
    let ratio = if mr.abs() > 1e-9 { mt / mr } else { f64::NAN };
    Ok(Some(MarketResult {
        top_decile_z: top_moves.iter().map(|&v| v as f32).collect(),
        rest_z: rest_moves.iter().map(|&v| v as f32).collect(),
        mean_top: mt as f32,
        mean_rest: mr as f32,
        ratio: ratio as f32,
    }))
}
