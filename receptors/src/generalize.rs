//! Generalization spot-checks for the transport operator `W`.
//!
//! The training corpus is only ~2 weeks wide, so the headline numbers (random
//! effect-hash split) could be flattered by *interval over-representation*: a
//! handful of loud stories (a rate decision, a heatwave, a tariff round) recur
//! all fortnight, and a random split trains and tests inside the same regime.
//! If W has merely memorised "docs about $HOT_TOPIC precede other docs about
//! $HOT_TOPIC", it will look great here and fail on anything genuinely new.
//!
//! These four probes each attack that specific failure mode, reusing the exact
//! mining / ridge-fit / eval code the headline uses:
//!
//!   E1  RANDOM split      — reproduce the headline as an anchor.
//!   E2  TEMPORAL split    — fit on the earlier half, predict cause->effect in
//!                           the unseen later half. The direct "does it survive
//!                           into time it never saw" test.
//!   E3  SPECIFICITY strat — split the SAME test pairs by the rarity of their
//!                           linking cause-entity. If the edge only holds on
//!                           common (hub / over-represented) entities and
//!                           collapses on rare ones, W is riding the loud topics.
//!   E4  SHUFFLE null      — refit W on permuted (a,b) pairings. Whatever
//!                           accuracy survives is geometry + time artifact, not
//!                           learned cause->effect structure. The honest floor.

use crate::data::Dataset;
use crate::pairs::{self, Pair};
use crate::{eval, linalg};
use anyhow::Result;
use ndarray::Array2;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Fit W on the train subset of `pairs`, return the transported cause rows.
fn fit_ahat(emb: &Array2<f32>, pairs: &[Pair], lambda: f32) -> Result<Array2<f32>> {
    let d = emb.ncols();
    let n_train = pairs.iter().filter(|p| !p.is_test).count();
    let mut a = Array2::<f32>::zeros((n_train, d));
    let mut b = Array2::<f32>::zeros((n_train, d));
    for (r, p) in pairs.iter().filter(|p| !p.is_test).enumerate() {
        a.row_mut(r).assign(&emb.row(p.a));
        b.row_mut(r).assign(&emb.row(p.b));
    }
    let w = linalg::fit_operator(&a, &b, lambda)?;
    Ok(linalg::transport(emb, &w))
}

/// Direction accuracy over an explicit subset of test pairs (content only).
fn dir_acc_on(subset: &[&Pair], ahat: &Array2<f32>, emb: &Array2<f32>) -> (usize, f32) {
    let (mut correct, mut n) = (0usize, 0usize);
    for p in subset {
        let fwd = ahat.row(p.a).dot(&emb.row(p.b));
        let rev = ahat.row(p.b).dot(&emb.row(p.a));
        if fwd > rev {
            correct += 1;
        }
        n += 1;
    }
    (n, if n > 0 { correct as f32 / n as f32 } else { 0.0 })
}

/// One line: fit, then report direction acc + retrieval R@10/MRR vs cosine.
fn run_split(label: &str, emb: &Array2<f32>, pairs: &[Pair], epochs: &[i64], lambda: f32) -> Result<()> {
    let ahat = fit_ahat(emb, pairs, lambda)?;
    let test: Vec<&Pair> = pairs.iter().filter(|p| p.is_test).collect();
    let (n, acc) = dir_acc_on(&test, &ahat, emb);

    let cos = eval::retrieval("cos", pairs, epochs, |a, j| emb.row(a).dot(&emb.row(j)));
    let trn = eval::retrieval("trn", pairs, epochs, |a, j| ahat.row(a).dot(&emb.row(j)));
    let lift = if cos.recall_at_10 > 0.0 {
        100.0 * (trn.recall_at_10 - cos.recall_at_10) / cos.recall_at_10
    } else {
        0.0
    };
    println!(
        "  {:<18} pairs={:<5} dir={:.3}   R@10 cos {:.3} -> W {:.3} ({:+.0}%)   MRR cos {:.3} -> W {:.3}",
        label, n, acc, cos.recall_at_10, trn.recall_at_10, lift, cos.mrr, trn.mrr
    );
    Ok(())
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64, test_frac: u64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let emb = ds.emb.clone();
    let epochs: Vec<i64> = ds.docs.iter().map(|d| d.published_epoch).collect();
    let n = ds.n();

    let mined = pairs::mine(&ds, specific_frac, test_frac);
    let base = &mined.pairs;
    println!(
        "loaded {} docs (span {} .. {}), {} mined pairs, lambda={}",
        n,
        ds.docs.first().map(|d| d.date.as_str()).unwrap_or("?"),
        ds.docs.last().map(|d| d.date.as_str()).unwrap_or("?"),
        base.len(),
        lambda,
    );
    println!("\n== generalization spot-checks (direction 0.5=chance; R@10 vs cosine) ==");

    // ---- E1: RANDOM split (headline anchor) ----
    run_split("E1 random", &emb, base, &epochs, lambda)?;

    // ---- E2: TEMPORAL split (train past -> predict future) ----
    // Cutoff at the 55th percentile of effect epochs; test = effect after cutoff.
    let mut eff_epochs: Vec<i64> = base.iter().map(|p| epochs[p.b]).collect();
    eff_epochs.sort_unstable();
    let cutoff = eff_epochs[(eff_epochs.len() as f64 * 0.55) as usize];
    let temporal: Vec<Pair> = base
        .iter()
        .map(|p| Pair { is_test: epochs[p.b] >= cutoff, ..*p })
        .collect();
    let n_tr = temporal.iter().filter(|p| !p.is_test).count();
    let n_te = temporal.len() - n_tr;
    println!(
        "  (temporal cutoff epoch {cutoff}: {n_tr} train pairs fully before, {n_te} test effects after)"
    );
    run_split("E2 temporal", &emb, &temporal, &epochs, lambda)?;

    // ---- E3: SPECIFICITY stratification on the random test set ----
    // For each test pair, rarity = min doc-frequency among the cause-entities of
    // b that a is actually about (the entity the edge was mined on). Low df =
    // rare/specific (cannot be an over-represented hub); high df = common.
    let mut df: HashMap<&str, usize> = HashMap::new();
    for d in &ds.docs {
        for e in &d.entities {
            *df.entry(e.as_str()).or_insert(0) += 1;
        }
    }
    let a_entset: Vec<HashSet<&str>> = ds
        .docs
        .iter()
        .map(|d| d.entities.iter().map(|s| s.as_str()).collect())
        .collect();
    let mut rated: Vec<(usize, &Pair)> = Vec::new(); // (min_df, pair)
    for p in base.iter().filter(|p| p.is_test) {
        let mut best: Option<usize> = None;
        for c in &ds.docs[p.b].cause_entities {
            if a_entset[p.a].contains(c.as_str()) {
                if let Some(&d) = df.get(c.as_str()) {
                    best = Some(best.map_or(d, |m| m.min(d)));
                }
            }
        }
        if let Some(d) = best {
            rated.push((d, p));
        }
    }
    rated.sort_by_key(|&(d, _)| d);
    let ahat_rand = fit_ahat(&emb, base, lambda)?;
    if rated.len() >= 6 {
        let t = rated.len() / 3;
        let rare: Vec<&Pair> = rated[..t].iter().map(|&(_, p)| p).collect();
        let hub: Vec<&Pair> = rated[rated.len() - t..].iter().map(|&(_, p)| p).collect();
        let rare_df = (rated[0].0, rated[t.saturating_sub(1)].0);
        let hub_df = (rated[rated.len() - t].0, rated[rated.len() - 1].0);
        let (nr, ar) = dir_acc_on(&rare, &ahat_rand, &emb);
        let (nh, ah) = dir_acc_on(&hub, &ahat_rand, &emb);
        println!(
            "  E3 RARE  entity   pairs={nr:<5} dir={ar:.3}   (linking-entity df {}..{}  = specific, NOT over-represented)",
            rare_df.0, rare_df.1
        );
        println!(
            "  E3 HUB   entity   pairs={nh:<5} dir={ah:.3}   (linking-entity df {}..{}  = common / over-represented)",
            hub_df.0, hub_df.1
        );
    }

    // ---- E4: SHUFFLE null (permuted correspondence) ----
    // Keep the train cause rows; pair each with a DIFFERENT train effect row (a
    // fixed-stride permutation). Breaks true a->b correspondence, preserves the
    // marginal row distributions and the time-ordering of the eval pool.
    let train_idx: Vec<usize> = base
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.is_test)
        .map(|(i, _)| i)
        .collect();
    let m = train_idx.len();
    let stride = m / 3 + 1; // coprime-ish, deterministic
    let mut shuffled = base.clone();
    for (k, &i) in train_idx.iter().enumerate() {
        let donor = base[train_idx[(k + stride) % m]].b;
        shuffled[i].b = donor; // cause a_i now points at an unrelated effect
    }
    let ahat_null = fit_ahat(&emb, &shuffled, lambda)?;
    let test_real: Vec<&Pair> = base.iter().filter(|p| p.is_test).collect();
    let (nn, an) = dir_acc_on(&test_real, &ahat_null, &emb);
    let cos = eval::retrieval("cos", base, &epochs, |a, j| emb.row(a).dot(&emb.row(j)));
    let nullr = eval::retrieval("null", base, &epochs, |a, j| ahat_null.row(a).dot(&emb.row(j)));
    println!(
        "  E4 shuffle-null    pairs={nn:<5} dir={an:.3}   R@10 cos {:.3} -> W_null {:.3}   (this is the floor: real E1 must beat it)",
        cos.recall_at_10, nullr.recall_at_10
    );

    println!("\nReading: E2 surviving = structure transfers to unseen time (not interval memorisation).");
    println!("         E3 rare ~ hub  = edge not carried by over-represented topics.");
    println!("         E4 near 0.5 / cosine = E1's lift is real learned direction, not geometry.");
    Ok(())
}
