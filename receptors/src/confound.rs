//! B3 — CONFOUNDER / COMMON-CAUSE DISCRIMINATOR.
//!
//! A real direct edge A->B and a spurious sibling pair (B1,B2) — where B1 and
//! B2 are correlated only because they SHARE A COMMON PARENT A (A->B1, A->B2) —
//! are topically similar, so cosine can't tell them apart. Siblings co-move, but
//! there is no direct mechanism between them. Question: does the learned
//! asymmetric transport operator carry a signal that separates a direct edge
//! from a confounded co-moving sibling pair?
//!
//! Setup:
//!   - Mine directed edges E = {(a,b)} for free from cause_entities.
//!   - Parent->children map from E.
//!   - POSITIVES = the direct edges (a,b) in E.
//!   - NEGATIVES = sibling pairs (b1,b2) sharing a parent, with NO direct edge.
//!   - Fit one GLOBAL W on all of E; ahat = transport(emb, W).
//!   - 4-dim feature per ordered pair (x,y):
//!       [ forward transport ahat(x)·emb(y),
//!         reverse transport ahat(y)·emb(x),
//!         cosine emb(x)·emb(y),
//!         entity jaccard(x,y) ].
//!   - Ridge probe (D=4) on label +1 direct / 0 sibling. Report probe AUC +
//!     accuracy vs a cosine-only AUC baseline (which should sit near 0.5 because
//!     siblings are cosine-similar).

use crate::data::Dataset;
use crate::eval;
use crate::linalg;
use crate::pairs::{self, Pair};
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde_json::json;
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Deterministic small hash for the 70/30 split — mixes x and y so the same
/// ordered pair always lands on the same side.
fn split_hash(x: usize, y: usize) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    for v in [x as u64, y as u64] {
        h ^= v;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Build the 4-dim feature row for an ordered pair (x, y).
fn feat(
    x: usize,
    y: usize,
    ahat: &Array2<f32>,
    emb: &Array2<f32>,
    docs_entities: &[Vec<String>],
) -> [f32; 4] {
    let fwd = ahat.row(x).dot(&emb.row(y));
    let rev = ahat.row(y).dot(&emb.row(x));
    let cos = emb.row(x).dot(&emb.row(y));
    let jac = eval::entity_jaccard(&docs_entities[x], &docs_entities[y]);
    [fwd, rev, cos, jac]
}

/// Mann-Whitney AUC of `score` separating positives (label==1) from negatives.
fn auc(scores: &[f32], labels: &[f32]) -> f32 {
    // rank-sum over ties-averaged ranks
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| scores[i].partial_cmp(&scores[j]).unwrap());
    // assign average ranks (1-based), handling ties
    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1 + j + 1) as f64) / 2.0; // average of ranks i+1..=j+1
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    let (mut n_pos, mut n_neg, mut sum_pos_rank) = (0.0f64, 0.0f64, 0.0f64);
    for k in 0..n {
        if labels[k] > 0.5 {
            n_pos += 1.0;
            sum_pos_rank += ranks[k];
        } else {
            n_neg += 1.0;
        }
    }
    if n_pos == 0.0 || n_neg == 0.0 {
        return 0.5;
    }
    ((sum_pos_rank - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)) as f32
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let mined = pairs::mine(&ds, specific_frac, 30);

    // Directed edge set E and its membership set.
    let edges: Vec<(usize, usize)> = mined.pairs.iter().map(|p| (p.a, p.b)).collect();
    let edge_set: HashSet<(usize, usize)> = edges.iter().copied().collect();
    anyhow::ensure!(edges.len() > 50, "not enough mined edges ({})", edges.len());

    // parent -> children
    let mut children: HashMap<usize, Vec<usize>> = HashMap::new();
    for &(a, b) in &edges {
        children.entry(a).or_default().push(b);
    }

    println!(
        "confounder discriminator (B3): {} docs, {} directed edges, {} parents",
        ds.n(),
        edges.len(),
        children.len()
    );

    // ---- fit the GLOBAL transport operator on ALL of E ----
    let d = ds.d();
    let mut a_rows = Array2::<f32>::zeros((edges.len(), d));
    let mut b_rows = Array2::<f32>::zeros((edges.len(), d));
    for (r, &(a, b)) in edges.iter().enumerate() {
        a_rows.row_mut(r).assign(&ds.emb.row(a));
        b_rows.row_mut(r).assign(&ds.emb.row(b));
    }
    let w = linalg::fit_transport(&a_rows, &b_rows, lambda)?;
    let ahat = linalg::transport(&ds.emb, &w);

    // ---- POSITIVES: direct edges (sample up to ~2000) ----
    const CAP: usize = 2000;
    let mut positives: Vec<(usize, usize)> = edges.clone();
    // deterministic subsample by hash if too many
    if positives.len() > CAP {
        positives.sort_by_key(|&(x, y)| split_hash(x, y));
        positives.truncate(CAP);
    }

    // ---- NEGATIVES: confounded sibling ordered pairs (b1,b2) sharing a parent,
    //      b1!=b2, (b1,b2) NOT a direct edge. Balanced count. ----
    let mut neg_seen: HashSet<(usize, usize)> = HashSet::new();
    let mut negatives: Vec<(usize, usize, usize)> = Vec::new(); // (parent, b1, b2)
    // iterate parents deterministically
    let mut parents: Vec<usize> = children.keys().copied().collect();
    parents.sort_unstable();
    for a in parents {
        let kids = &children[&a];
        if kids.len() < 2 {
            continue;
        }
        // dedup children for this parent, keep order stable
        let mut seen_kid = HashSet::new();
        let uniq: Vec<usize> = kids
            .iter()
            .copied()
            .filter(|k| seen_kid.insert(*k))
            .collect();
        for (i, &b1) in uniq.iter().enumerate() {
            for &b2 in uniq.iter().skip(i + 1) {
                if b1 == b2 {
                    continue;
                }
                // both ordered directions, if not a real edge and not already taken
                for &(x, y) in &[(b1, b2), (b2, b1)] {
                    if edge_set.contains(&(x, y)) {
                        continue;
                    }
                    if neg_seen.insert((x, y)) {
                        negatives.push((a, x, y));
                    }
                }
            }
        }
    }

    anyhow::ensure!(
        !negatives.is_empty(),
        "no confounded sibling pairs found — parents have no co-children"
    );

    // balance: match negative count to positive count
    negatives.sort_by_key(|&(_, x, y)| split_hash(x, y));
    let n_bal = positives.len().min(negatives.len());
    let pos_bal: Vec<(usize, usize)> = positives.into_iter().take(n_bal).collect();
    let neg_bal: Vec<(usize, usize, usize)> = negatives.iter().copied().take(n_bal).collect();

    println!(
        "  balanced sample: {} direct positives, {} sibling negatives",
        pos_bal.len(),
        neg_bal.len()
    );

    // ---- assemble labelled feature rows + deterministic 70/30 split ----
    let entities: Vec<Vec<String>> = ds.docs.iter().map(|d| d.entities.clone()).collect();
    struct Row {
        f: [f32; 4],
        label: f32,
        train: bool,
    }
    let mut rows: Vec<Row> = Vec::with_capacity(2 * n_bal);
    for &(x, y) in &pos_bal {
        let train = split_hash(x, y) % 100 < 70;
        rows.push(Row {
            f: feat(x, y, &ahat, &ds.emb, &entities),
            label: 1.0,
            train,
        });
    }
    for &(_, x, y) in &neg_bal {
        let train = split_hash(x, y) % 100 < 70;
        rows.push(Row {
            f: feat(x, y, &ahat, &ds.emb, &entities),
            label: 0.0,
            train,
        });
    }

    let n_train = rows.iter().filter(|r| r.train).count();
    let n_test = rows.len() - n_train;
    anyhow::ensure!(
        n_train > 20 && n_test > 10,
        "split too small (train {} test {})",
        n_train,
        n_test
    );

    // ---- fit the 4-dim ridge probe on the training rows ----
    let mut xtr = Array2::<f32>::zeros((n_train, 4));
    let mut ytr = Array1::<f32>::zeros(n_train);
    let mut r = 0;
    for row in rows.iter().filter(|r| r.train) {
        for c in 0..4 {
            xtr[[r, c]] = row.f[c];
        }
        ytr[r] = row.label;
        r += 1;
    }
    let probe = linalg::ridge_probe(&xtr, &ytr, lambda)?;

    // ---- evaluate on the held-out test rows ----
    let test_rows: Vec<&Row> = rows.iter().filter(|r| !r.train).collect();
    let mut probe_scores = Vec::with_capacity(test_rows.len());
    let mut cos_scores = Vec::with_capacity(test_rows.len());
    let mut labels = Vec::with_capacity(test_rows.len());
    let mut correct = 0usize;
    for row in &test_rows {
        let s: f32 = (0..4).map(|c| probe[c] * row.f[c]).sum();
        probe_scores.push(s);
        cos_scores.push(row.f[2]); // cosine-only feature
        labels.push(row.label);
        let pred = if s >= 0.5 { 1.0 } else { 0.0 };
        if (pred - row.label).abs() < 0.5 {
            correct += 1;
        }
    }
    let probe_acc = correct as f32 / test_rows.len() as f32;
    let probe_auc = auc(&probe_scores, &labels);
    let cos_auc = auc(&cos_scores, &labels);

    println!("\n== DISCRIMINATE direct edge vs confounded sibling ==");
    println!("  test pairs                 : {}", test_rows.len());
    println!("  cosine-only AUC (baseline) : {:.3}", cos_auc);
    println!("  learned probe AUC          : {:.3}", probe_auc);
    println!("  learned probe accuracy@0.5 : {:.3}", probe_acc);
    println!(
        "  lift over cosine           : {:+.3} AUC",
        probe_auc - cos_auc
    );

    // ---- which features carry the weight ----
    let names = [
        "fwd_transport(x->y)",
        "rev_transport(y->x)",
        "cosine(x,y)",
        "entity_jaccard(x,y)",
    ];
    let mut ranked: Vec<(usize, f32)> = (0..4).map(|i| (i, probe[i].abs())).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\n== probe feature weights (|w|, descending) ==");
    for (i, _) in &ranked {
        println!("  {:<22} {:>+8.4}", names[*i], probe[*i]);
    }
    println!(
        "  -> dominant feature: {} (|w|={:.4})",
        names[ranked[0].0], ranked[0].1
    );

    // ---- concrete examples of confounding: parent a -> b1, b2 ----
    println!("\n== example confounded sibling triples (parent -> b1, b2) ==");
    let short = |v: &[String]| -> String {
        let mut s = v.iter().take(4).cloned().collect::<Vec<_>>().join(", ");
        if v.len() > 4 {
            s.push_str(", …");
        }
        s
    };
    for (k, &(a, b1, b2)) in neg_bal.iter().take(3).enumerate() {
        println!(
            "  [{}] parent {} [{}]",
            k + 1,
            ds.docs[a].doc_id,
            short(&ds.docs[a].entities)
        );
        println!(
            "        -> b1 {} [{}]",
            ds.docs[b1].doc_id,
            short(&ds.docs[b1].entities)
        );
        println!(
            "        -> b2 {} [{}]",
            ds.docs[b2].doc_id,
            short(&ds.docs[b2].entities)
        );
        let c = ds.emb.row(b1).dot(&ds.emb.row(b2));
        println!("        sibling cosine={:.3} (co-move, but no direct edge)", c);
    }

    // ---- honest verdict ----
    let lift = probe_auc - cos_auc;
    println!("\n== honest note ==");
    if cos_auc <= 0.58 {
        println!(
            "  cosine AUC={:.3} confirms siblings are cosine-similar to real edges —",
            cos_auc
        );
        println!("  co-movement alone does NOT distinguish a direct edge from a shared cause.");
    } else {
        println!(
            "  cosine AUC={:.3} is higher than expected: real edges are somewhat more",
            cos_auc
        );
        println!("  cosine-similar than siblings here, so the confound is only partial.");
    }
    if lift >= 0.05 && probe_auc >= 0.62 {
        println!(
            "  The learned probe adds {:+.3} AUC (to {:.3}); the transport-asymmetry channel",
            lift, probe_auc
        );
        println!(
            "  carries real signal — receptor B3 CAN screen confounded co-movement to a"
        );
        println!("  useful degree (dominant feature: {}).", names[ranked[0].0]);
    } else if lift >= 0.02 {
        println!(
            "  The probe adds only {:+.3} AUC (to {:.3}): a weak, marginal screen — the",
            lift, probe_auc
        );
        println!("  operator sees some direction signal but cannot cleanly reject siblings.");
    } else {
        println!(
            "  The probe adds {:+.3} AUC (to {:.3}): essentially NO screening power. Direct",
            lift, probe_auc
        );
        println!(
            "  edges and confounded siblings are indistinguishable in this geometry —"
        );
        println!("  the receptor cannot separate real edges from common-cause co-movement.");
    }

    let labels_u8: Vec<u8> = labels.iter().map(|&l| if l > 0.5 { 1u8 } else { 0u8 }).collect();
    crate::report::save(
        dir,
        "confound",
        &json!({
            "probe_auc": probe_auc,
            "cosine_auc": cos_auc,
            "n_test": labels_u8.len(),
            "probe_scores": probe_scores,
            "cosine_scores": cos_scores,
            "labels": labels_u8,
            "feature_weights": {
                "fwd_transport": probe[0],
                "rev_transport": probe[1],
                "cosine": probe[2],
                "entity_jaccard": probe[3],
            },
        }),
    )?;

    Ok(())
}
