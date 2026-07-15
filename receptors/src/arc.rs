//! C3 — NARRATIVE-ARC / STORY-CONTINUATION operator.
//!
//! Unlike the causal receptors (cause->effect), this one learns which document
//! *follows up* another the next day about the SAME entities — story
//! continuation — and then predicts continuations. There is no causal claim:
//! the relation A->B is "B is a same-entity story written within two days after
//! A". We mine those continuation pairs ourselves (NOT `pairs::mine`, which mines
//! a different, cause-attribution relation), fit a ridge transport operator on
//! the follow-up geometry, and ask whether that operator predicts the next-day
//! follow-up better than plain cosine similarity.

use crate::data::Dataset;
use crate::eval;
use crate::linalg;
use crate::pairs::{self, Pair};
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

const TWO_DAYS: i64 = 2 * 86_400;
const MAX_PAIRS: usize = 40_000;

/// Build (A, B) row matrices for a set of continuation pairs over `emb`.
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

/// Mine same-entity next-two-days continuation pairs.
fn mine_continuations(
    ds: &Dataset,
    specific_frac: f64,
    test_frac: u64,
) -> (Vec<Pair>, HashMap<String, usize>, HashMap<String, f32>) {
    let n = ds.n();

    // doc-frequency + inverted index per entity.
    let mut df: HashMap<&str, usize> = HashMap::new();
    let mut postings: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, d) in ds.docs.iter().enumerate() {
        // one posting per (entity, doc) — dedup entities within a doc.
        let mut seen = std::collections::HashSet::new();
        for e in &d.entities {
            if seen.insert(e.as_str()) {
                *df.entry(e.as_str()).or_insert(0) += 1;
                postings.entry(e.as_str()).or_default().push(i);
            }
        }
    }
    let specific_cap = (specific_frac * n as f64).ceil() as usize;
    let idf: HashMap<String, f32> = df
        .iter()
        .map(|(e, &c)| (e.to_string(), ((n as f32) / (c as f32)).ln()))
        .collect();

    // For each specific entity, form candidate (a,b) pairs among its docs within
    // the 2-day forward window. Accumulate summed idf weight per (a,b).
    let mut acc: HashMap<(usize, usize), f32> = HashMap::new();
    let mut df_of: HashMap<String, usize> = HashMap::new();

    for (e, &c) in &df {
        if c > specific_cap {
            continue; // not a specific entity
        }
        df_of.insert(e.to_string(), c);
        let w = *idf.get(*e).unwrap_or(&0.0);
        let Some(list) = postings.get(e) else { continue };
        // sort this entity's docs by epoch so we can window forward cheaply.
        let mut docs: Vec<usize> = list.iter().copied().collect();
        docs.sort_by_key(|&i| ds.docs[i].published_epoch);
        for (pi, &a) in docs.iter().enumerate() {
            let ta = ds.docs[a].published_epoch;
            if ta <= 0 {
                continue;
            }
            for &b in &docs[pi + 1..] {
                let tb = ds.docs[b].published_epoch;
                if tb <= 0 {
                    continue;
                }
                let dt = tb - ta;
                if dt <= 0 {
                    continue; // b must be strictly later
                }
                if dt > TWO_DAYS {
                    break; // docs sorted by epoch -> no later doc qualifies
                }
                if a == b {
                    continue;
                }
                *acc.entry((a, b)).or_insert(0.0) += w;
            }
        }
    }

    // Cap total pairs by keeping the heaviest edges.
    let mut edges: Vec<((usize, usize), f32)> = acc.into_iter().collect();
    if edges.len() > MAX_PAIRS {
        edges.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
        edges.truncate(MAX_PAIRS);
    }

    let mut pairs: Vec<Pair> = edges
        .into_iter()
        .map(|((a, b), weight)| {
            let is_test = pairs::fnv1a(&ds.docs[b].doc_id) % 100 < test_frac;
            Pair { a, b, weight, is_test }
        })
        .collect();
    pairs.sort_by(|x, y| (x.a, x.b).cmp(&(y.a, y.b)));
    (pairs, df_of, idf)
}

/// Shared specific-entity preview for a pair (a,b): entities present in both,
/// limited to specific ones, sorted by idf descending.
fn shared_specific(
    ds: &Dataset,
    a: usize,
    b: usize,
    df_of: &HashMap<String, usize>,
    idf: &HashMap<String, f32>,
    k: usize,
) -> Vec<String> {
    let sb: std::collections::HashSet<&str> =
        ds.docs[b].entities.iter().map(|s| s.as_str()).collect();
    let mut shared: Vec<(&str, f32)> = ds.docs[a]
        .entities
        .iter()
        .map(|s| s.as_str())
        .filter(|e| sb.contains(e) && df_of.contains_key(*e))
        .map(|e| (e, *idf.get(e).unwrap_or(&0.0)))
        .collect();
    shared.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    shared.dedup_by(|x, y| x.0 == y.0);
    shared.into_iter().take(k).map(|(e, _)| e.to_string()).collect()
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64, test_frac: u64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);

    let (mined, df_of, idf) = mine_continuations(&ds, specific_frac, test_frac);
    let train: Vec<&Pair> = mined.iter().filter(|p| !p.is_test).collect();
    let test: Vec<&Pair> = mined.iter().filter(|p| p.is_test).collect();
    anyhow::ensure!(
        train.len() > 50 && test.len() > 20,
        "not enough continuation pairs (train {}, test {})",
        train.len(),
        test.len()
    );
    println!(
        "narrative-arc (C3): {} docs, {} continuation pairs ({} train / {} test)",
        ds.n(),
        mined.len(),
        train.len(),
        test.len()
    );
    println!(
        "  relation: B follows A within 2 days about >=1 shared SPECIFIC entity (df<=ceil({:.3}*n))",
        specific_frac
    );

    // ---- fit the follow-up operator ----
    let (a, b) = stack(&ds.emb, &train);
    let w = linalg::fit_transport(&a, &b, lambda)?;
    let ahat = linalg::transport(&ds.emb, &w);

    // ---- retrieval eval: cosine vs arc operator ----
    let epochs: Vec<i64> = ds.docs.iter().map(|d| d.published_epoch).collect();

    let cos = eval::retrieval("cosine", &mined, &epochs, |a, j| {
        ds.emb.row(a).dot(&ds.emb.row(j))
    });
    let arc = eval::retrieval("arc op", &mined, &epochs, |a, j| {
        ahat.row(a).dot(&ds.emb.row(j))
    });

    println!("\n== next-two-day continuation retrieval (macro over test causes) ==");
    println!("  {:<10} {:>7} {:>11} {:>8}", "scorer", "causes", "Recall@10", "MRR");
    for r in [&cos, &arc] {
        println!(
            "  {:<10} {:>7} {:>11.3} {:>8.3}",
            r.label, r.causes, r.recall_at_10, r.mrr
        );
    }
    let win_r = arc.recall_at_10 - cos.recall_at_10;
    let win_m = arc.mrr - cos.mrr;
    println!(
        "  arc op vs cosine: dR@10 = {:+.3}, dMRR = {:+.3}",
        win_r, win_m
    );

    // ---- sample continuations: for a few TEST causes, the doc the arc operator
    //      ranks #1 among later docs, marked HIT if it is a true continuation. ----
    // gather test targets per cause + candidate pools mirroring eval::retrieval.
    let mut test_targets: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    let mut train_targets: HashMap<usize, std::collections::HashSet<usize>> = HashMap::new();
    for p in &mined {
        if p.is_test {
            test_targets.entry(p.a).or_default().insert(p.b);
        } else {
            train_targets.entry(p.a).or_default().insert(p.b);
        }
    }

    // deterministic order: causes sorted by index.
    let mut causes: Vec<usize> = test_targets.keys().copied().collect();
    causes.sort_unstable();

    println!("\n== sample #1-ranked continuations (arc operator, TEST causes) ==");
    let empty = std::collections::HashSet::new();
    let mut shown = 0;
    for &a in &causes {
        if shown >= 6 {
            break;
        }
        let ta = epochs[a];
        let excl = train_targets.get(&a).unwrap_or(&empty);
        let truth = &test_targets[&a];

        // rank later docs by arc score, pick the top one. scores held in an
        // Array1 so we lean on ndarray's dot throughout.
        let arow: Array1<f32> = ahat.row(a).to_owned();
        let mut best: Option<(f32, usize)> = None;
        for j in 0..ds.n() {
            if j == a || epochs[j] <= ta || excl.contains(&j) {
                continue;
            }
            let s = arow.dot(&ds.emb.row(j));
            if best.map_or(true, |(bs, _)| s > bs) {
                best = Some((s, j));
            }
        }
        let Some((_, top)) = best else { continue };

        let hit = truth.contains(&top);
        let da = if ds.docs[a].date.is_empty() {
            ta.to_string()
        } else {
            ds.docs[a].date.clone()
        };
        let db = if ds.docs[top].date.is_empty() {
            epochs[top].to_string()
        } else {
            ds.docs[top].date.clone()
        };
        let ents = shared_specific(&ds, a, top, &df_of, &idf, 3);
        let ent_prev = if ents.is_empty() {
            "(no specific overlap)".to_string()
        } else {
            ents.join(", ")
        };
        println!(
            "  [{}] {} ({})  ->  {} ({})",
            if hit { "HIT " } else { "miss" },
            ds.docs[a].doc_id,
            da,
            ds.docs[top].doc_id,
            db
        );
        println!("       shared: {}", ent_prev);
        shown += 1;
    }
    if shown == 0 {
        println!("  (no test causes had a later-doc candidate pool)");
    }

    // ---- honest note ----
    println!("\n== honest note ==");
    println!(
        "  This is TOPICAL story-continuation (same-entity next-2-day follow-up), NOT causation."
    );
    if win_m > 0.0 || win_r > 0.0 {
        println!(
            "  The arc operator BEATS cosine at predicting the next-day follow-up (dR@10 {:+.3}, dMRR {:+.3}):",
            win_r, win_m
        );
        println!(
            "  learning the follow-up geometry adds signal over raw topical similarity."
        );
    } else {
        println!(
            "  The arc operator does NOT beat cosine here (dR@10 {:+.3}, dMRR {:+.3}):",
            win_r, win_m
        );
        println!(
            "  continuation appears to be mostly plain topical similarity — the operator adds little."
        );
    }

    crate::report::save(
        dir,
        "arc",
        &json!({
            "n_causes": cos.causes,
            "cosine_r10": cos.recall_at_10,
            "cosine_mrr": cos.mrr,
            "arc_r10": arc.recall_at_10,
            "arc_mrr": arc.mrr,
        }),
    )?;

    Ok(())
}
