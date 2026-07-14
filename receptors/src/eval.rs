//! Evaluate whether the de-gisted asymmetric transport operator recovers the
//! directed cause->effect structure better than plain cosine.
//!
//! Two tasks, both on held-out pairs:
//!   1. Direction: given an *unordered* {a,b}, guess which is the cause using
//!      only content (sign of score(a->b) - score(b->a)). Symmetric cosine
//!      can't -> 0.5. The operator's asymmetry is the whole point.
//!   2. Retrieval: given a cause a, rank all later docs; how high is the true
//!      effect? Recall@10 + MRR, macro-averaged over causes.

use crate::pairs::Pair;
use ndarray::{Array1, Array2};
use std::collections::{HashMap, HashSet};

pub struct DirResult {
    pub n: usize,
    pub transport_acc: f32,
    pub cosine_acc: f32, // symmetric baseline == 0.5 by construction
}

/// `ahat_i` = predicted-effect embedding of doc i (cause row through W).
/// `resid_j` = de-gisted embedding of doc j.
pub fn direction(pairs: &[Pair], ahat: &Array2<f32>, resid: &Array2<f32>) -> DirResult {
    let score = |from: usize, to: usize| ahat.row(from).dot(&resid.row(to));
    let (mut correct, mut n) = (0usize, 0usize);
    for p in pairs.iter().filter(|p| p.is_test) {
        let fwd = score(p.a, p.b);
        let rev = score(p.b, p.a);
        if fwd > rev {
            correct += 1;
        }
        n += 1;
    }
    DirResult {
        n,
        transport_acc: if n > 0 { correct as f32 / n as f32 } else { 0.0 },
        cosine_acc: 0.5,
    }
}

pub struct RetrievalResult {
    pub label: String,
    pub causes: usize,
    pub recall_at_10: f32,
    pub mrr: f32,
}

/// Generic retrieval eval. `scorer(cause, cand) -> f32`, higher = better.
/// Candidate pool for a cause = all docs strictly later in time, minus that
/// cause's *train* targets (so known edges don't count as distractors).
pub fn retrieval<F>(
    label: &str,
    pairs: &[Pair],
    epochs: &[i64],
    scorer: F,
) -> RetrievalResult
where
    F: Fn(usize, usize) -> f32,
{
    // test targets and train targets, per cause
    let mut test_targets: HashMap<usize, HashSet<usize>> = HashMap::new();
    let mut train_targets: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in pairs {
        if p.is_test {
            test_targets.entry(p.a).or_default().insert(p.b);
        } else {
            train_targets.entry(p.a).or_default().insert(p.b);
        }
    }

    let n = epochs.len();
    let (mut sum_recall, mut sum_mrr, mut causes) = (0.0f32, 0.0f32, 0usize);

    for (&a, truth) in &test_targets {
        let ta = epochs[a];
        let empty = HashSet::new();
        let excl = train_targets.get(&a).unwrap_or(&empty);

        // candidate pool: strictly later docs, excluding known train targets
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
        // rank descending by score
        cands.sort_by(|x, y| y.0.partial_cmp(&x.0).unwrap());

        // Recall@10
        let hits10 = cands
            .iter()
            .take(10)
            .filter(|(_, j)| truth.contains(j))
            .count();
        sum_recall += hits10 as f32 / truth.len().min(10) as f32;

        // reciprocal rank of first true hit
        let mut rr = 0.0;
        for (rank, (_, j)) in cands.iter().enumerate() {
            if truth.contains(j) {
                rr = 1.0 / (rank as f32 + 1.0);
                break;
            }
        }
        sum_mrr += rr;
        causes += 1;
    }

    RetrievalResult {
        label: label.to_string(),
        causes,
        recall_at_10: if causes > 0 { sum_recall / causes as f32 } else { 0.0 },
        mrr: if causes > 0 { sum_mrr / causes as f32 } else { 0.0 },
    }
}

/// Jaccard overlap of two docs' entity sets — the sparse/structural channel.
pub fn entity_jaccard(a: &[String], b: &[String]) -> f32 {
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }
    let sa: HashSet<&str> = a.iter().map(|s| s.as_str()).collect();
    let sb: HashSet<&str> = b.iter().map(|s| s.as_str()).collect();
    let inter = sa.intersection(&sb).count() as f32;
    let union = sa.union(&sb).count() as f32;
    if union > 0.0 {
        inter / union
    } else {
        0.0
    }
}

/// Convenience: build an owned score vector row for a doc (unused helper kept
/// small; scoring is done inline via closures above).
#[allow(dead_code)]
pub fn row_vec(m: &Array2<f32>, i: usize) -> Array1<f32> {
    m.row(i).to_owned()
}
