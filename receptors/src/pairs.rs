//! Mine directed cause->effect article pairs for free from the already-extracted
//! `cause_entities` labels — no new LLM inference.
//!
//! Edge A -> B exists when: an entity that B attributes as a *cause*
//! (B.cause_entities) is something A is *about* (A.entities), and A precedes B
//! in time. Pairs are weighted by entity specificity (IDF) and we require at
//! least one reasonably specific shared entity so we don't link on generic
//! tokens like "strike".

use crate::data::Dataset;
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct Pair {
    pub a: usize, // cause (earlier)
    pub b: usize, // effect (later)
    pub weight: f32,
    pub is_test: bool,
}

/// Deterministic FNV-1a hash -> reproducible train/test split by effect doc.
pub fn fnv1a(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in s.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub struct Mined {
    pub pairs: Vec<Pair>,
    pub idf: HashMap<String, f32>,
    pub n_specific_entities: usize,
}

/// `specific_frac`: an entity counts as "specific" if it appears in at most
/// this fraction of docs. `test_frac`: share of effect-docs held out.
pub fn mine(ds: &Dataset, specific_frac: f64, test_frac: u64) -> Mined {
    let n = ds.n();

    // Document frequency per entity + inverted index entity -> docs mentioning it.
    let mut df: HashMap<&str, usize> = HashMap::new();
    let mut postings: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, d) in ds.docs.iter().enumerate() {
        for e in &d.entities {
            *df.entry(e.as_str()).or_insert(0) += 1;
            postings.entry(e.as_str()).or_default().push(i);
        }
    }
    let idf: HashMap<String, f32> = df
        .iter()
        .map(|(e, &c)| (e.to_string(), ((n as f32) / (c as f32)).ln()))
        .collect();
    let specific_cap = (specific_frac * n as f64).ceil() as usize;
    let n_specific = df.values().filter(|&&c| c <= specific_cap).count();

    // For each effect doc B, walk its cause_entities, pull earlier docs that
    // mention them, accumulate weighted directed edges.
    let mut acc: HashMap<(usize, usize), (f32, bool)> = HashMap::new();
    for (b, d) in ds.docs.iter().enumerate() {
        let tb = d.published_epoch;
        for c in &d.cause_entities {
            let Some(dfc) = df.get(c.as_str()) else { continue };
            let specific = *dfc <= specific_cap;
            let w = *idf.get(c).unwrap_or(&0.0);
            if let Some(list) = postings.get(c.as_str()) {
                for &a in list {
                    if a == b {
                        continue;
                    }
                    if ds.docs[a].published_epoch >= tb {
                        continue; // strict precedence; same-day epoch excluded
                    }
                    let e = acc.entry((a, b)).or_insert((0.0, false));
                    e.0 += w;
                    e.1 |= specific;
                }
            }
        }
    }

    let mut pairs = Vec::new();
    for ((a, b), (weight, has_specific)) in acc {
        if !has_specific {
            continue; // require at least one specific shared cause entity
        }
        let is_test = fnv1a(&ds.docs[b].doc_id) % 100 < test_frac;
        pairs.push(Pair { a, b, weight, is_test });
    }
    pairs.sort_by(|x, y| (x.a, x.b).cmp(&(y.a, y.b)));

    Mined { pairs, idf, n_specific_entities: n_specific }
}
