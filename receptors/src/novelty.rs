//! Relational novelty (first-story detection). For each chunk at time t:
//!   novelty = 1 - max cosine similarity to any chunk published STRICTLY earlier.
//! High = genuinely new information; low = an echo / restatement of prior news.
//! This is a per-document property computed against the corpus history — it is
//! not derivable from the chunk's own text, so it's invisible to single-doc cosine.

use anyhow::{Context, Result};
use ndarray::{s, Array1, Array2};
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::collections::BTreeMap;
use std::path::Path;

#[derive(Deserialize)]
struct ChunkMeta {
    chunk_id: String,
    doc_id: String,
    epoch: i64,
}

pub fn run(dir: &Path) -> Result<()> {
    let mut chunks: Array2<f32> = read_npy(dir.join("chunks.npy")).context("chunks.npy")?;
    crate::linalg::normalize_rows(&mut chunks);
    let meta: Vec<ChunkMeta> = std::fs::read_to_string(dir.join("chunks.jsonl"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<ChunkMeta>(l).unwrap())
        .collect();
    let n = chunks.nrows();
    anyhow::ensure!(n == meta.len(), "row mismatch");
    let epoch: Vec<i64> = meta.iter().map(|m| m.epoch).collect();

    // novelty[i] = 1 - max_{epoch[j] < epoch[i]} cos(i,j); nn = the matched prior chunk
    let mut novelty = vec![f32::NAN; n]; // NaN = no prior (earliest day)
    let mut nn = vec![usize::MAX; n];
    let block = 2000usize;
    let mut b = 0;
    while b < n {
        let e = (b + block).min(n);
        let sims = chunks.slice(s![b..e, ..]).dot(&chunks.t()); // (blk, n) BLAS
        for (r, i) in (b..e).enumerate() {
            let mut best = f32::MIN;
            let mut bj = usize::MAX;
            let row = sims.row(r);
            for j in 0..n {
                if epoch[j] < epoch[i] && row[j] > best {
                    best = row[j];
                    bj = j;
                }
            }
            if bj != usize::MAX {
                novelty[i] = 1.0 - best;
                nn[i] = bj;
            }
        }
        b = e;
    }

    // ---- summary ----
    let mut vals: Vec<f32> = novelty.iter().copied().filter(|v| !v.is_nan()).collect();
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = vals.len();
    let pct = |p: f32| vals[((m as f32 * p) as usize).min(m - 1)];
    let echo = vals.iter().filter(|&&v| v < 0.10).count() as f32 / m as f32;
    let novel = vals.iter().filter(|&&v| v > 0.40).count() as f32 / m as f32;
    println!("novelty over {} chunks ({} have prior-day history)", n, m);
    println!("  percentiles: p10 {:.3}  p50 {:.3}  p90 {:.3}", pct(0.10), pct(0.50), pct(0.90));
    println!("  echo rate  (novelty<0.10, near-duplicate of prior): {:.1}%", echo * 100.0);
    println!("  novel rate (novelty>0.40, far from all prior)     : {:.1}%", novel * 100.0);

    // ---- novelty by day (does the corpus get more repetitive over time?) ----
    let mut byday: BTreeMap<i64, (f64, usize)> = BTreeMap::new();
    for i in 0..n {
        if !novelty[i].is_nan() {
            let e = byday.entry(epoch[i]).or_insert((0.0, 0));
            e.0 += novelty[i] as f64;
            e.1 += 1;
        }
    }
    println!("\n  mean novelty by day (epoch | n | mean-novelty):");
    for (ep, (s, c)) in &byday {
        println!("    {:>12} | {:>5} | {:.3}", ep, c, s / *c as f64);
    }

    // ---- write per-chunk novelty + nearest-prior match for validation ----
    let mut out = String::new();
    for i in 0..n {
        if novelty[i].is_nan() {
            continue;
        }
        out.push_str(&format!(
            "{{\"chunk_id\":\"{}\",\"doc_id\":\"{}\",\"novelty\":{:.4},\"nn_doc\":\"{}\"}}\n",
            meta[i].chunk_id,
            meta[i].doc_id,
            novelty[i],
            meta[nn[i]].doc_id
        ));
    }
    std::fs::write(dir.join("novelty.jsonl"), out)?;
    println!("\n  wrote novelty.jsonl (chunk_id, doc_id, novelty, nearest-prior doc)");
    Ok(())
}

/// Relevance-conditioned novelty: for each query, retrieve the topical set, tag
/// each result with novelty vs the EARLIER relevant results, and de-duplicate.
/// Measures how many DISTINCT developments the top-K contains vs restatements.
pub fn run_query(dir: &Path) -> Result<()> {
    let mut chunks: Array2<f32> = read_npy(dir.join("chunks.npy")).context("chunks.npy")?;
    crate::linalg::normalize_rows(&mut chunks);
    let meta: Vec<ChunkMeta> = std::fs::read_to_string(dir.join("chunks.jsonl"))?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<ChunkMeta>(l).unwrap())
        .collect();
    let d = chunks.ncols();

    // build doc-level embeddings (mean-pool chunks) + min epoch
    use std::collections::HashMap;
    let mut idx: HashMap<&str, usize> = HashMap::new();
    let mut ids: Vec<&str> = Vec::new();
    let mut acc: Vec<f64> = Vec::new();
    let mut cnt: Vec<f32> = Vec::new();
    let mut ep: Vec<i64> = Vec::new();
    for (ci, m) in meta.iter().enumerate() {
        let di = *idx.entry(m.doc_id.as_str()).or_insert_with(|| {
            ids.push(m.doc_id.as_str());
            acc.extend(std::iter::repeat(0.0).take(d));
            cnt.push(0.0);
            ep.push(m.epoch);
            ids.len() - 1
        });
        for j in 0..d {
            acc[di * d + j] += chunks[[ci, j]] as f64;
        }
        cnt[di] += 1.0;
        ep[di] = ep[di].min(m.epoch);
    }
    let nd = ids.len();
    let mut demb = Array2::<f32>::zeros((nd, d));
    for i in 0..nd {
        for j in 0..d {
            demb[[i, j]] = (acc[i * d + j] / cnt[i] as f64) as f32;
        }
    }
    crate::linalg::normalize_rows(&mut demb);

    let q: Array2<f32> = read_npy(dir.join("questions_custom.npy")).context("questions_custom.npy")?;
    let qtxt = std::fs::read_to_string(dir.join("questions_custom.txt"))?;
    let qtexts: Vec<&str> = qtxt.lines().filter(|l| !l.trim().is_empty()).collect();

    const POOL: usize = 30;
    const K: usize = 10;
    const DISTINCT: f32 = 0.55; // max-sim below this => a distinct development
    let mut nat_red = 0.0f64;
    let mut mmr_red = 0.0f64;
    let mut nat_dist = 0.0f64;
    let mut mmr_dist = 0.0f64;
    let mut out = String::from("[\n");

    for (qi, qt) in qtexts.iter().enumerate() {
        let qr = q.row(qi).to_owned();
        let rel: Array1<f32> = demb.dot(&qr);
        let mut pool: Vec<usize> = (0..nd).collect();
        pool.sort_by(|&a, &b| rel[b].partial_cmp(&rel[a]).unwrap());
        pool.truncate(POOL);

        // within-topic novelty (vs earlier relevant docs, by epoch)
        let nov = |di: usize| -> f32 {
            let mut best = f32::MIN;
            for &dj in &pool {
                if ep[dj] < ep[di] {
                    best = best.max(demb.row(di).dot(&demb.row(dj)));
                }
            }
            if best == f32::MIN { 1.0 } else { 1.0 - best }
        };

        // naive top-K (by relevance) — redundancy + distinct count
        let naive: Vec<usize> = pool[..K].to_vec();
        let redundancy = |set: &[usize]| -> f64 {
            let (mut s, mut c) = (0.0f64, 0u32);
            for a in 0..set.len() {
                for b in a + 1..set.len() {
                    s += demb.row(set[a]).dot(&demb.row(set[b])) as f64;
                    c += 1;
                }
            }
            if c > 0 { s / c as f64 } else { 0.0 }
        };
        let distinct = |set: &[usize]| -> f64 {
            let mut kept: Vec<usize> = Vec::new();
            for &di in set {
                let m = kept.iter().map(|&k| demb.row(di).dot(&demb.row(k))).fold(0.0f32, f32::max);
                if m < DISTINCT {
                    kept.push(di);
                }
            }
            kept.len() as f64
        };

        // MMR diversification from the pool (λ=0.7)
        let lam = 0.7f32;
        let mut sel: Vec<usize> = Vec::new();
        let mut cand = pool.clone();
        while sel.len() < K && !cand.is_empty() {
            let mut bi = 0usize;
            let mut bs = f32::MIN;
            for (ii, &di) in cand.iter().enumerate() {
                let maxsim = sel.iter().map(|&s| demb.row(di).dot(&demb.row(s))).fold(0.0f32, f32::max);
                let score = lam * rel[di] - (1.0 - lam) * maxsim;
                if score > bs {
                    bs = score;
                    bi = ii;
                }
            }
            sel.push(cand.remove(bi));
        }

        nat_red += redundancy(&naive);
        mmr_red += redundancy(&sel);
        nat_dist += distinct(&naive);
        mmr_dist += distinct(&sel);

        if qi < 3 {
            let j = |set: &[usize]| set.iter().map(|&di| format!(
                "{{\"doc\":\"{}\",\"rel\":{:.3},\"nov\":{:.3}}}", ids[di], rel[di], nov(di)))
                .collect::<Vec<_>>().join(",");
            out.push_str(&format!(
                "  {{\"q\": {}, \"naive\": [{}], \"mmr\": [{}]}}{}\n",
                serde_json::to_string(qt).unwrap(), j(&naive), j(&sel),
                if qi < 2 { "," } else { "" }));
        }
    }
    out.push_str("]\n");
    std::fs::write(dir.join("novelq.json"), out)?;
    let nq = qtexts.len() as f64;
    println!("relevance-conditioned novelty over {} queries (top-{} relevant, distinct<{})", qtexts.len(), K, DISTINCT);
    println!("  {:<22} {:>12} {:>16}", "top-K set", "intra-redund", "distinct-stories");
    println!("  {:<22} {:>12.3} {:>16.2}", "naive (cosine)", nat_red / nq, nat_dist / nq);
    println!("  {:<22} {:>12.3} {:>16.2}", "novelty-diversified", mmr_red / nq, mmr_dist / nq);
    println!("  wrote novelq.json (naive vs diversified top-10 w/ rel+novelty, first 3 queries)");
    Ok(())
}
