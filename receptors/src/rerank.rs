//! RAG reranking experiment: does the transport operator W improve relevance
//! when used to rerank a naive-cosine chunk shortlist?
//!
//! Query = an effect claim. Gold = earlier articles about its cause_entities.
//! Baseline = cosine top-N chunks -> max-pool to docs. Rerank = same shortlist,
//! reordered by cos + beta * cos(chunk·W, query). Compare nDCG@10 / Recall / MRR.

use crate::data::{Dataset, Doc};
use crate::linalg;
use crate::pairs;
use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::path::Path;

#[derive(Deserialize)]
struct ChunkMeta {
    #[allow(dead_code)]
    chunk_id: String,
    doc_id: String,
    epoch: i64,
}
#[derive(Deserialize)]
struct Query {
    #[allow(dead_code)]
    query_id: String,
    effect_doc_id: String,
    epoch: i64,
    #[allow(dead_code)]
    claim: String,
    cause_entities: Vec<String>,
}

/// Split predicate. Temporal mode: a query is TEST iff its effect is at/after the
/// time cutoff (train = strictly earlier). Random mode: FNV hash of effect doc.
fn is_test(cutoff: Option<i64>, epoch: i64, doc_id: &str, test_frac: u64) -> bool {
    match cutoff {
        Some(c) => epoch >= c,
        None => pairs::fnv1a(doc_id) % 100 < test_frac,
    }
}

/// Epoch quantile over queries (for the temporal cutoff).
fn epoch_quantile(queries: &[Query], frac: f32) -> i64 {
    let mut es: Vec<i64> = queries.iter().map(|q| q.epoch).collect();
    es.sort_unstable();
    let idx = ((es.len() as f32 * frac) as usize).min(es.len().saturating_sub(1));
    es[idx]
}

fn load_jsonl<T: for<'de> Deserialize<'de>>(p: &Path) -> Result<Vec<T>> {
    let s = std::fs::read_to_string(p).with_context(|| format!("read {p:?}"))?;
    s.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str::<T>(l).context("parse jsonl"))
        .collect()
}

struct Metrics {
    ndcg: f32,
    recall: f32,
    mrr: f32,
    hit: f32, // 1 if >=1 gold in top-k, else 0
}
/// Rank a doc list (already sorted desc by score) against a gold set.
fn score_ranking(ranked_docs: &[usize], gold: &HashSet<usize>, k: usize) -> Metrics {
    let mut dcg = 0.0f32;
    let mut first_hit = 0.0f32;
    let mut hits = 0usize;
    for (r, &d) in ranked_docs.iter().take(k).enumerate() {
        if gold.contains(&d) {
            dcg += 1.0 / ((r as f32) + 2.0).log2();
            if first_hit == 0.0 {
                first_hit = 1.0 / (r as f32 + 1.0);
            }
            hits += 1;
        }
    }
    let ideal_n = gold.len().min(k);
    let idcg: f32 = (0..ideal_n).map(|r| 1.0 / ((r as f32) + 2.0).log2()).sum();
    Metrics {
        ndcg: if idcg > 0.0 { dcg / idcg } else { 0.0 },
        recall: if !gold.is_empty() {
            hits as f32 / gold.len() as f32
        } else {
            0.0
        },
        mrr: first_hit,
        hit: if hits > 0 { 1.0 } else { 0.0 },
    }
}

pub fn run(
    dir: &Path,
    lambda: f32,
    specific_frac: f64,
    test_frac: u64,
    shortlist: usize,
    betas: &[f32],
) -> Result<()> {
    // --- article dataset (for doc entities / gold) ---
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let d = ds.d();
    let (df, specific_cap) = doc_df(&ds.docs, specific_frac);

    // --- RAG pool: chunks ---
    let chunks: Array2<f32> = read_npy(dir.join("chunks.npy")).context("chunks.npy")?;
    let cmeta: Vec<ChunkMeta> = load_jsonl(&dir.join("chunks.jsonl"))?;
    anyhow::ensure!(chunks.nrows() == cmeta.len(), "chunk row mismatch");
    let chunk_doc: Vec<Option<usize>> = cmeta
        .iter()
        .map(|c| ds.id2idx.get(&c.doc_id).copied())
        .collect();
    let chunk_epoch: Vec<i64> = cmeta.iter().map(|c| c.epoch).collect();
    // article index -> its chunk rows
    let mut chunks_of: Vec<Vec<usize>> = vec![Vec::new(); ds.n()];
    for (ci, dd) in chunk_doc.iter().enumerate() {
        if let Some(a) = dd {
            chunks_of[*a].push(ci);
        }
    }

    // --- queries ---
    let queries: Vec<Query> = load_jsonl(&dir.join("queries.jsonl"))?;
    let qvec: Array2<f32> = read_npy(dir.join("queries.npy")).context("queries.npy")?;
    anyhow::ensure!(qvec.nrows() == queries.len(), "query row mismatch");

    // --- fit W ---
    // Default: article->article pairs (mined). RECEPTORS_INDIST=1: train on the
    // distribution W is actually scored on — (cause_chunk -> effect_claim) pairs
    // built from TRAIN queries only.
    // Temporal split: train on earlier effects, test on later. Cutoff = the
    // (1 - test_frac) epoch quantile so the latest ~test_frac% of queries are test.
    let temporal = std::env::var("RECEPTORS_SPLIT").as_deref() == Ok("temporal");
    let cutoff: Option<i64> = if temporal {
        let c = epoch_quantile(&queries, 1.0 - test_frac as f32 / 100.0);
        println!("temporal split: cutoff epoch {c} (train < cutoff, test >= cutoff)");
        Some(c)
    } else {
        None
    };

    let indist = std::env::var("RECEPTORS_INDIST").as_deref() == Ok("1");
    let w = if indist {
        let (a_mat, b_mat) = indist_pairs(
            &ds, &df, specific_cap, &queries, &qvec, &chunks, &chunks_of, test_frac, cutoff,
        );
        println!("in-distribution fit: {} (cause_chunk -> effect_claim) pairs", a_mat.nrows());
        linalg::fit_operator(&a_mat, &b_mat, lambda)?
    } else {
        let mined = pairs::mine(&ds, specific_frac, test_frac);
        let n_train = mined.pairs.iter().filter(|p| !p.is_test).count();
        let mut a_mat = Array2::<f32>::zeros((n_train, d));
        let mut b_mat = Array2::<f32>::zeros((n_train, d));
        for (r, p) in mined.pairs.iter().filter(|p| !p.is_test).enumerate() {
            a_mat.row_mut(r).assign(&ds.emb.row(p.a));
            b_mat.row_mut(r).assign(&ds.emb.row(p.b));
        }
        linalg::fit_operator(&a_mat, &b_mat, lambda)?
    };
    let cw = linalg::transport(&chunks, &w); // chunk·W, once (BLAS)

    println!(
        "articles {}  chunks {}  queries {}  (shortlist N={}, fit={})",
        ds.n(),
        chunks.nrows(),
        queries.len(),
        shortlist,
        if indist { "in-dist" } else { "article" },
    );

    // FIRST-STAGE RETRIEVAL test: can W *find* over the whole corpus, not just
    // rerank cosine's output? Rank the FULL earlier pool by each scorer.
    if std::env::var("RECEPTORS_MODE").as_deref() == Ok("retrieve") {
        return retrieval_eval(
            &ds, &df, specific_cap, &queries, &qvec, &chunks, &cw, &chunk_doc, &chunk_epoch,
            test_frac, cutoff,
        );
    }
    if std::env::var("RECEPTORS_MODE").as_deref() == Ok("multihop") {
        return multihop_eval(
            &ds, &df, specific_cap, &queries, &qvec, &chunks, &w, &chunk_doc, test_frac, cutoff,
        );
    }

    // accumulate metrics: [recall@10, recall@30, hit@10, hit@30, mrr]
    let mut base = [0.0f32; 5];
    let mut rer: Vec<[f32; 5]> = vec![[0.0; 5]; betas.len()];
    let mut n_eval = 0usize;
    let mut sl_recall10 = 0.0f32; // ceiling within shortlist
    let mut sl_recall30 = 0.0f32;
    let best_bi = betas.iter().position(|&b| b == 0.5).unwrap_or(0);
    let (mut wins, mut losses) = (0usize, 0usize); // per-query hit@30 change at best beta

    for (qi, q) in queries.iter().enumerate() {
        if !is_test(cutoff, q.epoch, &q.effect_doc_id, test_frac) {
            continue; // test queries only (same split as W training)
        }
        // gold: earlier article docs sharing a SPECIFIC cause entity
        let gold = gold_docs(&ds.docs, &df, specific_cap, &q.cause_entities, q.epoch);
        if gold.is_empty() {
            continue;
        }

        let qrow = qvec.row(qi).to_owned();
        let cos: Array1<f32> = chunks.dot(&qrow); // (M) BLAS gemv
        let dir_s: Array1<f32> = cw.dot(&qrow); // (M) BLAS gemv

        // candidate chunks: strictly earlier than the effect
        let mut cand: Vec<usize> = (0..chunks.nrows())
            .filter(|&i| chunk_epoch[i] < q.epoch)
            .collect();
        if cand.len() < 10 {
            continue;
        }
        // shortlist: top-N by cosine (stage-1 retrieval)
        cand.sort_by(|&i, &j| cos[j].partial_cmp(&cos[i]).unwrap());
        cand.truncate(shortlist);

        // oracle: how much gold is even in the shortlist (reranker headroom)
        let shortlist_docs: HashSet<usize> =
            cand.iter().filter_map(|&i| chunk_doc[i]).collect();
        let in_sl = gold.iter().filter(|g| shortlist_docs.contains(g)).count() as f32
            / gold.len() as f32;
        sl_recall10 += in_sl; // (shortlist has no rank cut; same ceiling for @10/@30)
        sl_recall30 += in_sl;

        // baseline doc ranking: max-pool cosine
        let base_docs = rank_docs(&cand, &chunk_doc, |i| cos[i]);
        let b10 = score_ranking(&base_docs, &gold, 10);
        let b30 = score_ranking(&base_docs, &gold, 30);
        base[0] += b10.recall;
        base[1] += b30.recall;
        base[2] += b10.hit;
        base[3] += b30.hit;
        base[4] += b10.mrr;

        // reranked doc ranking per beta: max-pool (cos + beta*dir)
        for (bi, &beta) in betas.iter().enumerate() {
            let rr = rank_docs(&cand, &chunk_doc, |i| cos[i] + beta * dir_s[i]);
            let m10 = score_ranking(&rr, &gold, 10);
            let m30 = score_ranking(&rr, &gold, 30);
            rer[bi][0] += m10.recall;
            rer[bi][1] += m30.recall;
            rer[bi][2] += m10.hit;
            rer[bi][3] += m30.hit;
            rer[bi][4] += m10.mrr;
            if bi == best_bi {
                if m30.hit > b30.hit + 1e-6 {
                    wins += 1;
                } else if m30.hit < b30.hit - 1e-6 {
                    losses += 1;
                }
            }
        }
        n_eval += 1;
    }

    anyhow::ensure!(n_eval > 0, "no evaluable queries");
    let nf = n_eval as f32;
    println!("\nevaluated {n_eval} test queries (gold = earlier cause articles)\n");
    println!(
        "  {:<20}  Rec@10  Rec@30  Hit@10  Hit@30   MRR",
        "config"
    );
    let b_hit30 = base[3] / nf;
    println!(
        "  {:<20}  {:>5.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>5.3}",
        "baseline cosine",
        base[0] / nf,
        base[1] / nf,
        base[2] / nf,
        b_hit30,
        base[4] / nf
    );
    for (bi, &beta) in betas.iter().enumerate() {
        let hit30 = rer[bi][3] / nf;
        let lift = if b_hit30 > 0.0 {
            100.0 * (hit30 - b_hit30) / b_hit30
        } else {
            0.0
        };
        println!(
            "  rerank +W (b={:<5})  {:>5.3}  {:>6.3}  {:>6.3}  {:>6.3}  {:>5.3}   (Hit@30 {:+.1}%)",
            beta,
            rer[bi][0] / nf,
            rer[bi][1] / nf,
            rer[bi][2] / nf,
            hit30,
            rer[bi][4] / nf,
            lift
        );
    }
    println!(
        "\n  shortlist recall ceiling (gold present in top-{shortlist} chunks): {:.3}",
        sl_recall30 / nf
    );
    println!(
        "  Hit@30 at b={}: {} queries gained a relevant doc, {} lost one, {} unchanged (of {})",
        betas[best_bi],
        wins,
        losses,
        n_eval - wins - losses,
        n_eval
    );
    Ok(())
}

/// Aggregate shortlist chunks to their docs by max score, return docs ranked desc.
fn rank_docs<F: Fn(usize) -> f32>(
    cand: &[usize],
    chunk_doc: &[Option<usize>],
    score: F,
) -> Vec<usize> {
    let mut best: HashMap<usize, f32> = HashMap::new();
    for &ci in cand {
        if let Some(doc) = chunk_doc[ci] {
            let s = score(ci);
            best.entry(doc)
                .and_modify(|v| {
                    if s > *v {
                        *v = s
                    }
                })
                .or_insert(s);
        }
    }
    let mut docs: Vec<(usize, f32)> = best.into_iter().collect();
    docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    docs.into_iter().map(|(d, _)| d).collect()
}

/// Build a directed doc-level causal graph from W: doc embeddings (mean-pooled
/// chunks), doc·W rows, validity mask, per-effect top-K softmax-weighted causes
/// (column-stochastic), and per-node in-degree. Edge i->j means "i is a cause
/// of j". `gamma` > 0 applies hub suppression: edge weights are divided by
/// (1+in_degree(i))^gamma before renormalising, so "cause-of-everything" nodes
/// (macro/geopolitical hubs) stop dominating propagation.
fn build_causal_graph(
    ds: &Dataset,
    chunks: &Array2<f32>,
    chunk_doc: &[Option<usize>],
    w: &Array2<f32>,
    k: usize,
    gamma: f32,
    content: bool,
) -> (Array2<f32>, Array2<f32>, Vec<bool>, Vec<Vec<(usize, f32)>>, Vec<u32>) {
    let n = ds.n();
    let d = ds.d();
    let mut doc_emb = Array2::<f32>::zeros((n, d));
    let mut cnt = vec![0u32; n];
    for (ci, dd) in chunk_doc.iter().enumerate() {
        if let Some(a) = dd {
            let row = chunks.row(ci).to_owned();
            let mut acc = doc_emb.row_mut(*a);
            acc += &row;
            cnt[*a] += 1;
        }
    }
    linalg::normalize_rows(&mut doc_emb);
    let valid: Vec<bool> = cnt.iter().map(|&c| c > 0).collect();
    let doc_w = linalg::transport(&doc_emb, w);
    let epoch: Vec<i64> = ds.docs.iter().map(|d| d.published_epoch).collect();
    let s_mat = doc_w.dot(&doc_emb.t()); // S[i,j] = "i causes j"

    // pass 1: raw top-K causes per effect (store raw scores) + in-degree
    let mut raw: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    let mut deg = vec![0u32; n];
    for j in 0..n {
        if !valid[j] {
            continue;
        }
        let mut cand: Vec<(f32, usize)> = (0..n)
            .filter(|&i| valid[i] && i != j && epoch[i] < epoch[j])
            .map(|i| (s_mat[[i, j]], i))
            .collect();
        if cand.is_empty() {
            continue;
        }
        cand.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        cand.truncate(k);
        for &(sc, i) in &cand {
            raw[j].push((i, sc));
            deg[i] += 1;
        }
    }

    // Content-based genericness: a node whose W-projection points at the corpus
    // *centroid* scores moderately-high to everything -> a "cause of everything"
    // hub (macro/boilerplate). gen[i] = doc_W[i]·mean(doc_emb); z-score it.
    // Genuine common roots (Iran/Hormuz) point at a focused cluster, not the mean.
    let mut mean_emb = Array1::<f32>::zeros(d);
    let mut nv = 0f32;
    for i in 0..n {
        if valid[i] {
            mean_emb += &doc_emb.row(i);
            nv += 1.0;
        }
    }
    if nv > 0.0 {
        mean_emb.mapv_inplace(|x| x / nv);
    }
    let gen: Vec<f32> = (0..n)
        .map(|i| if valid[i] { doc_w.row(i).dot(&mean_emb) } else { 0.0 })
        .collect();
    let gmean = gen.iter().enumerate().filter(|(i, _)| valid[*i]).map(|(_, &g)| g).sum::<f32>() / nv.max(1.0);
    let gstd = (gen.iter().enumerate().filter(|(i, _)| valid[*i]).map(|(_, &g)| (g - gmean).powi(2)).sum::<f32>() / nv.max(1.0)).sqrt().max(1e-6);
    let zgen: Vec<f32> = gen.iter().map(|&g| (g - gmean) / gstd).collect();

    // pass 2: softmax weights with hub penalty (content or degree), renormalise
    let mut col_causes: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
    for j in 0..n {
        if raw[j].is_empty() {
            continue;
        }
        let mx = raw[j].iter().map(|&(_, s)| s).fold(f32::MIN, f32::max);
        let mut z = 0.0f32;
        let mut es: Vec<(usize, f32)> = raw[j]
            .iter()
            .map(|&(i, sc)| {
                let pen = if gamma <= 0.0 {
                    1.0
                } else if content {
                    (-gamma * zgen[i]).exp().clamp(1e-3, 1e3) // generic -> down-weighted
                } else {
                    1.0 / (1.0 + deg[i] as f32).powf(gamma)
                };
                let e = ((sc - mx) * 4.0).exp() * pen;
                z += e;
                (i, e)
            })
            .collect();
        for e in es.iter_mut() {
            e.1 /= z;
        }
        col_causes[j] = es;
    }
    (doc_emb, doc_w, valid, col_causes, deg)
}

/// Reciprocal-rank fusion of two doc rankings -> fused ranking (best first).
fn rrf(a: &[usize], b: &[usize], take: usize) -> Vec<usize> {
    let mut score: HashMap<usize, f32> = HashMap::new();
    for (r, &d) in a.iter().take(take).enumerate() {
        *score.entry(d).or_insert(0.0) += 1.0 / (60.0 + r as f32);
    }
    for (r, &d) in b.iter().take(take).enumerate() {
        *score.entry(d).or_insert(0.0) += 1.0 / (60.0 + r as f32);
    }
    let mut v: Vec<(usize, f32)> = score.into_iter().collect();
    v.sort_by(|x, y| y.1.partial_cmp(&x.1).unwrap());
    v.into_iter().map(|(d, _)| d).collect()
}

/// PPR power iteration: s = α·seed + (1-α)·P·s, over the causal graph.
fn ppr(seed: &Array1<f32>, col_causes: &[Vec<(usize, f32)>], alpha: f32, iters: usize) -> Array1<f32> {
    let n = seed.len();
    let pmul = |x: &Array1<f32>| {
        let mut y = Array1::<f32>::zeros(n);
        for (j, causes) in col_causes.iter().enumerate() {
            let xj = x[j];
            if xj == 0.0 {
                continue;
            }
            for &(i, wt) in causes {
                y[i] += wt * xj;
            }
        }
        y
    };
    let mut s = seed.clone();
    for _ in 0..iters {
        let ps = pmul(&s);
        s = &seed.mapv(|x| x * alpha) + &ps.mapv(|x| x * (1.0 - alpha));
    }
    s
}

/// Multi-hop causal retrieval. Build a directed doc-level causal k-NN graph from
/// W (edge i->j = "i is a cause of j"), then for each effect query propagate the
/// 1-hop causal seed back through the graph:  s = Σ_h βʰ Pʰ s0, where P moves
/// mass from an effect to its causes. Tests whether reaching causes-of-causes
/// improves recall over a single hop.
#[allow(clippy::too_many_arguments)]
fn multihop_eval(
    ds: &Dataset,
    df: &HashMap<String, usize>,
    specific_cap: usize,
    queries: &[Query],
    qvec: &Array2<f32>,
    chunks: &Array2<f32>,
    w: &Array2<f32>,
    chunk_doc: &[Option<usize>],
    test_frac: u64,
    cutoff: Option<i64>,
) -> Result<()> {
    let n = ds.n();
    let d = ds.d();
    let getf = |k: &str, dv: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(dv);
    let k_edges: usize = getf("RECEPTORS_K", 10.0) as usize;
    let gamma: f32 = getf("RECEPTORS_HUBSUP", 1.0);
    let alpha: f32 = getf("RECEPTORS_ALPHA", 0.9);
    let iters: usize = getf("RECEPTORS_ITERS", 20.0) as usize;
    let xv_thresh: f32 = getf("RECEPTORS_XV", 0.35); // gold below this cosine = cross-vocab
    let cw_cos: f32 = getf("RECEPTORS_WCOS", 1.0); // RRF weight on cosine in weighted fuse
    let cw_chain: f32 = getf("RECEPTORS_WCHAIN", 2.0); // RRF weight on chain in weighted fuse
    let seed_top: usize = getf("RECEPTORS_SEEDTOP", 0.0) as usize; // 0 = dense seed; M = keep top-M direct causes
    let K = k_edges;
    // raw graph (for in-degree/hub set) and content-suppressed causal graph
    let (doc_emb, doc_w, valid, _col_raw, deg) =
        build_causal_graph(ds, chunks, chunk_doc, w, K, 0.0, false);
    let (_, _, _, col_content, _) = build_causal_graph(ds, chunks, chunk_doc, w, K, gamma, true);
    let epoch: Vec<i64> = ds.docs.iter().map(|d| d.published_epoch).collect();

    // hub set = top 2% of nodes by in-degree (the "cause-of-everything" macro docs)
    let mut degs: Vec<u32> = deg.clone();
    degs.sort_unstable_by(|a, b| b.cmp(a));
    let hub_thresh = degs[(n as f32 * 0.02) as usize].max(1);
    let is_hub: Vec<bool> = deg.iter().map(|&dg| dg >= hub_thresh).collect();

    let ks = [10usize, 30, 100];
    let methods = [
        "cosine (topical)",
        "1-hop W",
        "PPR hub-content",
        "union RRF",
        "abductive H->R",
        "abductive -hubs",   // NEW: best method, post-filter in-degree hubs
        "weighted fuse",     // NEW: RRF, chain up-weighted vs cosine (cross-vocab tilt)
    ];
    let nm = methods.len();
    let mut chain_r: Vec<Vec<f32>> = vec![vec![0.0; ks.len()]; nm]; // chain (1+2 hop) recall
    let mut direct_r: Vec<Vec<f32>> = vec![vec![0.0; ks.len()]; nm]; // direct (1-hop) recall
    let mut xv_r: Vec<Vec<f32>> = vec![vec![0.0; ks.len()]; nm]; // CROSS-VOCAB chain recall (the right attribute)
    let mut hubrate: Vec<f32> = vec![0.0; nm]; // hub-rate@10 (lower = cleaner)
    let mut chain_extra = 0.0f32;
    let mut xv_frac = 0.0f32; // avg share of chain gold that is cross-vocab
    let mut n_xv_q = 0usize; // queries with a non-empty cross-vocab gold set
    let mut n_eval = 0usize;

    for (qi, q) in queries.iter().enumerate() {
        if !is_test(cutoff, q.epoch, &q.effect_doc_id, test_frac) {
            continue;
        }
        let gold = gold_docs(&ds.docs, df, specific_cap, &q.cause_entities, q.epoch);
        // chain gold = 1-hop causes ∪ their causes (2 hops back)
        let mut gold_chain = gold.clone();
        for &c in &gold {
            let g2 = gold_docs(&ds.docs, df, specific_cap, &ds.docs[c].cause_entities, epoch[c]);
            gold_chain.extend(g2);
        }
        // curated "good sample": ≥2 direct causes AND a real 2-hop expansion
        if gold.len() < 2 || gold_chain.len() < gold.len() + 3 {
            continue;
        }
        chain_extra += (gold_chain.len() - gold.len()) as f32;

        let qrow = qvec.row(qi).to_owned();
        let seed_full: Array1<f32> = doc_w.dot(&qrow);
        let mut seed = Array1::<f32>::zeros(n);
        for i in 0..n {
            if valid[i] && epoch[i] < q.epoch {
                seed[i] = seed_full[i].max(0.0);
            }
        }
        // Optional: concentrate the seed on the top-M direct causes so the chain
        // walk starts from a focused hypothesis instead of spraying mass on hubs.
        if seed_top > 0 {
            let mut idx: Vec<usize> = (0..n).collect();
            idx.sort_by(|&i, &j| seed[j].partial_cmp(&seed[i]).unwrap());
            for &i in idx.iter().skip(seed_top) {
                seed[i] = 0.0;
            }
        }
        let ssum: f32 = seed.sum();
        if ssum > 0.0 {
            seed.mapv_inplace(|x| x / ssum);
        }
        let cos_full: Array1<f32> = doc_emb.dot(&qrow);
        let rank_by = |score: &Array1<f32>| -> Vec<usize> {
            let mut v: Vec<(f32, usize)> = (0..n)
                .filter(|&i| valid[i] && epoch[i] < q.epoch)
                .map(|i| (score[i], i))
                .collect();
            v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            v.into_iter().map(|(_, i)| i).collect()
        };
        let mask_future = |mut s: Array1<f32>| {
            for i in 0..n {
                if epoch[i] >= q.epoch {
                    s[i] = 0.0;
                }
            }
            s
        };

        let r_cos = rank_by(&cos_full);
        let r_1hop = rank_by(&seed);
        let hyp = ppr(&seed, &col_content, alpha, iters); // hypothesis chain (multi-hop)
        let r_pcontent = rank_by(&mask_future(hyp.clone()));
        let r_union = rrf(&r_1hop, &r_pcontent, 100);

        // ABDUCTIVE final pass: take the top-of-chain nodes as a hypothesis,
        // form a query centroid, run naive cosine RAG to pull *supporting*
        // documents for that hypothesized cause, then fuse (RRF) the three
        // signals: direct causes (1-hop), the chain (PPR), and the corroborating
        // evidence (cosine-on-hypothesis). No re-PPR -> avoids re-hubbing.
        let r_abd = {
            let mut hv: Vec<(f32, usize)> =
                (0..n).filter(|&i| valid[i]).map(|i| (hyp[i], i)).collect();
            hv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut qhyp = Array1::<f32>::zeros(d);
            for &(wt, i) in hv.iter().take(15) {
                qhyp = &qhyp + &doc_emb.row(i).mapv(|x| x * wt);
            }
            let nn = qhyp.dot(&qhyp).sqrt();
            if nn > 0.0 {
                qhyp.mapv_inplace(|x| x / nn);
            }
            let support: Array1<f32> = doc_emb.dot(&qhyp); // naive cosine RAG (gemv)
            let r_support = rank_by(&support);
            // 4-way RRF: naive cosine (effect-anchored) + direct + chain + support
            let mut sc: HashMap<usize, f32> = HashMap::new();
            for lst in [&r_cos, &r_1hop, &r_pcontent, &r_support] {
                for (r, &d) in lst.iter().take(100).enumerate() {
                    *sc.entry(d).or_insert(0.0) += 1.0 / (60.0 + r as f32);
                }
            }
            let mut v: Vec<(usize, f32)> = sc.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v.into_iter().map(|(d, _)| d).collect::<Vec<usize>>()
        };

        // NEW variant A: abductive with in-degree hubs stripped out post-hoc.
        let r_abd_nohub: Vec<usize> = r_abd.iter().copied().filter(|&i| !is_hub[i]).collect();
        // NEW variant B: weighted RRF — up-weight the chain signal (1-hop ⊕ PPR)
        // relative to cosine, tilting the fusion toward cross-vocabulary reach.
        let r_weighted = {
            let mut sc: HashMap<usize, f32> = HashMap::new();
            for (r, &dd) in r_cos.iter().take(100).enumerate() {
                *sc.entry(dd).or_insert(0.0) += cw_cos / (60.0 + r as f32);
            }
            for lst in [&r_1hop, &r_pcontent] {
                for (r, &dd) in lst.iter().take(100).enumerate() {
                    *sc.entry(dd).or_insert(0.0) += cw_chain / (60.0 + r as f32);
                }
            }
            let mut v: Vec<(usize, f32)> = sc.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v.into_iter().map(|(dd, _)| dd).collect::<Vec<usize>>()
        };

        // cross-vocab gold = chain gold the query embedding ranks low topically
        // (cos < threshold) — the docs cosine structurally cannot surface.
        let gold_xv: HashSet<usize> = gold_chain
            .iter()
            .copied()
            .filter(|&i| cos_full[i] < xv_thresh)
            .collect();
        if !gold_xv.is_empty() {
            xv_frac += gold_xv.len() as f32 / gold_chain.len() as f32;
            n_xv_q += 1;
        }

        let rankings = [&r_cos, &r_1hop, &r_pcontent, &r_union, &r_abd, &r_abd_nohub, &r_weighted];
        for (mi, ranked) in rankings.iter().enumerate() {
            for (ki, &k) in ks.iter().enumerate() {
                let hc = ranked.iter().take(k).filter(|i| gold_chain.contains(i)).count();
                chain_r[mi][ki] += hc as f32 / gold_chain.len() as f32;
                let hd = ranked.iter().take(k).filter(|i| gold.contains(i)).count();
                direct_r[mi][ki] += hd as f32 / gold.len() as f32;
                if !gold_xv.is_empty() {
                    let hx = ranked.iter().take(k).filter(|i| gold_xv.contains(i)).count();
                    xv_r[mi][ki] += hx as f32 / gold_xv.len() as f32;
                }
            }
            let hubs = ranked.iter().take(10).filter(|&&i| is_hub[i]).count();
            hubrate[mi] += hubs as f32 / 10.0;
        }
        n_eval += 1;
    }
    anyhow::ensure!(n_eval > 0, "no evaluable queries in the curated sample");
    let nf = n_eval as f32;
    println!("\n== MULTI-HOP CAUSAL RETRIEVAL — judgeable scoreboard ==");
    println!(
        "  curated sample: {n_eval} effect queries (≥2 direct causes, real 2-hop chain);",
    );
    println!(
        "  chain gold adds {:.1} extra 2-hop docs on avg; cross-vocab (cos<{xv_thresh}) = {:.0}% of chain gold ({} qs)",
        chain_extra / nf,
        100.0 * xv_frac / n_xv_q.max(1) as f32,
        n_xv_q
    );
    println!(
        "  config: K={K} α={alpha} iters={iters} γ={gamma} wcos={cw_cos} wchain={cw_chain}\n"
    );
    println!("  --- recall vs DIRECT gold (1-hop causes) ---");
    println!("  {:<18} {:>7} {:>7} {:>7}", "method", "R@10", "R@30", "R@100");
    for (mi, name) in methods.iter().enumerate() {
        println!(
            "  {:<18} {:>7.3} {:>7.3} {:>7.3}",
            name, direct_r[mi][0] / nf, direct_r[mi][1] / nf, direct_r[mi][2] / nf
        );
    }
    println!("\n  --- recall vs CHAIN gold (1+2 hop) + hub-rate@10 (lower=cleaner) ---");
    println!(
        "  {:<18} {:>7} {:>7} {:>7} {:>9}",
        "method", "R@10", "R@30", "R@100", "hub@10"
    );
    for (mi, name) in methods.iter().enumerate() {
        println!(
            "  {:<18} {:>7.3} {:>7.3} {:>7.3} {:>9.3}",
            name,
            chain_r[mi][0] / nf,
            chain_r[mi][1] / nf,
            chain_r[mi][2] / nf,
            hubrate[mi] / nf
        );
    }
    // THE RIGHT ATTRIBUTE: recall on cross-vocab chain gold (docs cosine misses).
    let xf = n_xv_q.max(1) as f32;
    println!("\n  --- recall vs CROSS-VOCAB chain gold (cos<{xv_thresh}; the docs cosine can't reach) ---");
    println!("  {:<18} {:>7} {:>7} {:>7}", "method", "R@10", "R@30", "R@100");
    for (mi, name) in methods.iter().enumerate() {
        println!(
            "  {:<18} {:>7.3} {:>7.3} {:>7.3}",
            name, xv_r[mi][0] / xf, xv_r[mi][1] / xf, xv_r[mi][2] / xf
        );
    }
    Ok(())
}

/// A/B demo: for each free-text question, retrieve top-k docs via naive cosine
/// and via the in-distribution W retriever; dump doc ids to JSON for the LLM step.
pub fn ask(dir: &Path, lambda: f32, specific_frac: f64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let (df, specific_cap) = doc_df(&ds.docs, specific_frac);

    let chunks: Array2<f32> = read_npy(dir.join("chunks.npy")).context("chunks.npy")?;
    let cmeta: Vec<ChunkMeta> = load_jsonl(&dir.join("chunks.jsonl"))?;
    let chunk_doc: Vec<Option<usize>> = cmeta
        .iter()
        .map(|c| ds.id2idx.get(&c.doc_id).copied())
        .collect();
    let mut chunks_of: Vec<Vec<usize>> = vec![Vec::new(); ds.n()];
    for (ci, dd) in chunk_doc.iter().enumerate() {
        if let Some(a) = dd {
            chunks_of[*a].push(ci);
        }
    }

    let queries: Vec<Query> = load_jsonl(&dir.join("queries.jsonl"))?;
    let qvec: Array2<f32> = read_npy(dir.join("queries.npy")).context("queries.npy")?;

    // train in-dist W on ALL fact pairs (test_frac=0 -> nothing held out)
    let (a_mat, b_mat) = indist_pairs(
        &ds, &df, specific_cap, &queries, &qvec, &chunks, &chunks_of, 0, None,
    );
    let w = linalg::fit_operator(&a_mat, &b_mat, lambda)?;
    eprintln!("trained in-dist W on {} pairs", a_mat.nrows());

    // content-hub-suppressed doc-level causal graph for the multi-hop retriever
    let n = ds.n();
    let d = ds.d();
    let (doc_emb, doc_w, valid, col_hub, _deg) =
        build_causal_graph(&ds, &chunks, &chunk_doc, &w, 10, 1.0, true);

    let qc: Array2<f32> = read_npy(dir.join("questions_custom.npy")).context("questions_custom.npy")?;
    let qtext = std::fs::read_to_string(dir.join("questions_custom.txt"))?;
    let qtexts: Vec<&str> = qtext.lines().filter(|l| !l.trim().is_empty()).collect();
    anyhow::ensure!(qc.nrows() == qtexts.len(), "question count mismatch");

    let k = 6usize;
    let all: Vec<usize> = (0..chunks.nrows()).collect();
    let full_rank = |score: &Array1<f32>| -> Vec<usize> {
        let mut v: Vec<(f32, usize)> = (0..n).filter(|&i| valid[i]).map(|i| (score[i], i)).collect();
        v.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        v.into_iter().map(|(_, i)| i).collect()
    };
    let ids = |docs: &[usize]| -> Vec<&str> {
        docs.iter().take(k).map(|&i| ds.docs[i].doc_id.as_str()).collect()
    };
    let mut out = String::from("[\n");
    for (i, qt) in qtexts.iter().enumerate() {
        let qrow = qc.row(i).to_owned();
        // standard vector (chunk max-pool)
        let cos: Array1<f32> = chunks.dot(&qrow);
        let cos_docs: Vec<&str> = rank_docs(&all, &chunk_doc, |c| cos[c])
            .into_iter()
            .take(k)
            .map(|d| ds.docs[d].doc_id.as_str())
            .collect();
        // 1-hop W, multi-hop PPR (hub-suppressed), and their RRF union
        let seed_full: Array1<f32> = doc_w.dot(&qrow);
        let mut seed = Array1::<f32>::zeros(n);
        for idx in 0..n {
            if valid[idx] {
                seed[idx] = seed_full[idx].max(0.0);
            }
        }
        let ss: f32 = seed.sum();
        if ss > 0.0 {
            seed.mapv_inplace(|x| x / ss);
        }
        let r_1hop = full_rank(&seed);
        let hyp = ppr(&seed, &col_hub, 0.9, 20);
        let r_mh = full_rank(&hyp);
        let r_union = rrf(&r_1hop, &r_mh, 100);
        // abductive: hypothesis centroid -> naive cosine support -> 3-way RRF
        let r_abd = {
            let mut hv: Vec<(f32, usize)> =
                (0..n).filter(|&i| valid[i]).map(|i| (hyp[i], i)).collect();
            hv.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            let mut qhyp = Array1::<f32>::zeros(d);
            for &(wt, i) in hv.iter().take(15) {
                qhyp = &qhyp + &doc_emb.row(i).mapv(|x| x * wt);
            }
            let nn = qhyp.dot(&qhyp).sqrt();
            if nn > 0.0 {
                qhyp.mapv_inplace(|x| x / nn);
            }
            let r_support = full_rank(&doc_emb.dot(&qhyp));
            let r_cosd = full_rank(&doc_emb.dot(&qrow)); // effect-anchored naive RAG
            let mut sc: HashMap<usize, f32> = HashMap::new();
            for lst in [&r_cosd, &r_1hop, &r_mh, &r_support] {
                for (r, &dd) in lst.iter().take(100).enumerate() {
                    *sc.entry(dd).or_insert(0.0) += 1.0 / (60.0 + r as f32);
                }
            }
            let mut v: Vec<(usize, f32)> = sc.into_iter().collect();
            v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            v.into_iter().map(|(dd, _)| dd).collect::<Vec<usize>>()
        };
        let j = |ids: &[&str]| ids.iter().map(|s| format!("\"{s}\"")).collect::<Vec<_>>().join(", ");
        out.push_str(&format!(
            "  {{\"question\": {}, \"cosine\": [{}], \"w1hop\": [{}], \"wmultihop\": [{}], \"union\": [{}], \"abductive\": [{}]}}{}\n",
            serde_json::to_string(qt).unwrap(),
            j(&cos_docs),
            j(&ids(&r_1hop)),
            j(&ids(&r_mh)),
            j(&ids(&r_union)),
            j(&ids(&r_abd)),
            if i + 1 < qtexts.len() { "," } else { "" }
        ));
    }
    out.push_str("]\n");
    std::fs::write(dir.join("ask_results.json"), out)?;
    println!("wrote {}", dir.join("ask_results.json").display());
    Ok(())
}

/// First-stage retrieval: rank the FULL earlier-chunk pool by each scorer,
/// max-pool to docs, measure recall/hit at several cutoffs. Answers "can the
/// operator FIND, not just rank cosine's output?"
#[allow(clippy::too_many_arguments)]
fn retrieval_eval(
    ds: &Dataset,
    df: &HashMap<String, usize>,
    specific_cap: usize,
    queries: &[Query],
    qvec: &Array2<f32>,
    chunks: &Array2<f32>,
    cw: &Array2<f32>,
    chunk_doc: &[Option<usize>],
    chunk_epoch: &[i64],
    test_frac: u64,
    cutoff: Option<i64>,
) -> Result<()> {
    let ks = [10usize, 30, 100, 300];
    // per scorer: [recall@k...] then [hit@k...]
    let mut cos_r = vec![0.0f32; ks.len()];
    let mut w_r = vec![0.0f32; ks.len()];
    let mut fuse_r = vec![0.0f32; ks.len()];
    let mut n_eval = 0usize;

    for (qi, q) in queries.iter().enumerate() {
        if !is_test(cutoff, q.epoch, &q.effect_doc_id, test_frac) {
            continue; // test queries only
        }
        let gold = gold_docs(&ds.docs, df, specific_cap, &q.cause_entities, q.epoch);
        if gold.is_empty() {
            continue;
        }
        let qrow = qvec.row(qi).to_owned();
        let cos: Array1<f32> = chunks.dot(&qrow);
        let dir_s: Array1<f32> = cw.dot(&qrow);
        let pool: Vec<usize> = (0..chunks.nrows())
            .filter(|&i| chunk_epoch[i] < q.epoch)
            .collect();
        if pool.len() < 10 {
            continue;
        }
        // three full-pool rankings (no shortlist)
        let cos_docs = rank_docs(&pool, chunk_doc, |i| cos[i]);
        let w_docs = rank_docs(&pool, chunk_doc, |i| dir_s[i]);
        let fuse_docs = rank_docs(&pool, chunk_doc, |i| cos[i] + dir_s[i]);
        for (ki, &k) in ks.iter().enumerate() {
            let rec = |docs: &[usize]| {
                docs.iter().take(k).filter(|d| gold.contains(d)).count() as f32
                    / gold.len() as f32
            };
            cos_r[ki] += rec(&cos_docs);
            w_r[ki] += rec(&w_docs);
            fuse_r[ki] += rec(&fuse_docs);
        }
        n_eval += 1;
    }
    anyhow::ensure!(n_eval > 0, "no evaluable queries");
    let nf = n_eval as f32;
    println!("\n== FIRST-STAGE RETRIEVAL (full corpus, no cosine shortlist) ==");
    println!("  recall of gold cause-docs, {n_eval} test queries\n");
    println!("  {:<16} {:>7} {:>7} {:>7} {:>7}", "retriever", "R@10", "R@30", "R@100", "R@300");
    let show = |name: &str, r: &[f32]| {
        println!(
            "  {:<16} {:>7.3} {:>7.3} {:>7.3} {:>7.3}",
            name, r[0] / nf, r[1] / nf, r[2] / nf, r[3] / nf
        );
    };
    show("cosine (naive)", &cos_r);
    show("W only", &w_r);
    show("cosine + W", &fuse_r);
    Ok(())
}

/// Build (cause_chunk_embedding, effect_claim_embedding) training pairs from
/// TRAIN-split queries only — the exact distribution the operator is scored on.
/// Caps gold docs/query and chunks/doc so the matrix stays bounded.
fn indist_pairs(
    ds: &Dataset,
    df: &HashMap<String, usize>,
    specific_cap: usize,
    queries: &[Query],
    qvec: &Array2<f32>,
    chunks: &Array2<f32>,
    chunks_of: &[Vec<usize>],
    test_frac: u64,
    cutoff: Option<i64>,
) -> (Array2<f32>, Array2<f32>) {
    const MAX_GOLD: usize = 20;
    const MAX_CHUNKS: usize = 3;
    const MAX_PAIRS: usize = 120_000;
    let mut rows: Vec<(usize, usize)> = Vec::new(); // (chunk_row, query_row)
    for (qi, q) in queries.iter().enumerate() {
        if is_test(cutoff, q.epoch, &q.effect_doc_id, test_frac) {
            continue; // TRAIN queries only
        }
        let gold = gold_docs(&ds.docs, df, specific_cap, &q.cause_entities, q.epoch);
        let mut gold: Vec<usize> = gold.into_iter().collect();
        gold.sort_unstable(); // deterministic subset selection
        for g in gold.into_iter().take(MAX_GOLD) {
            for &ci in chunks_of[g].iter().take(MAX_CHUNKS) {
                rows.push((ci, qi));
                if rows.len() >= MAX_PAIRS {
                    break;
                }
            }
        }
        if rows.len() >= MAX_PAIRS {
            break;
        }
    }
    let d = ds.d();
    let mut a = Array2::<f32>::zeros((rows.len(), d));
    let mut b = Array2::<f32>::zeros((rows.len(), d));
    for (r, &(ci, qi)) in rows.iter().enumerate() {
        a.row_mut(r).assign(&chunks.row(ci));
        b.row_mut(r).assign(&qvec.row(qi));
    }
    (a, b)
}

fn doc_df(docs: &[Doc], specific_frac: f64) -> (HashMap<String, usize>, usize) {
    let mut df: HashMap<String, usize> = HashMap::new();
    for d in docs {
        for e in &d.entities {
            *df.entry(e.clone()).or_insert(0) += 1;
        }
    }
    let cap = (specific_frac * docs.len() as f64).ceil() as usize;
    (df, cap)
}

fn gold_docs(
    docs: &[Doc],
    df: &HashMap<String, usize>,
    specific_cap: usize,
    cause_entities: &[String],
    effect_epoch: i64,
) -> HashSet<usize> {
    let specific: HashSet<&str> = cause_entities
        .iter()
        .filter(|c| df.get(c.as_str()).map_or(false, |&n| n <= specific_cap))
        .map(|s| s.as_str())
        .collect();
    if specific.is_empty() {
        return HashSet::new();
    }
    let mut gold = HashSet::new();
    for (i, d) in docs.iter().enumerate() {
        if d.published_epoch >= effect_epoch {
            continue;
        }
        if d.entities.iter().any(|e| specific.contains(e.as_str())) {
            gold.insert(i);
        }
    }
    gold
}
