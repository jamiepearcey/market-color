//! §3 qualitative program — spectral reading + anisotropy.
//!
//! (1) Spectral reading: the ridge transport operator W is already a compact
//!     summary of the corpus's causal geometry. Its top singular channels
//!     (v_in -> u_out) are the dominant cause->effect directions. We name each
//!     channel by the entities of the docs that load most strongly on its input
//!     and output directions — reading the latent causal grammar straight out of
//!     the operator, at near-zero cost.
//!
//! (2) Anisotropy accounting: how much of the embedding variance is topical
//!     "gist" (a few leading directions) vs the residual where the causal signal
//!     lives — and how direction accuracy moves as we strip gist directions.

use crate::data::Dataset;
use crate::linalg;
use crate::pairs::{self, Pair};
use anyhow::Result;
use ndarray::{Array1, Array2};
use serde_json::json;
use std::collections::HashMap;
use std::path::Path;

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

/// Top entity strings among the docs whose embedding projects most on `dir`.
fn top_entities(ds: &Dataset, proj: &Array1<f32>, use_cause: bool, top_docs: usize) -> String {
    let mut idx: Vec<usize> = (0..proj.len()).collect();
    idx.sort_by(|&i, &j| proj[j].partial_cmp(&proj[i]).unwrap());
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for &i in idx.iter().take(top_docs) {
        let ents = if use_cause {
            &ds.docs[i].cause_entities
        } else {
            &ds.docs[i].entities
        };
        for e in ents {
            *counts.entry(e.as_str()).or_default() += 1;
        }
    }
    let mut cv: Vec<(&str, usize)> = counts.into_iter().collect();
    cv.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    cv.iter()
        .take(6)
        .map(|(e, _)| *e)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn run(dir: &Path, lambda: f32, specific_frac: f64, test_frac: u64) -> Result<()> {
    let mut ds = Dataset::load(dir)?;
    linalg::normalize_rows(&mut ds.emb);
    let mined = pairs::mine(&ds, specific_frac, test_frac);
    let train: Vec<&Pair> = mined.pairs.iter().filter(|p| !p.is_test).collect();
    anyhow::ensure!(train.len() > 50, "not enough pairs");

    // ---- fit global W and read its singular channels ----
    let (a, b) = stack(&ds.emb, &train);
    let w = linalg::fit_transport(&a, &b, lambda)?;
    let channels = linalg::top_singular(&w, 8);

    println!("== spectral reading of the transport operator W ({} channels) ==", channels.len());
    println!("   each channel: causes ABOUT (v_in)  -->  effects that BLAME (u_out)\n");
    for (k, (sigma, v_in, u_out)) in channels.iter().enumerate() {
        let in_proj = ds.emb.dot(v_in); // (N,) load of each doc on input dir
        let out_proj = ds.emb.dot(u_out);
        println!("  channel {k}  (σ={:.2})", sigma);
        println!("    cause about  : {}", top_entities(&ds, &in_proj, false, 25));
        println!("    effect blames: {}", top_entities(&ds, &out_proj, true, 25));
    }

    // ---- anisotropy: variance concentration in the embedding cloud ----
    let gram = ds.emb.t().dot(&ds.emb);
    // eigenvalues via the gist_subspace path (reuse symmetric eig): grab full spectrum
    let evals = gram_eigenvalues(&gram);
    let total: f32 = evals.iter().sum();
    println!("\n== anisotropy: share of embedding variance in the top-k directions ==");
    let mut cum = 0.0;
    for (k, &ev) in evals.iter().take(20).enumerate() {
        cum += ev;
        if k < 5 || k == 9 || k == 19 {
            println!("  top {:>2} dirs: {:>5.1}% cumulative", k + 1, 100.0 * cum / total);
        }
    }

    // ---- direction accuracy vs gist stripped ----
    let test: Vec<&Pair> = mined.pairs.iter().filter(|p| p.is_test).collect();
    println!("\n== direction accuracy vs # gist directions stripped ==");
    println!("  {:>6}  {:>9}", "gist_k", "dir-acc");
    let mut gist_sweep: Vec<(usize, f32)> = Vec::new();
    for &gk in &[0usize, 1, 2, 4, 8, 16] {
        let gist = linalg::gist_subspace(&ds.emb, gk)?;
        let resid = if gk == 0 {
            ds.emb.clone()
        } else {
            linalg::degist(&ds.emb, &gist)
        };
        let (ar, br) = stack(&resid, &train);
        let wr = linalg::fit_transport(&ar, &br, lambda)?;
        let ahat = linalg::transport(&resid, &wr);
        let mut ok = 0usize;
        for p in &test {
            let fwd = ahat.row(p.a).dot(&resid.row(p.b));
            let rev = ahat.row(p.b).dot(&resid.row(p.a));
            if fwd > rev {
                ok += 1;
            }
        }
        let acc = ok as f32 / test.len() as f32;
        println!("  {:>6}  {:>9.3}", gk, acc);
        gist_sweep.push((gk, acc));
    }

    // ---- emit spectrum bundle ----
    let sigmas: Vec<f32> = channels.iter().map(|c| c.0).collect();
    let cumvar: Vec<f32> = {
        let total: f32 = evals.iter().sum();
        let mut c = 0.0;
        evals.iter().take(40).map(|&e| { c += e; 100.0 * c / total }).collect()
    };
    crate::report::save(
        dir,
        "spectrum",
        &json!({
            "channel_sigmas": sigmas,
            "cumulative_variance_pct": cumvar,
            "gist_sweep": gist_sweep.iter().map(|(k, a)| json!({"gist_k": k, "acc": a})).collect::<Vec<_>>(),
        }),
    )?;
    Ok(())
}

/// Descending eigenvalues of a symmetric matrix.
fn gram_eigenvalues(gram: &Array2<f32>) -> Vec<f32> {
    use nalgebra::{DMatrix, SymmetricEigen};
    let m = DMatrix::from_row_iterator(gram.nrows(), gram.ncols(), gram.iter().copied());
    let se = SymmetricEigen::new(m);
    let mut ev: Vec<f32> = se.eigenvalues.iter().copied().collect();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
    ev
}
