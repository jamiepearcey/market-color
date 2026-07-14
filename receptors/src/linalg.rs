//! The actual matrix algebra.
//!
//! Heavy matmuls (E^T E, A·W, ridge normal equations) go through ndarray's
//! `.dot()`, which is dispatched to Apple's Accelerate BLAS (the `blas`
//! feature + `blas-src`). The two small dense-linear-algebra kernels that
//! ndarray doesn't provide — a symmetric eigendecomposition and a matrix
//! inverse, both only 384x384 — use nalgebra.

use anyhow::{anyhow, Result};
use nalgebra::{DMatrix, SymmetricEigen};
use ndarray::{Array1, Array2, Axis};

/// L2-normalise each row in place (safe against zero rows).
pub fn normalize_rows(m: &mut Array2<f32>) {
    for mut row in m.axis_iter_mut(Axis(0)) {
        let n = row.dot(&row).sqrt();
        if n > 1e-9 {
            row.mapv_inplace(|x| x / n);
        }
    }
}

fn to_na(a: &Array2<f32>) -> DMatrix<f32> {
    DMatrix::from_row_iterator(a.nrows(), a.ncols(), a.iter().copied())
}
fn from_na(m: &DMatrix<f32>) -> Array2<f32> {
    Array2::from_shape_fn((m.nrows(), m.ncols()), |(i, j)| m[(i, j)])
}

/// Top-`k` right singular directions of `E` (the "gist" subspace), via the
/// leading eigenvectors of the DxD gram matrix E^T E.
/// Returns (D, k) with orthonormal columns.
pub fn gist_subspace(emb: &Array2<f32>, k: usize) -> Result<Array2<f32>> {
    let gram = emb.t().dot(emb); // GEMM (BLAS)
    let se = SymmetricEigen::new(to_na(&gram));
    // Sort eigen-pairs by eigenvalue descending.
    let mut idx: Vec<usize> = (0..se.eigenvalues.len()).collect();
    idx.sort_by(|&i, &j| {
        se.eigenvalues[j]
            .partial_cmp(&se.eigenvalues[i])
            .unwrap()
    });
    let d = gram.nrows();
    let k = k.min(d);
    let mut out = Array2::<f32>::zeros((d, k));
    for (col, &ev) in idx.iter().take(k).enumerate() {
        for row in 0..d {
            out[[row, col]] = se.eigenvectors[(row, ev)];
        }
    }
    Ok(out)
}

/// E' = E - (E V)(V^T), then re-normalise: binding on the orthogonal-to-topic residual.
pub fn degist(emb: &Array2<f32>, gist: &Array2<f32>) -> Array2<f32> {
    let coeff = emb.dot(gist); // (N, k)  BLAS
    let proj = coeff.dot(&gist.t()); // (N, D)  BLAS
    let mut resid = emb - &proj;
    normalize_rows(&mut resid);
    resid
}

/// Ridge transport operator: B ~= A W,  W = (A^T A + lambda I)^{-1} A^T B.
/// Asymmetric by construction -> encodes direction.
pub fn fit_transport(a: &Array2<f32>, b: &Array2<f32>, lambda: f32) -> Result<Array2<f32>> {
    let d = a.ncols();
    let mut g = a.t().dot(a); // (D, D)  BLAS
    for i in 0..d {
        g[[i, i]] += lambda;
    }
    let rhs = a.t().dot(b); // (D, D)  BLAS
    let ginv = to_na(&g)
        .try_inverse()
        .ok_or_else(|| anyhow!("ridge matrix not invertible (raise lambda)"))?;
    Ok(from_na(&ginv).dot(&rhs)) // (D, D)  BLAS
}

/// Refit the transport operator with a margin/contrastive objective (warm-started
/// from `w0`, typically the ridge solution). For each positive cause->effect pair
/// (a,b) we require the bilinear score aᵀWb to beat two hard negatives by a margin:
///   - the REVERSED pair (bᵀWa)  -> teaches direction
///   - a random later effect     -> teaches specificity (in-batch negative)
/// Full-batch subgradient descent; every step is a handful of BLAS matmuls.
pub fn fit_transport_margin(
    a: &Array2<f32>,
    b: &Array2<f32>,
    w0: &Array2<f32>,
    iters: usize,
    lr: f32,
    margin: f32,
    reg: f32,
) -> Array2<f32> {
    let n = a.nrows();
    let d = a.ncols();
    let mut w = w0.clone();
    for it in 0..iters {
        let aw = a.dot(&w); // (n,d)  aᵀW rows
        let bw = b.dot(&w); // (n,d)  bᵀW rows
        let s_pos = (&aw * b).sum_axis(Axis(1)); // aᵀWb
        let s_rev = (&bw * a).sum_axis(Axis(1)); // bᵀWa (reversed)

        // deterministic derangement -> one random later-effect negative per pos
        let mut perm = vec![0usize; n];
        for i in 0..n {
            let mut j = (i.wrapping_mul(2654435761).wrapping_add(it.wrapping_mul(40503)) + 12345) % n;
            if j == i {
                j = (j + 1) % n;
            }
            perm[i] = j;
        }
        let mut b_perm = Array2::<f32>::zeros((n, d));
        for i in 0..n {
            b_perm.row_mut(i).assign(&b.row(perm[i]));
        }
        let s_neg = (&aw * &b_perm).sum_axis(Axis(1)); // aᵀW b_rand

        // hinge violation masks
        let mut v_neg = Array1::<f32>::zeros(n);
        let mut v_rev = Array1::<f32>::zeros(n);
        for i in 0..n {
            if margin - s_pos[i] + s_neg[i] > 0.0 {
                v_neg[i] = 1.0;
            }
            if margin - s_pos[i] + s_rev[i] > 0.0 {
                v_rev[i] = 1.0;
            }
        }

        // grad of aᵀWb wrt W is the outer product a·bᵀ; accumulate over violations
        // via row-scaled BLAS matmuls:  G = Aᵀdiag(vn)(Bperm-B) + diag(vr)(BᵀA - AᵀB)
        let a_vn = a * &v_neg.view().insert_axis(Axis(1));
        let a_vr = a * &v_rev.view().insert_axis(Axis(1));
        let b_vr = b * &v_rev.view().insert_axis(Axis(1));
        let g = a_vn.t().dot(&b_perm) - a_vn.t().dot(b) + b_vr.t().dot(a) - a_vr.t().dot(b);

        let scale = lr / n as f32;
        w = &w - &g.mapv(|x| x * scale) - &w.mapv(|x| x * (lr * reg));
    }
    w
}

/// Apply the operator to every cause row and re-normalise: predicted-effect rows.
pub fn transport(rows: &Array2<f32>, w: &Array2<f32>) -> Array2<f32> {
    let mut out = rows.dot(w); // (N, D)  BLAS
    normalize_rows(&mut out);
    out
}

#[allow(dead_code)]
pub fn cos(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    a.dot(b)
}

/// Fit the operator on cause rows `a` and effect rows `b`. Ridge by default;
/// set RECEPTORS_FIT=margin to refine with the contrastive objective.
pub fn fit_operator(a: &Array2<f32>, b: &Array2<f32>, lambda: f32) -> anyhow::Result<Array2<f32>> {
    let w = fit_transport(a, b, lambda)?;
    if std::env::var("RECEPTORS_FIT").as_deref() == Ok("margin") {
        let get = |k: &str, d: f32| std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d);
        let iters = get("MARGIN_ITERS", 150.0) as usize;
        Ok(fit_transport_margin(
            a, b, &w, iters,
            get("MARGIN_LR", 0.5),
            get("MARGIN_M", 0.1),
            get("MARGIN_REG", 0.01),
        ))
    } else {
        Ok(w)
    }
}
