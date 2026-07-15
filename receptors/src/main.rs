//! Receptor / transport-operator experiment.
//!
//! Question: can pure BLAS matrix algebra over embeddings recover directed
//! cause->effect structure between news articles — using only the free labels
//! already sitting in `cause_entities` — well enough to beat plain cosine,
//! with NO per-chunk LLM inference?
//!
//! Pipeline:  load embeddings -> de-gist (SVD) -> mine directed pairs ->
//!            ridge-fit asymmetric transport operator W -> evaluate.

extern crate blas_src; // links Accelerate BLAS for ndarray's `.dot()`

mod arc;
mod atlas;
mod confound;
mod consensus;
mod data;
mod eval;
mod facts;
mod generalize;
mod impact;
mod latency;
mod linalg;
mod mechanism;
mod metric;
mod modality;
mod novelty;
mod pairs;
mod polarity;
mod price;
mod regime;
mod report;
mod rerank;
mod source;
mod spectrum;
mod spillover;

use anyhow::Result;
use ndarray::Array2;
use std::path::PathBuf;

struct Cfg {
    dir: PathBuf,
    gist_k: usize,     // topical directions to strip
    lambda: f32,       // ridge regularisation
    specific_frac: f64,// entity "specific" if in <= this frac of docs
    test_frac: u64,    // percent of effect-docs held out
}

fn getenv_or<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn main() -> Result<()> {
    let cfg = Cfg {
        dir: PathBuf::from(
            std::env::var("RECEPTORS_DATA").unwrap_or_else(|_| "data".into()),
        ),
        gist_k: getenv_or("RECEPTORS_K", 0usize), // de-gist off by default (it hurt direction)
        lambda: getenv_or("RECEPTORS_LAMBDA", 0.1f32),
        specific_frac: getenv_or("RECEPTORS_SPECIFIC", 0.03f64),
        test_frac: getenv_or("RECEPTORS_TEST", 30u64),
    };

    // Subcommand: A/B retrieval demo for sample questions.
    if std::env::args().nth(1).as_deref() == Some("ask") {
        return rerank::ask(&cfg.dir, cfg.lambda, cfg.specific_frac);
    }

    // Subcommand: generalization spot-checks (temporal / specificity / null).
    if std::env::args().nth(1).as_deref() == Some("generalize") {
        return generalize::run(&cfg.dir, cfg.lambda, cfg.specific_frac, cfg.test_frac);
    }

    // Subcommand: price-grounded receptor.
    if std::env::args().nth(1).as_deref() == Some("price") {
        return price::run(&cfg.dir, cfg.lambda);
    }
    if std::env::args().nth(1).as_deref() == Some("polarity") {
        return polarity::run(&cfg.dir, cfg.lambda);
    }
    if std::env::args().nth(1).as_deref() == Some("modality") {
        return modality::run(&cfg.dir, cfg.lambda);
    }
    if std::env::args().nth(1).as_deref() == Some("novelty") {
        return novelty::run(&cfg.dir);
    }
    if std::env::args().nth(1).as_deref() == Some("novelq") {
        return novelty::run_query(&cfg.dir);
    }

    // ---- second-generation receptors ----
    match std::env::args().nth(1).as_deref() {
        Some("mechanism") => return mechanism::run(&cfg.dir, cfg.lambda, cfg.specific_frac, cfg.test_frac),
        Some("spectrum") => return spectrum::run(&cfg.dir, cfg.lambda, cfg.specific_frac, cfg.test_frac),
        Some("atlas") => return atlas::run(&cfg.dir, cfg.lambda),
        Some("impact") => return impact::run(&cfg.dir, cfg.lambda),
        Some("latency") => return latency::run(&cfg.dir, cfg.lambda, cfg.specific_frac, cfg.test_frac),
        Some("spillover") => return spillover::run(&cfg.dir, cfg.lambda),
        Some("confound") => return confound::run(&cfg.dir, cfg.lambda, cfg.specific_frac),
        Some("source") => return source::run(&cfg.dir, cfg.lambda),
        Some("metric") => return metric::run(&cfg.dir, cfg.lambda, cfg.specific_frac, cfg.test_frac),
        Some("consensus") => return consensus::run(&cfg.dir),
        Some("regime") => return regime::run(&cfg.dir),
        Some("arc") => return arc::run(&cfg.dir, cfg.lambda, cfg.specific_frac, cfg.test_frac),
        _ => {}
    }

    // Subcommand: RAG reranking experiment.
    if std::env::args().nth(1).as_deref() == Some("rerank") {
        return rerank::run(
            &cfg.dir,
            cfg.lambda,
            cfg.specific_frac,
            cfg.test_frac,
            getenv_or("RECEPTORS_SHORTLIST", 100usize),
            &[0.25, 0.5, 1.0, 2.0],
        );
    }

    let mut ds = data::Dataset::load(&cfg.dir)?;
    linalg::normalize_rows(&mut ds.emb);
    println!(
        "loaded {} docs x {} dims  (span {} .. {})",
        ds.n(),
        ds.d(),
        ds.docs.first().map(|d| d.date.as_str()).unwrap_or("?"),
        ds.docs.last().map(|d| d.date.as_str()).unwrap_or("?"),
    );

    // --- de-gist: strip the dominant topical subspace ---
    let gist = linalg::gist_subspace(&ds.emb, cfg.gist_k)?;
    let resid = linalg::degist(&ds.emb, &gist);
    println!("de-gisted: stripped top-{} topical directions", cfg.gist_k);

    // --- mine directed cause->effect pairs from cause_entities (free labels) ---
    let mined = pairs::mine(&ds, cfg.specific_frac, cfg.test_frac);
    let n_train = mined.pairs.iter().filter(|p| !p.is_test).count();
    let n_test = mined.pairs.iter().filter(|p| p.is_test).count();
    println!(
        "mined {} directed pairs ({} train / {} test) over {} specific entities",
        mined.pairs.len(),
        n_train,
        n_test,
        mined.n_specific_entities
    );
    anyhow::ensure!(n_train > 10 && n_test > 10, "not enough pairs to evaluate");

    // --- fit the transport operator W on TRAIN pairs ---
    // Fit on BOTH raw and de-gisted rows so we can isolate what de-gisting buys.
    let d = ds.d();
    let fit_on = |src: &Array2<f32>| -> Result<(Array2<f32>, Array2<f32>)> {
        let mut a_mat = Array2::<f32>::zeros((n_train, d));
        let mut b_mat = Array2::<f32>::zeros((n_train, d));
        for (r, p) in mined.pairs.iter().filter(|p| !p.is_test).enumerate() {
            a_mat.row_mut(r).assign(&src.row(p.a));
            b_mat.row_mut(r).assign(&src.row(p.b));
        }
        let w = linalg::fit_operator(&a_mat, &b_mat, cfg.lambda)?;
        let ahat = linalg::transport(src, &w);
        Ok((w, ahat))
    };
    let (_w_raw, ahat_raw) = fit_on(&ds.emb)?;
    let (_w, ahat) = fit_on(&resid)?;
    println!("fit transport operator W ({d}x{d}) via ridge, lambda={}", cfg.lambda);

    let epochs: Vec<i64> = ds.docs.iter().map(|d| d.published_epoch).collect();

    // ===================== DIRECTION TASK =====================
    let dir_raw = eval::direction(&mined.pairs, &ahat_raw, &ds.emb);
    let dir = eval::direction(&mined.pairs, &ahat, &resid);
    println!("\n== DIRECTION (unordered pair -> which is cause? content only) ==");
    println!("  test pairs                  : {}", dir.n);
    println!("  symmetric cosine (baseline) : {:.3}  (0.5 by construction)", dir.cosine_acc);
    println!("  transport, no de-gist       : {:.3}", dir_raw.transport_acc);
    println!("  transport, de-gisted        : {:.3}", dir.transport_acc);

    // ===================== RETRIEVAL TASK =====================
    // Scorers over the SAME time-masked candidate pool.
    let raw = eval::retrieval("raw cosine        ", &mined.pairs, &epochs, |a, j| {
        ds.emb.row(a).dot(&ds.emb.row(j))
    });
    let deg = eval::retrieval("de-gisted cosine  ", &mined.pairs, &epochs, |a, j| {
        resid.row(a).dot(&resid.row(j))
    });
    let trn = eval::retrieval("transport (ours)  ", &mined.pairs, &epochs, |a, j| {
        ahat.row(a).dot(&resid.row(j))
    });
    let cos_trn = eval::retrieval("cosine + transport", &mined.pairs, &epochs, |a, j| {
        ds.emb.row(a).dot(&ds.emb.row(j)) + 0.3 * ahat.row(a).dot(&resid.row(j))
    });
    let fused = eval::retrieval("cosine+transp+ent ", &mined.pairs, &epochs, |a, j| {
        let c = ds.emb.row(a).dot(&ds.emb.row(j));
        let t = ahat.row(a).dot(&resid.row(j));
        let e = eval::entity_jaccard(&ds.docs[a].entities, &ds.docs[j].entities);
        c + 0.3 * t + 0.5 * e
    });

    println!("\n== RETRIEVAL (rank the true later effect of each cause) ==");
    println!("  {:<20}  causes  Recall@10   MRR", "scorer");
    for r in [&raw, &deg, &trn, &cos_trn, &fused] {
        println!(
            "  {:<20}  {:>6}   {:>7.3}  {:>6.3}",
            r.label, r.causes, r.recall_at_10, r.mrr
        );
    }

    // ===================== QUALITATIVE =====================
    println!("\n== sample directed edges the operator ranks #1 (test causes) ==");
    show_examples(&ds, &mined, &ahat, &resid, &epochs, 6);

    Ok(())
}

fn show_examples(
    ds: &data::Dataset,
    mined: &pairs::Mined,
    ahat: &Array2<f32>,
    resid: &Array2<f32>,
    epochs: &[i64],
    limit: usize,
) {
    use std::collections::{HashMap, HashSet};
    let mut truth: HashMap<usize, HashSet<usize>> = HashMap::new();
    for p in mined.pairs.iter().filter(|p| p.is_test) {
        truth.entry(p.a).or_default().insert(p.b);
    }
    let mut causes: Vec<usize> = truth.keys().copied().collect();
    causes.sort();

    let mut shown = 0;
    for a in causes {
        let ta = epochs[a];
        let mut best = (f32::MIN, usize::MAX);
        for j in 0..ds.n() {
            if j == a || epochs[j] <= ta {
                continue;
            }
            let s = ahat.row(a).dot(&resid.row(j));
            if s > best.0 {
                best = (s, j);
            }
        }
        if best.1 == usize::MAX {
            continue;
        }
        let b = best.1;
        let hit = truth[&a].contains(&b);
        let da = &ds.docs[a];
        let db = &ds.docs[b];
        println!(
            "  [{}] {} {}  -->  {} {}",
            if hit { "HIT " } else { "miss" },
            da.date,
            short(&da.doc_id),
            db.date,
            short(&db.doc_id),
        );
        println!("        cause about : {}", preview(&da.entities));
        println!("        effect blames: {}", preview(&db.cause_entities));
        shown += 1;
        if shown >= limit {
            break;
        }
    }
}

fn short(id: &str) -> &str {
    &id[..id.len().min(8)]
}
fn preview(v: &[String]) -> String {
    v.iter().take(6).cloned().collect::<Vec<_>>().join(", ")
}
