//! WORKFLOW orchestrator (standalone rayon runner).
//!
//!   load feeds
//!     -> par fetch_feed         (ACTIVITY 1)
//!     -> resolve_url + dedup    (ACTIVITY 2) vs on-disk state
//!     -> par fetch_and_extract  (ACTIVITY 3)
//!     -> write_partitions       (ACTIVITY 4)
//!     -> emit last_crawl_report.json
//!
//! Blocking HTTP + rayon means no async runtime. Each stage is a retryable,
//! idempotent activity, so the whole job is at-least-once safe: dedup by
//! content_hash + canonical_url makes re-runs converge. See `temporal.rs` for the
//! durable-orchestration drop-in.

use super::activities::{
    self, build_agent, corpus_size, fetch_and_extract, fetch_feed, load_existing_state,
    resolve_url, write_partitions, DomainThrottle, Semaphore,
};
use super::{load_all_feeds, Article, FeedSpec};
use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

pub struct ScrapeConfig {
    pub feeds: Vec<PathBuf>,
    pub out: PathBuf,
    pub since_hours: i64,
    pub max_per_feed: usize,
    pub concurrency: usize,
    pub limit: Option<usize>,
    pub per_domain_delay: f64,
    pub timeout_secs: u64,
}

#[derive(serde::Serialize)]
pub struct ScrapeReport {
    pub run_utc: String,
    pub sources: usize,
    pub discovered: usize,
    pub fetched: usize,
    pub kept_this_run: usize,
    pub full_body_this_run: usize,
    pub per_source_discovered: HashMap<String, usize>,
    pub partitions_written_this_run: HashMap<String, usize>,
    pub corpus_total_rows: usize,
    pub corpus_partitions: usize,
    pub gnews_total: usize,
    pub gnews_decoded: usize,
    pub gnews_skipped: usize,
    pub gnews_decode_rate: f64,
}

pub fn run(cfg: &ScrapeConfig) -> Result<ScrapeReport> {
    let sources = load_all_feeds(&cfg.feeds)?;
    let now = Utc::now();
    let since = if cfg.since_hours > 0 {
        Some(now - ChronoDuration::hours(cfg.since_hours))
    } else {
        None
    };

    std::fs::create_dir_all(&cfg.out)?;
    let (seen_ids, seen_hashes) = load_existing_state(&cfg.out);
    eprintln!(
        "[discovery] {} sources | since={} | resume-known={} docs",
        sources.len(),
        since.map(|s| s.to_rfc3339()).unwrap_or_else(|| "all".into()),
        seen_ids.len()
    );

    // Bounded rayon pool = concurrency cap (mirrors the Python semaphore).
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(cfg.concurrency.clamp(1, 64))
        .build()?;

    let agent = build_agent(cfg.timeout_secs);
    let throttle = DomainThrottle::new(cfg.per_domain_delay);
    // Cap concurrent hits to news.google.com (batchexecute rate-limit safety).
    let gnews_sem = Semaphore::new(6);
    let gnews_total = AtomicUsize::new(0);
    let gnews_decoded = AtomicUsize::new(0);
    let gnews_skipped = AtomicUsize::new(0);

    // ---- ACTIVITY 1 (parallel): fetch every feed ----
    type Work = (FeedSpec, String, &'static str, super::FeedItem);
    let per_source: Mutex<HashMap<String, usize>> = Mutex::new(HashMap::new());
    let discovered: Vec<Work> = pool.install(|| {
        sources
            .par_iter()
            .flat_map(|spec| {
                let items = fetch_feed(&agent, spec, since, cfg.max_per_feed);
                per_source
                    .lock()
                    .unwrap()
                    .insert(spec.name.clone(), items.len());
                let kind: &'static str = if spec.method == "google_news_rss" {
                    "google_news"
                } else {
                    "rss"
                };
                // ---- ACTIVITY 2: resolve google-news redirects ----
                let is_gnews = spec.method == "google_news_rss";
                let mut out = Vec::new();
                for it in items {
                    if is_gnews {
                        gnews_total.fetch_add(1, Ordering::Relaxed);
                    }
                    match resolve_url(&agent, &gnews_sem, &it, &spec.method) {
                        Some(u) => {
                            if is_gnews {
                                gnews_decoded.fetch_add(1, Ordering::Relaxed);
                            }
                            out.push((spec.clone(), u, kind, it));
                        }
                        None => {
                            if is_gnews {
                                gnews_skipped.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
                out
            })
            .collect()
    });
    let per_source_discovered = per_source.into_inner().unwrap();

    // Dedup canonical URLs across sources + against on-disk doc_ids.
    let mut seen_canon: HashSet<String> = HashSet::new();
    let mut work: Vec<Work> = Vec::new();
    for (spec, url, kind, item) in discovered {
        let canon = super::urls::canonicalize(&url);
        if !seen_canon.insert(canon.clone()) {
            continue;
        }
        let doc_id = super::urls::stable_id(&[&canon]);
        if seen_ids.contains(&doc_id) {
            continue; // already collected in a prior run
        }
        work.push((spec, url, kind, item));
    }
    if let Some(limit) = cfg.limit {
        work.truncate(limit);
    }
    let discovered_n = work.len();
    let g_total = gnews_total.load(Ordering::Relaxed);
    let g_dec = gnews_decoded.load(Ordering::Relaxed);
    let g_rate = if g_total > 0 {
        g_dec as f64 / g_total as f64
    } else {
        0.0
    };
    eprintln!(
        "[discovery] {} unique URLs to fetch | google-news decoded {}/{} ({:.0}%)",
        discovered_n,
        g_dec,
        g_total,
        g_rate * 100.0
    );

    // ---- ACTIVITY 3 (parallel): fetch + extract ----
    let seen_hashes = Mutex::new(seen_hashes);
    let run_ids = Mutex::new(HashSet::<String>::new());
    let articles: Vec<Article> = pool.install(|| {
        work.par_iter()
            .filter_map(|(spec, url, kind, item)| {
                let art = fetch_and_extract(&agent, url, spec, kind, item, &throttle, now)?;
                // dedup vs prior-run + this-run content hashes
                if let Some(h) = &art.row.content_hash {
                    let mut hs = seen_hashes.lock().unwrap();
                    if hs.contains(h) {
                        return None;
                    }
                    hs.insert(h.clone());
                }
                let mut ids = run_ids.lock().unwrap();
                if !ids.insert(art.row.doc_id.clone()) {
                    return None;
                }
                Some(art)
            })
            .collect()
    });

    let fetched = articles.len();
    let full_body = articles.iter().filter(|a| a.row.extraction_ok).count();

    // ---- ACTIVITY 4: date-partitioned merge-write ----
    let partitions_written_this_run = write_partitions(&cfg.out, &articles)?;
    let (corpus_total_rows, corpus_partitions) = corpus_size(&cfg.out);

    let report = ScrapeReport {
        run_utc: now.to_rfc3339(),
        sources: sources.len(),
        discovered: discovered_n,
        fetched,
        kept_this_run: fetched,
        full_body_this_run: full_body,
        per_source_discovered,
        partitions_written_this_run,
        corpus_total_rows,
        corpus_partitions,
        gnews_total: g_total,
        gnews_decoded: g_dec,
        gnews_skipped: gnews_skipped.load(Ordering::Relaxed),
        gnews_decode_rate: g_rate,
    };
    let report_path = cfg.out.join("last_crawl_report.json");
    std::fs::write(&report_path, serde_json::to_string_pretty(&report)?)?;
    eprintln!(
        "fetched(kept,deduped)={} full_body={} -> corpus now {} rows across {} partitions -> {}",
        fetched,
        full_body,
        corpus_total_rows,
        corpus_partitions,
        cfg.out.display()
    );

    // Keep the module import used even if the parse_published helper is unused.
    let _ = activities::parse_published;
    Ok(report)
}
