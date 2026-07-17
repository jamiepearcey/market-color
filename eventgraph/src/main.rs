//! eventgraph CLI: ingest a feed -> narrative extraction -> Postgres upsert SQL
//! + DuckLake fact load.
//!
//!   eventgraph ingest --feed docs.jsonl --out out/ [--model M | --mock] [--limit N]
//!                     [--apply-lake] [--catalog ...] [--data-path ...]

use anyhow::Result;
use clap::{Parser, Subcommand};
use eventgraph::extract::{Extractor, GroqExtractor, MockExtractor, PROMPT_VERSION};
use eventgraph::feed::{Feed, JsonlFeed, ParquetFeed};
use eventgraph::model::SCHEMA_VERSION;
use eventgraph::pipeline::{self, RunMeta};
use eventgraph::scrape::workflow::{self, ScrapeConfig};
use eventgraph::{lake, pg};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "eventgraph", about = "feeds -> event/sensitivity graph -> Postgres + DuckLake")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Ingest a raw-doc feed (JSONL or scraped Parquet corpus) and emit Postgres
    /// SQL + DuckLake facts.
    Ingest(Ingest),
    /// Scrape all configured news feeds into a date-partitioned Parquet corpus.
    Scrape(Scrape),
}

#[derive(Parser)]
struct Scrape {
    /// Comma-separated feeds JSON files (feeds.json,feeds_em.json shape).
    #[arg(long, default_value = "../config/feeds.json,../config/feeds_em.json", value_delimiter = ',')]
    feeds: Vec<PathBuf>,
    /// Corpus output root (date-partitioned dt=YYYY-MM-DD/part-000.parquet).
    #[arg(long, default_value = "../data/news_corpus")]
    out: PathBuf,
    /// Recency window in hours (0 = all).
    #[arg(long, default_value = "48")]
    since_hours: i64,
    /// Max items taken per feed.
    #[arg(long, default_value = "200")]
    max_per_feed: usize,
    /// Parallel fetch workers.
    #[arg(long, default_value = "12")]
    concurrency: usize,
    /// Global cap on total items fetched (smoke-testing).
    #[arg(long)]
    limit: Option<usize>,
    /// Seconds between hits to the same domain (polite throttle).
    #[arg(long, default_value = "1.0")]
    per_domain_delay: f64,
    /// Per-request read/write timeout in seconds.
    #[arg(long, default_value = "30")]
    timeout: u64,
}

#[derive(Parser)]
struct Ingest {
    #[arg(long)]
    feed: PathBuf,
    #[arg(long, default_value = "out")]
    out: PathBuf,
    #[arg(long, default_value = "openai/gpt-oss-120b")]
    model: String,
    /// Use the offline mock extractor (no API key needed).
    #[arg(long)]
    mock: bool,
    #[arg(long)]
    limit: Option<usize>,
    /// Parallel extraction workers (network-bound; raise for throughput).
    #[arg(long, default_value = "8")]
    concurrency: usize,
    /// Ignore any existing checkpoint and re-extract from scratch.
    #[arg(long)]
    fresh: bool,
    /// After writing, load the facts into DuckLake via the duckdb CLI.
    #[arg(long)]
    apply_lake: bool,
    #[arg(long, default_value = "ducklake:eventgraph_catalog.ducklake")]
    catalog: String,
    #[arg(long, default_value = "eg_data/")]
    data_path: String,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Ingest(a) => ingest(a),
        Cmd::Scrape(a) => scrape(a),
    }
}

fn scrape(a: Scrape) -> Result<()> {
    let cfg = ScrapeConfig {
        feeds: a.feeds,
        out: a.out,
        since_hours: a.since_hours,
        max_per_feed: a.max_per_feed,
        concurrency: a.concurrency,
        limit: a.limit,
        per_domain_delay: a.per_domain_delay,
        timeout_secs: a.timeout,
    };
    let report = workflow::run(&cfg)?;
    println!("== eventgraph scrape ==");
    println!(
        "sources {}  discovered {}  kept {}  full_body {}",
        report.sources, report.discovered, report.kept_this_run, report.full_body_this_run,
    );
    println!(
        "google-news decoded {}/{} ({:.0}%)  skipped {}",
        report.gnews_decoded,
        report.gnews_total,
        report.gnews_decode_rate * 100.0,
        report.gnews_skipped
    );
    println!(
        "corpus {} rows / {} partitions -> {}",
        report.corpus_total_rows,
        report.corpus_partitions,
        cfg.out.display()
    );
    Ok(())
}

fn ingest(a: Ingest) -> Result<()> {
    std::fs::create_dir_all(&a.out)?;
    let now = chrono::Utc::now();
    let meta = RunMeta {
        run_id: format!("run_{}", now.format("%Y%m%dT%H%M%SZ")),
        model: if a.mock { "mock".into() } else { a.model.clone() },
        prompt_version: PROMPT_VERSION.into(),
        schema_version: SCHEMA_VERSION.into(),
        created_at: now.to_rfc3339(),
    };

    // Pick the reader by shape: a `.parquet` file or a dt=*/ corpus dir -> the
    // Parquet drop-in; anything else -> the raw-doc JSONL reader.
    let is_parquet = a.feed.extension().and_then(|e| e.to_str()) == Some("parquet")
        || (a.feed.is_dir()
            && std::fs::read_dir(&a.feed).map(|mut d| {
                d.any(|e| {
                    e.ok()
                        .and_then(|e| e.file_name().into_string().ok())
                        .is_some_and(|n| n.starts_with("dt="))
                })
            }).unwrap_or(false));
    let feed: Box<dyn Feed> = if is_parquet {
        Box::new(ParquetFeed::new(&a.feed, a.limit))
    } else {
        Box::new(JsonlFeed::new(&a.feed, a.limit))
    };
    let extractor: Box<dyn Extractor> = if a.mock {
        Box::new(MockExtractor)
    } else {
        Box::new(GroqExtractor::from_env(&a.model)?)
    };

    // Auto-checkpoint to out/extractions.jsonl so an interrupted run resumes.
    let checkpoint = a.out.join("extractions.jsonl");
    if a.fresh {
        let _ = std::fs::remove_file(&checkpoint);
    }
    let batch = pipeline::run(&*feed, &*extractor, &meta, a.concurrency, Some(&checkpoint))?;

    // Postgres tier
    let pg_sql = pg::upsert_sql(&batch, &meta);
    let pg_path = a.out.join("pg_upsert.sql");
    std::fs::write(&pg_path, pg_sql)?;

    // DuckLake tier
    let lake_dir = a.out.join("lake");
    let tables = lake::write_lake(&lake_dir, &batch)?;
    let ddl_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("lake");
    let script = lake::load_script(&tables, &a.catalog, &a.data_path, &ddl_dir);
    let script_path = lake::write_script(&a.out, &script)?;

    println!("== eventgraph ingest ==");
    println!(
        "entities {}  aliases {}  documents {}  chunks {}",
        batch.entities.len(), batch.aliases.len(), batch.documents.len(), batch.chunks.len()
    );
    println!(
        "events {}  causal_edges {}  sensitivities {}  sentiments {}  propositions {}  figures {}",
        batch.events.len(), batch.causal_edges.len(), batch.sensitivities.len(),
        batch.sentiments.len(), batch.propositions.len(), batch.figures.len()
    );
    println!("wrote {} (Postgres) and {} (DuckLake load)", pg_path.display(), script_path.display());

    if a.apply_lake {
        println!("applying DuckLake load ...");
        lake::apply(&script)?;
        println!("DuckLake facts loaded.");
    } else {
        println!("run:  duckdb :memory: < {}   # to load DuckLake", script_path.display());
    }
    Ok(())
}
