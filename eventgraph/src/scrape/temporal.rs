//! Temporal drop-in orchestration adapter (DOCUMENTATION + feature-gated skeleton).
//!
//! HONEST NOTE: the shipped orchestrator is the standalone rayon runner in
//! `workflow.rs`. There is NO hard dependency on a Temporal SDK and NO running
//! Temporal server is required. This module exists to document how the job maps
//! onto Temporal when durable, distributed orchestration is wanted, and to pin a
//! feature-gated worker signature (compiled only under `--features temporal`).
//!
//! ## Activity <-> Temporal mapping
//!
//! Every function in `activities.rs` is already a Temporal-shaped activity:
//! deterministic inputs, bounded timeout, retryable, idempotent (re-running is
//! safe because dedup by content_hash + canonical_url converges). Register each
//! as a Temporal Activity:
//!
//! | activity (this crate)      | Temporal activity            | idempotency key            |
//! |----------------------------|------------------------------|----------------------------|
//! | `fetch_feed(spec)`         | `FetchFeed`                  | feed url                    |
//! | `resolve_url(item)`        | `ResolveUrl`                 | item link                   |
//! | `fetch_and_extract(url)`   | `FetchAndExtract`            | canonical_url / doc_id      |
//! | `write_partitions(rows)`   | `WritePartitions`            | doc_id (upsert-merge)       |
//!
//! Suggested activity options: StartToClose timeout = per-request timeout;
//! RetryPolicy = 3 attempts, initial 400ms, backoff x2 (matches `with_retry`).
//!
//! ## Workflow mapping
//!
//! `workflow::run` becomes a Temporal Workflow. The fan-out (`par_iter` over
//! feeds, then over resolved URLs) becomes N child activity invocations awaited
//! together; Temporal supplies the durability + at-least-once execution that the
//! rayon pool approximates in-process. Dedup state (`seen_ids` / `seen_hashes`)
//! that we load from disk becomes either workflow state or a first
//! `LoadExistingState` activity.
//!
//! ## Wiring steps left as follow-on (NOT done here)
//!
//! 1. Add a Temporal Rust SDK dep behind the `temporal` feature (e.g. the
//!    `temporal-sdk` / `temporal-client` crates once API-stable), OR shell out to
//!    a Temporal worker in another language.
//! 2. Implement `#[activity]` wrappers delegating to `activities::*` (they are
//!    already pure; just serde the inputs/outputs).
//! 3. Implement the `#[workflow]` that orchestrates them per the mapping above.
//! 4. Stand up a Temporal server (`temporal server start-dev`) + a task queue,
//!    register the worker, and trigger the workflow on a schedule (cron) in place
//!    of `cron_crawl.sh`.
//!
//! Until then, run the standalone job: `eventgraph scrape ...`.

#[cfg(feature = "temporal")]
pub mod worker {
    //! Feature-gated worker skeleton. Compiles under `--features temporal` but is
    //! intentionally a signature-only stub -- it pulls no external SDK. Fill in
    //! the bodies once a Temporal SDK dep is added (see follow-on steps above).

    use crate::scrape::{Article, FeedItem, FeedSpec};

    /// Marker for a registered Temporal task queue.
    pub const TASK_QUEUE: &str = "market-color-scrape";

    /// Activity: fetch one feed. Delegates to `activities::fetch_feed`.
    pub fn act_fetch_feed(_spec: FeedSpec) -> Vec<FeedItem> {
        unimplemented!("register with a Temporal SDK worker; body delegates to activities::fetch_feed")
    }

    /// Activity: resolve one item's URL. Delegates to `activities::resolve_url`.
    pub fn act_resolve_url(_item: FeedItem, _method: String) -> Option<String> {
        unimplemented!("delegates to activities::resolve_url")
    }

    /// Activity: fetch + extract one URL. Delegates to `activities::fetch_and_extract`.
    pub fn act_fetch_and_extract(_url: String) -> Option<Article> {
        unimplemented!("delegates to activities::fetch_and_extract")
    }

    /// Workflow entrypoint: orchestrates the activities above. Mirrors
    /// `workflow::run` but under Temporal's durable execution.
    pub fn scrape_workflow() {
        unimplemented!("register as a Temporal workflow; see module docs for the mapping")
    }
}
