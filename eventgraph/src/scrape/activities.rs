//! Idempotent, retryable ACTIVITIES. Each is pure w.r.t. external state (safe to
//! re-run) and self-contained so it maps directly onto a Temporal activity
//! (see `temporal.rs`). Every activity bounds its own timeout + retries and
//! degrades gracefully on failure (returns None / empty, never panics the run).

use super::model::{read_parquet, write_parquet, Row};
use super::urls::{
    canonicalize, content_hash, decode_gnews_url, domain_of, extract_pub_ts, parse_dt, stable_id,
};
use super::{Article, FeedItem, FeedSpec};
use anyhow::Result;
use chrono::{DateTime, Utc};
use scraper::{Html, Selector};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};

pub const UA: &str =
    "Mozilla/5.0 (compatible; market-color-crawler/0.2; +research corpus builder)";

/// Build a blocking ureq agent with bounded connect/read timeouts.
pub fn build_agent(timeout_secs: u64) -> ureq::Agent {
    ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(10))
        .timeout_read(Duration::from_secs(timeout_secs))
        .timeout_write(Duration::from_secs(timeout_secs))
        .user_agent(UA)
        .build()
}

/// Retry a fallible op up to `tries` times with exponential backoff. Idempotent
/// activities make at-least-once execution safe (dedup handles duplicates).
fn with_retry<T>(tries: u32, mut op: impl FnMut() -> Option<T>) -> Option<T> {
    let mut delay = Duration::from_millis(400);
    for attempt in 0..tries {
        if let Some(v) = op() {
            return Some(v);
        }
        if attempt + 1 < tries {
            std::thread::sleep(delay);
            delay *= 2;
        }
    }
    None
}

/// Polite per-domain throttle: schedules each domain's next allowed hit without
/// holding the lock across the sleep.
pub struct DomainThrottle {
    delay: Duration,
    next: Mutex<HashMap<String, Instant>>,
}

impl DomainThrottle {
    pub fn new(delay_secs: f64) -> Self {
        Self {
            delay: Duration::from_secs_f64(delay_secs.max(0.0)),
            next: Mutex::new(HashMap::new()),
        }
    }

    pub fn wait(&self, dom: &str) {
        if self.delay.is_zero() {
            return;
        }
        let now = Instant::now();
        let scheduled = {
            let mut map = self.next.lock().unwrap();
            let earliest = map.get(dom).copied().unwrap_or(now).max(now);
            map.insert(dom.to_string(), earliest + self.delay);
            earliest
        };
        if scheduled > now {
            std::thread::sleep(scheduled - now);
        }
    }
}

fn http_get(agent: &ureq::Agent, url: &str) -> Option<(String, String)> {
    with_retry(3, || match agent.get(url).call() {
        Ok(resp) => {
            let final_url = resp.get_url().to_string();
            resp.into_string().ok().map(|body| (final_url, body))
        }
        Err(_) => None,
    })
}

/// Bounded counting semaphore (std-only). Caps concurrent hits to
/// news.google.com so we stay polite + avoid batchexecute rate-limits.
pub struct Semaphore {
    count: Mutex<usize>,
    cv: Condvar,
}

pub struct SemGuard<'a>(&'a Semaphore);

impl Semaphore {
    pub fn new(n: usize) -> Self {
        Self {
            count: Mutex::new(n.max(1)),
            cv: Condvar::new(),
        }
    }
    pub fn acquire(&self) -> SemGuard<'_> {
        let mut c = self.count.lock().unwrap();
        while *c == 0 {
            c = self.cv.wait(c).unwrap();
        }
        *c -= 1;
        SemGuard(self)
    }
}

impl Drop for SemGuard<'_> {
    fn drop(&mut self) {
        let mut c = self.0.count.lock().unwrap();
        *c += 1;
        self.0.cv.notify_one();
    }
}

const GNEWS_HOST: &str = "news.google.com";
// Small politeness delay per google-news request (in addition to the semaphore).
const GNEWS_DELAY: Duration = Duration::from_millis(120);

// --------------------------------------------------------------------------- //
// ACTIVITY 1: fetch_feed(spec) -> Vec<FeedItem>   (RSS/Atom parse)
// --------------------------------------------------------------------------- //
pub fn fetch_feed(
    agent: &ureq::Agent,
    spec: &FeedSpec,
    since: Option<DateTime<Utc>>,
    max_per_feed: usize,
) -> Vec<FeedItem> {
    let Some((_final, body)) = http_get(agent, &spec.url) else {
        return Vec::new();
    };
    let parsed = match feed_rs::parser::parse(body.as_bytes()) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for e in parsed.entries {
        let link = e
            .links
            .iter()
            .map(|l| l.href.clone())
            .find(|h| !h.is_empty())
            .unwrap_or_default();
        if link.is_empty() {
            continue;
        }
        let published = e
            .published
            .or(e.updated)
            .map(|d| d.with_timezone(&Utc));
        if let (Some(cut), Some(pub_dt)) = (since, published) {
            if pub_dt < cut {
                continue;
            }
        }
        let summary = e
            .summary
            .map(|s| clean_text(&s.content))
            .filter(|s| !s.is_empty())
            .or_else(|| {
                e.content
                    .and_then(|c| c.body)
                    .map(|b| clean_text(&b))
                    .filter(|s| !s.is_empty())
            });
        let title = e.title.map(|t| clean_text(&t.content));
        out.push(FeedItem {
            link,
            published,
            summary,
            title,
        });
        if out.len() >= max_per_feed {
            break;
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// ACTIVITY 2: resolve_url(item) -> Option<Url>   (google-news decode)
// --------------------------------------------------------------------------- //
/// Resolve a feed item's link to a publisher URL.
///
/// For `google_news_rss`: (1) fast-path the older base64 `CBMi…` form that embeds
/// a plain http URL; (2) for the opaque `AU_yqL…` batchexecute form, PORT the
/// maintained `googlenewsdecoder` flow in pure Rust -- GET the article page for
/// the `data-n-a-sg`/`data-n-a-ts` signature+timestamp, POST them to
/// `/_/DotsSplashUi/data/batchexecute`, and parse the publisher URL out of the
/// response. On any failure it returns None (skip + count) -- never breaks the run.
/// Google is rate-limited via `sem` (concurrency cap) + a small per-request delay.
pub fn resolve_url(
    agent: &ureq::Agent,
    sem: &Semaphore,
    item: &FeedItem,
    method: &str,
) -> Option<String> {
    if method != "google_news_rss" {
        return Some(item.link.clone());
    }
    // Fast path: older CBMi… form with an embedded plain URL.
    if let Some(u) = decode_gnews_url(&item.link) {
        if !u.contains(GNEWS_HOST) {
            return Some(u);
        }
    }
    // Opaque form -> batchexecute (rate-limited).
    let id = gnews_id(&item.link)?;
    let _permit = sem.acquire();
    std::thread::sleep(GNEWS_DELAY);
    let (sig, ts) = gnews_decoding_params(agent, &id)?;
    let url = gnews_batch_decode(agent, &id, &ts, &sig)?;
    if url.starts_with("http") && !url.contains(GNEWS_HOST) {
        Some(url)
    } else {
        None
    }
}

/// Extract the base64 article id from a `news.google.com/(rss/)?articles/<id>` link.
fn gnews_id(link: &str) -> Option<String> {
    let p = url::Url::parse(link).ok()?;
    if p.host_str()? != GNEWS_HOST {
        return None;
    }
    let segs: Vec<&str> = p.path_segments()?.collect();
    let n = segs.len();
    if n >= 2 && (segs[n - 2] == "articles" || segs[n - 2] == "read") {
        Some(segs[n - 1].to_string())
    } else {
        None
    }
}

/// GET the article page and scrape the batchexecute signature + timestamp from
/// the `c-wiz > div[jscontroller]` element (data-n-a-sg / data-n-a-ts). Tries the
/// bare `/articles/` form first, then the `/rss/articles/` fallback.
fn gnews_decoding_params(agent: &ureq::Agent, id: &str) -> Option<(String, String)> {
    // Single-attempt each (no retry storm on the expected-404 form); RSS form
    // first since it reliably carries the c-wiz signature block.
    for base in [
        "https://news.google.com/rss/articles/",
        "https://news.google.com/articles/",
    ] {
        std::thread::sleep(GNEWS_DELAY);
        if let Ok(resp) = agent.get(&format!("{base}{id}")).call() {
            if let Ok(html) = resp.into_string() {
                if let Some(pair) = parse_wiz_params(&html) {
                    return Some(pair);
                }
            }
        }
    }
    None
}

fn parse_wiz_params(html: &str) -> Option<(String, String)> {
    let doc = Html::parse_document(html);
    let sel = Selector::parse("c-wiz > div[jscontroller]").ok()?;
    for el in doc.select(&sel) {
        if let (Some(sg), Some(ts)) = (
            el.value().attr("data-n-a-sg"),
            el.value().attr("data-n-a-ts"),
        ) {
            return Some((sg.to_string(), ts.to_string()));
        }
    }
    None
}

/// POST the batchexecute Fbv4je RPC and parse the decoded publisher URL. Mirrors
/// googlenewsdecoder's `decode_url`: response is `)]}'\n\n[[...]]`; the array's
/// `[0][2]` is a JSON string whose `[1]` is the URL.
fn gnews_batch_decode(agent: &ureq::Agent, id: &str, ts: &str, sig: &str) -> Option<String> {
    let inner = format!(
        "[\"garturlreq\",[[\"X\",\"X\",[\"X\",\"X\"],null,null,1,1,\"US:en\",null,1,null,null,null,null,null,0,1],\"X\",\"X\",1,[1,1,1],1,1,null,0,0,null,0],\"{id}\",{ts},\"{sig}\"]"
    );
    let freq = serde_json::to_string(&serde_json::json!([[["Fbv4je", inner]]])).ok()?;
    let body = format!("f.req={}", percent_encode_form(&freq));
    let resp = with_retry(2, || {
        agent
            .post("https://news.google.com/_/DotsSplashUi/data/batchexecute?rpcids=Fbv4je")
            .set(
                "Content-Type",
                "application/x-www-form-urlencoded;charset=UTF-8",
            )
            .set("Referer", "https://news.google.com/")
            .send_string(&body)
            .ok()
            .and_then(|r| r.into_string().ok())
    })?;
    let after = resp.split("\n\n").nth(1)?.trim();
    let v: serde_json::Value = serde_json::from_str(after).ok()?;
    let inner_str = v.get(0)?.get(2)?.as_str()?;
    let iv: serde_json::Value = serde_json::from_str(inner_str).ok()?;
    iv.get(1)?.as_str().map(|s| s.to_string())
}

/// Percent-encode for `application/x-www-form-urlencoded` (matches Python
/// `urllib.parse.quote`: keep unreserved, encode everything else).
fn percent_encode_form(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// --------------------------------------------------------------------------- //
// ACTIVITY 3: fetch_and_extract(url) -> Option<Article>   (HTTP GET + HTML->text)
// --------------------------------------------------------------------------- //
#[allow(clippy::too_many_arguments)]
pub fn fetch_and_extract(
    agent: &ureq::Agent,
    url: &str,
    spec: &FeedSpec,
    discovery: &str,
    item: &FeedItem,
    throttle: &DomainThrottle,
    now: DateTime<Utc>,
) -> Option<Article> {
    let dom = domain_of(url);
    throttle.wait(&dom);
    let (final_url, html) = http_get(agent, url)?;
    if html.is_empty() {
        return None;
    }
    let (mut body, page_title) = extract_body_title(&html);

    // Graceful degradation: thin page body -> fall back to the RSS summary.
    if body.chars().count() < 200 {
        if let Some(sum) = &item.summary {
            if sum.chars().count() > body.chars().count() {
                body = sum.clone();
            }
        }
    }
    let title = if !page_title.is_empty() {
        page_title
    } else {
        item.title.clone().unwrap_or_default()
    };

    let meta_dt = extract_pub_ts(&html);
    let hint_dt = item.published;
    let pub_dt = meta_dt.or(hint_dt);
    let estimated = pub_dt.is_none();
    let pub_dt = pub_dt.unwrap_or(now);

    let extraction_ok = !body.is_empty() && body.chars().count() >= 300;
    let canon = canonicalize(&final_url);
    let content_hash_val = if extraction_ok {
        Some(content_hash(&body))
    } else {
        None
    };
    let word_count = body.split_whitespace().count() as i64;

    let row = Row {
        doc_id: stable_id(&[&canon]),
        source_name: spec.name.clone(),
        source_scope: spec.scope.clone(),
        source_tier: spec.tier,
        source_access: spec.access.clone(),
        source_method: spec.method.clone(),
        source_domain: dom,
        desks: spec.desks.clone(),
        discovery: discovery.to_string(),
        title,
        body_text: if body.is_empty() { None } else { Some(body) },
        author: None,
        url: final_url,
        canonical_url: canon,
        lang: spec.lang.clone(),
        word_count,
        extraction_ok,
        published_utc: Some(pub_dt.to_rfc3339()),
        published_date: pub_dt.date_naive().to_string(),
        published_is_estimated: estimated,
        fetched_utc: now.to_rfc3339(),
        content_hash: content_hash_val,
    };
    Some(Article { row })
}

// --------------------------------------------------------------------------- //
// ACTIVITY 4: write_partitions(articles) -> Report   (date-partitioned merge)
// --------------------------------------------------------------------------- //
/// Group articles by published_date and merge-write each `dt=DATE/part-000.parquet`
/// (append-merge: current-run rows win, prior rows kept unless doc_id collides).
/// Idempotent: re-running with the same articles yields the same partitions.
pub fn write_partitions(out_root: &Path, articles: &[Article]) -> Result<HashMap<String, usize>> {
    let mut by_date: HashMap<String, HashMap<String, Row>> = HashMap::new();
    for a in articles {
        by_date
            .entry(a.row.published_date.clone())
            .or_default()
            .insert(a.row.doc_id.clone(), a.row.clone());
    }
    let mut written = HashMap::new();
    for (date, docs) in by_date {
        let part_dir = out_root.join(format!("dt={date}"));
        std::fs::create_dir_all(&part_dir)?;
        let path = part_dir.join("part-000.parquet");
        let mut rows: Vec<Row> = docs.values().cloned().collect();
        let seen: std::collections::HashSet<String> =
            rows.iter().map(|r| r.doc_id.clone()).collect();
        if path.exists() {
            match read_parquet(&path) {
                Ok(prev) => {
                    for r in prev {
                        if !seen.contains(&r.doc_id) {
                            rows.push(r);
                        }
                    }
                }
                Err(e) => {
                    // Refuse to clobber a partition we cannot read (Python parity).
                    eprintln!("  [write] REFUSING to overwrite {path:?}: {e}");
                    continue;
                }
            }
        }
        write_parquet(&path, &rows)?;
        written.insert(date, rows.len());
    }
    Ok(written)
}

/// Total rows + partition file count across the corpus on disk.
pub fn corpus_size(out_root: &Path) -> (usize, usize) {
    let mut rows = 0;
    let mut files = 0;
    let Ok(entries) = std::fs::read_dir(out_root) else {
        return (0, 0);
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("dt=")) {
            let part = p.join("part-000.parquet");
            if part.exists() {
                if let Ok(r) = read_parquet(&part) {
                    rows += r.len();
                    files += 1;
                }
            }
        }
    }
    (rows, files)
}

/// Existing doc_ids + content_hashes on disk, for cross-run dedup (Python resume).
pub fn load_existing_state(
    out_root: &Path,
) -> (std::collections::HashSet<String>, std::collections::HashSet<String>) {
    let mut ids = std::collections::HashSet::new();
    let mut hashes = std::collections::HashSet::new();
    let Ok(entries) = std::fs::read_dir(out_root) else {
        return (ids, hashes);
    };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() && p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("dt=")) {
            let part = p.join("part-000.parquet");
            if part.exists() {
                if let Ok(rows) = read_parquet(&part) {
                    for r in rows {
                        ids.insert(r.doc_id);
                        if let Some(h) = r.content_hash {
                            hashes.insert(h);
                        }
                    }
                }
            }
        }
    }
    (ids, hashes)
}

// --------------------------------------------------------------------------- //
// HTML -> text extraction (scraper). Concatenate <article>/<p> text, dropping
// script/style/nav; documented future swap for a trafilatura-grade extractor.
// --------------------------------------------------------------------------- //
fn extract_body_title(html: &str) -> (String, String) {
    let doc = Html::parse_document(html);
    let title = og_title(&doc);
    // Prefer paragraphs inside <article>; fall back to all <p>.
    let article_body = collect_paragraphs(&doc, "article p");
    let body = if article_body.chars().count() >= 200 {
        article_body
    } else {
        collect_paragraphs(&doc, "p")
    };
    (clean_text(&body), title)
}

fn collect_paragraphs(doc: &Html, sel: &str) -> String {
    let Ok(selector) = Selector::parse(sel) else {
        return String::new();
    };
    let mut parts = Vec::new();
    for el in doc.select(&selector) {
        let t: String = el.text().collect::<Vec<_>>().join(" ");
        let t = clean_text(&t);
        if !t.is_empty() {
            parts.push(t);
        }
    }
    parts.join("\n")
}

fn og_title(doc: &Html) -> String {
    if let Ok(sel) = Selector::parse("meta[property='og:title']") {
        if let Some(el) = doc.select(&sel).next() {
            if let Some(c) = el.value().attr("content") {
                let c = clean_text(c);
                if !c.is_empty() {
                    return c;
                }
            }
        }
    }
    if let Ok(sel) = Selector::parse("title") {
        if let Some(el) = doc.select(&sel).next() {
            return clean_text(&el.text().collect::<Vec<_>>().join(" "));
        }
    }
    String::new()
}

/// Strip residual tags + collapse whitespace (Python `clean_text`).
pub fn clean_text(text: &str) -> String {
    // Remove any residual angle-bracket tags, then collapse whitespace.
    let mut no_tags = String::with_capacity(text.len());
    let mut in_tag = false;
    for c in text.chars() {
        match c {
            '<' => in_tag = true,
            '>' => {
                in_tag = false;
                no_tags.push(' ');
            }
            _ if !in_tag => no_tags.push(c),
            _ => {}
        }
    }
    no_tags.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Best-effort parse used by the window filter (re-export of urls::parse_dt).
pub fn parse_published(s: &str) -> Option<DateTime<Utc>> {
    parse_dt(s)
}
