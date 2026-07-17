//! URL + time helpers, ported 1:1 from `../crawl_corpus.py`:
//! `canonicalize`, `domain_of`, `stable_id`, `decode_gnews_url`,
//! `resolve_gnews`, `parse_dt`, `extract_pub_ts`. Kept pure so each is trivially
//! testable and re-runnable (idempotent activity building blocks).

use base64::Engine;
use chrono::{DateTime, Datelike, TimeZone, Timelike, Utc};
use regex::Regex;
use sha2::{Digest, Sha256};

// Tracking params stripped by canonicalize() -- identical set to the Python.
const TRACKING_PARAMS: &[&str] = &[
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_id",
    "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "cmpid", "guccounter",
];

/// Host without a leading `www.`, lowercased (Python `domain_of`).
pub fn domain_of(u: &str) -> String {
    match url::Url::parse(u) {
        Ok(p) => {
            let host = p.host_str().unwrap_or("").to_lowercase();
            host.strip_prefix("www.").unwrap_or(&host).to_string()
        }
        Err(_) => String::new(),
    }
}

/// `scheme://netloc` (Python `site_root`).
pub fn site_root(u: &str) -> Option<String> {
    let p = url::Url::parse(u).ok()?;
    let host = p.host_str()?;
    match p.port() {
        Some(port) => Some(format!("{}://{}:{}", p.scheme(), host, port)),
        None => Some(format!("{}://{}", p.scheme(), host)),
    }
}

/// Strip tracking params + fragment, lowercase host, drop trailing slash
/// (Python `canonicalize`). Falls back to the raw string on parse failure.
pub fn canonicalize(u: &str) -> String {
    let parsed = match url::Url::parse(u) {
        Ok(p) => p,
        Err(_) => return u.to_string(),
    };
    let scheme = parsed.scheme();
    let host = parsed.host_str().unwrap_or("").to_lowercase();
    let netloc = match parsed.port() {
        Some(port) => format!("{host}:{port}"),
        None => host,
    };
    let mut path = parsed.path().trim_end_matches('/').to_string();
    if path.is_empty() {
        path = "/".to_string();
    }
    let mut kept: Vec<(String, String)> = Vec::new();
    for (k, v) in parsed.query_pairs() {
        if v.is_empty() {
            continue; // keep_blank_values=False
        }
        if TRACKING_PARAMS.contains(&k.to_lowercase().as_str()) {
            continue;
        }
        kept.push((k.into_owned(), v.into_owned()));
    }
    let mut out = format!("{scheme}://{netloc}{path}");
    if !kept.is_empty() {
        let q = kept
            .iter()
            .map(|(k, v)| format!("{}={}", urlencode(k), urlencode(v)))
            .collect::<Vec<_>>()
            .join("&");
        out.push('?');
        out.push_str(&q);
    }
    out
}

fn urlencode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push_str("+"),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

/// sha256 of the parts joined by NUL, first 32 hex chars (Python `stable_id`).
pub fn stable_id(parts: &[&str]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update(p.trim().as_bytes());
        h.update(b"\x00");
    }
    hex_lower(&h.finalize())[..32].to_string()
}

/// sha256 of the first 2000 bytes of body, first 16 hex chars (Python content_hash).
pub fn content_hash(body: &str) -> String {
    let slice: String = body.chars().take(2000).collect();
    let mut h = Sha256::new();
    h.update(slice.as_bytes());
    hex_lower(&h.finalize())[..16].to_string()
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// --------------------------------------------------------------------------- //
// Google-News URL decoding (Python decode_gnews_url + resolve_gnews_sync).
// --------------------------------------------------------------------------- //

/// Decode a `news.google.com/rss/articles/CBMi...` link to the publisher URL by
/// pulling the length-delimited http string out of the base64-protobuf blob.
/// The newer opaque `AU_yqL...` batchexecute form carries no plain URL here, so
/// it returns None (skipped + counted -- we deliberately do NOT fetch it).
pub fn decode_gnews_url(u: &str) -> Option<String> {
    if !u.contains("news.google.com") {
        return Some(u.to_string());
    }
    let art_re = Regex::new(r"/articles/([A-Za-z0-9_\-]+)").ok()?;
    let enc = art_re.captures(u)?.get(1)?.as_str().to_string();
    let mut enc = enc;
    while enc.len() % 4 != 0 {
        enc.push('=');
    }
    let raw = base64::engine::general_purpose::URL_SAFE
        .decode(enc.as_bytes())
        .ok()?;
    // First embedded http(s) URL, cut at the first control byte.
    let url_re = Regex::new(r#"https?://[^\x00-\x1f"'\\<> ]+"#).ok()?;
    let text = String::from_utf8_lossy(&raw);
    let m = url_re.find(&text)?;
    let candidate: String = m
        .as_str()
        .chars()
        .take_while(|c| !c.is_control())
        .collect();
    if candidate.starts_with("http") {
        Some(candidate)
    } else {
        None
    }
}

/// Resolve a Google-News redirect to a publisher URL (Python `resolve_gnews_sync`,
/// minus the external `googlenewsdecoder` lib -- the opaque form is skipped).
/// Returns None if it stays a news.google.com link.
pub fn resolve_gnews(link: &str) -> Option<String> {
    if !link.contains("news.google.com") {
        return Some(link.to_string());
    }
    match decode_gnews_url(link) {
        Some(d) if !d.contains("news.google.com") => Some(d),
        _ => None,
    }
}

// --------------------------------------------------------------------------- //
// Timestamp parsing (Python parse_dt + extract_pub_ts).
// --------------------------------------------------------------------------- //

/// Best-effort parse of a date/datetime string -> UTC (Python `parse_dt` via
/// dateutil). Tries RFC3339, RFC2822, and a handful of common patterns; naive
/// values are assumed UTC.
pub fn parse_dt(value: &str) -> Option<DateTime<Utc>> {
    let v = value.trim();
    if v.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(v) {
        return Some(dt.with_timezone(&Utc));
    }
    if let Ok(dt) = DateTime::parse_from_rfc2822(v) {
        return Some(dt.with_timezone(&Utc));
    }
    // datetime with explicit offset, a few spellings
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S%.f%z",
        "%Y-%m-%d %H:%M:%S%z",
    ] {
        if let Ok(dt) = DateTime::parse_from_str(v, fmt) {
            return Some(dt.with_timezone(&Utc));
        }
    }
    // naive datetime -> assume UTC
    use chrono::NaiveDateTime;
    for fmt in [
        "%Y-%m-%dT%H:%M:%S%.f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M",
    ] {
        if let Ok(ndt) = NaiveDateTime::parse_from_str(v, fmt) {
            return Some(Utc.from_utc_datetime(&ndt));
        }
    }
    // date-only -> midnight UTC
    use chrono::NaiveDate;
    for fmt in ["%Y-%m-%d", "%Y/%m/%d"] {
        if let Ok(nd) = NaiveDate::parse_from_str(v, fmt) {
            return Some(Utc.from_utc_datetime(&nd.and_hms_opt(0, 0, 0)?));
        }
    }
    None
}

thread_local! {
    static TS_PATTERNS: Vec<Regex> = {
        [
            r#""datePublished"\s*:\s*"([^"]{8,40})""#,
            r#"<meta[^>]+property=["']article:published_time["'][^>]+content=["']([^"']{8,40})["']"#,
            r#"<meta[^>]+content=["']([^"']{8,40})["'][^>]+property=["']article:published_time["']"#,
            r#"<meta[^>]+name=["']parsely-pub-date["'][^>]+content=["']([^"']{8,40})["']"#,
            r#"<meta[^>]+itemprop=["']datePublished["'][^>]+content=["']([^"']{8,40})["']"#,
            r#"<meta[^>]+name=["'](?:sailthru\.date|pubdate|dc\.date\.issued)["'][^>]+content=["']([^"']{8,40})["']"#,
            r#"<time[^>]+datetime=["']([^"']{8,40})["']"#,
        ]
        .iter()
        .filter_map(|p| Regex::new(&format!("(?i){p}")).ok())
        .collect()
    };
}

/// Full publication timestamp (with time-of-day preferred) from HTML metadata,
/// else None (Python `extract_pub_ts`). Scans only the first 200k chars.
pub fn extract_pub_ts(html: &str) -> Option<DateTime<Utc>> {
    let head: String = html.chars().take(200_000).collect();
    let mut best: Option<DateTime<Utc>> = None;
    TS_PATTERNS.with(|pats| {
        for pat in pats {
            for cap in pat.captures_iter(&head) {
                let Some(g) = cap.get(1) else { continue };
                let Some(dt) = parse_dt(g.as_str()) else {
                    continue;
                };
                if !(2000..=2100).contains(&dt.year()) {
                    continue;
                }
                let has_time = (dt.hour(), dt.minute(), dt.second()) != (0, 0, 0);
                if has_time {
                    best = Some(dt);
                    return; // early return: a time-of-day match wins outright
                }
                best = best.or(Some(dt));
            }
        }
    });
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalize_strips_tracking_and_slash() {
        let got = canonicalize("https://WWW.Example.com/a/b/?utm_source=x&id=7#frag");
        assert_eq!(got, "https://www.example.com/a/b?id=7");
    }

    #[test]
    fn gnews_opaque_is_skipped() {
        // An opaque batchexecute id decodes to no plain URL -> None (skip).
        let u = "https://news.google.com/rss/articles/AU_yqLABCDEF?oc=5";
        assert!(resolve_gnews(u).is_none());
    }

    #[test]
    fn non_gnews_passthrough() {
        assert_eq!(
            resolve_gnews("https://reuters.com/x").as_deref(),
            Some("https://reuters.com/x")
        );
    }

    #[test]
    fn parse_dt_iso_and_dateonly() {
        assert!(parse_dt("2026-07-15T13:45:00Z").is_some());
        let d = parse_dt("2026-07-15").unwrap();
        assert_eq!((d.hour(), d.minute()), (0, 0));
    }
}
