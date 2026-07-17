//! Small pure helpers: stable hashing, slugs, entity-name normalization, and the
//! canonical 120-word chunker (identical to the receptor scheme).

use std::collections::HashMap;

/// FNV-1a 64-bit as hex -- stable content-addressed ids/keys across runs.
pub fn fnv1a(s: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{h:016x}")
}

/// slug: lowercase alnum, other runs -> '_'.
pub fn slug(s: &str) -> String {
    let mut out = String::new();
    let mut last_us = false;
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            last_us = false;
        } else if !last_us {
            out.push('_');
            last_us = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn aliases() -> HashMap<&'static str, &'static str> {
    [
        ("us", "united states"), ("u.s.", "united states"), ("usa", "united states"),
        ("fed", "federal reserve"), ("the fed", "federal reserve"),
        ("uk", "united kingdom"), ("ecb", "european central bank"),
        ("boj", "bank of japan"),
    ]
    .into_iter()
    .collect()
}

const SUFFIXES: &[&str] = &[
    "inc", "corp", "corporation", "co", "ltd", "plc", "llc", "llp", "sa", "ag",
    "nv", "group", "holdings", "holding", "bhd",
];

/// Canonicalize an entity surface form: lowercase, unify dashes/quotes, strip a
/// trailing corporate suffix, apply the alias map. This is the fragmentation fix.
pub fn norm_name(s: &str) -> String {
    let mut t: String = s.trim().to_lowercase();
    t = t.replace(['\u{2019}', '\u{2018}'], "'");
    for d in ['\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}'] {
        t = t.replace(d, "-");
    }
    t = t.split_whitespace().collect::<Vec<_>>().join(" ");
    t = t.trim_matches(|c| c == '.' || c == ',').to_string();
    // strip one trailing suffix token
    if let Some(pos) = t.rfind(' ') {
        let last = t[pos + 1..].trim_matches('.');
        if SUFFIXES.contains(&last) {
            t = t[..pos].trim().to_string();
        }
    }
    aliases().get(t.as_str()).map(|s| s.to_string()).unwrap_or(t)
}

const WORDS: usize = 120;
const MAXC: usize = 6;

/// 120-word chunks (max 6), title-prefixed -- matches receptors/export_rag.py.
pub fn chunk_text(title: &str, body: &str) -> Vec<String> {
    let ws: Vec<&str> = body.split_whitespace().collect();
    let mut out = Vec::new();
    let cap = ws.len().min(WORDS * MAXC);
    let mut i = 0;
    while i < cap {
        let seg = ws[i..(i + WORDS).min(cap)].join(" ");
        let mut c = format!("{} — {}", title.trim(), seg);
        if c.len() > 2000 {
            let mut end = 2000;
            while !c.is_char_boundary(end) { end -= 1; } // don't split a multi-byte char
            c.truncate(end);
        }
        out.push(c);
        i += WORDS;
    }
    if out.is_empty() {
        let t = title.trim();
        out.push(if t.is_empty() { "untitled".into() } else { t.chars().take(2000).collect() });
    }
    out
}
