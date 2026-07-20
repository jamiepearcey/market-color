//! Extraction: raw doc + numbered chunks -> the narrative plane (`DocExtraction`).
//! `MockExtractor` runs offline (compile/CI/dev); `GroqExtractor` calls the proven
//! gpt-oss-120b path with the rich taxonomy prompt.

use crate::feed::RawDoc;
use crate::model::{DocExtraction, ExEntity};
use anyhow::{anyhow, Result};
use serde::de::DeserializeOwned;
use serde_json::Value;

pub trait Extractor: Sync {
    fn extract(&self, doc: &RawDoc, chunks: &[String]) -> Result<DocExtraction>;
}

/// Deterministic, no network: one entity from the headline. Exercises the whole
/// pipeline without an API key.
pub struct MockExtractor;
impl Extractor for MockExtractor {
    fn extract(&self, doc: &RawDoc, _chunks: &[String]) -> Result<DocExtraction> {
        let name = doc
            .headline
            .split_whitespace()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");
        Ok(DocExtraction {
            doc_type: Some("news".into()),
            entities: if name.is_empty() {
                vec![]
            } else {
                vec![ExEntity {
                    name,
                    etype: Some("other".into()),
                    suggested_ticker: None,
                    id_scheme: None,
                    listing_status_hint: None,
                    parent_company_hint: None,
                    aliases: vec![],
                    sector: None,
                    country: None,
                    role: Some("subject".into()),
                    evidence_chunk: None,
                }]
            },
            ..Default::default()
        })
    }
}

const GROQ_URL: &str = "https://api.groq.com/openai/v1/chat/completions";
pub const PROMPT_VERSION: &str = "eg-rich-v2";

pub struct GroqExtractor {
    pub api_key: String,
    pub model: String,
    agent: ureq::Agent, // shared connection pool -> keep-alive reuse, no per-call TLS handshake
}

impl GroqExtractor {
    pub fn from_env(model: &str) -> Result<Self> {
        let api_key = std::env::var("GROQ_API_KEY")
            .map_err(|_| anyhow!("GROQ_API_KEY not set (source data/tmp/groq.env)"))?;
        let agent = ureq::AgentBuilder::new()
            .timeout_read(std::time::Duration::from_secs(180))
            .max_idle_connections_per_host(64)
            .build();
        Ok(Self { api_key, model: model.to_string(), agent })
    }

    fn prompt(&self, doc: &RawDoc, chunks: &[String]) -> String {
        let numbered: String = chunks
            .iter()
            .enumerate()
            .map(|(i, c)| format!("[{i}] {c}"))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            r#"HEADLINE: {headline}

ARTICLE (numbered chunks):
{numbered}

Extract ONE JSON object (only what the article asserts; do not speculate). Every causal
edge, relation, sensitivity, sentiment, event and proposition MUST cite the chunk it comes
from (evidence_chunk = the [i]) and quote the exact supporting words verbatim.

{{
  "doc_type": "news|analysis|opinion|preview|recap|press_release|interview|data_release|non_financial",
  "sentiment_overall": <-1..1 or null>,
  "entities": [{{"name":"canonical (no tickers/Inc.)","type":"company|bank|central_bank|sovereign|regulator|commodity|currency|equity_index|rate_or_bond|sector|person|exchange|economic_indicator|other","suggested_ticker":"best-guess ticker/ISO or null (a GUESS)","id_scheme":"ticker|iso4217|iso3166|index_code|other|null","listing_status_hint":"listed|private|government|index|person|unknown","parent_company_hint":"owner if brand/subsidiary, else null","aliases":["other surface names in the text"],"sector":"coarse sector or null","country":"ISO-3166 alpha-2 or null","role":"subject|driver|counterparty|mentioned","evidence_chunk":0}}],
  "relations": [{{"source":"entities.name","target":"entities.name","relation":"acquires|merges_with|parent_of|brand_of|stake_in|supplies|customer_of|competes_with|officer_of|regulates|other","deal_value":null,"deal_currency":null,"status_hint":"rumored|proposed|agreed|completed|blocked|null","evidence_chunk":0,"quote":"verbatim"}}],
  "events": [{{"event_type":"rate_decision|cpi|employment|gdp|pmi|earnings|guidance|mergers_acquisitions|debt_auction|rating_review|election|opec_meeting|eia_inventory|other","series_hint":"FOMC|US-CPI|<TICKER>-EARN or null","issuer":"entity name or null","region":null,"scheduled":true,"event_time":"ISO or null","event_time_text":"verbatim date phrase or null","event_time_kind":"absolute|relative|period|deictic|none","period_text":"reference period e.g. 'May' or null","period_hint":"ISO period or null","expected":null,"actual":null,"prior":null,"unit":null,"evidence_chunk":0,"quote":"verbatim"}}],
  "causal_edges": [{{"cause":"entities.name","effect":"a DIFFERENT entities.name","cause_kind":"entity|factor|event|other","effect_kind":"entity|factor|event|other","mechanism":"monetary_policy|rate_decision|supply_shock|demand_change|earnings|guidance|mergers_acquisitions|default|rating_action|regulation|geopolitics|fund_flows|data_surprise|contagion|other","effect_direction":"up|down|widen|tighten|volatile|unchanged","effect_verbatim":"the EXACT words stating the move, e.g. 'shares jumped 7%'","magnitude_value":null,"magnitude_unit":"pct|bps|usd|local_ccy|ratio|count|other|null","magnitude_text":"verbatim e.g. '$53 billion' or null","modality":"happened|ongoing|forecast|hypothetical|denied","timing_text":"when it happened/is expected, verbatim or null","timing_kind":"absolute|relative|period|deictic|none","attribution_source":"reporter|named_analyst|company|official|market_consensus|market_implied","lag":"immediate|days|longer","confidence":"strong|tentative","evidence_chunk":0,"quote":"verbatim contiguous span"}}],
  "sensitivities": [{{"asset":"entities.name","asset_class":"equity|credit|rates|fx|commodity|vol","factor":"e.g. oil|rates|usd|credit spread|<name>","factor_type":"rates|credit_spread|oil|commodity|usd|fx|equity_beta|inflation|specific|other","sign":1,"magnitude_qual":"high|medium|low","magnitude_value":null,"magnitude_unit":"beta|years|pct_of_cost|null","magnitude_text":"verbatim or null","basis":"fundamental|stated_beta|historical","evidence_chunk":0,"quote":"verbatim"}}],
  "sentiments": [{{"target":"entities.name","polarity":-1.0,"intensity":"strong|mild","type":"directional|risk|credit|surprise","source":"reporter|named_analyst|official|market_implied","horizon":"immediate|short|long","evidence_chunk":0,"quote":"verbatim"}}],
  "propositions": [{{"text":"resolvable forward statement","subject":"entities.name or null","resolution_date":"ISO or null","resolution_date_text":"verbatim date phrase or null","resolution_date_kind":"absolute|relative|period|deictic|none","resolution_criteria":"objective settle condition","probability":null,"source":"reporter|named_analyst|market_implied","source_instrument":"fed_funds_future|cds|option|null","contract_hint":null,"evidence_chunk":0,"quote":"verbatim"}}],
  "figures": [{{"entity":"entities.name or null","kind":"price|level|pct|bps|volume|estimate|money|market_cap|revenue|eps|yield|spread","value":0.0,"unit":null,"currency":"ISO-4217 or null","as_of_text":"verbatim as-of phrase or null","evidence_chunk":0,"quote":"verbatim"}}]
}}

Rules:
- entities 3-8; every cause/effect/asset/target/subject and every relation source&target MUST be one of entities[].name (verbatim).
- a causal_edge's cause and effect MUST be DIFFERENT entities (if a story is one firm's own news, the cause is the driver/event, not the firm itself).
- effect_direction describes the EFFECT ASSET's move; effect_verbatim MUST be the exact words that state that move. In a bid/M&A the TARGET usually trades UP (a $X bid for T -> T 'up'), the acquirer often 'down'.
- suggested_ticker is a GUESS -- fill only if reasonably confident, else null; NEVER invent a ticker for private/government/index/person entities (set listing_status_hint instead).
- TEMPORAL (IMPORTANT): whenever the article names ANY date or time reference -- absolute ("June 30", "2013"), relative ("next month", "last week"), a weekday ("on Thursday"), or a period ("Q2", "May", "first half") -- you MUST copy it VERBATIM into the matching *_text field (event_time_text / period_text / resolution_date_text / timing_text / as_of_text) and set the *_kind. Never leave a *_text null when the article states a time. *_hint / *_kind are your own normalization; never fabricate a date not present in the text.
- modality: 'denied' when the article says something did NOT happen / was refuted; 'hypothetical' for conditional/if statements; 'forecast' for future expectations; else 'happened'/'ongoing'.
- use vocab values EXACTLY; empty arrays where nothing applies; every quote copied verbatim from the cited chunk."#,
            headline = doc.headline,
            numbered = numbered
        )
    }
}

impl GroqExtractor {
    fn body(&self, doc: &RawDoc, chunks: &[String]) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 3200,          // headroom so the rich JSON isn't truncated
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": "You are a financial news information extractor. Output ONLY the JSON object."},
                {"role": "user", "content": self.prompt(doc, chunks)}
            ]
        });
        if self.model.starts_with("openai/gpt-oss") {
            body["reasoning_effort"] = serde_json::json!("low");
        }
        body
    }

    fn call_once(&self, body: &serde_json::Value) -> Result<DocExtraction> {
        let resp: serde_json::Value = self
            .agent
            .post(GROQ_URL)
            .set("Authorization", &format!("Bearer {}", self.api_key))
            .send_json(body.clone())
            .map_err(|e| anyhow!("groq request: {e}"))?
            .into_json()
            .map_err(|e| anyhow!("groq decode: {e}"))?;
        let txt = resp["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow!("no content"))?;
        clean_to_value(txt)
            .map(lenient_extraction)
            .ok_or_else(|| anyhow!("unparseable extraction"))
    }
}

// ---------------------------------------------------------------------------
// Failure cleansing (lifts weak-model, e.g. 8B, doc yield to ~90%+).
// ---------------------------------------------------------------------------

/// Recover a JSON value from possibly-malformed model output: raw parse -> strip
/// code fences / take the outer object -> drop trailing commas -> balance a
/// truncated tail.
fn clean_to_value(txt: &str) -> Option<Value> {
    if let Ok(v) = serde_json::from_str::<Value>(txt) {
        return Some(v);
    }
    let mut s = txt.trim();
    if let Some(p) = s.find("```") {
        s = &s[p + 3..];
        if let Some(nl) = s.find('\n') {
            if s[..nl].trim().eq_ignore_ascii_case("json") {
                s = &s[nl + 1..];
            }
        }
        if let Some(e) = s.rfind("```") {
            s = &s[..e];
        }
    }
    let s = match (s.find('{'), s.rfind('}')) {
        (Some(a), Some(b)) if b > a => &s[a..=b],
        _ => s,
    };
    for cand in [strip_trailing_commas(s), strip_trailing_commas(&balance(s))] {
        if let Ok(v) = serde_json::from_str::<Value>(&cand) {
            return Some(v);
        }
    }
    None
}

/// Drop commas that immediately precede a `}` or `]` (string-aware).
fn strip_trailing_commas(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let (mut in_str, mut esc) = (false, false);
    for i in 0..b.len() {
        let c = b[i] as char;
        if in_str {
            out.push(c);
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
            out.push(c);
            continue;
        }
        if c == ',' {
            let mut j = i + 1;
            while j < b.len() && (b[j] as char).is_whitespace() {
                j += 1;
            }
            if j < b.len() && (b[j] == b'}' || b[j] == b']') {
                continue; // drop the trailing comma
            }
        }
        out.push(c);
    }
    out
}

/// Close an open string + unbalanced `{`/`[` left by a max_tokens cutoff.
fn balance(s: &str) -> String {
    let mut stack = Vec::new();
    let (mut in_str, mut esc) = (false, false);
    let mut last_sig = ' ';
    for c in s.chars() {
        if in_str {
            if esc {
                esc = false;
            } else if c == '\\' {
                esc = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        match c {
            '"' => in_str = true,
            '{' | '[' => stack.push(c),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
        if !c.is_whitespace() {
            last_sig = c;
        }
    }
    let mut out = s.to_string();
    if in_str {
        out.push('"');
    }
    if last_sig == ':' && !in_str {
        out.push_str("null");
    }
    while let Some(open) = stack.pop() {
        out.push(if open == '{' { '}' } else { ']' });
    }
    out
}

/// Build a DocExtraction leniently: array fields parsed element-by-element so a
/// single malformed item is skipped rather than discarding the whole document.
/// Reused by the checkpoint reader (pipeline.rs) so a resumed extraction with one
/// bad element keeps its good facts instead of falling back to a mock extraction.
pub fn lenient_extraction(v: Value) -> DocExtraction {
    fn arr<T: DeserializeOwned>(v: &Value, key: &str) -> Vec<T> {
        v.get(key)
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|e| serde_json::from_value::<T>(e.clone()).ok())
                    .collect()
            })
            .unwrap_or_default()
    }
    DocExtraction {
        doc_type: v.get("doc_type").and_then(|x| x.as_str()).map(str::to_string),
        sentiment_overall: v.get("sentiment_overall").and_then(|x| x.as_f64()),
        entities: arr(&v, "entities"),
        events: arr(&v, "events"),
        causal_edges: arr(&v, "causal_edges"),
        sensitivities: arr(&v, "sensitivities"),
        sentiments: arr(&v, "sentiments"),
        propositions: arr(&v, "propositions"),
        figures: arr(&v, "figures"),
        relations: arr(&v, "relations"),
    }
}

impl Extractor for GroqExtractor {
    fn extract(&self, doc: &RawDoc, chunks: &[String]) -> Result<DocExtraction> {
        let body = self.body(doc, chunks);
        let mut last = anyhow!("no attempt");
        for attempt in 0..3 {
            match self.call_once(&body) {
                Ok(v) => return Ok(v),
                Err(e) => {
                    last = e;
                    std::thread::sleep(std::time::Duration::from_millis(500 * (attempt + 1)));
                }
            }
        }
        Err(last)
    }
}
