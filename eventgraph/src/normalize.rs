//! Narrative -> normalized graph rows, WITH quality gates against hallucination.
//!
//! Gates (deterministic, no extra LLM):
//!   1. Provenance: a causal edge / sensitivity survives ONLY if its verbatim
//!      quote actually locates in the source chunks. No quote -> no edge. This is
//!      the primary anti-hallucination gate (the model cannot assert a link into
//!      existence). We NEVER fall back to a claimed chunk index.
//!   2. Entity grounding: an entity's name must appear in the article, else its
//!      LLM-guessed ticker is rejected (a wrong ticker poisons the realised join).
//!   3. Date sanity: event dates outside a plausible window of the publish date
//!      are nulled (kills the "opec 2026 / m&a 2023" hallucinations).
//! A GateStats report surfaces the catch rate for every run.

use crate::feed::RawDoc;
use crate::model::*;
use crate::util::{chunk_text, fnv1a, norm_name, slug};
use std::collections::{HashMap, HashSet};

const ENTITY_TYPES: &[&str] = &[
    "company", "bank", "central_bank", "sovereign", "supranational", "regulator",
    "government_agency", "commodity", "currency", "equity_index", "rate_or_bond",
    "sector", "person", "exchange", "economic_indicator", "asset_class", "market", "other",
];
fn snap_type(t: &Option<String>) -> String {
    let t = t.clone().unwrap_or_default().to_lowercase();
    if ENTITY_TYPES.contains(&t.as_str()) { t } else { "other".into() }
}

/// Per-run quality-gate metrics (a hallucination dashboard).
#[derive(Default)]
pub struct GateStats {
    pub docs: usize,
    pub entities_seen: usize,
    pub entities_unseen: usize,
    pub edges_in: usize,
    pub edges_kept: usize,
    pub edges_ungrounded: usize,
    pub self_loops: usize,
    pub sens_in: usize,
    pub sens_kept: usize,
    pub sens_ungrounded: usize,
    pub dates_nulled: usize,
    pub tickers_kept: usize,
    pub tickers_rejected: usize,
}
impl GateStats {
    pub fn report(&self) -> String {
        let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
        format!(
            "gates: causal edges kept {}/{} ({:.0}% grounded, {} ungrounded dropped, {} self-loops) | \
             sensitivities kept {}/{} | entities in-source {}/{} | tickers kept {}/{} | dates nulled {}",
            self.edges_kept, self.edges_in, pct(self.edges_kept, self.edges_in),
            self.edges_ungrounded, self.self_loops,
            self.sens_kept, self.sens_in,
            self.entities_seen, self.entities_seen + self.entities_unseen,
            self.tickers_kept, self.tickers_kept + self.tickers_rejected,
            self.dates_nulled,
        )
    }
}

/// Global canonical registry across all docs.
#[derive(Default)]
pub struct EntityResolver {
    seen_entity: HashSet<String>,
    seen_alias: HashSet<(String, String)>,
    pub new_entities: Vec<EntityRow>,
    pub new_aliases: Vec<AliasRow>,
    /// One canonical entity type per normalized name, resolved run-wide (see
    /// [`canonical_entity_types`]). Keys the entity_id so a name the 8B typed
    /// inconsistently (`china` as `sovereign` in one doc, `other` in another)
    /// collapses to ONE node instead of `china__sovereign` + `china__other`.
    canonical_type: HashMap<String, String>,
}

/// Resolve one canonical entity type per name across a whole run: the
/// most-frequent NON-`other` type a name is given; `other` only when a name is
/// never assigned a real type. Run this over ALL extractions before normalize so
/// entity nodes and edge endpoints agree on type (and thus on entity_id).
pub fn canonical_entity_types<'a>(exs: impl Iterator<Item = &'a DocExtraction>) -> HashMap<String, String> {
    let mut votes: HashMap<String, HashMap<String, usize>> = HashMap::new();
    for ex in exs {
        for e in &ex.entities {
            let name = norm_name(&e.name);
            if name.is_empty() {
                continue;
            }
            *votes.entry(name).or_default().entry(snap_type(&e.etype)).or_insert(0) += 1;
        }
    }
    votes
        .into_iter()
        .map(|(name, v)| {
            let best = v
                .iter()
                .filter(|(t, _)| t.as_str() != "other")
                .max_by_key(|(_, c)| **c)
                .map(|(t, _)| t.clone())
                .unwrap_or_else(|| "other".to_string());
            (name, best)
        })
        .collect()
}

impl EntityResolver {
    /// A resolver seeded with a run-wide canonical name→type map.
    pub fn with_canonical_types(canonical_type: HashMap<String, String>) -> Self {
        Self { canonical_type, ..Default::default() }
    }

    /// The canonical type for a name (run-wide vote), or the given fallback.
    fn typed(&self, name: &str, fallback: impl FnOnce() -> String) -> String {
        self.canonical_type.get(name).cloned().unwrap_or_else(fallback)
    }

    #[allow(clippy::too_many_arguments)]
    fn intern(
        &mut self,
        canonical: String,
        etype: String,
        identifier: Option<String>,
        suggested_ticker: Option<String>,
        resolution_status: String,
        resolution_source: Option<String>,
        sector: Option<String>,
        country: Option<String>,
        surface: &str,
    ) -> String {
        let id = format!("{}__{}", slug(&canonical), etype);
        if self.seen_entity.insert(id.clone()) {
            self.new_entities.push(EntityRow {
                entity_id: id.clone(),
                canonical_name: canonical,
                r#type: etype,
                identifier,
                suggested_ticker,
                resolution_status,
                resolution_source,
                sector,
                country,
            });
        }
        let key = (surface.to_string(), id.clone());
        if !surface.is_empty() && self.seen_alias.insert(key) {
            self.new_aliases.push(AliasRow { alias: surface.to_string(), entity_id: id.clone() });
        }
        id
    }
    /// Resolve a fully-typed entity. v2 identity doctrine: the LLM's ticker guess is
    /// NEVER written to `identifier` (the verified slot) here -- it is kept as
    /// `suggested_ticker` for the downstream resolve_yahoo/resolve_llm gates to
    /// verify. `seen` (name appears in source, gate 2) only sets a weak resolution
    /// tier; a real symbol arrives later. This keeps a hallucinated ticker out of
    /// the merge/realised-join path entirely.
    fn resolve_entity(&mut self, e: &ExEntity, seen: bool, g: &mut GateStats) -> String {
        let guess = e.suggested_ticker.clone().filter(|s| plausible_ticker(s));
        let status = match (&guess, seen) {
            (Some(_), true) => { g.tickers_kept += 1; "seen_in_source" }
            (Some(_), false) => { g.tickers_rejected += 1; "unresolved" }
            (None, _) => "unresolved",
        };
        let name = norm_name(&e.name);
        let etype = self.typed(&name, || snap_type(&e.etype));
        self.intern(
            name,
            etype,         // run-wide canonical type (no per-mention splitting)
            None,          // identifier (verified) -- filled by resolvers, never here
            guess,         // suggested_ticker -- the audited 8B guess
            status.into(),
            None,
            e.sector.clone(),
            e.country.clone(),
            &e.name,
        )
    }
    /// A reference by bare name (edge cause/effect etc.): reuse the per-doc map,
    /// else mint an 'other'-typed node with no ticker.
    fn resolve_ref(&mut self, name: &str, map: &HashMap<String, String>) -> Option<String> {
        if name.trim().is_empty() {
            return None;
        }
        let n = norm_name(name);
        if let Some(id) = map.get(&n) {
            return Some(id.clone());
        }
        // Edge endpoint minted as a bare node: use the run-wide canonical type so
        // it lands on the SAME id as the typed entity (china -> china__sovereign,
        // not china__other).
        let etype = self.typed(&n, || "other".to_string());
        Some(self.intern(n, etype, None, None, "unresolved".into(), None, None, None, name))
    }
    pub fn drain(&mut self, b: &mut GraphBatch) {
        b.entities.append(&mut self.new_entities);
        b.aliases.append(&mut self.new_aliases);
    }
}

/// Cheap lexicon direction-audit: does the verbatim effect phrase agree with the
/// tagged `effect_direction`? Returns "ok" | "conflict" | "unchecked". Catches
/// PayPal-class mis-tags (a bid target trades UP, tagged 'down') for downstream review.
fn audit_direction(effect_dir: &Option<String>, verbatim: &Option<String>) -> String {
    let (Some(dir), Some(v)) = (effect_dir.as_deref(), verbatim.as_deref()) else {
        return "unchecked".into();
    };
    let v = v.to_lowercase();
    let up = ["jump", "surg", "soar", "rall", "rose", "rise", "gain", "climb", "advanc", "up ", "higher", "spike", "boost"];
    let down = ["fell", "fall", "slid", "slip", "tumbl", "plung", "drop", "sank", "sink", "decline", "lower", "down ", "loss", "lost", "retreat"];
    let widen = ["widen", "blowout", "blew out"];
    let tighten = ["tighten", "narrow"];
    let hit = |kws: &[&str]| kws.iter().any(|k| v.contains(k));
    let phrase_dir = if hit(&up) { Some("up") }
        else if hit(&down) { Some("down") }
        else if hit(&widen) { Some("widen") }
        else if hit(&tighten) { Some("tighten") }
        else { None };
    match phrase_dir {
        None => "unchecked".into(),
        Some(pd) if pd == dir => "ok".into(),
        Some(_) => "conflict".into(),
    }
}

/// A plausible ticker / ISO code: 1-7 chars of alnum/./-, at least one alnum.
fn plausible_ticker(s: &str) -> bool {
    let s = s.trim();
    (1..=7).contains(&s.len())
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '.' || c == '-')
        && s.chars().any(|c| c.is_ascii_alphanumeric())
}

/// Is the entity actually named in the article (gate 2 input)?
fn seen_in_source(name: &str, article_lc: &str) -> bool {
    let n = norm_name(name);
    if n.len() >= 3 && article_lc.contains(&n) {
        return true;
    }
    n.split(|c: char| !c.is_alphanumeric())
        .any(|t| t.len() >= 4 && article_lc.contains(t))
}

fn canon_chunk(s: &str) -> String {
    s.to_lowercase()
        .replace(['\u{2019}', '\u{2018}'], "'")
        .replace(['\u{2010}', '\u{2011}', '\u{2012}', '\u{2013}', '\u{2014}', '\u{2212}'], "-")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// GATE 1: return the chunk id ONLY if the quote verifiably locates in the source
/// (exact, or the longest fragment of an ellipsis-stitched quote). No index
/// fallback -- an unverifiable quote yields None, which callers treat as failure.
fn grounded_chunk(doc_id: &str, quote: &Option<String>, canon: &[String]) -> Option<String> {
    let q = canon_chunk(quote.as_deref()?);
    if q.len() < 12 {
        return None;
    }
    if let Some(j) = canon.iter().position(|c| c.contains(&q)) {
        return Some(format!("{doc_id}:{j}"));
    }
    let mut frags: Vec<&str> = q
        .split("...")
        .flat_map(|p| p.split('\u{2026}'))
        .map(|s| s.trim())
        .filter(|s| s.len() >= 15)
        .collect();
    frags.sort_by_key(|f| std::cmp::Reverse(f.len()));
    frags
        .iter()
        .find_map(|f| canon.iter().position(|c| c.contains(*f)).map(|j| format!("{doc_id}:{j}")))
}

fn date_epoch(s: &str) -> Option<i64> {
    chrono::DateTime::parse_from_rfc3339(s)
        .ok()
        .map(|d| d.timestamp())
        .or_else(|| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok().map(|d| d.and_utc().timestamp()))
        .or_else(|| chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d").ok().map(|d| d.and_hms_opt(0, 0, 0).unwrap().and_utc().timestamp()))
}
fn epoch_of(published_at: &Option<String>) -> Option<i64> {
    date_epoch(published_at.as_ref()?)
}

/// GATE 3: keep an event date only if it parses AND sits within [-90d, +550d] of
/// the publish date (scheduled events look ahead; news looks slightly back).
fn sane_event_time(t: &Option<String>, doc_epoch: Option<i64>) -> (Option<String>, bool) {
    let Some(s) = t else { return (None, false) };
    match (date_epoch(s), doc_epoch) {
        (None, _) => (None, true),                 // unparseable -> drop (also protects the load)
        (Some(_), None) => (Some(s.clone()), false),
        (Some(e), Some(d)) => {
            let day = 86_400i64;
            if e >= d - 90 * day && e <= d + 550 * day {
                (Some(s.clone()), false)
            } else {
                (None, true)
            }
        }
    }
}

/// Turn one extracted document into normalized rows, applying the quality gates.
pub fn build_batch(
    res: &mut EntityResolver,
    g: &mut GateStats,
    doc: &RawDoc,
    ex: &DocExtraction,
    run_id: &str,
) -> GraphBatch {
    let mut b = GraphBatch::default();
    g.docs += 1;
    let chunks = chunk_text(&doc.headline, &doc.article);
    let canon: Vec<String> = chunks.iter().map(|c| canon_chunk(c)).collect();
    let article_lc = format!("{} {}", doc.headline, doc.article).to_lowercase();
    let epoch = epoch_of(&doc.published_at);

    b.documents.push(DocumentRow {
        doc_id: doc.doc_id.clone(),
        source: doc.source.clone(),
        url: doc.url.clone(),
        headline: Some(doc.headline.clone()),
        published_at: doc.published_at.clone(),
        doc_type: ex.doc_type.clone(),
        sentiment_overall: ex.sentiment_overall,
        run_id: run_id.to_string(),
    });
    for (i, c) in chunks.iter().enumerate() {
        b.chunks.push(ChunkRow {
            chunk_id: format!("{}:{}", doc.doc_id, i),
            doc_id: doc.doc_id.clone(),
            seq: i as i64,
            epoch,
            text: c.clone(),
        });
    }

    // entities first -> per-doc name lookup for edges/refs (gate 2 applied here)
    let mut map: HashMap<String, String> = HashMap::new();
    for e in &ex.entities {
        let seen = seen_in_source(&e.name, &article_lc);
        if seen { g.entities_seen += 1 } else { g.entities_unseen += 1 }
        let id = res.resolve_entity(e, seen, g);
        map.insert(norm_name(&e.name), id);
    }

    for (i, ev) in ex.events.iter().enumerate() {
        let (event_time, nulled) = sane_event_time(&ev.event_time, epoch);
        if nulled {
            g.dates_nulled += 1;
        }
        b.events.push(EventRow {
            event_id: fnv1a(&format!("{}|evt|{}|{}", doc.doc_id, ev.event_type, i)),
            event_type: ev.event_type.clone(),
            series_id: ev.series_hint.as_ref().map(|s| slug(s)),
            issuer_entity: ev.issuer.as_ref().and_then(|n| res.resolve_ref(n, &map)),
            region: ev.region.clone(),
            scheduled: ev.scheduled.unwrap_or(true),
            event_time,
            event_time_text: ev.event_time_text.clone(),
            event_time_kind: ev.event_time_kind.clone(),
            period_text: ev.period_text.clone(),
            period_hint: ev.period_hint.clone(),
            expected: ev.expected,
            actual: ev.actual,
            prior: ev.prior,
            unit: ev.unit.clone(),
            doc_id: doc.doc_id.clone(),
            chunk_id: grounded_chunk(&doc.doc_id, &ev.quote, &canon),
        });
    }

    for ce in &ex.causal_edges {
        let cause = res.resolve_ref(&ce.cause, &map);
        let effect = res.resolve_ref(&ce.effect, &map);
        if cause.is_some() && cause == effect {
            g.self_loops += 1;
            continue;
        }
        g.edges_in += 1;
        // GATE 1: require a verifiable quote. An ungrounded causal link is dropped.
        let Some(chunk_id) = grounded_chunk(&doc.doc_id, &ce.quote, &canon) else {
            g.edges_ungrounded += 1;
            continue;
        };
        g.edges_kept += 1;
        b.causal_edges.push(CausalEdgeRow {
            edge_key: fnv1a(&format!("{}|{}|{}|{}", doc.doc_id, ce.cause, ce.effect, ce.quote.clone().unwrap_or_default())),
            cause_entity: cause,
            effect_entity: effect,
            cause_kind: ce.cause_kind.clone(),
            effect_kind: ce.effect_kind.clone(),
            mechanism: ce.mechanism.clone(),
            effect_dir: ce.effect_direction.clone(),
            effect_verbatim: ce.effect_verbatim.clone(),
            magnitude_value: ce.magnitude_value,
            magnitude_unit: ce.magnitude_unit.clone(),
            magnitude_text: ce.magnitude_text.clone(),
            timing_text: ce.timing_text.clone(),
            timing_kind: ce.timing_kind.clone(),
            modality: ce.modality.clone().unwrap_or_else(|| "happened".into()),
            attribution: ce.attribution_source.clone(),
            lag: ce.lag.clone(),
            confidence: ce.confidence.clone(),
            direction_audit: audit_direction(&ce.effect_direction, &ce.effect_verbatim),
            event_id: None,
            doc_id: doc.doc_id.clone(),
            chunk_id: Some(chunk_id),
            quote: ce.quote.clone(),
        });
    }

    for s in &ex.sensitivities {
        let Some(asset) = res.resolve_ref(&s.asset, &map) else { continue };
        g.sens_in += 1;
        let Some(chunk_id) = grounded_chunk(&doc.doc_id, &s.quote, &canon) else {
            g.sens_ungrounded += 1;
            continue;
        };
        g.sens_kept += 1;
        b.sensitivities.push(SensitivityRow {
            sens_key: fnv1a(&format!("{}|sens|{}|{}", doc.doc_id, s.asset, s.factor)),
            asset_entity: asset,
            asset_class: s.asset_class.clone(),
            factor_id: Some(slug(&s.factor)),
            factor_entity: None,
            sign: s.sign,
            magnitude_qual: s.magnitude_qual.clone(),
            magnitude_value: s.magnitude_value,
            magnitude_unit: s.magnitude_unit.clone(),
            magnitude_text: s.magnitude_text.clone(),
            basis: s.basis.clone(),
            doc_id: doc.doc_id.clone(),
            chunk_id: Some(chunk_id),
            quote: s.quote.clone(),
        });
    }

    for s in &ex.sentiments {
        if let Some(target) = res.resolve_ref(&s.target, &map) {
            b.sentiments.push(SentimentRow {
                sent_key: fnv1a(&format!("{}|sent|{}|{:?}", doc.doc_id, s.target, s.stype)),
                target_entity: target,
                polarity: s.polarity,
                intensity: s.intensity.clone(),
                r#type: s.stype.clone().unwrap_or_else(|| "directional".into()),
                source: s.source.clone().unwrap_or_else(|| "reporter".into()),
                horizon: s.horizon.clone(),
                as_of: doc.published_at.clone(),
                doc_id: doc.doc_id.clone(),
                chunk_id: grounded_chunk(&doc.doc_id, &s.quote, &canon),
                quote: s.quote.clone(),
            });
        }
    }

    for p in &ex.propositions {
        let pid = fnv1a(&format!("prop|{}", norm_name(&p.text)));
        b.propositions.push(PropositionRow {
            proposition_id: pid.clone(),
            text: p.text.clone(),
            subject_entity: p.subject.as_ref().and_then(|n| res.resolve_ref(n, &map)),
            resolution_date: p.resolution_date.clone(),
            resolution_criteria: p.resolution_criteria.clone(),
            status: "open".into(),
            contract_ref: p.contract_hint.clone(),
            implied_instrument: p.source_instrument.clone(),
            doc_id: doc.doc_id.clone(),
            chunk_id: grounded_chunk(&doc.doc_id, &p.quote, &canon),
        });
        if let Some(prob) = p.probability {
            b.probabilities.push(ProbabilityRow {
                proposition_id: pid,
                probability: prob,
                source: p.source.clone().unwrap_or_else(|| "reporter".into()),
                source_instrument: p.source_instrument.clone(),
                as_of: doc.published_at.clone(),
                doc_id: doc.doc_id.clone(),
            });
        }
    }

    for f in &ex.figures {
        b.figures.push(FigureRow {
            doc_id: doc.doc_id.clone(),
            chunk_id: grounded_chunk(&doc.doc_id, &f.quote, &canon),
            entity_id: f.entity.as_ref().and_then(|n| res.resolve_ref(n, &map)),
            kind: f.kind.clone(),
            value: f.value,
            unit: f.unit.clone(),
            currency: f.currency.clone(),
            as_of_text: f.as_of_text.clone(),
            quote: f.quote.clone(),
        });
    }

    // v2 entity<->entity relations. Grounded like edges: an ungrounded relation
    // (quote does not locate in the source) is dropped. source/target resolve via
    // the per-doc entity map (or mint an 'other' node).
    for r in &ex.relations {
        let source = res.resolve_ref(&r.source, &map);
        let target = res.resolve_ref(&r.target, &map);
        if source.is_none() && target.is_none() {
            continue;
        }
        let Some(chunk_id) = grounded_chunk(&doc.doc_id, &r.quote, &canon) else {
            continue;
        };
        b.relations.push(RelationRow {
            rel_key: fnv1a(&format!("{}|rel|{}|{}|{:?}", doc.doc_id, r.source, r.target, r.relation)),
            source_entity: source,
            target_entity: target,
            relation: r.relation.clone(),
            deal_value: r.deal_value,
            deal_currency: r.deal_currency.clone(),
            status_hint: r.status_hint.clone(),
            doc_id: doc.doc_id.clone(),
            chunk_id: Some(chunk_id),
            quote: r.quote.clone(),
        });
    }

    res.drain(&mut b);
    b
}
