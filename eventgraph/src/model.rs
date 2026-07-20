//! The eventgraph type system, in two layers:
//!   1. NARRATIVE  -- what the LLM returns per document (`DocExtraction`).
//!   2. GRAPH ROWS -- the normalized relational rows that mirror schema/*.sql
//!      and are what we load into Postgres and export to DuckLake (`GraphBatch`).
//! normalize.rs turns (1) into (2), resolving entities to canonical ids.

use serde::{Deserialize, Serialize};

pub const SCHEMA_VERSION: &str = "0.2.0";

// ===========================================================================
// 1. NARRATIVE plane -- the LLM extraction contract (one object per document)
// ===========================================================================

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct DocExtraction {
    #[serde(default)]
    pub doc_type: Option<String>,
    #[serde(default)]
    pub sentiment_overall: Option<f64>,
    #[serde(default)]
    pub entities: Vec<ExEntity>,
    #[serde(default)]
    pub events: Vec<ExEvent>,
    #[serde(default)]
    pub causal_edges: Vec<ExCausal>,
    #[serde(default)]
    pub sensitivities: Vec<ExSensitivity>,
    #[serde(default)]
    pub sentiments: Vec<ExSentiment>,
    #[serde(default)]
    pub propositions: Vec<ExProposition>,
    #[serde(default)]
    pub figures: Vec<ExFigure>,
    /// v2: entity<->entity relations (M&A roles, parent/brand, officer_of, ...).
    #[serde(default)]
    pub relations: Vec<ExRelation>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExEntity {
    pub name: String,
    #[serde(rename = "type", default)]
    pub etype: Option<String>,
    // v2 IDENTITY doctrine: what the 8B EMITS is a SUGGESTION, never trusted as a
    // merge key. `identifier` (the v1 name) is accepted as an alias so old
    // extractions still deserialize. The VERIFIED symbol lives only on EntityRow,
    // written by the resolve_yahoo/resolve_llm gates.
    #[serde(default, alias = "identifier")]
    pub suggested_ticker: Option<String>,
    /// what kind of id `suggested_ticker` is: ticker|iso4217|iso3166|index_code|other
    #[serde(default)]
    pub id_scheme: Option<String>,
    /// listed|private|government|index|person|unknown -- gates whether resolvers even try.
    #[serde(default)]
    pub listing_status_hint: Option<String>,
    /// brand/subsidiary -> owner ("Instagram" -> "Meta Platforms"). Resolved via entity gates.
    #[serde(default)]
    pub parent_company_hint: Option<String>,
    /// alternate surface forms seen in the doc ("Alphabet"/"Google") -> alias table fuel.
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub sector: Option<String>,
    #[serde(default)]
    pub country: Option<String>,
    #[serde(default)]
    pub role: Option<String>,
    /// first-mention chunk -> entities become auditable like facts.
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
}

/// v2: an entity->entity relationship asserted in the text. Covers M&A roles
/// (acquirer vs target), parent/brand links, person->org affiliation, supply chain.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExRelation {
    pub source: String,
    pub target: String,
    /// acquires|merges_with|parent_of|brand_of|stake_in|supplies|customer_of|competes_with|officer_of|regulates|other
    #[serde(default)]
    pub relation: Option<String>,
    #[serde(default)]
    pub deal_value: Option<f64>,
    #[serde(default)]
    pub deal_currency: Option<String>,
    /// deal lifecycle -- market-moving in itself: rumored|proposed|agreed|completed|blocked
    #[serde(default)]
    pub status_hint: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExEvent {
    pub event_type: String,
    #[serde(default)]
    pub series_hint: Option<String>,
    #[serde(default)]
    pub issuer: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub scheduled: Option<bool>,
    /// model's ISO-8601 normalization attempt (the existing date-sanity gate reads this).
    #[serde(default)]
    pub event_time: Option<String>,
    /// v2 temporal-hint triple: verbatim phrase + kind, resolved vs published_at downstream.
    #[serde(default)]
    pub event_time_text: Option<String>,
    /// absolute|relative|period|deictic|none
    #[serde(default)]
    pub event_time_kind: Option<String>,
    /// the REFERENCE period ("May" in "May CPI") -- distinct from release timing; needed for the realised join.
    #[serde(default)]
    pub period_text: Option<String>,
    #[serde(default)]
    pub period_hint: Option<String>,
    #[serde(default)]
    pub expected: Option<f64>,
    #[serde(default)]
    pub actual: Option<f64>,
    #[serde(default)]
    pub prior: Option<f64>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExCausal {
    pub cause: String,
    pub effect: String,
    /// v2: entity|factor|event|other -- lets the loader referential-check each side by kind.
    #[serde(default)]
    pub cause_kind: Option<String>,
    #[serde(default)]
    pub effect_kind: Option<String>,
    #[serde(default)]
    pub mechanism: Option<String>,
    #[serde(default)]
    pub effect_direction: Option<String>,
    /// v2: the exact phrase stating the move ("shares jumped 7%"). Anchors the
    /// direction-audit that catches PayPal-class mis-tags.
    #[serde(default)]
    pub effect_verbatim: Option<String>,
    #[serde(default)]
    pub magnitude_value: Option<f64>,
    #[serde(default)]
    pub magnitude_unit: Option<String>,
    /// v2: verbatim magnitude ("$53 billion") -> magnitude_value/unit become verifiable by re-parse.
    #[serde(default)]
    pub magnitude_text: Option<String>,
    /// v2: when the effect happened/is expected -- richer than the `lag` enum.
    #[serde(default)]
    pub timing_text: Option<String>,
    #[serde(default)]
    pub timing_kind: Option<String>,
    #[serde(default)]
    pub modality: Option<String>,
    #[serde(default)]
    pub attribution_source: Option<String>,
    #[serde(default)]
    pub lag: Option<String>,
    #[serde(default)]
    pub confidence: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExSensitivity {
    pub asset: String,
    pub factor: String,
    #[serde(default)]
    pub factor_type: Option<String>,
    #[serde(default)]
    pub sign: Option<i64>,
    #[serde(default)]
    pub magnitude_qual: Option<String>,
    #[serde(default)]
    pub magnitude_value: Option<f64>,
    #[serde(default)]
    pub magnitude_unit: Option<String>,
    /// v2: verbatim magnitude phrase (verifiable by re-parse).
    #[serde(default)]
    pub magnitude_text: Option<String>,
    /// v2: which class of the issuer this sensitivity is about -- equity|credit|rates|fx|commodity|vol.
    /// "Ford credit widens" vs "Ford equity falls" are DIFFERENT edges; unrecoverable later.
    #[serde(default)]
    pub asset_class: Option<String>,
    #[serde(default)]
    pub basis: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExSentiment {
    pub target: String,
    pub polarity: f64,
    #[serde(default)]
    pub intensity: Option<String>,
    #[serde(rename = "type", default)]
    pub stype: Option<String>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub horizon: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExProposition {
    pub text: String,
    #[serde(default)]
    pub subject: Option<String>,
    #[serde(default)]
    pub resolution_date: Option<String>,
    /// v2 temporal-hint triple for the resolution date.
    #[serde(default)]
    pub resolution_date_text: Option<String>,
    #[serde(default)]
    pub resolution_date_kind: Option<String>,
    #[serde(default)]
    pub resolution_criteria: Option<String>,
    #[serde(default)]
    pub probability: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub source_instrument: Option<String>,
    #[serde(default)]
    pub contract_hint: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ExFigure {
    #[serde(default)]
    pub entity: Option<String>,
    /// price|level|pct|bps|volume|estimate|money|market_cap|revenue|eps|yield|spread
    pub kind: String,
    pub value: f64,
    #[serde(default)]
    pub unit: Option<String>,
    /// v2: ISO-4217 currency for money/price figures.
    #[serde(default)]
    pub currency: Option<String>,
    /// v2: "as of Thursday's close" verbatim -> resolved vs published_at.
    #[serde(default)]
    pub as_of_text: Option<String>,
    #[serde(default)]
    pub evidence_chunk: Option<i64>,
    #[serde(default)]
    pub quote: Option<String>,
}

// ===========================================================================
// 2. GRAPH ROWS -- normalized, mirror schema/*.sql. Serialize -> DuckLake JSONL.
// ===========================================================================

#[derive(Debug, Clone, Serialize, Default)]
pub struct GraphBatch {
    pub entities: Vec<EntityRow>,
    pub aliases: Vec<AliasRow>,
    pub documents: Vec<DocumentRow>,
    pub chunks: Vec<ChunkRow>,
    pub events: Vec<EventRow>,
    pub propositions: Vec<PropositionRow>,
    pub probabilities: Vec<ProbabilityRow>,
    pub causal_edges: Vec<CausalEdgeRow>,
    pub sensitivities: Vec<SensitivityRow>,
    pub sentiments: Vec<SentimentRow>,
    pub figures: Vec<FigureRow>,
    pub relations: Vec<RelationRow>,
}

impl GraphBatch {
    pub fn merge(&mut self, o: GraphBatch) {
        self.entities.extend(o.entities);
        self.aliases.extend(o.aliases);
        self.documents.extend(o.documents);
        self.chunks.extend(o.chunks);
        self.events.extend(o.events);
        self.propositions.extend(o.propositions);
        self.probabilities.extend(o.probabilities);
        self.causal_edges.extend(o.causal_edges);
        self.sensitivities.extend(o.sensitivities);
        self.sentiments.extend(o.sentiments);
        self.figures.extend(o.figures);
        self.relations.extend(o.relations);
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityRow {
    pub entity_id: String,
    pub canonical_name: String,
    pub r#type: String,
    /// VERIFIED symbol -- written ONLY by the resolve_yahoo/resolve_llm gates. None until verified.
    pub identifier: Option<String>,
    /// the 8B's raw guess, kept for audit; NEVER a merge key.
    pub suggested_ticker: Option<String>,
    /// unresolved|seen_in_source|verified_yahoo|verified_llm_yahoo|unlisted_private|unlisted_gov|unlisted_index|unlisted_person|resolution_failed
    pub resolution_status: String,
    pub resolution_source: Option<String>,
    pub sector: Option<String>,
    pub country: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct AliasRow {
    pub alias: String,
    pub entity_id: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct DocumentRow {
    pub doc_id: String,
    pub source: String,
    pub url: Option<String>,
    pub headline: Option<String>,
    pub published_at: Option<String>,
    pub doc_type: Option<String>,
    pub sentiment_overall: Option<f64>,
    pub run_id: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct ChunkRow {
    pub chunk_id: String,
    pub doc_id: String,
    pub seq: i64,
    pub epoch: Option<i64>,
    pub text: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct EventRow {
    pub event_id: String,
    pub event_type: String,
    pub series_id: Option<String>,
    pub issuer_entity: Option<String>,
    pub region: Option<String>,
    pub scheduled: bool,
    pub event_time: Option<String>,
    pub event_time_text: Option<String>,
    pub event_time_kind: Option<String>,
    pub period_text: Option<String>,
    pub period_hint: Option<String>,
    pub expected: Option<f64>,
    pub actual: Option<f64>,
    pub prior: Option<f64>,
    pub unit: Option<String>,
    // `surprise` is computed in a lake view (actual-expected vs calendar_event),
    // never stored on the fact row.
    pub doc_id: String,
    pub chunk_id: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct PropositionRow {
    pub proposition_id: String,
    pub text: String,
    pub subject_entity: Option<String>,
    pub resolution_date: Option<String>,
    pub resolution_criteria: Option<String>,
    pub status: String,
    pub contract_ref: Option<String>,
    pub implied_instrument: Option<String>,
    pub doc_id: String,
    pub chunk_id: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct ProbabilityRow {
    pub proposition_id: String,
    pub probability: f64,
    pub source: String,
    pub source_instrument: Option<String>,
    pub as_of: Option<String>,
    pub doc_id: String,
}
#[derive(Debug, Clone, Serialize)]
pub struct CausalEdgeRow {
    pub edge_key: String,
    pub cause_entity: Option<String>,
    pub effect_entity: Option<String>,
    pub cause_kind: Option<String>,
    pub effect_kind: Option<String>,
    pub mechanism: Option<String>,
    pub effect_dir: Option<String>,
    pub effect_verbatim: Option<String>,
    pub magnitude_value: Option<f64>,
    pub magnitude_unit: Option<String>,
    pub magnitude_text: Option<String>,
    pub timing_text: Option<String>,
    pub timing_kind: Option<String>,
    pub modality: String,
    pub attribution: Option<String>,
    pub lag: Option<String>,
    pub confidence: Option<String>,
    /// cheap lexicon audit vs effect_verbatim: ok|conflict|unchecked.
    pub direction_audit: String,
    pub event_id: Option<String>,
    pub doc_id: String,
    pub chunk_id: Option<String>,
    pub quote: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SensitivityRow {
    pub sens_key: String,
    pub asset_entity: String,
    pub asset_class: Option<String>,
    pub factor_id: Option<String>,
    pub factor_entity: Option<String>,
    pub sign: Option<i64>,
    pub magnitude_qual: Option<String>,
    pub magnitude_value: Option<f64>,
    pub magnitude_unit: Option<String>,
    pub magnitude_text: Option<String>,
    pub basis: Option<String>,
    pub doc_id: String,
    pub chunk_id: Option<String>,
    pub quote: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct SentimentRow {
    pub sent_key: String,
    pub target_entity: String,
    pub polarity: f64,
    pub intensity: Option<String>,
    pub r#type: String,
    pub source: String,
    pub horizon: Option<String>,
    pub as_of: Option<String>,
    pub doc_id: String,
    pub chunk_id: Option<String>,
    pub quote: Option<String>,
}
#[derive(Debug, Clone, Serialize)]
pub struct FigureRow {
    pub doc_id: String,
    pub chunk_id: Option<String>,
    pub entity_id: Option<String>,
    pub kind: String,
    pub value: f64,
    pub unit: Option<String>,
    pub currency: Option<String>,
    pub as_of_text: Option<String>,
    pub quote: Option<String>,
}

/// v2 entity->entity relation row (append-only fact, grounded like an edge).
#[derive(Debug, Clone, Serialize)]
pub struct RelationRow {
    pub rel_key: String,
    pub source_entity: Option<String>,
    pub target_entity: Option<String>,
    pub relation: Option<String>,
    pub deal_value: Option<f64>,
    pub deal_currency: Option<String>,
    pub status_hint: Option<String>,
    pub doc_id: String,
    pub chunk_id: Option<String>,
    pub quote: Option<String>,
}
