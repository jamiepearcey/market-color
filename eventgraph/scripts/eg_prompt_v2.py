"""eg-rich-v2 extraction prompt (schema_version 0.2.0). Single source of truth for
the Python Groq path; kept verbatim-aligned with src/extract.rs PROMPT_VERSION.

v2 vs v1: suggested_ticker (was identifier) + id_scheme/listing_status_hint/
parent_company_hint/aliases; entity<->entity `relations`; temporal-hint triples
(*_text/*_kind) on events/causal/propositions/figures; effect_verbatim +
cause_kind/effect_kind + magnitude_text + timing on causal; asset_class +
magnitude_text on sensitivity; currency + as_of on figures; closed doc_type/kind
enums; explicit anti-hallucination rules (referential integrity, M&A direction,
denied/hypothetical, no fabricated dates).
"""
PROMPT_VERSION = "eg-rich-v2"
SCHEMA_VERSION = "0.2.0"

SYSTEM = "You are a financial news information extractor. Output ONLY the JSON object."

# chunking identical to scripts/extract.py (120 words, first PROMPT_CHUNKS chunks)
WORDS, MAXC, PROMPT_CHUNKS = 120, 6, 4

def chunk_text(title, body):
    ws = (body or "").split()
    out = []
    for i in range(0, min(len(ws), WORDS * MAXC), WORDS):
        out.append((f"{(title or '').strip()} — " + " ".join(ws[i:i + WORDS]))[:2000])
    return out or [((title or "").strip() or "untitled")[:2000]]

_TEMPLATE = """HEADLINE: {headline}

ARTICLE (numbered chunks):
{numbered}

Extract ONE JSON object (only what the article asserts; do not speculate). Every causal edge,
relation, sensitivity, sentiment, event and proposition MUST cite the chunk it comes from
(evidence_chunk = the [i]) and quote the exact supporting words verbatim.

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
- TEMPORAL (IMPORTANT): whenever the article names ANY date or time reference -- absolute ("June 30", "2013"), relative ("next month", "last week", "a year ago"), a weekday ("on Thursday"), or a period ("Q2", "May", "first half") -- you MUST copy it VERBATIM into the matching *_text field (event_time_text / period_text / resolution_date_text / timing_text / as_of_text) and set the *_kind. Never leave a *_text null when the article states a time. *_hint / *_kind are your own normalization; never fabricate a date not present in the text.
- modality: 'denied' when the article says something did NOT happen / was refuted; 'hypothetical' for conditional/if statements; 'forecast' for future expectations; else 'happened'/'ongoing'.
- use vocab values EXACTLY; empty arrays where nothing applies; every quote copied verbatim from the cited chunk."""

def build_prompt(headline, chunks):
    numbered = "\n".join(f"[{i}] {c}" for i, c in enumerate(chunks))
    return _TEMPLATE.format(headline=headline, numbered=numbered)
