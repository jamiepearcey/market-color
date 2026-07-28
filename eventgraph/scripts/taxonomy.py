# /// script
# requires-python = ">=3.10"
# ///
"""
STANDARDISED CLASSIFICATION TAXONOMY — the shared contract for a sector /
identifier / event-taxonomy layer over the extracted news graphs (Bloomberg
eg100k + India india2021), so the two graphs are comparable and downstream
studies (news -> sector -> volume, exposure, attribution) run on standard tags
instead of the raw, dup-riddled LLM vocab.

DOCTRINE (borrowed from eg-rich-v2):
- ADDITIVE side-car: this layer is emitted as separate classification_*.jsonl
  files joined by entity_id / doc_id / event_id. The extraction lake is never
  mutated.
- SUGGESTED vs VERIFIED: `suggested_ticker` from the LLM is a hint, NEVER a merge
  key. The India nifty50_resolved.json is demonstrably hallucinated (Jindal Steel
  -> JSWSTEEL, Rain Industries -> GRASIM), so India tickers are marked
  resolution_status="suggested_unverified" until a real NSE/BSE + price-history
  verification pass runs. Sector is therefore taken from the free-text sector
  (more reliable than the bad ticker) or a verified US ticker, and each carries a
  `sector_source` so consumers know how much to trust it.
- CONTROLLED VOCAB: every observed event_type / mechanism maps to a canonical
  enum here; anything unmapped -> "other" and MUST be counted+reported, never
  silently dropped.

Consumers: classify_layer.py (the classifier), and any study needing a standard
sector/event tag. `gics.py` remains the US ticker->GICS source.
"""

# ---------------------------------------------------------------------------
# 1. STANDARD EVENT TYPE (from event.event_type) — canonical enum + group.
# ---------------------------------------------------------------------------
EVENT_GROUP = {  # canonical event_type -> high-level group
    "monetary_policy": "macro", "inflation": "macro", "employment": "macro",
    "growth": "macro", "trade_balance": "macro", "fiscal_policy": "macro",
    "econ_indicator": "macro", "debt_issuance": "macro",
    "earnings": "corporate", "guidance": "corporate", "m_and_a": "corporate",
    "ipo": "corporate", "rating_action": "corporate", "mgmt_change": "corporate",
    "capital_action": "corporate", "product": "corporate",
    "legal_regulatory": "corporate", "credit_event": "corporate",
    "partnership": "corporate",
    "supply_shock": "commodity", "inventory": "commodity", "opec": "commodity",
    "geopolitics": "market", "election": "market", "disaster": "market",
    "strike": "market", "fund_flows": "market", "market_move": "market",
    "contagion": "market",
    "operations": "corporate",
    "clinical_trial": "corporate",
    "other": "other",
}
STD_EVENT_TYPES = sorted(EVENT_GROUP)

# observed event.event_type value (lowercased) -> canonical. Covers the full
# high-frequency vocab seen across both graphs; extend the long tail as needed.
EVENT_TYPE_MAP = {
    "rate_decision": "monetary_policy", "monetary_policy": "monetary_policy",
    "cpi": "inflation", "inflation": "inflation", "ppi": "inflation",
    "employment": "employment", "payrolls": "employment", "unemployment": "employment", "jobs": "employment",
    "gdp": "growth", "pmi": "growth", "growth": "growth",
    "economic indicator": "econ_indicator", "economic_indicator": "econ_indicator",
    "data_release": "econ_indicator", "data_surprise": "econ_indicator", "data release": "econ_indicator",
    "trade_balance": "trade_balance", "fiscal_policy": "fiscal_policy",
    "debt_auction": "debt_issuance", "bond_issuance": "debt_issuance",
    "mergers_acquisitions": "m_and_a", "m&a": "m_and_a", "acquisition": "m_and_a", "merger": "m_and_a",
    "earnings": "earnings", "results": "earnings",
    "guidance": "guidance", "forecast": "guidance", "outlook": "guidance",
    "ipo": "ipo",
    "rating_review": "rating_action", "rating_action": "rating_action", "downgrade": "rating_action", "upgrade": "rating_action",
    "appointment": "mgmt_change", "leadership": "mgmt_change", "resignation": "mgmt_change", "management_change": "mgmt_change",
    "funding": "capital_action", "investment": "capital_action", "buyback": "capital_action",
    "dividend": "capital_action", "split": "capital_action", "capital_raise": "capital_action", "stake": "capital_action",
    "launch": "product", "product_launch": "product", "product": "product",
    "regulation": "legal_regulatory", "lawsuit": "legal_regulatory", "litigation": "legal_regulatory",
    "investigation": "legal_regulatory", "fine": "legal_regulatory", "approval": "legal_regulatory",
    "bankruptcy": "credit_event", "default": "credit_event", "restructuring": "credit_event",
    "partnership": "partnership", "deal": "partnership",
    "supply_shock": "supply_shock", "eia_inventory": "inventory", "inventory": "inventory",
    "opec_meeting": "opec", "opec": "opec",
    "geopolitics": "geopolitics", "war": "geopolitics", "sanctions": "geopolitics",
    "election": "election", "weather": "disaster", "disaster": "disaster", "strike": "strike",
    "fund_flows": "fund_flows", "price_change": "market_move", "market movement": "market_move",
    "market_move": "market_move", "valuation": "market_move", "contagion": "contagion",
    "meeting": "other", "event": "other", "announcement": "other", "null": "other", "other": "other",
}

# ---------------------------------------------------------------------------
# 2. STANDARD MECHANISM (from causal_event_edge.mechanism) — transmission axis.
# ---------------------------------------------------------------------------
STD_MECHANISMS = sorted({
    "monetary_policy", "fiscal_policy", "supply_shock", "demand_change", "regulation",
    "geopolitics", "rating_action", "earnings", "guidance", "m_and_a", "credit_default",
    "contagion", "fund_flows", "data_surprise", "competition", "mgmt_change",
    "corporate_action", "factor", "market_move", "partnership", "other",
})
MECHANISM_MAP = {
    "monetary_policy": "monetary_policy", "rate_decision": "monetary_policy",
    "fiscal_policy": "fiscal_policy",
    "supply_shock": "supply_shock", "demand_change": "demand_change",
    "regulation": "regulation", "geopolitics": "geopolitics",
    "mergers_acquisitions": "m_and_a", "m&a": "m_and_a",
    "default": "credit_default", "credit_event": "credit_default",
    "contagion": "contagion", "data_surprise": "data_surprise",
    "guidance": "guidance", "forecast": "guidance",
    "fund_flows": "fund_flows", "rating_action": "rating_action",
    "earnings": "earnings", "competition": "competition",
    "investment": "corporate_action", "capital_action": "corporate_action",
    "appointment": "mgmt_change", "leadership": "mgmt_change", "officer_of": "mgmt_change",
    "factor": "factor", "valuation": "market_move",
    "market movement": "market_move", "market_move": "market_move",
    "partnership": "partnership",
    "event": "other", "other": "other", "null": "other",
}

# ---------------------------------------------------------------------------
# 3. STANDARD SECTOR — GICS 11 (+ MACRO for non-corporate subjects, UNK).
# ---------------------------------------------------------------------------
GICS_SECTORS = [
    "Financials", "Information Technology", "Communication Services", "Energy",
    "Industrials", "Consumer Discretionary", "Consumer Staples", "Health Care",
    "Materials", "Utilities", "Real Estate", "MACRO", "UNK",
]

# free-text LLM `sector` / entity type (lowercased, substring-matched) -> GICS.
# MACRO = the subject is not a company (index/sovereign/rate/fx/commodity), so a
# corporate sector doesn't apply.
SECTOR_TEXT_MAP = {
    "bank": "Financials", "financ": "Financials", "insur": "Financials", "nbfc": "Financials",
    "broker": "Financials", "asset manag": "Financials", "lend": "Financials", "fintech": "Financials",
    "payment": "Financials",
    "software": "Information Technology", "tech": "Information Technology", "it ": "Information Technology",
    "semiconduct": "Information Technology", "hardware": "Information Technology", "internet": "Information Technology",
    "oil": "Energy", "gas": "Energy", "energy": "Energy", "petrol": "Energy", "refin": "Energy", "coal": "Energy",
    "pharma": "Health Care", "health": "Health Care", "hospital": "Health Care", "biotech": "Health Care", "medic": "Health Care",
    "auto": "Consumer Discretionary", "retail": "Consumer Discretionary", "apparel": "Consumer Discretionary",
    "hotel": "Consumer Discretionary", "media": "Communication Services", "entertain": "Communication Services",
    "telecom": "Communication Services", "telecommunicat": "Communication Services",
    "fmcg": "Consumer Staples", "food": "Consumer Staples", "beverage": "Consumer Staples",
    "consumer stapl": "Consumer Staples", "tobacco": "Consumer Staples",
    "metal": "Materials", "steel": "Materials", "mining": "Materials", "cement": "Materials",
    "chemical": "Materials", "material": "Materials", "paper": "Materials",
    "power": "Utilities", "utilit": "Utilities", "electricity": "Utilities",
    "realt": "Real Estate", "real estate": "Real Estate", "property": "Real Estate", "reit": "Real Estate",
    "industr": "Industrials", "infrastructure": "Industrials", "engineering": "Industrials",
    "capital goods": "Industrials", "aerospace": "Industrials", "defen": "Industrials",
    "airline": "Industrials", "aviation": "Industrials", "logistics": "Industrials", "construction": "Industrials",
    # non-corporate subjects -> MACRO
    "equit": "MACRO", "index": "MACRO", "market": "MACRO", "sovereign": "MACRO",
    "government": "MACRO", "central_bank": "MACRO", "central bank": "MACRO",
    "currenc": "MACRO", "fx": "MACRO", "rate": "MACRO", "bond": "MACRO", "commodit": "MACRO",
    "economic": "MACRO",
}

# entity `type` values that are inherently non-corporate -> MACRO (used before
# the free-text sector, since these are the most reliable non-company signal).
MACRO_ENTITY_TYPES = {
    "equity_index", "sovereign", "central_bank", "currency", "commodity",
    "rate_or_bond", "economic_indicator", "sector", "market", "exchange", "index",
}


# ---------------------------------------------------------------------------
# SUBSTRING FALLBACK for the long tail. Exact-match alone left 12.5% of events in
# `other` spread over 1,540 distinct strings, so no realistic number of dict keys
# fixes it -- the tail is variants (court_case / court_ruling / court_hearing /
# court_decision). Same technique SECTOR_TEXT_MAP already uses for sectors.
#
# ORDER IS SIGNIFICANT: first match wins, so specific patterns precede generic
# ones ("economic_growth" must hit growth before the generic "econom" rule).
#
# NOT INFORMATIVE is checked FIRST and deliberately: `event`, `meeting`,
# `announcement`, `report`, `release`, `statement` are the two largest strings in
# the tail and they are not event classes -- they are the extractor declining to
# name one. Mapping them anywhere would manufacture classification that does not
# exist. They stay `other`, honestly.
EVENT_TYPE_NON_INFORMATIVE = {
    "other", "null", "none", "n/a", "event", "meeting", "announcement", "report",
    "release", "statement", "update", "news", "initiative", "review", "general",
    "activity", "development", "unknown", "misc", "game", "holiday", "speech",
    "visit_", "nomination", "appearance", "interview", "comment",
}

EVENT_TYPE_PATTERNS = [
    # legal / regulatory / enforcement
    (("court", "lawsuit", "litigat", "legal", "judicial", "verdict", "ruling",
      "hearing", "settlement", "arrest", "indict", "penalt", "fine", "sanction_case",
      "regulat", "antitrust", "compliance", "investigat", "probe", "subpoena",
      "legislat", "bill_pass", "law_", "criminal", "fraud"), "legal_regulatory"),
    # growth BEFORE the generic economic rule
    (("gdp", "economic_growth", "growth", "expansion", "recession", "contraction",
      "slowdown"), "growth"),
    (("inflation", "cpi", "ppi", "price_index", "deflation"), "inflation"),
    (("unemploy", "employ", "payroll", "jobless", "labour_market", "labor_market"),
     "employment"),
    (("econom", "macro_data", "data_release", "statistic", "indicator",
      "retail_sales", "pmi", "survey_data"), "econ_indicator"),
    (("budget", "fiscal", "tax_", "_tax", "spending_", "stimulus", "austerity",
      "subsid"), "fiscal_policy"),
    (("central_bank", "monetary", "interest_rate", "rate_decision", "rate_cut",
      "rate_hike", "qe_", "quantitative"), "monetary_policy"),
    # markets & flows
    (("price_movement", "price movement", "price_change", "price change",
      "market_movement", "market movement", "selloff", "sell-off", "rally",
      "volatility", "correction_", "crash"), "market_move"),
    (("auction", "issuance", "bond_sale", "debt_sale", "syndicat", "placement"),
     "debt_issuance"),
    (("ipo", "listing", "flotation", "public_offering"), "ipo"),
    (("buyback", "dividend", "share_repurchase", "capital_raise", "rights_issue",
      "split"), "capital_action"),
    (("default", "restructur", "bankrupt", "insolven", "moratorium", "haircut",
      "debt_relief"), "credit_event"),
    (("downgrade", "upgrade", "rating", "outlook_change", "creditwatch"),
     "rating_action"),
    # corporate operations
    (("merger", "acquisit", "takeover", "buyout", "m&a", "divest", "spin"),
     "m_and_a"),
    (("earnings", "results", "profit", "revenue", "quarterly_report"), "earnings"),
    (("guidance", "forecast", "outlook", "warning"), "guidance"),
    # Pharma/biotech catalysts. Added because the data asked for it: Health Care's
    # `other` bucket ran 2.40x the sector control while every other sector's sat
    # near 1.0x, and inspecting it showed clinical/FDA events with no home in the
    # 30-class vocab. Ordered AFTER legal_regulatory on purpose -- bare "trial" is
    # ambiguous, and criminal_fraud_trial / administrative_trial must route to
    # legal first. Residual contamination is known and small (~3 of ~170:
    # "oil discovery" lands here wrongly).
    (("clinical", "fda", "drug", "vaccin", "therap", "efficacy", "study",
      "studies", "phase_1", "phase_2", "phase_3", "trial", "discovery"),
     "clinical_trial"),
    (("recall", "product", "launch", "approval", "patent", "trial_result"),
     "product"),
    (("partnership", "joint_venture", "alliance", "contract_win", "deal_signed"),
     "partnership"),
    (("resign", "appoint", "ceo", "cfo", "succession", "mgmt", "management_change",
      "board_change"), "mgmt_change"),
    # exogenous
    (("natural_disaster", "natural disaster", "earthquake", "hurricane", "flood",
      "storm", "wildfire", "drought", "weather", "accident", "incident",
      "explosion", "fire_", "outage", "pandemic", "epidemic", "disaster"),
     "disaster"),
    (("strike", "walkout", "industrial_action", "labour_dispute", "labor_dispute",
      "union_"), "strike"),
    (("war", "conflict", "attack", "terror", "invasion", "military", "protest",
      "unrest", "coup", "diplomat", "sanction", "geopolit", "border", "treaty"),
     "geopolitics"),
    (("election", "referendum", "vote", "poll", "ballot", "parliament"), "election"),
    (("opec",), "opec"),
    (("supply", "shortage", "disruption", "export_ban", "embargo"), "supply_shock"),
    (("inventor", "stockpile", "reserves_"), "inventory"),
    (("fund_flow", "inflow", "outflow", "allocation_"), "fund_flows"),
    (("contagion", "spillover", "systemic"), "contagion"),
    # operations: output/sales/capacity. A genuinely new class -- `sales` and
    # `production` were among the largest unmapped strings and belong to none of
    # the existing 30, so inventing a home is more honest than forcing them into
    # earnings (a reported figure) or growth (a macro aggregate).
    (("production", "output", "sales", "capacity", "manufactur", "shipment",
      "volume_", "operation"), "operations"),
]


def std_event_type(raw):
    """raw event.event_type -> (canonical_type, group). Unknown -> ('other','other').

    Exact map first (fast, curated), then ordered substring rules for the long
    tail of variants. Strings that are the extractor declining to classify are
    caught before either and left as `other` on purpose -- see
    EVENT_TYPE_NON_INFORMATIVE."""
    s = (raw or "").strip().lower()
    t = EVENT_TYPE_MAP.get(s)
    if t is None:
        if s in EVENT_TYPE_NON_INFORMATIVE or not s:
            t = "other"
        else:
            t = "other"
            for needles, canon in EVENT_TYPE_PATTERNS:
                if any(n in s for n in needles):
                    t = canon
                    break
    return t, EVENT_GROUP.get(t, "other")


def std_mechanism(raw):
    return MECHANISM_MAP.get((raw or "").strip().lower(), "other")


def sector_from_text(text):
    """free-text sector / entity-type string -> GICS sector or None (substring)."""
    s = (text or "").strip().lower()
    if not s:
        return None
    for key, sec in SECTOR_TEXT_MAP.items():
        if key in s:
            return sec
    return None


# ---------------------------------------------------------------------------
# 5. HEADLINE CLASSIFICATION — prose triggers, for the primitive pipeline.
# ---------------------------------------------------------------------------
# WHY A SECOND SET. EVENT_TYPE_MAP / EVENT_TYPE_PATTERNS read an LLM-emitted
# event_type SLUG ("rate_decision", "policy_change"). Applied to prose they miss
# most of it, because a newswire writes VERBS: "acquisit" matches "acquisition"
# but not "acquires", "buys" or "bid for". Measured on the 26,362 eg100k
# headlines that name a verified firm, slug patterns alone leave 61.6%
# unclassified; adding these prose triggers takes it to 41.1%.
#
# TWO EXCLUSION CLASSES, and they matter as much as the event classes:
#   roundup  "Colombian Stocks: Ecopetrol, Rubiales, Canacol" -- names many firms,
#            describes none of them. These are the bystander documents that F39
#            showed were poisoning every bucket via doc-union; catching them here
#            fixes it at source rather than downstream.
#   opinion  bylined Bloomberg columns ("...: David Reilly"). Commentary, not an
#            event, and it should never create a cell.
# Both are CLASSIFICATIONS, not failures -- callers should drop them, not try
# harder on them.

import re as _re

HEADLINE_EXCLUDE = [
    (_re.compile(r"^\s*[A-Z][\w.&' ]+(,\s*[A-Z][\w.&' ]+){2,}\s*:"), "roundup"),
    (_re.compile(r"\b(Stocks?|Equity|Market)\s*(Preview|Movers|Roundup|Report|Wrap)\b"
                 r"|^\w+ Stocks:|\bin Court News\b|\bIntellectual Property\s*$", _re.I), "roundup"),
    (_re.compile(r":\s*[A-Z][a-z]+ [A-Z][a-z]+\s*$"), "opinion"),
]

HEADLINE_PATTERNS = [
    (r"\b(bids? for|bid to buy|buys?|bought|to acquire|acquires?|acquired|takes? over|"
     r"agreed? to buy|agrees? to buy|merge[sd]? with|tie-?up with|stake in|"
     r"sells? .{0,30}\bunit\b|sells? .{0,30}\bbusiness\b|divests?)\b", "m_and_a"),
    (r"\b(sells? .{0,25}(bonds?|notes?|bills?|debt)|issues? .{0,25}(bonds?|notes?|debt)|"
     r"raises? .{0,20}in (bonds?|debt)|bond sale|note sale|debt sale|tap(s|ped)? .{0,15}market)\b",
     "debt_issuance"),
    (r"\b(partners? with|partnership with|joint venture|alliance with|teams? up with|"
     r"signs? .{0,25}(deal|agreement|contract)|wins? .{0,20}contract)\b", "partnership"),
    (r"\b(profit|earnings|revenue|sales) (rose|fell|climbed|dropped|beat|missed|gained|slid)\b"
     r"|\breports? .{0,25}(profit|loss|earnings|revenue)|\b(first|second|third|fourth|1st|2nd|3rd|4th)"
     r"[- ]quarter (profit|loss|earnings|revenue|results)|\bposts? .{0,20}(profit|loss)\b", "earnings"),
    (r"\b(forecasts?|sees|expects?|projects?|guides?|raises? .{0,15}(outlook|forecast)|"
     r"cuts? .{0,15}(outlook|forecast)|warns?)\b", "guidance"),
    (r"\b(cut to|raised to|lowered to|upgraded? to|downgraded? to|puts? .{0,20}on review|"
     r"outlook to (negative|positive|stable)|affirms? .{0,15}rating)\b", "rating_action"),
    (r"\b(sues?|sued|lawsuit|settles?|settlement|fined?|penalt|probe[sd]?|investigat|"
     r"court|judge|ruling|appeals?|charged with|indicted|antitrust|regulator)\b", "legal_regulatory"),
    (r"\b(stocks?|shares?|index|futures|bonds?|yields?|currency|peso|euro|dollar|franc|"
     r"yen|pound|swaps?|spreads?) .{0,25}\b(advance[sd]?|gain(s|ed)?|rise[sn]?|rose|climb(s|ed)?|"
     r"rall(y|ies|ied)|drop(s|ped)?|fall(s|en)?|fell|decline[sd]?|slide[sd]?|slid|weaken[sd]?|"
     r"strengthen[sd]?|erase[sd]?|tumble[sd]?|surge[sd]?)\b", "market_move"),
    (r"\b(raises?|cuts?|holds?|keeps?) .{0,20}(rate|rates)\b|\b(central bank|fed|ecb|boe|boj)\b"
     r".{0,40}\b(rate|policy|purchase|stimulus|easing)\b|\bbond (buying|program|purchase)",
     "monetary_policy"),
    (r"\b(default(s|ed)?|restructur|bankrupt|insolven|writedown|write-down|haircut|"
     r"maturity extension|debt relief|bailout|credit facility)\b", "credit_event"),
    (r"\b(shuts?|halts?|closes?|reopens?|resumes?|output|production|capacity|shipment|"
     r"refinery|plant|mine|field)\b", "operations"),
    (r"\b(shortage|disruption|embargo|export ban|glut|surplus|stockpile)\b", "supply_shock"),
    (r"\b(elect(s|ed|ion)?|vote[sd]?|referendum|parliament|poll)\b", "election"),
]
HEADLINE_PATTERNS = [(_re.compile(p, _re.I), c) for p, c in HEADLINE_PATTERNS]


def classify_headline(headline):
    """headline text -> (class, group). Returns ('roundup'|'opinion', 'exclude')
    for documents that should never create a cell. Slug patterns run first so the
    two vocabularies stay consistent where they overlap."""
    h = (headline or "").strip()
    if not h:
        return "other", "other"
    slug, grp = std_event_type(h)
    if slug != "other":
        return slug, grp
    for rx, lbl in HEADLINE_EXCLUDE:
        if rx.search(h):
            return lbl, "exclude"
    for rx, canon in HEADLINE_PATTERNS:
        if rx.search(h):
            return canon, EVENT_GROUP.get(canon, "other")
    return "other", "other"
