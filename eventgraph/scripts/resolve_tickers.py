# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Entity -> tradeable-symbol resolution (the security master), deterministic, no LLM.

Per entity, in order:
  1. MACRO   : exact curated map (country/currency/commodity/index) -> US ETF/index
               proxy (clean free daily bars). Handles 'X futures/prices' + 'u.s. X'.
  2. SEC     : company_tickers.json (free; all US-listed + ADRs; exact then fuzzy).
  3. OpenFIGI: opt-in (--openfigi N / OPENFIGI_API_KEY) for the global/ADR tail.
  4. FACTOR  : abstraction/instrument -> a risk-factor proxy (credit->HYG,
               yields->^TNX, oil->USO, inflation->TIP, ...). This is Fable's
               "route abstraction nodes to the factor vocabulary" -- makes concept
               nodes tradable instead of dropping them.

Prioritizes by causal-edge EFFECT frequency and reports coverage on the
TRADEABLE-type effects (persons/pure abstractions are excluded from the denominator).
Output: <graph-dir>/entity_symbol.jsonl {entity_id,name,type,symbol,source,kind,edge_freq}.

Usage: uv run eventgraph/scripts/resolve_tickers.py --graph-dir /tmp/eg_6k [--openfigi 400]
"""
import argparse, json, re, difflib, collections, time, os
from pathlib import Path
import httpx

MACRO = {
    # broad
    "united states": "SPY", "us economy": "SPY", "u.s. economy": "SPY",
    "global economy": "ACWI", "world economy": "ACWI", "emerging markets": "EEM",
    # countries -> country ETF
    "europe": "FEZ", "european union": "FEZ", "euro area": "FEZ", "eurozone": "FEZ",
    "european debt crisis": "FEZ", "germany": "EWG", "france": "EWQ", "greece": "GREK",
    "italy": "EWI", "spain": "EWP", "portugal": "PGAL", "ireland": "EIRL",
    "netherlands": "EWN", "switzerland": "EWL", "china": "FXI", "japan": "EWJ",
    "brazil": "EWZ", "russia": "RSX", "india": "INDA", "united kingdom": "EWU",
    "britain": "EWU", "mexico": "EWW", "south korea": "EWY", "korea": "EWY",
    "canada": "EWC", "australia": "EWA", "turkey": "TUR", "south africa": "EZA",
    "indonesia": "EIDO", "thailand": "THD", "malaysia": "EWM", "poland": "EPOL",
    "taiwan": "EWT", "hong kong": "EWH", "singapore": "EWS", "argentina": "ARGT",
    "saudi arabia": "KSA", "israel": "EIS", "ukraine": "RSX",
    # central banks / rate hubs -> yield / bond proxy
    "federal reserve": "^TNX", "european central bank": "BUND", "bank of japan": "EWJ",
    "bank of england": "EWU", "central bank": "^TNX", "reserve bank of india": "INDA",
    "international monetary fund": "ACWI",
    # commodities
    "crude oil": "USO", "oil": "USO", "brent crude": "BNO", "oil prices": "USO",
    "gold": "GLD", "silver": "SLV", "natural gas": "UNG", "copper": "CPER",
    "palm oil": "USO", "wheat": "WEAT", "corn": "CORN", "commodities": "DBC",
    "aluminum": "JJU", "iron ore": "PICK",
    # currencies -> FX ETF
    "u.s. dollar": "UUP", "us dollar": "UUP", "dollar": "UUP", "euro": "FXE",
    "yen": "FXY", "japanese yen": "FXY", "pound": "FXB", "sterling": "FXB",
    "australian dollar": "FXA", "swiss franc": "FXF",
    # indices
    "s&p 500": "SPY", "stoxx europe 600": "FEZ", "msci asia pacific index": "AAXJ",
    "msci asia pacific": "AAXJ", "dow jones": "DIA", "nasdaq": "QQQ",
    "ftse 100": "EWU", "dax": "EWG", "nikkei": "EWJ", "hang seng": "EWH",
    # concepts
    "inflation": "TIP", "investors": "SPY",
}
# substring -> factor proxy (checked LAST, for abstractions/instruments)
FACTOR = [
    ("credit", "HYG"), ("high yield", "HYG"), ("treasur", "TLT"), ("govt bond", "TLT"),
    ("government bond", "TLT"), ("bond", "TLT"), ("yield", "^TNX"), ("interest rate", "^TNX"),
    ("rate", "^TNX"), ("inflation", "TIP"), ("brent", "BNO"), ("crude", "USO"),
    ("oil", "USO"), ("natural gas", "UNG"), ("gold", "GLD"), ("silver", "SLV"),
    ("copper", "CPER"), ("dollar", "UUP"), ("euro", "FXE"), ("yen", "FXY"),
    ("volatilit", "VIXY"), ("vix", "VIXY"), ("equit", "SPY"), ("stock", "SPY"),
    ("share", "SPY"), ("wheat", "WEAT"), ("corn", "CORN"), ("recovery", "SPY"),
    ("economic growth", "SPY"), ("economy", "SPY"), ("recession", "SPY"),
]
MACRO_TYPES = {"country", "sovereign", "supranational", "currency", "commodity",
               "equity_index", "rate_or_bond", "central_bank", "economic_indicator",
               "asset_class", "market"}
TRADEABLE_TYPES = {"company", "bank", "equity_index", "commodity", "currency", "exchange",
                   "central_bank", "sovereign", "rate_or_bond", "economic_indicator", "asset_class"}

SUFFIX = re.compile(r"\b(inc|corp|corporation|co|ltd|plc|llc|llp|sa|ag|nv|group|holdings?|the|company|ord|adr|spon)\b", re.I)
def norm(s):
    s = (s or "").lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = SUFFIX.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()

def macro_lookup(nm, macro):
    variants = {nm,
                re.sub(r"\b(futures|prices|price|markets|market|index)\b", " ", nm),
                re.sub(r"^(u s|us|the) ", "", nm)}
    for c in variants:
        c = re.sub(r"\s+", " ", c).strip()
        if c in macro:
            return (macro[c], "macro", "etf_proxy")
    return None

def factor_lookup(nm):
    toks = nm.split()
    for kw, sym in FACTOR:
        if " " in kw:                       # multiword: substring is safe
            if kw in nm:
                return (sym, "factor_proxy", "factor")
        else:                               # single word: match whole token / stem
            for t in toks:                  # (avoids 'rate' matching 'infratel')
                if t == kw or (len(kw) >= 4 and t.startswith(kw)):
                    return (sym, "factor_proxy", "factor")
    return None

# factor routing applies ONLY to abstraction/instrument types, never to a
# company/bank/person (those must get their own ticker or stay unresolved).
FACTOR_OK = {"other", "economic_indicator", "sector", "rate_or_bond",
             "asset_class", "market", "commodity", "currency"}

def load_entities(pg_sql):
    pat = re.compile(r"INSERT INTO entity \([^)]*\) VALUES \('([^']+)','((?:[^']|'')*)','([^']*)'")
    return {m.group(1): (m.group(2).replace("''", "'"), m.group(3))
            for m in (pat.search(l) for l in open(pg_sql)) if m}

def load_freq(lake):
    freq = collections.Counter()
    for fn, w_eff, w_other in [("causal_event_edge.jsonl", 2, 1), ("sensitivity_edge.jsonl", 2, 0)]:
        p = lake / fn
        if not p.exists(): continue
        for l in open(p):
            j = json.loads(l)
            for k, w in [("effect_entity", w_eff), ("asset_entity", w_eff), ("cause_entity", w_other)]:
                if j.get(k): freq[j[k]] += w
    return freq

def build_sec_index():
    print("downloading SEC company_tickers.json (free) ...")
    try:
        r = httpx.get("https://www.sec.gov/files/company_tickers.json",
                      headers={"User-Agent": "eventgraph research jamiep.tigereye@gmail.com"}, timeout=30)
        r.raise_for_status(); data = r.json()
    except Exception as e:
        print(f"  SEC download failed ({e})"); return {}, {}
    exact, tok2 = {}, collections.defaultdict(set)
    for v in data.values():
        nm, tk = norm(v["title"]), v["ticker"]
        if nm and nm not in exact:
            exact[nm] = tk
            for t in nm.split():
                if len(t) >= 4: tok2[t].add(nm)
    print(f"  indexed {len(exact)} US-listed names")
    return exact, tok2

def sec_match(name, exact, tok2):
    n = norm(name)
    if not n: return None
    if n in exact: return (exact[n], "sec_exact", "security")
    cands = set()
    for t in n.split():
        if len(t) >= 4: cands |= tok2.get(t, set())
    best = difflib.get_close_matches(n, list(cands), n=1, cutoff=0.90) if cands else []
    return (exact[best[0]], "sec_fuzzy", "security") if best else None

EQUITY_TYPES = ("Common Stock", "ADR", "ETP", "REIT", "Depositary Receipt")
def openfigi_search(name, key):
    h = {"Content-Type": "application/json"}
    if key: h["X-OPENFIGI-APIKEY"] = key
    # US-listed first (best free daily-bar coverage), then any exchange as fallback.
    for payload in ({"query": name, "exchCode": "US"}, {"query": name}):
        for attempt in range(2):
            try:
                r = httpx.post("https://api.openfigi.com/v3/search", headers=h, json=payload, timeout=20)
            except Exception:
                break
            if r.status_code == 429:                 # rate limited -> back off once
                time.sleep(6); continue
            if r.status_code != 200:
                break
            for d in r.json().get("data", []):
                if d.get("securityType") in EQUITY_TYPES:
                    tag = "openfigi_us" if payload.get("exchCode") == "US" else f"openfigi_{d.get('exchCode','x')}"
                    return (d.get("ticker"), tag, "security")
            break
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="/tmp/eg_6k")
    ap.add_argument("--openfigi", type=int, default=0)
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    ents = load_entities(gd / "pg_upsert.sql")
    freq = load_freq(gd / "lake")
    print(f"{len(ents)} entities; {len(freq)} appear in edges")
    exact, tok2 = build_sec_index()
    macro = {norm(k): v for k, v in MACRO.items()}
    of_key = os.environ.get("OPENFIGI_API_KEY", "")
    of_budget = a.openfigi
    if of_budget and not of_key:
        print("  (OpenFIGI keyless: rate-limited; set OPENFIGI_API_KEY for real coverage)")
    OF_TYPES = ("company", "bank", "exchange")   # only real securities -> OpenFIGI

    # OpenFIGI result cache (resumable; avoids re-querying across runs).
    of_cache, cache_path = {}, Path(a.graph_dir) / "openfigi_cache.jsonl"
    if cache_path.exists():
        for l in open(cache_path):
            try:
                j = json.loads(l); of_cache[j["k"]] = tuple(j["r"]) if j["r"] else None
            except Exception: pass
    if of_cache: print(f"  loaded {len(of_cache)} cached OpenFIGI lookups")
    cache_f = open(cache_path, "a")

    out, src = {}, collections.Counter()
    for eid, (name, etype) in sorted(ents.items(), key=lambda kv: -freq.get(kv[0], 0)):
        nm = norm(name)
        res = macro_lookup(nm, macro)
        if not res and etype in ("company", "bank", "exchange", "regulator", "other"):
            res = sec_match(name, exact, tok2)
            if not res and etype in OF_TYPES:
                if nm in of_cache:
                    res = of_cache[nm]                       # cache hit (may be None)
                elif of_budget > 0:
                    res = openfigi_search(name, of_key); of_budget -= 1
                    of_cache[nm] = res
                    cache_f.write(json.dumps({"k": nm, "r": list(res) if res else None}) + "\n"); cache_f.flush()
                    if not of_key: time.sleep(1.3)
        if not res and etype in FACTOR_OK:
            res = factor_lookup(nm)
        if res:
            out[eid] = {"entity_id": eid, "canonical_name": name, "type": etype,
                        "symbol": res[0], "source": res[1], "kind": res[2], "edge_freq": freq.get(eid, 0)}
            src[res[1]] += 1
        else:
            src["unresolved"] += 1

    with open(gd / "entity_symbol.jsonl", "w") as f:
        for v in out.values(): f.write(json.dumps(v) + "\n")

    # coverage on TRADEABLE-type effect edges (the realised-loop denominator)
    eff = collections.Counter()
    ce = gd / "lake" / "causal_event_edge.jsonl"
    if ce.exists():
        for l in open(ce):
            j = json.loads(l)
            if j.get("effect_entity"): eff[j["effect_entity"]] += 1
    tradeable = {e: c for e, c in eff.items() if e in ents and ents[e][1] in TRADEABLE_TYPES}
    cov = sum(c for e, c in tradeable.items() if e in out)
    print("\n=== resolution ===")
    for k, v in src.most_common(): print(f"  {k:12} {v}")
    print(f"resolved {len(out)}/{len(ents)} entities")
    if tradeable:
        print(f"TRADEABLE-EFFECT coverage: {cov}/{sum(tradeable.values())} effect-edges "
              f"({100*cov/sum(tradeable.values()):.0f}%)  <-- the realised-loop number")
    print(f"wrote {gd/'entity_symbol.jsonl'}")

if __name__ == "__main__":
    main()
