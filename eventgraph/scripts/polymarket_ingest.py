# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Polymarket -> eventgraph proposition ingest (LLM-FREE by default).

Prediction markets are the forward twin of the news graph: every market is a
resolvable proposition with (a) an explicit RESOLUTION DATE -> populates a forward
event calendar, (b) a live market-implied PROBABILITY -> a continuous sensitivity
annotation, and (c) VOLUME/liquidity -> the attention observable the news timestamps
cannot give. The graph already reserves proposition.contract_ref / probability for
exactly this.

Pull (free public API, no key):
  * Gamma /events  -> markets nested with tags, outcomePrices, volume, endDate.
  * CLOB  /prices-history (--prices) -> the implied-probability PATH per market
    (the forward repricing/attention series to test against realised vol).

Resolve each market's entities -> tradeable US-preferred tickers, DETERMINISTICALLY:
  1. MACRO/crypto curated map (countries/central-banks/commodities/indices/coins).
  2. Yahoo search over capitalised candidate phrases, gated exactly like
     resolve_yahoo.py (quoteType==EQUITY + price-exists + issuer name-similarity),
     US listings preferred.
  3. --llm-tail (OPT-IN, costs nothing unless set): a single batched gpt-oss-120b
     pass over the still-unresolved questions, every hit re-verified against Yahoo.

Emits into <graph-dir>/lake (additive; no news re-extraction, no spend by default):
  proposition.jsonl        one row per market (contract_ref, resolution_date, prob,
                           volume, tags, resolved symbols + provenance).
  proposition_price.jsonl  (--prices) implied-prob path {prop_id, t, d, p}.

Usage:
  uv run eventgraph/scripts/polymarket_ingest.py --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/polymarket_ingest.py --graph-dir eventgraph/data/eg_live \
      --max-events 400 --prices                 # + fetch implied-prob paths
  ... --llm-tail                                # opt-in LLM tail (source data/tmp/groq.env)
"""
import argparse, json, re, time, difflib, hashlib, os, collections, datetime as dt
from pathlib import Path
import httpx

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
US_EXCH = {"NMS", "NYQ", "NGM", "NCM", "NSM", "ASE", "PCX", "PNK", "OQB", "OQX", "NAS", "NYS"}

# curated macro/crypto map (checked as lowercased-question substrings; longest wins).
# values are Yahoo symbols with clean daily bars (ETF proxies / index / crypto pairs).
MACRO = {
    "s&p 500": "SPY", "nasdaq": "QQQ", "dow jones": "DIA", "russell 2000": "IWM",
    "vix": "^VIX", "volatility index": "^VIX",
    "federal reserve": "^TNX", "the fed": "^TNX", "fed funds": "^TNX", "fomc": "^TNX",
    "interest rate": "^TNX", "10-year": "^TNX", "treasury yield": "^TNX",
    "european central bank": "FEZ", "ecb": "FEZ", "bank of japan": "EWJ",
    "recession": "SPY", "inflation": "TIP", "cpi": "TIP", "jobs report": "SPY",
    "gdp": "SPY", "unemployment": "SPY",
    "crude oil": "USO", "oil price": "USO", "brent": "BNO", "wti": "USO",
    "gold price": "GLD", "gold": "GLD", "silver": "SLV", "natural gas": "UNG",
    "us dollar": "UUP", "dollar index": "UUP", "dxy": "UUP", "euro": "FXE",
    "japanese yen": "FXY", "british pound": "FXB",
    "china": "FXI", "japan": "EWJ", "germany": "EWG", "india": "INDA",
    "brazil": "EWZ", "mexico": "EWW", "emerging markets": "EEM",
    "bitcoin": "BTC-USD", "btc": "BTC-USD", "ethereum": "ETH-USD", "ether": "ETH-USD",
    "eth": "ETH-USD", "solana": "SOL-USD", "dogecoin": "DOGE-USD", "xrp": "XRP-USD",
    "ripple": "XRP-USD", "cardano": "ADA-USD", "litecoin": "LTC-USD",
}
# tags that mark a market as market-relevant. The finance tag is the KEEP GATE:
# a market is ingested only if it is finance-tagged, so spurious ticker matches on
# sports/election/weather questions ("eth"->Ethiopia, "Argentina"->Lithium Argentina)
# can never pull non-financial markets into the graph.
FINANCE_TAGS = {"business", "economy", "finance", "crypto", "bitcoin", "ethereum",
                "fed", "interest rates", "inflation", "recession", "stocks", "earnings",
                "markets", "tech", "economics", "commodities", "oil", "companies",
                "gold", "stock market", "s&p 500", "nasdaq", "solana", "xrp", "dogecoin",
                "federal reserve", "gdp", "jobs", "cpi", "treasuries", "bonds"}
# hard-exclude domains even if some finance tag co-occurs (sports/politics dominate volume).
EXCLUDE_TAGS = {"sports", "soccer", "football", "basketball", "baseball", "esports",
                "games", "gaming", "pop culture", "weather", "entertainment", "awards",
                "fifa world cup", "nfl", "nba", "mlb", "tennis", "cricket", "mention markets"}
CANDIDATE = re.compile(r"\b([A-Z][a-zA-Z0-9.&'-]+(?:\s+[A-Z][a-zA-Z0-9.&'-]+){0,3})\b")
STOP = {"Will", "Yes", "No", "The", "What", "Who", "When", "How", "Which", "US", "U.S.",
        "United States", "President", "Trump", "Biden", "America", "American", "New",
        "This", "That", "January", "February", "March", "April", "May", "June", "July",
        "August", "September", "October", "November", "December", "Q1", "Q2", "Q3", "Q4"}


def gget(sess, path, **params):
    r = sess.get(f"{GAMMA}{path}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def yahoo_search(sess, q, cache, cf):
    if q in cache:
        return cache[q]
    try:
        r = sess.get("https://query2.finance.yahoo.com/v1/finance/search",
                     params={"q": q, "quotesCount": 6, "newsCount": 0}, timeout=15)
        quotes = r.json().get("quotes", []) if r.status_code == 200 else []
    except Exception:
        quotes = []
    cache[q] = quotes
    cf.write(json.dumps({"q": q, "r": quotes}) + "\n"); cf.flush()
    time.sleep(0.2)
    return quotes


def has_prices(sess, sym):
    p2 = int(dt.datetime(2026, 12, 31, tzinfo=dt.UTC).timestamp())
    p1 = int(dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp())
    try:
        r = sess.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"period1": p1, "period2": p2, "interval": "1d"}, timeout=20)
        cl = r.json()["chart"]["result"][0]["indicators"]["quote"][0].get("close") or []
        return sum(1 for c in cl if c is not None) > 120
    except Exception:
        return False


def resolve_yahoo(sess, name, cache, cf, min_sim, verified):
    """gated Yahoo-search resolution (EQUITY + price-exists + name-sim), US preferred."""
    quotes = yahoo_search(sess, name, cache, cf)
    scored = []
    for x in quotes:
        if x.get("quoteType") != "EQUITY" or not x.get("symbol"):
            continue
        nm = x.get("shortname") or x.get("longname") or ""
        sim = difflib.SequenceMatcher(None, name.lower(), nm.lower()).ratio()
        scored.append((sim, x, nm))
    ok = [t for t in scored if t[0] >= min_sim]
    if not ok:
        return None
    ok.sort(key=lambda t: (0 if (t[1].get("exchange") in US_EXCH or "." not in t[1]["symbol"]) else 1, -t[0]))
    sim, top, nm = ok[0]; sym = top["symbol"]
    if sym not in verified:
        verified[sym] = has_prices(sess, sym)
    if not verified[sym]:
        return None
    return {"name": name, "symbol": sym, "kind": "security", "via": "yahoo_search",
            "match_name": nm, "name_sim": round(sim, 2)}


def resolve_market(sess, question, cache, cf, min_sim, verified, yahoo_companies=False):
    """Resolution dicts (deduped by symbol).

    DEFAULT = curated macro/crypto only, which is clean and high-precision (countries,
    central banks, commodities, indices, coins). Naive Yahoo-search over capitalised
    words in raw QUESTION text is deliberately OFF: question text is full of place
    names / titles / product names that spuriously match small-cap tickers
    ("Bab el-Mandeb"->Alibaba). Company resolution is the verified --llm-tail's job.
    --yahoo-companies re-enables the naive path for experimentation only.
    """
    ql = question.lower()
    res, seen = [], set()
    for kw in sorted(MACRO, key=len, reverse=True):
        # word-boundary so short keys don't match inside words (eth != Ethiopia, btc != ...)
        if re.search(r"\b" + re.escape(kw) + r"\b", ql) and MACRO[kw] not in seen:
            seen.add(MACRO[kw]); res.append({"name": kw, "symbol": MACRO[kw], "kind": "macro", "via": "curated"})
    if yahoo_companies:
        for m in CANDIDATE.finditer(question):
            cand = m.group(1).strip()
            if len(cand) < 3 or cand in STOP or cand.split()[0] in STOP:
                continue
            r = resolve_yahoo(sess, cand, cache, cf, min_sim, verified)
            if r and r["symbol"] not in seen:
                seen.add(r["symbol"]); res.append(r)
    return res


def categorize(tags):
    t = {x.lower() for x in tags}
    for c in ("crypto", "bitcoin", "ethereum"):
        if c in t: return "crypto"
    if t & {"fed", "interest rates", "inflation", "recession", "economy", "economics"}: return "macro"
    if t & {"stocks", "earnings", "companies", "tech", "business"}: return "company"
    if t & {"oil", "commodities"}: return "commodity"
    return "other"


def prop_id(cond):
    return "pm_" + hashlib.sha1((cond or "").encode()).hexdigest()[:16]


def price_history(sess, token, fidelity=1440):
    try:
        r = sess.get(f"{CLOB}/prices-history", params={"market": token, "interval": "max", "fidelity": fidelity}, timeout=25)
        return r.json().get("history", []) if r.status_code == 200 else []
    except Exception:
        return []


def llm_tail(unresolved, key, model, sess, cache, cf, min_sim, verified):
    """opt-in: one batched gpt-oss-120b pass proposing ticker+issuer, each re-verified vs Yahoo."""
    SYSTEM = ('For each prediction-market question about a company/asset, name the single most-affected '
              'PUBLICLY-LISTED US ticker and its issuer, or null if none (elections, sports, macro-only, '
              'private co). Return ONLY JSON {"results":[{"i":<int>,"ticker":"AAPL|null","issuer":"Apple Inc"}]}')
    out = {}
    for i in range(0, len(unresolved), 30):
        batch = unresolved[i:i + 30]
        lines = [f'{k+1}. "{q[:160]}"' for k, (pid, q) in enumerate(batch)]
        body = {"model": model, "temperature": 0, "max_tokens": 1400, "response_format": {"type": "json_object"},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]}
        try:
            r = httpx.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
            txt = r.json()["choices"][0]["message"]["content"]; a, b = txt.find("{"), txt.rfind("}")
            rows = {x["i"]: x for x in json.loads(txt[a:b + 1]).get("results", []) if isinstance(x, dict)}
        except Exception:
            rows = {}
        for k, (pid, q) in enumerate(batch):
            x = rows.get(k + 1) or {}
            tkr = (x.get("ticker") or "").strip().upper()
            if not tkr or tkr in ("NULL", "NONE"):
                continue
            iss = x.get("issuer") or tkr
            # verify: price exists AND Yahoo issuer name matches the LLM's issuer (anti-hallucination)
            if tkr not in verified:
                verified[tkr] = has_prices(sess, tkr)
            if not verified[tkr]:
                continue
            quotes = yahoo_search(sess, iss, cache, cf)
            nm = next((y.get("shortname") or y.get("longname") or "" for y in quotes if y.get("symbol") == tkr), "")
            if not nm:
                nm = next((y.get("shortname") or "" for y in quotes if y.get("quoteType") == "EQUITY"), "")
            sim = difflib.SequenceMatcher(None, iss.lower(), nm.lower()).ratio()
            if sim < min_sim:
                continue
            out[pid] = {"name": iss, "symbol": tkr, "kind": "security", "via": "llm_verified",
                        "match_name": nm, "name_sim": round(sim, 2)}
        print(f"  llm-tail batch {i//30+1}: +{len(out)} verified so far")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--max-events", type=int, default=300, help="active Gamma events to scan (by 24h volume)")
    ap.add_argument("--include-closed", type=int, default=0, metavar="N",
                    help="ALSO pull N resolved/historical events (by total volume) -> ground-truth "
                         "labels + full price paths = the supervised set for the vol/attention test")
    ap.add_argument("--min-volume", type=float, default=5000.0, help="skip thin markets below this total USD volume")
    ap.add_argument("--min-sim", type=float, default=0.6, help="issuer name-similarity gate")
    ap.add_argument("--prices", action="store_true", help="also fetch CLOB implied-prob paths")
    ap.add_argument("--llm-tail", action="store_true", help="opt-in gpt-oss-120b pass over unresolved (needs GROQ_API_KEY)")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--all", action="store_true", help="keep markets even with no finance tag and no symbol")
    ap.add_argument("--yahoo-companies", action="store_true",
                    help="(experimental, noisy) also naive-Yahoo-search capitalised words in questions")
    a = ap.parse_args()

    gd = Path(a.graph_dir); lake = gd / "lake"; lake.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})
    cache_path = gd / "yahoo_search_cache.jsonl"; cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            j = json.loads(l); cache[j["q"]] = j["r"]
    cf = open(cache_path, "a"); verified = {}

    # 1. pull events (markets nested with tags). Active by 24h volume; closed by total volume.
    def pull(n, **q):
        acc, off = [], 0
        while len(acc) < n:
            page = gget(sess, "/events", limit=100, offset=off, **q)
            if not page:
                break
            acc.extend(page); off += 100
        return acc[:n]

    events = pull(a.max_events, closed="false", active="true", order="volume24hr", ascending="false")
    print(f"pulled {len(events)} active events")
    if a.include_closed:
        closed = pull(a.include_closed, closed="true", order="volume", ascending="false")
        print(f"pulled {len(closed)} resolved/historical events")
        events += closed

    props = []; dropped_nonfin = 0
    for e in events:
        tags = [t.get("label", "") for t in (e.get("tags") or [])]
        tagset = {t.lower() for t in tags}
        is_fin = bool(tagset & FINANCE_TAGS) and not (tagset & EXCLUDE_TAGS)
        if not is_fin and not a.all:
            dropped_nonfin += len(e.get("markets") or [])
            continue
        for m in e.get("markets") or []:
            q = m.get("question") or ""
            try:
                vol = float(m.get("volumeNum") or m.get("volume") or 0)
            except Exception:
                vol = 0.0
            if vol < a.min_volume or not m.get("conditionId"):
                continue
            try:
                prices = json.loads(m.get("outcomePrices") or "[]")
                outs = json.loads(m.get("outcomes") or "[]")
            except Exception:
                prices, outs = [], []
            prob_yes = None
            if outs and prices and len(outs) == len(prices):
                yi = next((i for i, o in enumerate(outs) if str(o).lower() in ("yes", "up", "above")), 0)
                try: prob_yes = float(prices[yi])
                except Exception: prob_yes = None
            res = resolve_market(sess, q, cache, cf, a.min_sim, verified, a.yahoo_companies)
            pid = prop_id(m["conditionId"])
            try:
                clobs = json.loads(m.get("clobTokenIds") or "[]")
            except Exception:
                clobs = []
            props.append({
                "prop_id": pid, "source": "polymarket", "contract_ref": m["conditionId"],
                "slug": m.get("slug"), "event_title": e.get("title"), "question": q,
                "resolution_date": (m.get("endDate") or "")[:10],
                "start_date": (m.get("startDate") or e.get("startDate") or "")[:10],
                "resolution_criteria": (m.get("description") or m.get("resolutionSource") or "")[:400],
                "tags": tags, "category": categorize(tags),
                "prob_yes": prob_yes, "probability_annotation": prob_yes,
                "outcomes": outs, "volume": vol, "volume24hr": float(m.get("volume24hr") or 0),
                "liquidity": float(m.get("liquidity") or 0),
                "one_day_price_change": m.get("oneDayPriceChange"),
                "one_week_price_change": m.get("oneWeekPriceChange"),
                "closed": bool(m.get("closed")),
                # for resolved markets outcomePrices settle to 0/1 -> ground-truth label
                "resolved_outcome": ("yes" if prob_yes >= 0.5 else "no")
                if (m.get("closed") and isinstance(prob_yes, (int, float))) else None,
                "clob_token_yes": clobs[0] if clobs else None,
                "symbols": [r["symbol"] for r in res], "resolution": res,
            })
        # progress
    print(f"built {len(props)} propositions from finance-tagged events "
          f"(dropped {dropped_nonfin} non-finance markets); "
          f"{sum(1 for p in props if p['symbols'])} resolve to >=1 ticker")

    # 2. opt-in LLM tail over questions that finance-tagged but resolved to nothing
    if a.llm_tail:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            print("  --llm-tail set but GROQ_API_KEY missing (source data/tmp/groq.env) -> skipping")
        else:
            need = [(p["prop_id"], p["question"]) for p in props if not p["symbols"]]
            print(f"  llm-tail: {len(need)} finance props unresolved -> re-judging")
            got = llm_tail(need, key, a.model, sess, cache, cf, a.min_sim, verified)
            by_id = {p["prop_id"]: p for p in props}
            for pid, r in got.items():
                by_id[pid]["symbols"] = [r["symbol"]]; by_id[pid]["resolution"] = [r]
            print(f"  llm-tail resolved {len(got)} more")

    # 3. write proposition table, MERGING by source (keep other sources e.g. kalshi)
    ppath = lake / "proposition.jsonl"
    kept = []
    if ppath.exists():
        kept = [json.loads(l) for l in open(ppath) if l.strip() and json.loads(l).get("source") != "polymarket"]
    with open(ppath, "w") as f:
        for p in kept + props:
            f.write(json.dumps(p) + "\n")
    resolved = sum(1 for p in props if p["symbols"])
    print(f"wrote {len(props)} -> {lake/'proposition.jsonl'}  ({resolved} resolved, "
          f"{resolved/max(len(props),1):.0%})")

    # 4. optional implied-prob paths
    if a.prices:
        pp = lake / "proposition_price.jsonl"; n = 0
        with open(pp, "w") as f:
            for p in props:
                if not p["clob_token_yes"]:
                    continue
                for pt in price_history(sess, p["clob_token_yes"]):
                    d = dt.datetime.fromtimestamp(pt["t"], dt.UTC).strftime("%Y-%m-%d")
                    f.write(json.dumps({"prop_id": p["prop_id"], "t": pt["t"], "d": d, "p": pt["p"]}) + "\n")
                    n += 1
        print(f"wrote {n} implied-prob points -> {pp}")

    by_cat = collections.Counter(p["category"] for p in props)
    print("by category:", dict(by_cat))


if __name__ == "__main__":
    main()
