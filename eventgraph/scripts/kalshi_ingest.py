# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Kalshi -> eventgraph proposition ingest (LLM-FREE by default).

Kalshi is the finance-native prediction market Polymarket is not: real economic-print
and single-name series (Fed decisions, CPI/NFP/GDP, Treasury yields/spreads, index
ranges, company events) organised under categories Financials/Economics/Commodities/
Crypto/Companies. That is the cross-sectional / macro-calendar breadth Polymarket's
16-symbol crypto-ladder feed lacks.

Same proposition schema + graph-dir as polymarket_ingest.py (source="kalshi"), MERGED
by source so both markets coexist in lake/proposition.jsonl. Reuses that module's
curated MACRO/crypto resolver + Yahoo verification (imported, not duplicated).

Resolution (deterministic default): curated MACRO/crypto map over the series title +
market question (Fed/rates/CPI/GDP/crypto/commodities/indices -> Yahoo proxies).
Single-name Companies are the OPT-IN --llm-tail's job (ticker+issuer proposed by
gpt-oss-120b, re-verified against Yahoo) -- same doctrine as Polymarket: raw question
text is too noisy for naive ticker search.

Public API, no key. Prices are the market yes-price (implied prob); Kalshi candlestick
history is a later add (noted).

Usage:
  uv run eventgraph/scripts/kalshi_ingest.py --graph-dir eventgraph/data/eg_live
  uv run eventgraph/scripts/kalshi_ingest.py --graph-dir eventgraph/data/eg_live \
      --include-settled --max-markets 4000        # + resolved history (labels)
  ... --llm-tail                                  # opt-in single-name resolution
"""
import argparse, json, re, time, os, collections, datetime as dt
from pathlib import Path
import httpx
# reuse the tested Polymarket resolver primitives (same dir -> importable under `uv run`)
from polymarket_ingest import MACRO, has_prices, yahoo_search, GROQ_URL, US_EXCH
import difflib

BASE = "https://api.elections.kalshi.com/trade-api/v2"
FIN_CATEGORIES = ["Financials", "Economics", "Commodities", "Crypto", "Companies"]
CAT_MAP = {"crypto": "crypto", "commodities": "commodity", "economics": "macro",
           "financials": "company", "companies": "company"}


def kget(sess, path, **params):
    r = sess.get(f"{BASE}{path}", params={k: v for k, v in params.items() if v is not None}, timeout=30)
    r.raise_for_status()
    return r.json()


def resolve_macro(question):
    """curated macro/crypto only (clean, high-precision); word-boundary matched."""
    ql = question.lower()
    res, seen = [], set()
    for kw in sorted(MACRO, key=len, reverse=True):
        if re.search(r"\b" + re.escape(kw) + r"\b", ql) and MACRO[kw] not in seen:
            seen.add(MACRO[kw]); res.append({"name": kw, "symbol": MACRO[kw], "kind": "macro", "via": "curated"})
    return res


# Kalshi event_title is a curated, CLEAN company name (unlike Polymarket questions):
# "Intel KPI" / "Urban Outfitters KPI" / "Sea Limited KPI". Strip the metric suffix and
# Yahoo-verify -> deterministic single-name resolution the noisy path can't do.
SUFFIX = re.compile(r"\b(KPI|stock|earnings?|revenue|IPO|guidance|report|price|index)\b.*$", re.I)


def resolve_company(name, sess, cache, cf, min_sim, verified):
    cand = SUFFIX.sub("", name).strip(" .-")
    if len(cand) < 3:
        return None
    quotes = yahoo_search(sess, cand, cache, cf)
    scored = []
    for x in quotes:
        if x.get("quoteType") != "EQUITY" or not x.get("symbol"):
            continue
        nm = x.get("shortname") or x.get("longname") or ""
        scored.append((difflib.SequenceMatcher(None, cand.lower(), nm.lower()).ratio(), x, nm))
    ok = [t for t in scored if t[0] >= min_sim]
    if not ok:
        return None
    ok.sort(key=lambda t: (0 if (t[1].get("exchange") in US_EXCH or "." not in t[1]["symbol"]) else 1, -t[0]))
    sim, top, nm = ok[0]; sym = top["symbol"]
    if "." in sym:                       # US-listed/ADR only (doctrine: foreign-local = timing noise)
        return None
    if sym not in verified:
        verified[sym] = has_prices(sess, sym)
    if not verified[sym]:
        return None
    return {"name": cand, "symbol": sym, "kind": "security", "via": "kalshi_title",
            "match_name": nm, "name_sim": round(sim, 2)}


def llm_tail(unresolved, key, model, sess, cache, cf, min_sim, verified):
    """opt-in gpt-oss-120b single-name resolution, each hit re-verified vs Yahoo."""
    SYSTEM = ('For each prediction-market title about a company/asset, name the single most-affected '
              'PUBLICLY-LISTED US ticker and its issuer, or null (macro-only, index, private, sports). '
              'Return ONLY JSON {"results":[{"i":<int>,"ticker":"AAPL|null","issuer":"Apple Inc"}]}')
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
            if tkr not in verified:
                verified[tkr] = has_prices(sess, tkr)
            if not verified[tkr]:
                continue
            quotes = yahoo_search(sess, iss, cache, cf)
            nm = next((y.get("shortname") or y.get("longname") or "" for y in quotes if y.get("symbol") == tkr), "")
            sim = difflib.SequenceMatcher(None, iss.lower(), nm.lower()).ratio()
            if sim < min_sim:
                continue
            out[pid] = {"name": iss, "symbol": tkr, "kind": "security", "via": "llm_verified",
                        "match_name": nm, "name_sim": round(sim, 2)}
        print(f"  llm-tail batch {i//30+1}: +{len(out)} verified so far")
    return out


def price(m):
    for f in ("last_price_dollars", "previous_price_dollars"):
        v = m.get(f)
        if isinstance(v, (int, float)):
            return float(v)
    yb, ya = m.get("yes_bid_dollars"), m.get("yes_ask_dollars")
    if isinstance(yb, (int, float)) and isinstance(ya, (int, float)) and (yb or ya):
        return (yb + ya) / 2
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="eventgraph/data/eg_live")
    ap.add_argument("--max-markets", type=int, default=3000, help="cap total markets pulled")
    ap.add_argument("--per-series", type=int, default=60, help="markets per series (paginate cap)")
    ap.add_argument("--min-volume", type=float, default=0.0, help="skip markets below this contract volume")
    ap.add_argument("--include-settled", action="store_true", help="also pull resolved markets (ground-truth labels)")
    ap.add_argument("--min-sim", type=float, default=0.6)
    ap.add_argument("--llm-tail", action="store_true", help="opt-in single-name resolution (needs GROQ_API_KEY)")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    a = ap.parse_args()

    gd = Path(a.graph_dir); lake = gd / "lake"; lake.mkdir(parents=True, exist_ok=True)
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})
    cache_path = gd / "yahoo_search_cache.jsonl"; cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            j = json.loads(l); cache[j["q"]] = j["r"]
    cf = open(cache_path, "a"); verified = {}

    # 1. finance series across the finance categories, ROUND-ROBIN so the market cap
    #    doesn't starve later categories (crypto/economics) by filling on Financials.
    by_cat = {}
    for cat in FIN_CATEGORIES:
        try:
            by_cat[cat] = [(s["ticker"], s.get("title") or "") for s in kget(sess, "/series", category=cat).get("series", []) if s.get("ticker")]
        except Exception as e:
            by_cat[cat] = []; print(f"  series {cat} ERR {e}")
    series = {}
    for rank in range(max((len(v) for v in by_cat.values()), default=0)):
        for cat in FIN_CATEGORIES:
            if rank < len(by_cat[cat]):
                tkr, title = by_cat[cat][rank]
                series.setdefault(tkr, {"title": title, "category": cat.lower()})
    print(f"{len(series)} finance series across {len(FIN_CATEGORIES)} categories "
          f"({', '.join(f'{c}:{len(v)}' for c, v in by_cat.items())})")

    statuses = ["open", "settled"] if a.include_settled else ["open"]
    props = []
    for i, (stkr, meta) in enumerate(series.items()):
        if len(props) >= a.max_markets:
            break
        for status in statuses:
            cursor, got = None, 0
            while got < a.per_series and len(props) < a.max_markets:
                try:
                    resp = kget(sess, "/markets", series_ticker=stkr, status=status, limit=100, cursor=cursor)
                except Exception:
                    break
                mkts = resp.get("markets", [])
                for m in mkts:
                    q = " ".join(x for x in [meta["title"], m.get("title"), m.get("subtitle") or m.get("yes_sub_title")] if x)
                    try:
                        vol = float(m.get("volume_fp") or 0)
                    except Exception:
                        vol = 0.0
                    if vol < a.min_volume or not m.get("ticker"):
                        continue
                    py = price(m)
                    res = resolve_macro(q)
                    # single-name: Yahoo-verify the clean event_title (deterministic, no LLM)
                    if not res and meta["category"] in ("financials", "companies"):
                        c = resolve_company(meta["title"], sess, cache, cf, a.min_sim, verified)
                        if c:
                            res = [c]
                    settled = (m.get("status") == "settled") or bool(m.get("result"))
                    result = (m.get("result") or "").lower() or None
                    props.append({
                        "prop_id": "kx_" + m["ticker"], "source": "kalshi", "contract_ref": m["ticker"],
                        "series": stkr, "event_title": meta["title"], "question": q,
                        "resolution_date": (m.get("close_time") or m.get("expiration_time") or "")[:10],
                        "start_date": (m.get("open_time") or "")[:10],
                        "resolution_criteria": (m.get("rules_primary") or "")[:400],
                        "tags": [meta["category"]], "category": CAT_MAP.get(meta["category"], "other"),
                        "prob_yes": py, "probability_annotation": py,
                        "outcomes": ["yes", "no"], "volume": vol,
                        "liquidity": float(m.get("liquidity_dollars") or 0),
                        "open_interest": m.get("open_interest_fp"),
                        "closed": settled,
                        "resolved_outcome": result if result in ("yes", "no") else None,
                        "strike_type": m.get("strike_type"), "floor_strike": m.get("floor_strike"),
                        "clob_token_yes": None,
                        "symbols": [r["symbol"] for r in res], "resolution": res,
                    })
                    got += 1
                cursor = resp.get("cursor")
                if not cursor or not mkts:
                    break
        if (i + 1) % 100 == 0:
            print(f"  scanned {i+1}/{len(series)} series -> {len(props)} markets")
    print(f"built {len(props)} kalshi propositions; {sum(1 for p in props if p['symbols'])} resolve to >=1 ticker")

    # 2. opt-in LLM single-name tail
    if a.llm_tail:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            print("  --llm-tail set but GROQ_API_KEY missing -> skipping")
        else:
            # dedupe by question so we pay once per distinct market title
            need, seenq = [], set()
            for p in props:
                if p["symbols"] or p["question"] in seenq:
                    continue
                seenq.add(p["question"]); need.append((p["question"], p["question"]))
            print(f"  llm-tail: {len(need)} distinct unresolved titles")
            got = llm_tail(need, key, a.model, sess, cache, cf, a.min_sim, verified)
            for p in props:
                r = got.get(p["question"])
                if r and not p["symbols"]:
                    p["symbols"] = [r["symbol"]]; p["resolution"] = [r]
            print(f"  llm-tail resolved {sum(1 for p in props if p['symbols'] and p['resolution'] and p['resolution'][0]['via']=='llm_verified')} markets")

    # 3. write proposition table, MERGING by source (keep polymarket etc.)
    ppath = lake / "proposition.jsonl"
    kept = []
    if ppath.exists():
        kept = [json.loads(l) for l in open(ppath) if l.strip() and json.loads(l).get("source") != "kalshi"]
    with open(ppath, "w") as f:
        for p in kept + props:
            f.write(json.dumps(p) + "\n")
    resolved = sum(1 for p in props if p["symbols"])
    print(f"wrote {len(props)} kalshi props ({resolved} resolved, {resolved/max(len(props),1):.0%}); "
          f"merged with {len(kept)} existing non-kalshi -> {ppath}")
    print("by category:", dict(collections.Counter(p["category"] for p in props)))


if __name__ == "__main__":
    main()
