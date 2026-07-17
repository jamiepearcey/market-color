# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Ticker-EMITTING resolution pass (LLM proposes, security master disposes).

The 8b extractor emits names, not tickers, and the deterministic passes (SEC / OpenFIGI /
Yahoo-search) miss brands, subsidiaries, and context-dependent references (waymo->GOOGL,
'intel foundry'->INTC). A capable model (gpt-oss-120b) disambiguates these from the
headline context and emits a ticker + the issuer name.

ANTI-HALLUCINATION GATE (same doctrine as provenance gating): a proposed ticker survives
ONLY if it VERIFIES against Yahoo — the ticker must have a real price history AND its actual
Yahoo issuer name must match the issuer the model claims. So a plausible-but-wrong ticker
(right format, wrong company) is dropped. US listing / ADR preferred (foreign local lines are
signal-noise). listed=false (private / gov / index / crypto / person) is respected.

Usage: source data/tmp/groq.env && uv run eventgraph/scripts/resolve_llm.py --graph-dir /tmp/eg_live
"""
import argparse, os, re, json, time, collections, difflib, datetime as dt
from pathlib import Path
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DIRSET = {"up", "down", "widen", "tighten"}
TYPES = {"company", "bank", "security"}
US_EXCH = {"NMS", "NYQ", "NGM", "NCM", "NSM", "ASE", "PCX", "PNK", "OQB", "OQX", "NAS", "NYS"}

SYSTEM = (
    "You map company/organization names to their primary publicly-traded stock ticker. Rules:\n"
    "- Prefer the US listing or US ADR ticker. If only a foreign listing exists, give it with the "
    "Yahoo Finance suffix (e.g. 005930.KS, AAL.L, 8035.T).\n"
    "- Return listed=false for anything that is NOT a single publicly-traded equity: private "
    "companies/startups, government bodies, regulators, central banks, non-profits, sports clubs, "
    "indices, crypto tokens/protocols, people, or brands/products that are not their own listed "
    "issuer. Map a brand/subsidiary to its LISTED PARENT only when unambiguous (e.g. waymo->GOOGL, "
    "'intel foundry'->INTC).\n"
    "- Do NOT guess. If you are not confident the ticker is correct, set listed=false.\n"
    "- Always include 'issuer': the exact official name of the company the ticker belongs to.\n"
    'Return ONLY JSON: {"results":[{"i":<int>,"listed":<bool>,"ticker":<str|null>,'
    '"issuer":<str|null>,"confidence":"high|med|low"}]}'
)


def ysearch(q, sess, cache, cf):
    if q in cache: return cache[q]
    try:
        r = sess.get("https://query2.finance.yahoo.com/v1/finance/search",
                     params={"q": q, "quotesCount": 6, "newsCount": 0}, timeout=15)
        out = r.json().get("quotes", []) if r.status_code == 200 else []
    except Exception:
        out = []
    cache[q] = out; cf.write(json.dumps({"q": q, "r": out}) + "\n"); cf.flush(); time.sleep(0.2)
    return out


def yahoo_name_and_prices(ticker, sess, cache, cf):
    """(issuer_name_on_yahoo, has_usable_prices, exchange) for a ticker."""
    hits = ysearch(ticker, sess, cache, cf)
    match = next((x for x in hits if x.get("symbol", "").upper() == ticker.upper()), None)
    name = (match.get("shortname") or match.get("longname") or "") if match else ""
    exch = match.get("exchange", "") if match else ""
    p2 = int(dt.datetime(2026, 12, 31, tzinfo=dt.UTC).timestamp()); p1 = int(dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp())
    try:
        r = sess.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
                     params={"period1": p1, "period2": p2, "interval": "1d"}, timeout=20)
        cl = r.json()["chart"]["result"][0]["indicators"]["quote"][0].get("close") or []
        ok = sum(1 for c in cl if c is not None) > 150
    except Exception:
        ok = False
    time.sleep(0.15)
    return name, ok, exch


def groq(entities, model, key):
    lines = [f'{i+1}. {e["name"]}  [{e["type"]}]  — context: {e["ctx"][:90]}' for i, e in enumerate(entities)]
    body = {"model": model, "temperature": 0, "max_tokens": 2200, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": "Resolve each to a ticker:\n" + "\n".join(lines)}]}
    for attempt in range(5):
        try:
            r = httpx.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
            if r.status_code == 429:
                ra = r.headers.get("retry-after"); time.sleep(min(float(ra), 20) if ra else 3 * (attempt + 1)); continue
            if r.status_code != 200:
                time.sleep(2 * (attempt + 1)); continue
            txt = r.json()["choices"][0]["message"]["content"]
            a, b = txt.find("{"), txt.rfind("}")
            return json.loads(txt[a:b + 1]).get("results", [])
        except Exception:
            time.sleep(2 * (attempt + 1))
    return []


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_live")
    ap.add_argument("--model", default="openai/gpt-oss-120b"); ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--limit", type=int, default=300); a = ap.parse_args()
    gd = Path(a.graph_dir); lake = gd / "lake"; esym = gd / "entity_symbol.jsonl"
    key = os.environ.get("GROQ_API_KEY")
    if not key: raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")

    resolved = {j["entity_id"] for j in map(json.loads, open(esym)) if j.get("kind") == "security"}
    have = {json.loads(l)["entity_id"] for l in open(esym)}
    hl = {json.loads(l)["doc_id"]: json.loads(l).get("headline", "") for l in open(lake / "document.jsonl")}
    freq = collections.Counter(); ctx = {}
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); e = j.get("effect_entity", "")
        if e.split("__")[-1] not in TYPES or j.get("effect_dir") not in DIRSET or e in resolved: continue
        freq[e] += 1
        if e not in ctx: ctx[e] = hl.get(j.get("doc_id"), "")
    targets = sorted(freq, key=lambda e: -freq[e])[:a.limit]
    ents = [{"eid": e, "name": e.split("__")[0].replace("_", " "), "type": e.split("__")[-1], "ctx": ctx.get(e, "")} for e in targets]
    print(f"{len(freq)} unresolved company/bank entities; emitting tickers for top {len(ents)} via {a.model}")

    proposals = []
    for i in range(0, len(ents), a.batch):
        batch = ents[i:i + a.batch]
        res = groq(batch, a.model, key)
        by_i = {r.get("i"): r for r in res if isinstance(r, dict)}
        for k, e in enumerate(batch):
            r = by_i.get(k + 1)
            if r and r.get("listed") and r.get("ticker"):
                proposals.append((e, r["ticker"].strip(), (r.get("issuer") or "").strip(), r.get("confidence", "low")))
        print(f"  batch {i//a.batch+1}: {len([1 for e in batch if by_i.get(0)])}  (cumulative proposals {len(proposals)})")

    # VERIFY against Yahoo
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})
    cache_path = gd / "yahoo_search_cache.jsonl"; cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            j = json.loads(l); cache[j["q"]] = j["r"]
    cf = open(cache_path, "a")
    accepted, rejected = [], []
    for e, ticker, issuer, conf in proposals:
        yname, ok, exch = yahoo_name_and_prices(ticker, sess, cache, cf)
        sim_iss = difflib.SequenceMatcher(None, issuer.lower(), yname.lower()).ratio() if yname else 0
        sim_ent = difflib.SequenceMatcher(None, e["name"].lower(), yname.lower()).ratio() if yname else 0
        good = ok and yname and (sim_iss >= 0.5 or sim_ent >= 0.5) and conf in ("high", "med")
        rec = {"entity_id": e["eid"], "canonical_name": e["name"], "type": e["type"], "symbol": ticker,
               "source": "llm_verified", "kind": "security", "edge_freq": freq[e["eid"]],
               "issuer_claim": issuer, "yahoo_name": yname, "sim": round(max(sim_iss, sim_ent), 2), "conf": conf}
        (accepted if good else rejected).append(rec)
        flag = "US" if (exch in US_EXCH or "." not in ticker) else exch
        print(f"  {'✓' if good else '✗'} {e['name']:28} -> {ticker:11} [{flag:4}] yahoo='{yname[:26]}' sim{rec['sim']} {conf}")

    new = [r for r in accepted if r["entity_id"] not in have]
    with open(esym, "a") as f:
        for r in new: f.write(json.dumps({k: r[k] for k in ("entity_id", "canonical_name", "type", "symbol", "source", "kind", "edge_freq")}) + "\n")
    us = sum(1 for r in accepted if "." not in r["symbol"])
    print(f"\nproposed {len(proposals)} | verified-accepted {len(accepted)} ({us} US-listed) | rejected {len(rejected)} | appended {len(new)}")


if __name__ == "__main__":
    main()
