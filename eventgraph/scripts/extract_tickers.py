# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Ticker-emitting re-extraction — the source-level fix.

The 8b rich extraction emits names (no tickers) and makes direction errors (a $53B bid
for PayPal tagged 'down'). This re-extracts the PRICE-RELEVANT causal edges with a strong,
ticker-aware model (gpt-oss-120b) that, seeing the FULL article, emits for each affected
listed company: its US ticker (context-disambiguated), the correct STOCK-PRICE direction
(a takeover bid sends the TARGET up), mechanism, cause, and a verbatim quote.

TWO anti-hallucination gates on build: (1) PROVENANCE — the quote must locate in the
article (else drop); (2) SECURITY MASTER — the ticker must verify on Yahoo (real prices).

Output: a self-contained graph-dir (default /tmp/eg_live_tkr) that every downstream script
(concentrated_coverage, attribution, signal_value) reads unchanged.

  source data/tmp/groq.env
  uv run eventgraph/scripts/extract_tickers.py --feed /tmp/eg_live_feed.jsonl --src /tmp/eg_live
  uv run eventgraph/scripts/extract_tickers.py --build-only --src /tmp/eg_live   # re-verify/assemble
"""
import os, re, json, argparse, asyncio, time, collections, difflib, datetime as dt
from pathlib import Path
import httpx

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MECHS = ["earnings", "guidance", "demand_change", "supply_shock", "mergers_acquisitions", "rating_action",
         "regulation", "competition", "monetary_policy", "rate_decision", "geopolitics", "default",
         "contagion", "data_surprise", "leadership_change", "legal", "product", "other"]
US_EXCH = {"NMS", "NYQ", "NGM", "NCM", "NSM", "ASE", "PCX", "PNK", "OQB", "OQX", "NAS", "NYS"}

SYSTEM = (
    "You extract PRICE-RELEVANT causal claims from a financial news article and map each affected "
    "company to its stock ticker. For each claim where a specific PUBLICLY-LISTED company's stock is "
    "affected, output an edge with:\n"
    "- ticker: the company's PRIMARY US-listed ticker or US ADR (AAPL, GOOGL, TD). If it lists only "
    "abroad, give the Yahoo-suffixed local ticker (005930.KS, AAL.L). If the affected party is NOT a "
    "single publicly-listed company (private/startup, government, index, sector, a person, crypto, a "
    "product), SKIP it. Never invent a ticker — if unsure, SKIP.\n"
    "- direction: the effect on THAT COMPANY'S STOCK PRICE, 'up' or 'down'. Reason about who benefits: "
    "a takeover BID sends the TARGET up; a lawsuit/recall/guidance-cut/probe sends the subject down; a "
    "major contract win sends the winner up; a downgrade sends it down.\n"
    f"- mechanism: one of {MECHS}.\n"
    "- cause: the short entity or event that caused the move.\n"
    "- quote: a VERBATIM sentence copied EXACTLY from the article that supports the claim.\n"
    "- confidence: high|med|low.\n"
    "Only include claims genuinely supported by the article text. "
    'Return ONLY JSON: {"edges":[{"ticker":..,"direction":"up|down","mechanism":..,"cause":..,"quote":..,"confidence":..}]}'
)


def salvage(txt):
    a, b = txt.find("{"), txt.rfind("}")
    if a < 0 or b <= a: return {"edges": []}
    try: return json.loads(txt[a:b + 1])
    except Exception:
        try: return json.loads(re.sub(r",\s*([}\]])", r"\1", txt[a:b + 1]))
        except Exception: return {"edges": []}


class TokenBucket:
    def __init__(self, per_min):
        self.cap = per_min; self.tokens = float(per_min); self.rate = per_min / 60.0
        self.last = time.monotonic(); self.lock = asyncio.Lock()
    async def acquire(self, n):
        n = min(n, self.cap)
        async with self.lock:
            while True:
                now = time.monotonic(); self.tokens = min(self.cap, self.tokens + (now - self.last) * self.rate); self.last = now
                if self.tokens >= n: self.tokens -= n; return
                await asyncio.sleep((n - self.tokens) / self.rate)


async def extract_one(client, sem, bucket, model, key, doc, maxtok):
    art = (doc.get("article") or "")[:2000]
    user = f"HEADLINE: {doc.get('headline','')}\n\nARTICLE:\n{art}"
    body = {"model": model, "temperature": 0, "max_tokens": maxtok, "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
    await bucket.acquire((len(SYSTEM) + len(user)) // 4 + maxtok)
    async with sem:
        for attempt in range(6):
            try:
                r = await client.post(GROQ_URL, json=body, headers={"Authorization": f"Bearer {key}"})
                if r.status_code == 429:
                    ra = r.headers.get("retry-after"); await asyncio.sleep(min(float(ra), 20) if ra else 2 * (attempt + 1)); continue
                if r.status_code >= 500: await asyncio.sleep(1.5 * (attempt + 1)); continue
                if r.status_code != 200: return doc["doc_id"], []
                return doc["doc_id"], salvage(r.json()["choices"][0]["message"]["content"]).get("edges", [])
            except Exception:
                await asyncio.sleep(1.5 * (attempt + 1))
        return doc["doc_id"], []


async def run_extract(feed, outdir, model, conc, limit, key):
    docs = [json.loads(l) for l in open(feed)][:limit]
    ckpt = outdir / "raw_edges.jsonl"; done = set()
    if ckpt.exists():
        for l in open(ckpt):
            try: done.add(json.loads(l)["doc_id"])
            except Exception: pass
    todo = [d for d in docs if d["doc_id"] not in done]
    print(f"extract: {len(docs)} docs, {len(done)} done, {len(todo)} to do  [{model} conc={conc}]")
    bucket = TokenBucket(240_000); sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=90, limits=httpx.Limits(max_connections=conc + 4)) as client:
        cf = open(ckpt, "a")
        n = 0
        tasks = [asyncio.create_task(extract_one(client, sem, bucket, model, key, d, 900)) for d in todo]
        for fut in asyncio.as_completed(tasks):
            did, edges = await fut
            cf.write(json.dumps({"doc_id": did, "edges": edges}) + "\n"); cf.flush()
            n += 1
            if n % 200 == 0: print(f"  {n}/{len(todo)} ...")
    print(f"extract done -> {ckpt}")


def norm_txt(s):
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def yverify(ticker, sess, cache, cf):
    if ticker in cache: return cache[ticker]
    name, exch, ok = "", "", False
    try:
        r = sess.get("https://query2.finance.yahoo.com/v1/finance/search", params={"q": ticker, "quotesCount": 6, "newsCount": 0}, timeout=15)
        m = next((x for x in r.json().get("quotes", []) if x.get("symbol", "").upper() == ticker.upper()), None)
        if m: name = m.get("shortname") or m.get("longname") or ""; exch = m.get("exchange", "")
    except Exception: pass
    p2 = int(dt.datetime(2026, 12, 31, tzinfo=dt.UTC).timestamp()); p1 = int(dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp())
    try:
        r = sess.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}", params={"period1": p1, "period2": p2, "interval": "1d"}, timeout=20)
        cl = r.json()["chart"]["result"][0]["indicators"]["quote"][0].get("close") or []
        ok = sum(1 for c in cl if c is not None) > 150
    except Exception: pass
    res = {"name": name, "exch": exch, "ok": ok}; cache[ticker] = res
    cf.write(json.dumps({"t": ticker, **res}) + "\n"); cf.flush(); time.sleep(0.2)
    return res


def build(outdir, src, verify=True):
    raw = outdir / "raw_edges.jsonl"
    # article text for provenance (from the feed) + document.jsonl for downstream
    src = Path(src)
    arts = {}
    feedpath = Path("/tmp/eg_live_feed.jsonl")
    for l in open(feedpath):
        d = json.loads(l); arts[d["doc_id"]] = norm_txt(d.get("article", "") + " " + d.get("headline", ""))
    lake = outdir / "lake"; lake.mkdir(parents=True, exist_ok=True)
    docsrc = src / "lake" / "document.jsonl"
    (lake / "document.jsonl").write_text(docsrc.read_text())      # reuse doc metadata

    DIR = {"up": "up", "down": "down"}
    vc_path = outdir / "yahoo_verify_cache.jsonl"; cache = {}
    if vc_path.exists():
        for l in open(vc_path):
            j = json.loads(l); cache[j["t"]] = {"name": j["name"], "exch": j["exch"], "ok": j["ok"]}
    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"}); cf = open(vc_path, "a")

    edges_out = open(lake / "causal_event_edge.jsonl", "w")
    seen_ticker = {}; kept = dropped_prov = dropped_ver = raw_n = 0
    for l in open(raw):
        j = json.loads(l); did = j["doc_id"]; art = arts.get(did, "")
        for e in j.get("edges", []):
            raw_n += 1
            t = (e.get("ticker") or "").strip().upper(); d = (e.get("direction") or "").lower()
            q = e.get("quote") or ""
            if not t or d not in DIR or t in ("NONE", "N/A"): continue
            # GATE 1: provenance — quote must locate in the article
            if len(norm_txt(q)) < 12 or norm_txt(q)[:80] not in art:
                dropped_prov += 1; continue
            # GATE 2: security master — ticker verifies on Yahoo
            if t not in seen_ticker:
                v = yverify(t, sess, cache, cf) if verify else {"name": t, "exch": "", "ok": True}
                seen_ticker[t] = v
            v = seen_ticker[t]
            if not v["ok"] or not v["name"]:
                dropped_ver += 1; continue
            eid = f"{t}__security"
            edges_out.write(json.dumps({
                "effect_entity": eid, "effect_dir": d, "mechanism": e.get("mechanism") or "other",
                "cause_entity": (e.get("cause") or "").strip().lower().replace(" ", "_") + "__cause",
                "quote": q, "confidence": e.get("confidence") or "", "modality": "happened", "doc_id": did,
            }) + "\n")
            kept += 1
    edges_out.close()

    # entity_symbol.jsonl from verified tickers
    with open(outdir / "entity_symbol.jsonl", "w") as f:
        for t, v in seen_ticker.items():
            if v["ok"] and v["name"]:
                f.write(json.dumps({"entity_id": f"{t}__security", "canonical_name": v["name"], "type": "company",
                                    "symbol": t, "source": "llm_extract", "kind": "security", "edge_freq": 0}) + "\n")
    # reuse the price cache
    pc = outdir / "prices"
    if not pc.exists():
        try: pc.symlink_to((src / "prices").resolve())
        except Exception: pc.mkdir(exist_ok=True)
    us_syms = sum(1 for t, v in seen_ticker.items() if v["ok"] and v["name"] and "." not in t)
    print(f"build: {raw_n} raw edges -> kept {kept} | dropped provenance {dropped_prov}, unverified-ticker {dropped_ver}")
    print(f"  {len([1 for v in seen_ticker.values() if v['ok'] and v['name']])} unique verified tickers ({us_syms} US-listed)")
    print(f"  wrote {lake/'causal_event_edge.jsonl'} + entity_symbol.jsonl -> graph-dir {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", default="/tmp/eg_live_feed.jsonl"); ap.add_argument("--src", default="/tmp/eg_live")
    ap.add_argument("--out", default="/tmp/eg_live_tkr"); ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--concurrency", type=int, default=8); ap.add_argument("--limit", type=int, default=100000)
    ap.add_argument("--build-only", action="store_true"); a = ap.parse_args()
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    if not a.build_only:
        key = os.environ.get("GROQ_API_KEY")
        if not key: raise SystemExit("set GROQ_API_KEY (source data/tmp/groq.env)")
        asyncio.run(run_extract(Path(a.feed), outdir, a.model, a.concurrency, a.limit, key))
    build(outdir, a.src)


if __name__ == "__main__":
    main()
