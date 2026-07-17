# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Supplementary resolution pass: Yahoo search over the UNRESOLVED company/bank/security
effect entities (the 8b model emits names, not tickers, and the SEC name-matcher misses
brands/subsidiaries/foreign majors — google->GOOG, samsung->005930.KS, anglo american->
AAL.L, canadian pacific kansas city->CP).

Verifiable, no LLM: Yahoo's search is a real security master. Gate: take the top
quoteType==EQUITY hit and require it to have a usable price history (tradeable). Genuinely
non-tradeable entities (openai, waymo, nexperia, sports clubs) return no EQUITY hit and
are correctly left unresolved. Results are APPENDED to entity_symbol.jsonl (source
"yahoo_exact", kind "security") and cached for reruns.

Usage: uv run eventgraph/scripts/resolve_yahoo.py --graph-dir /tmp/eg_live [--limit 300]
"""
import argparse, json, time, collections, difflib, datetime as dt
from pathlib import Path
import httpx

DIRSET = {"up", "down", "widen", "tighten"}
TRADEABLE_TYPES = {"company", "bank", "security"}


def ysearch(q, sess):
    try:
        r = sess.get("https://query2.finance.yahoo.com/v1/finance/search",
                     params={"q": q, "quotesCount": 6, "newsCount": 0}, timeout=15)
        return r.json().get("quotes", []) if r.status_code == 200 else []
    except Exception:
        return []


def has_prices(sym, sess):
    """verify a usable daily history exists (tradeable)."""
    p2 = int(dt.datetime(2026, 12, 31, tzinfo=dt.UTC).timestamp()); p1 = int(dt.datetime(2024, 1, 1, tzinfo=dt.UTC).timestamp())
    try:
        r = sess.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}",
                     params={"period1": p1, "period2": p2, "interval": "1d"}, timeout=20)
        res = r.json()["chart"]["result"][0]
        cl = res["indicators"]["quote"][0].get("close") or []
        return sum(1 for c in cl if c is not None) > 150
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="/tmp/eg_live")
    ap.add_argument("--limit", type=int, default=300); ap.add_argument("--min-freq", type=int, default=1)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"
    esym = gd / "entity_symbol.jsonl"

    resolved_sec = set()
    existing = []
    for l in open(esym):
        j = json.loads(l); existing.append(j)
        if j.get("kind") == "security": resolved_sec.add(j["entity_id"])

    # unresolved tradeable-typed effect entities by edge frequency
    freq = collections.Counter()
    names = {}
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); e = j.get("effect_entity", "")
        if not e or j.get("effect_dir") not in DIRSET: continue
        if e.split("__")[-1] not in TRADEABLE_TYPES or e in resolved_sec: continue
        freq[e] += 1; names[e] = e.split("__")[0].replace("_", " ")
    targets = [(e, freq[e]) for e in freq if freq[e] >= a.min_freq]
    targets.sort(key=lambda x: -x[1]); targets = targets[:a.limit]
    print(f"{len(freq)} unresolved tradeable-typed entities; resolving top {len(targets)} by edge freq ...")

    cache_path = gd / "yahoo_search_cache.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path):
            j = json.loads(l); cache[j["q"]] = j["r"]
        print(f"  loaded {len(cache)} cached searches")

    sess = httpx.Client(headers={"User-Agent": "Mozilla/5.0"})
    out = []; hit = 0
    cf = open(cache_path, "a")
    for e, fr in targets:
        q = names[e]
        if q in cache:
            quotes = cache[q]
        else:
            quotes = ysearch(q, sess); time.sleep(0.25)
            cf.write(json.dumps({"q": q, "r": quotes}) + "\n"); cf.flush()
        eq = [x for x in quotes if x.get("quoteType") == "EQUITY" and x.get("symbol")]
        if not eq:
            continue
        # PREFER a US listing / ADR (clean US-session bars aligned with the news timing);
        # foreign local listings (.KS/.NS/.T/...) diluted the signal to noise in testing.
        US_EXCH = {"NMS", "NYQ", "NGM", "NCM", "NSM", "ASE", "PCX", "PNK", "OQB", "OQX", "NAS", "NYS"}
        eq.sort(key=lambda x: 0 if (x.get("exchange") in US_EXCH or "." not in x.get("symbol", "")) else 1)
        top = eq[0]; sym = top["symbol"]; nm = top.get("shortname", top.get("longname", ""))
        if not has_prices(sym, sess):
            continue
        sim = difflib.SequenceMatcher(None, q.lower(), nm.lower()).ratio()
        out.append({"entity_id": e, "canonical_name": q, "type": e.split("__")[-1],
                    "symbol": sym, "source": "yahoo_exact", "kind": "security", "edge_freq": fr,
                    "match_name": nm, "name_sim": round(sim, 2)})
        hit += 1
        print(f"  {fr:3}  {q:34} -> {sym:12} ({nm[:32]}) sim {sim:.2f}")

    # append (idempotent: only new entity_ids)
    have = {j["entity_id"] for j in existing}
    new = [o for o in out if o["entity_id"] not in have]
    with open(esym, "a") as f:
        for o in new: f.write(json.dumps(o) + "\n")
    print(f"\nresolved {hit}/{len(targets)} via Yahoo search; appended {len(new)} new securities to {esym}")


if __name__ == "__main__":
    main()
