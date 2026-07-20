# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
eventgraph :: eg-rich-v2 extraction quality assessment.

Joins <out>/extractions.jsonl with the source feed (for quote grounding) and scores
the v2 contract: parse rate, enum validity, REFERENTIAL INTEGRITY (edge/relation
endpoints -> entities.name), QUOTE GROUNDING (verbatim quote locates in cited
chunk), suggested-vs-verified identity, temporal-hint capture, relations, and the
direction-audit guardrail. Emits the same GateStats view the Rust normalize applies.

Usage: uv run assess_v2.py --feed /tmp/eg100_feed.jsonl --ex /tmp/eg100/extractions.jsonl
"""
import argparse, json, re, sys
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eg_prompt_v2 import chunk_text, PROMPT_CHUNKS

ENUM = {
 "doc_type": set("news analysis opinion preview recap press_release interview data_release non_financial".split()),
 "entity_type": set("company bank central_bank sovereign supranational regulator government_agency commodity currency equity_index rate_or_bond sector person exchange economic_indicator asset_class market other".split()),
 "role": set("subject driver counterparty mentioned".split()),
 "listing_status_hint": set("listed private government index person unknown".split()),
 "mechanism": set("monetary_policy rate_decision supply_shock demand_change earnings guidance mergers_acquisitions default rating_action regulation geopolitics fund_flows data_surprise contagion other".split()),
 "effect_direction": set("up down widen tighten volatile unchanged".split()),
 "modality": set("happened ongoing forecast hypothetical denied".split()),
 "attribution_source": set("reporter named_analyst company official market_consensus market_implied".split()),
 "lag": set("immediate days longer".split()),
 "confidence": set("strong tentative".split()),
 "factor_type": set("rates credit_spread oil commodity usd fx equity_beta inflation specific other".split()),
 "asset_class": set("equity credit rates fx commodity vol".split()),
 "relation": set("acquires merges_with parent_of brand_of stake_in supplies customer_of competes_with officer_of regulates other".split()),
 "status_hint": set("rumored proposed agreed completed blocked".split()),
}
# NB: match against VALUES only (not keys) -- a key like `effect_verbatim` must not trip this.
ECHO_LITERALS = ("entities.name", "a different entities", "up|down|widen", "best-guess",
                 "verbatim contiguous", "ticker/iso", "canonical (no tickers", "reference period e.g")

def _values(o):
    if isinstance(o, dict): return " ".join(_values(v) for v in o.values())
    if isinstance(o, list): return " ".join(_values(v) for v in o)
    return o if isinstance(o, str) else ""

def norm(s): return re.sub(r"\s+", " ", (s or "").lower()).replace("’", "'").replace("–", "-").strip()
def plausible_ticker(s):
    s = (s or "").strip()
    return bool(1 <= len(s) <= 7 and re.fullmatch(r"[A-Za-z0-9.\-]+", s) and re.search(r"[A-Za-z0-9]", s))

UP = ["jump","surg","soar","rall","rose","rise","gain","climb","advanc","higher","spike","boost"]
DOWN = ["fell","fall","slid","slip","tumbl","plung","drop","sank","sink","declin","lower","loss","lost","retreat"]
def lexicon_dir(v):
    v = (v or "").lower()
    if any(k in v for k in UP): return "up"
    if any(k in v for k in DOWN): return "down"
    if "widen" in v: return "widen"
    if "tighten" in v or "narrow" in v: return "tighten"
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feed", required=True)
    ap.add_argument("--ex", required=True)
    args = ap.parse_args()

    feed = {}
    for l in open(args.feed):
        d = json.loads(l); feed[d["doc_id"]] = d
    rows = [json.loads(l) for l in open(args.ex)]
    N = len(rows)

    parsed = 0; echo = 0
    enum_ok = Counter(); enum_tot = Counter()
    fills = Counter(); counts = Counter()
    ref_ok = ref_tot = 0
    edges = edges_diff = 0
    q_tot = q_grounded = 0; ev_ok = ev_tot = 0
    sugg = sugg_plausible = 0
    dir_checked = dir_conflict = 0
    temporal_text = temporal_tot = 0
    ent_evidence = ent_tot = 0
    samples = []

    for r in rows:
        ex = r.get("ex")
        if not isinstance(ex, dict):
            continue
        parsed += 1
        blob = _values(ex).lower()
        if any(lit in blob for lit in ECHO_LITERALS): echo += 1
        doc = feed.get(r["doc_id"], {})
        chunks = [norm(c) for c in chunk_text(doc.get("headline", ""), doc.get("article", ""))[:PROMPT_CHUNKS]]
        ents = ex.get("entities") or []
        names = {norm(e.get("name")) for e in ents if isinstance(e, dict)}

        if "doc_type" in ex:
            enum_tot["doc_type"] += 1; enum_ok["doc_type"] += ex.get("doc_type") in ENUM["doc_type"]
        for key in ["entities","relations","events","causal_edges","sensitivities","sentiments","propositions","figures"]:
            v = ex.get(key)
            if isinstance(v, list):
                counts[key] += len(v)
                if v: fills[key] += 1

        for e in ents:
            if not isinstance(e, dict): continue
            ent_tot += 1
            enum_tot["entity_type"] += 1; enum_ok["entity_type"] += e.get("type") in ENUM["entity_type"]
            enum_tot["role"] += 1; enum_ok["role"] += e.get("role") in ENUM["role"]
            if e.get("listing_status_hint") is not None:
                enum_tot["listing_status_hint"] += 1
                enum_ok["listing_status_hint"] += e.get("listing_status_hint") in ENUM["listing_status_hint"]
            st = e.get("suggested_ticker")
            if st:
                sugg += 1; sugg_plausible += plausible_ticker(st)
            if isinstance(e.get("evidence_chunk"), int): ent_evidence += 1

        def ground(item, ck_field="evidence_chunk"):
            nonlocal q_tot, q_grounded, ev_tot, ev_ok
            q = item.get("quote")
            if not q: return
            q_tot += 1
            ec = item.get(ck_field)
            valid = isinstance(ec, int) and 0 <= ec < len(chunks)
            ev_tot += 1; ev_ok += valid
            hay = chunks[ec] if valid else " ".join(chunks)
            q_grounded += norm(q) in hay

        def ref(item, keys):
            nonlocal ref_ok, ref_tot
            for k in keys:
                if item.get(k):
                    ref_tot += 1; ref_ok += norm(item[k]) in names

        for ce in (ex.get("causal_edges") or []):
            if not isinstance(ce, dict): continue
            edges += 1
            for f in ["mechanism","effect_direction","modality","attribution_source","lag","confidence"]:
                if f in ce: enum_tot[f] += 1; enum_ok[f] += ce.get(f) in ENUM[f]
            c, e = norm(ce.get("cause")), norm(ce.get("effect"))
            ref_tot += 2; ref_ok += (c in names) + (e in names)
            edges_diff += bool(c != e and c and e)
            ground(ce)
            ld = lexicon_dir(ce.get("effect_verbatim"))
            if ld and ce.get("effect_direction"):
                dir_checked += 1; dir_conflict += (ld != ce.get("effect_direction"))
        for rel in (ex.get("relations") or []):
            if not isinstance(rel, dict): continue
            if "relation" in rel: enum_tot["relation"] += 1; enum_ok["relation"] += rel.get("relation") in ENUM["relation"]
            if rel.get("status_hint") is not None: enum_tot["status_hint"] += 1; enum_ok["status_hint"] += rel.get("status_hint") in ENUM["status_hint"]
            ref(rel, ["source","target"]); ground(rel)
        for se in (ex.get("sensitivities") or []):
            if not isinstance(se, dict): continue
            for f in ["factor_type","asset_class"]:
                if f in se: enum_tot[f] += 1; enum_ok[f] += se.get(f) in ENUM[f]
            ref(se, ["asset"]); ground(se)
        for st in (ex.get("sentiments") or []):
            if isinstance(st, dict): ref(st, ["target"]); ground(st)
        for ev in (ex.get("events") or []):
            if isinstance(ev, dict):
                ground(ev)
                temporal_tot += 1; temporal_text += bool(ev.get("event_time_text") or ev.get("period_text"))
        for pr in (ex.get("propositions") or []):
            if isinstance(pr, dict):
                ground(pr)
                temporal_tot += 1; temporal_text += bool(pr.get("resolution_date_text"))
        for fg in (ex.get("figures") or []):
            if isinstance(fg, dict): ground(fg); ref(fg, ["entity"])

        if len(samples) < 2 and (ex.get("causal_edges") or ex.get("relations")):
            samples.append(r)

    def pct(a, b): return f"{100*a/b:5.1f}% ({a}/{b})" if b else "   n/a"
    print(f"\n{'='*64}\n  eg-rich-v2 quality  (N={N})\n{'='*64}")
    print(f"JSON parse success:        {pct(parsed,N)}")
    print(f"template-echo (schema copy): {pct(echo,parsed)}   <-- want ~0%")
    print("\n-- ENUM validity --")
    for k in ["doc_type","entity_type","role","listing_status_hint","mechanism","effect_direction","modality","attribution_source","lag","confidence","factor_type","asset_class","relation","status_hint"]:
        print(f"  {k:20s} {pct(enum_ok[k],enum_tot[k])}")
    print("\n-- Referential integrity & grounding --")
    print(f"  edge/rel/ref endpoints -> entities.name: {pct(ref_ok,ref_tot)}")
    print(f"  causal cause != effect:                  {pct(edges_diff,edges)}")
    print(f"  evidence_chunk in range:                 {pct(ev_ok,ev_tot)}")
    print(f"  quote verbatim in cited chunk:           {pct(q_grounded,q_tot)}")
    print("\n-- v2 identity / temporal / guardrails --")
    print(f"  entities w/ evidence_chunk:              {pct(ent_evidence,ent_tot)}")
    print(f"  suggested_ticker plausible format:       {pct(sugg_plausible,sugg)}")
    print(f"  temporal *_text captured (events+props): {pct(temporal_text,temporal_tot)}")
    print(f"  direction-audit CONFLICTS (want low):    {pct(dir_conflict,dir_checked)}")
    print("\n-- Fill rates (docs with >=1) & totals --")
    for k in ["entities","relations","events","causal_edges","sensitivities","sentiments","propositions","figures"]:
        print(f"  {k:16s} docs {pct(fills[k],parsed)}   total {counts[k]}")

    print("\n-- sample extraction --")
    for r in samples[:1]:
        ex = r["ex"]
        print(f"  [{r['doc_id']}] {r.get('headline')}")
        print("  entities:", [(e.get('name'), e.get('type'), e.get('suggested_ticker')) for e in (ex.get('entities') or [])][:6])
        for ce in (ex.get('causal_edges') or [])[:2]:
            print(f"    edge: {ce.get('cause')} --{ce.get('mechanism')}/{ce.get('effect_direction')}--> {ce.get('effect')}"
                  f" | verbatim={ce.get('effect_verbatim')!r}")
        for rel in (ex.get('relations') or [])[:2]:
            print(f"    rel:  {rel.get('source')} --{rel.get('relation')}({rel.get('status_hint')})--> {rel.get('target')}")

if __name__ == "__main__":
    main()
