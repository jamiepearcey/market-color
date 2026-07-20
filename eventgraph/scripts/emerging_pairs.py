# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx", "numpy"]
# ///
"""
EMERGING CROSS-SECTOR PAIRS — the non-obvious live demo. Same-sector elevated pairs
(INTC~TXN) are trivia; the VALIDATED, non-obvious content of the graph is:
  (1) CROSS-SECTOR pairs bound by a named driver (sector membership cannot explain them;
      cause_sim survived the SIC control at 3x the sector effect), and
  (2) EMERGING correlation — rising NOW vs its own prior baseline (the leading-correlation
      result: news-linked pairs' correlations rise/persist while unlinked mean-revert).
Link sources = the validated 10x expansion: shared CAUSE, shared SENSITIVITY factor,
direct RELATION. Output ranks cross-sector risers with the driver + provenance headline.

Usage: uv run scripts/emerging_pairs.py --graph-dir ../data/eg_runs/eg_live2
"""
import argparse, json, collections, sys, datetime as dt
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import persistence_signal as ps

SICN = {"60":"banks","61":"credit","62":"brokers","63":"insurance","67":"holding","28":"pharma/chem",
        "29":"refining","13":"oil&gas","10":"mining","20":"food","35":"machinery","36":"electronics",
        "37":"autos/aero","38":"instruments","48":"telecom","49":"utilities","73":"software/svcs",
        "45":"airlines","40":"rail","50":"wholesale","53":"retail","58":"restaurants","80":"health",
        "27":"media/print","78":"film","70":"hotels","33":"metals","16":"construction","12":"coal","44":"shipping"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--graph-dir", default="../data/eg_runs/eg_live2")
    ap.add_argument("--recent", type=int, default=21); ap.add_argument("--min-corr", type=float, default=0.15)
    ap.add_argument("--max-doc-names", type=int, default=6)
    a = ap.parse_args(); gd = Path(a.graph_dir); lake = gd / "lake"; cache = gd / "prices"
    sec = json.loads((gd / "sector.json").read_text()); sec = {k: v for k, v in sec.items() if v}

    sym = {}
    for l in open(gd / "entity_symbol.jsonl"):
        j = json.loads(l)
        if j["kind"] == "security" and "." not in j["symbol"] and not j["symbol"].startswith("^") and j["symbol"] not in ps.SKIP:
            sym[j["entity_id"]] = j["symbol"]
    docdate = {json.loads(l)["doc_id"]: (json.loads(l).get("published_at") or "")[:10] for l in open(lake / "document.jsonl")}
    dochead = {json.loads(l)["doc_id"]: json.loads(l).get("headline") or "" for l in open(lake / "document.jsonl")}

    # ROUNDUP FILTER: docs whose facts span many resolved names (analyst-call listicles,
    # market roundups) manufacture spurious shared-driver links (every name inherits every
    # sensitivity). Count distinct symbols per doc across all fact tables; drop link
    # formation from docs above --max-doc-names (focused single-story docs pass).
    doc_syms = collections.defaultdict(set)
    for fname, flds in [("causal_event_edge.jsonl", ("effect_entity",)),
                        ("sensitivity_edge.jsonl", ("asset_entity",)),
                        ("relation_edge.jsonl", ("source_entity", "target_entity"))]:
        for l in open(lake / fname):
            j = json.loads(l)
            for f in flds:
                e = j.get(f)
                if e in sym: doc_syms[j.get("doc_id")].add(sym[e])
    roundup = {d for d, ss in doc_syms.items() if len(ss) > a.max_doc_names}
    print(f"roundup filter: {len(roundup)}/{len(doc_syms)} docs excluded (> {a.max_doc_names} names)")

    # three link sources over the live window
    links = collections.defaultdict(lambda: {"drivers": set(), "kind": set(), "docs": [], "signs": []})
    def add(pair, driver, kind, doc, signpair=None):
        k = tuple(sorted(pair))
        if k[0] == k[1]: return
        links[k]["drivers"].add(driver); links[k]["kind"].add(kind); links[k]["docs"].append(doc)
        if signpair: links[k]["signs"].append(signpair)
    by_cause = collections.defaultdict(set); by_fac = collections.defaultdict(dict)
    cause_doc = {}; fac_doc = {}
    for l in open(lake / "causal_event_edge.jsonl"):
        j = json.loads(l); e = j.get("effect_entity"); c = j.get("cause_entity")
        if j.get("doc_id") in roundup: continue
        if e in sym and c: by_cause[c].add(sym[e]); cause_doc[(c, sym[e])] = j.get("doc_id")
    for l in open(lake / "sensitivity_edge.jsonl"):
        j = json.loads(l); s_ = j.get("asset_entity"); f = j.get("factor_entity") or j.get("factor_id")
        if j.get("doc_id") in roundup: continue
        if s_ in sym and f:
            by_fac[f][sym[s_]] = j.get("sign")   # keep the extracted exposure SIGN
            fac_doc[(f, sym[s_])] = j.get("doc_id")
    for c, ss in by_cause.items():
        ss = sorted(ss)
        for x in range(len(ss)):
            for y in range(x+1, len(ss)): add((ss[x], ss[y]), c.split("__")[0], "cause", cause_doc.get((c, ss[x])))
    for f, sgn in by_fac.items():
        ss = sorted(sgn)
        for x in range(len(ss)):
            for y in range(x+1, len(ss)):
                # SIGN LOGIC: shared-factor exposure predicts corr ~ sign_i*sign_j. Same-sign
                # -> co-movement predicted; mixed-sign (e.g. "shielded from oil" vs "lifted
                # by oil") predicts the OPPOSITE — label, don't headline.
                add((ss[x], ss[y]), f.split("__")[0], "sens", fac_doc.get((f, ss[x])),
                    signpair=(sgn[ss[x]], sgn[ss[y]]))
    for l in open(lake / "relation_edge.jsonl"):
        j = json.loads(l); s1, s2 = j.get("source_entity"), j.get("target_entity")
        if j.get("doc_id") in roundup: continue
        if s1 in sym and s2 in sym:
            add((sym[s1], sym[s2]), j.get("relation") or "related", "rel", j.get("doc_id"))
    names = sorted({s for pr in links for s in pr})
    print(f"{len(links)} linked pairs over {len(names)} US names (cause+sensitivity+relation)")

    p1 = int(dt.datetime(2026, 1, 1, tzinfo=dt.UTC).timestamp()); p2 = int(dt.datetime.now(dt.UTC).timestamp())
    prox = ps.fetch_returns(ps.PROXIES, cache, p1, p2)
    px = ps.fetch_returns(names, cache, p1, p2)
    ar = ps.neutralize(px, {p: prox[p] for p in ps.PROXIES if p in prox})
    alldays = sorted(set().union(*[set(v) for v in ar.values()])) if ar else []
    recent, prior = alldays[-a.recent:], alldays[-(a.recent+45):-a.recent]
    print(f"{len(ar)} priced; recent window {recent[0]}..{recent[-1]}, prior {prior[0]}..{prior[-1]}\n")

    rows = []
    for pr, meta in links.items():
        if pr[0] not in ar or pr[1] not in ar: continue
        rc = ps.win_corr(ar[pr[0]], ar[pr[1]], recent, min_common=12)
        pc = ps.win_corr(ar[pr[0]], ar[pr[1]], prior, min_common=20)
        if rc is None or pc is None or rc < a.min_corr: continue
        s1, s2 = sec.get(pr[0]), sec.get(pr[1])
        xs = bool(s1 and s2 and s1 != s2)
        doc = next((d for d in meta["docs"] if d), None)
        sp = meta.get("signs") or []
        if any(a_ and b_ and a_ == b_ for a_, b_ in sp) or meta["kind"] - {"sens"}:
            sign_class = "same"          # co-movement predicted (or non-sens link present)
        elif any(a_ and b_ and a_ != b_ for a_, b_ in sp):
            sign_class = "MIXED"         # graph predicts hedge; positive corr = disagreement
        else:
            sign_class = "na"
        rows.append({"pair": pr, "sign_class": sign_class,
                     "recent": rc, "prior": pc, "delta": rc - pc, "cross_sector": xs,
                     "sectors": (SICN.get(s1, s1), SICN.get(s2, s2)),
                     "drivers": sorted(meta["drivers"])[:3], "kinds": sorted(meta["kind"]),
                     "headline": dochead.get(doc, "")[:70]})
    xs_rise = [r for r in rows if r["cross_sector"] and r["delta"] > 0.10]
    xs_rise.sort(key=lambda r: (r["sign_class"] == "MIXED", -r["delta"]))
    print(f"=== EMERGING CROSS-SECTOR pairs (recent corr >= {a.min_corr}, delta > +0.10, sector CANNOT explain) ===")
    print(f"  {'pair':14} {'prior->recent':>14} {'Δ':>6}  {'sectors':24} driver [link kinds]")
    for r in xs_rise[:14]:
        tag = " ⚠MIXED-SIGN(graph predicted hedge)" if r["sign_class"] == "MIXED" else ""
        print(f"  {'~'.join(r['pair']):14} {r['prior']:+.2f} -> {r['recent']:+.2f} {r['delta']:+6.2f}  "
              f"{'/'.join(str(s) for s in r['sectors']):24} {', '.join(r['drivers'])} {r['kinds']}{tag}")
        if r["headline"]: print(f"    ↳ \"{r['headline']}\"")
    same_rise = sorted([r for r in rows if not r["cross_sector"] and r["delta"] > 0.10], key=lambda r: -r["delta"])
    print(f"\n  (context: {len(xs_rise)} cross-sector risers vs {len(same_rise)} same-sector/unknown risers; "
          f"{len(rows)} linked pairs met corr floor)")


if __name__ == "__main__":
    main()
