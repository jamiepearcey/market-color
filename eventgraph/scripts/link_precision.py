# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
LINK PRECISION — which edges are about the company, and which merely mention it?

THE PROBLEM, found by drilling into the news-weight table. Across 160 names, the
MEDIAN captured "event day" moves the stock 0.68 sigma — LESS than an ordinary
non-event day (0.75). Only 27% clear the gate. Inspecting them shows why: the
causal extraction links a ticker to any article naming it, so AAPL picks up
"Murdoch, Moguls Head to Sun Valley as Mobile Shapes Media" and a fund's position
list ("Omega Cuts Sunoco, Regal Entertainment, Buys Wellpoint, Apple") classified
as `earnings`. These are passing mentions, not events about the company, and they
dilute every per-name statistic built on top.

WHAT THIS DOES. Scores each (ticker, day) cell by signals already present on the
edge or cheaply derivable, then tests EACH SIGNAL INDEPENDENTLY against the only
outcome that matters: does the filtered set actually move more?

    median idiosyncratic sigma, and share of days above 2 sigma,
    against the 0.75-sigma control and the 0.68/12.0% unfiltered baseline

A signal only earns its place if it raises those. This is deliberately not a
learned model — a filter that cannot be stated in one sentence cannot be argued
with, and the point is to be able to say WHY a link was dropped.

Usage:
    uv run scripts/link_precision.py
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
OUT = G / "link_precision.json"
STOP = {"inc", "corp", "co", "ltd", "plc", "group", "holdings", "the", "company",
        "sa", "nv", "ag", "&", "and", "of", "international", "industries"}


def main() -> None:
    C = json.loads((G / "mcp_cache.json").read_text())
    days = C["days"]; di = {d: i for i, d in enumerate(days)}

    ent, ent_name = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        tk = j.get("resolved_ticker")
        if j.get("resolution_status") == "resolved_security" and tk:
            ent[j["entity_id"]] = tk.upper()
            ent_name[j["entity_id"]] = (j.get("name") or "").lower()

    docmeta = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docmeta[j["doc_id"]] = {"date": d, "head": (j.get("headline") or "").lower()}

    # gather edges, keeping the signals
    edges = []
    per_doc_entities = collections.defaultdict(set)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        if e not in ent or doc not in docmeta:
            continue
        per_doc_entities[doc].add(e)
        chunk = j.get("chunk_id") or ""
        m = re.search(r":(\d+)$", chunk)
        edges.append({
            "tk": ent[e], "eid": e, "doc": doc, "date": docmeta[doc]["date"],
            "conf": j.get("confidence"), "modality": j.get("modality"),
            "attribution": j.get("attribution"), "lag": j.get("lag"),
            "chunk": int(m.group(1)) if m else 99,
        })
    print(f"{len(edges)} resolved edges over {len(per_doc_entities)} documents")

    # derived signals
    for e in edges:
        nm = ent_name.get(e["eid"], "")
        toks = [t for t in re.split(r"[^a-z0-9]+", nm) if t and t not in STOP and len(t) > 2]
        head = docmeta[e["doc"]]["head"]
        e["in_headline"] = bool(toks) and any(t in head for t in toks)
        e["n_entities_in_doc"] = len(per_doc_entities[e["doc"]])
        e["lead_chunk"] = e["chunk"] == 0

    # outcome: the idiosyncratic move on that (ticker, day)
    def sigma(tk, d):
        i = str(di.get(d, -1))
        return C["sigma"].get(tk, {}).get(i)

    for e in edges:
        e["sig"] = sigma(e["tk"], e["date"])
    have = [e for e in edges if e["sig"] is not None]
    print(f"{len(have)} edges land on a measurable (ticker, day)\n")

    base = np.array([e["sig"] for e in have])
    CTRL = 0.75

    def score(sel, label):
        s = np.array([e["sig"] for e in sel])
        if len(s) < 40:
            return None
        return {"filter": label, "n": len(s), "kept": round(len(s) / len(have), 3),
                "median": round(float(np.median(s)), 3),
                "mean": round(float(s.mean()), 3),
                "pct_gt2": round(float((s > 2).mean()), 4),
                "pct_gt_ctrl": round(float((s > CTRL).mean()), 4)}

    rows = [score(have, "ALL EDGES (no filter)")]
    tests = [
        ("confidence == strong", lambda e: e["conf"] == "strong"),
        ("modality == happened", lambda e: e["modality"] == "happened"),
        ("entity in HEADLINE", lambda e: e["in_headline"]),
        ("lead chunk (chunk 0)", lambda e: e["lead_chunk"]),
        ("doc names <= 2 companies", lambda e: e["n_entities_in_doc"] <= 2),
        ("doc names <= 4 companies", lambda e: e["n_entities_in_doc"] <= 4),
        ("doc names >= 8 companies (list article)", lambda e: e["n_entities_in_doc"] >= 8),
        ("attribution == reporter", lambda e: e["attribution"] == "reporter"),
        ("lag == immediate", lambda e: e["lag"] == "immediate"),
    ]
    for label, fn in tests:
        r = score([e for e in have if fn(e)], label)
        if r:
            rows.append(r)

    print(f"{'filter':42}{'n':>7}{'kept':>7}{'median σ':>10}{'>2σ':>8}{'>ctrl':>8}")
    for r in rows:
        mark = ""
        if r["filter"] != rows[0]["filter"]:
            mark = "  <<" if r["median"] > rows[0]["median"] + 0.05 else ""
        print(f"{r['filter']:42}{r['n']:7}{r['kept']:7.0%}{r['median']:10.2f}"
              f"{r['pct_gt2']:8.1%}{r['pct_gt_ctrl']:8.1%}{mark}")
    print(f"\ncontrol (ordinary non-event day) = {CTRL:.2f}σ. A filter is only worth "
          f"having if it lifts the median ABOVE that line.")

    # best combination of the signals that individually helped
    combos = [
        ("headline AND <=4 companies",
         lambda e: e["in_headline"] and e["n_entities_in_doc"] <= 4),
        ("headline AND strong",
         lambda e: e["in_headline"] and e["conf"] == "strong"),
        ("headline AND happened AND <=4 companies",
         lambda e: e["in_headline"] and e["modality"] == "happened"
         and e["n_entities_in_doc"] <= 4),
        ("headline AND lead chunk",
         lambda e: e["in_headline"] and e["lead_chunk"]),
    ]
    print(f"\n{'combination':42}{'n':>7}{'kept':>7}{'median σ':>10}{'>2σ':>8}{'>ctrl':>8}")
    best = None
    for label, fn in combos:
        r = score([e for e in have if fn(e)], label)
        if r:
            rows.append(r)
            print(f"{r['filter']:42}{r['n']:7}{r['kept']:7.0%}{r['median']:10.2f}"
                  f"{r['pct_gt2']:8.1%}{r['pct_gt_ctrl']:8.1%}")
            if best is None or r["median"] > best["median"]:
                best = r
    OUT.write_text(json.dumps({"control": CTRL, "results": rows, "best": best}))
    print(f"\nwrote {OUT}")
    if best:
        print(f"best: {best['filter']} — median {best['median']}σ on {best['kept']:.0%} of edges "
              f"(unfiltered {rows[0]['median']}σ)")


if __name__ == "__main__":
    main()
