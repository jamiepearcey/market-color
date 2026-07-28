# /// script
# requires-python = ">=3.10"
# ///
"""Standardised classification side-car over the two extracted news graphs
(eg100k_graph, india2021). ADDITIVE ONLY: never mutates lake/*.jsonl, writes new
files under <graph-dir>/classification/. Joins on entity_id (the "name__type"
slug already used as cause_entity/effect_entity/issuer_entity across both
graphs) and doc_id/event_id.

Entity source per graph (see docstring in taxonomy.py for why):
- eg100k_graph: entity_symbol.jsonl at the GRAPH ROOT (11,012 rows: entity_id,
  canonical_name, type, symbol). extractions.jsonl entities are per-doc mentions
  with NO entity_id (can't be joined without a name-matching heuristic we were
  told not to invent risk around), so they are NOT used as an entity source
  here -- only entity_symbol.jsonl is authoritative for eg100k entities.
- india2021: NO entity_symbol.jsonl at either lake/ or graph root (checked
  both). Entity records instead come from extractions.jsonl ex.entities[]
  (name/type/suggested_ticker/sector/country/aliases), matched to the graph's
  entity_id slugs by normalising the entity name the same way the slugs were
  built (lowercase, non-alnum -> "_", collapsed). Unmatched extraction entities
  are simply not joined (honestly reported).

Every entity_id actually appearing in the graph (event.issuer_entity or
causal_event_edge cause/effect_entity with kind=="entity") is included in
entity_class.jsonl even if no entity record was found for it -- entity_type
then falls back to the "__type" suffix of the slug, and sector resolution
proceeds from there (often ending in UNK, which is an honest result).
"""
import json, re, collections, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import taxonomy as tax
import gics

ROOT = Path(__file__).parent.parent
DATA = (ROOT / ".." / "data" / "eg_runs").resolve()

# gics.py uses abbreviated GICS sector labels; normalise to taxonomy.GICS_SECTORS
GICS_ABBR = {
    "InfoTech": "Information Technology",
    "CommSvcs": "Communication Services",
    "ConsDisc": "Consumer Discretionary",
    "Staples": "Consumer Staples",
    "Health": "Health Care",
}


def gics_sector(ticker):
    if not ticker:
        return None
    s = gics.sector(ticker.strip().upper())
    if s == "UNK":
        return None
    return GICS_ABBR.get(s, s)


def norm_name(s):
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")


def clean(v):
    """Collapse the extraction pipeline's literal string "null" / "" to None."""
    if v is None:
        return None
    if isinstance(v, str) and v.strip().lower() in ("null", ""):
        return None
    return v


def load_jsonl(path):
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_sector(entity_type, ticker, free_text_sector, name, kind=None):
    """Precedence from the task spec (+ kind-based proxy rule). Returns
    (std_sector, sector_source)."""
    if entity_type in tax.MACRO_ENTITY_TYPES:
        return "MACRO", "entity_type"
    if kind in ("factor", "etf_proxy"):
        # entity_symbol.jsonl proxies: macro/factor series or sector-ETF proxy,
        # not a tradable company -> MACRO regardless of its (often "other") type.
        return "MACRO", "kind_proxy"
    g = gics_sector(ticker)
    if g:
        return g, "gics_ticker"
    s = tax.sector_from_text(free_text_sector)
    if s:
        return s, "sector_text"
    s = tax.sector_from_text(name) or tax.sector_from_text(entity_type)
    if s:
        return s, "name_text"
    return "UNK", "none"


def classify_graph(gdir: Path, label: str):
    out_dir = gdir / "classification"
    out_dir.mkdir(exist_ok=True)

    # ---- 1. entity universe from the graph itself -------------------------
    entity_ids = set()
    for j in load_jsonl(gdir / "lake" / "event.jsonl"):
        iss = j.get("issuer_entity")
        if iss:
            entity_ids.add(iss)
    edge_rows = list(load_jsonl(gdir / "lake" / "causal_event_edge.jsonl"))
    for j in edge_rows:
        if j.get("cause_kind") == "entity" and j.get("cause_entity"):
            entity_ids.add(j["cause_entity"])
        if j.get("effect_kind") == "entity" and j.get("effect_entity"):
            entity_ids.add(j["effect_entity"])

    # ---- 2. entity records ------------------------------------------------
    # eg100k: entity_symbol.jsonl at graph root (authoritative).
    sym_path = gdir / "entity_symbol.jsonl"
    sym_by_id = {}
    for j in load_jsonl(sym_path):
        sym_by_id[j["entity_id"]] = j

    # NORMALISED-NAME FALLBACK. The resolver and the graph slug company names
    # independently, so exact entity_id equality misses legitimate matches:
    # the resolver holds "amazon_com__company" -> AMZN while the graph refers to
    # "amazon__company". A normalised join recovers these. Corporate-form suffixes
    # are stripped, and a match is only accepted when the normalised key is UNIQUE
    # on both sides and the entity types agree, so no ambiguous name is promoted.
    _SUFFIX = ("com", "inc", "corp", "corporation", "co", "plc", "ltd", "limited",
               "group", "holdings", "holding", "sa", "ag", "nv", "se", "spa", "the")

    def _norm_key(eid: str) -> str:
        name, _, typ = eid.rpartition("__")
        parts = [p for p in (name or eid).split("_") if p]
        while parts and parts[-1] in _SUFFIX:
            parts.pop()
        return ("".join(parts), typ)

    _sym_norm: dict = {}
    for k in sym_by_id:
        _sym_norm.setdefault(_norm_key(k), []).append(k)
    sym_norm = {k: v[0] for k, v in _sym_norm.items() if len(v) == 1}

    # extractions.jsonl ex.entities[] (streamed, one line at a time -- never
    # materialised in full; eg100k's file is ~400MB): used for BOTH graphs as
    # a free-text sector / country / (unverified) suggested_ticker source,
    # matched to entity_id slugs by normalised-name lookup. This is additive
    # to entity_symbol.jsonl (eg100k), not a replacement for it -- symbol/kind
    # from entity_symbol.jsonl always wins for suggested_ticker/resolution.
    name_index = collections.defaultdict(list)  # norm_name -> [entity record]
    extraction_mentions = 0
    for j in load_jsonl(gdir / "extractions.jsonl"):
        ex = j.get("ex")
        if not ex:
            continue
        for e in ex.get("entities", []) or []:
            if not isinstance(e, dict):
                continue
            extraction_mentions += 1
            nm = norm_name(e.get("name"))
            if nm:
                name_index[nm].append(e)
            for alias in e.get("aliases") or []:
                an = norm_name(alias)
                if an:
                    name_index[an].append(e)

    if sym_path.exists():
        entity_source_note = (
            f"entity_symbol.jsonl ({len(sym_by_id)} rows, graph root, resolved "
            f"ticker+kind) + extractions.jsonl ex.entities[] ({extraction_mentions} "
            "mention rows, name-matched, free-text sector/country only)"
        )
    elif extraction_mentions:
        entity_source_note = (
            f"extractions.jsonl ex.entities[] ({extraction_mentions} mention rows, "
            "name-matched) -- no entity_symbol.jsonl found at root or lake/"
        )
    else:
        entity_source_note = "NONE FOUND"

    matched_by_name = 0
    entity_rows = []
    for eid in sorted(entity_ids):
        name_slug, _, type_suffix = eid.rpartition("__")
        name_slug = name_slug or eid
        rec = sym_by_id.get(eid)
        matched_via = "entity_id" if rec else None
        if rec is None:
            alt = sym_norm.get(_norm_key(eid))
            if alt is not None:
                rec = sym_by_id[alt]
                matched_via = "normalised_name"
        kind = rec.get("kind") if rec else None
        suggested_ticker = None
        free_text_sector = None
        country = None
        entity_type = type_suffix or "other"
        canonical_name = name_slug.replace("_", " ").strip()

        if rec:
            entity_type = clean(rec.get("type")) or entity_type
            suggested_ticker = clean(rec.get("symbol"))
            canonical_name = clean(rec.get("canonical_name")) or canonical_name

        # extraction name-match: additive free-text sector/country/(unverified)
        # ticker source, applied for BOTH graphs regardless of whether
        # entity_symbol.jsonl already gave us a record.
        cand = name_index.get(name_slug) or name_index.get(norm_name(canonical_name))
        if cand:
            matched_by_name += 1
            e = cand[0]
            if not rec:
                entity_type = clean(e.get("type")) or entity_type
                suggested_ticker = clean(e.get("suggested_ticker"))
                canonical_name = clean(e.get("name")) or canonical_name
            free_text_sector = clean(e.get("sector"))
            country = clean(e.get("country"))

        std_sector, sector_source = resolve_sector(
            entity_type, suggested_ticker, free_text_sector, canonical_name, kind
        )

        # resolution_status: entity_symbol.jsonl (rec) IS the verified Yahoo/LLM
        # resolution pipeline output for eg100k -- kind tells us what was
        # resolved. India (and any eg100k entity without a rec) only ever has
        # an LLM "suggested" ticker from extractions, never promoted.
        resolved_ticker = None
        if rec and suggested_ticker:
            resolved_ticker = suggested_ticker.strip().upper()
            resolution_status = {
                "security": "resolved_security",
                "etf_proxy": "resolved_proxy",
                "factor": "resolved_factor",
            }.get(kind, "resolved_other")
        elif suggested_ticker:
            resolution_status = "suggested_unverified"
        else:
            resolution_status = "unresolved"

        entity_rows.append(
            {
                "entity_id": eid,
                "name": canonical_name,
                "entity_type": entity_type,
                "std_sector": std_sector,
                "sector_source": sector_source,
                "suggested_ticker": suggested_ticker,
                "resolved_ticker": resolved_ticker,
                "resolution_status": resolution_status,
                "country": country,
            }
        )

    sector_by_id = {r["entity_id"]: r["std_sector"] for r in entity_rows}

    # ---- 3. events ----------------------------------------------------------
    event_rows = []
    event_type_counts = collections.Counter()
    event_unmapped_raw = collections.Counter()
    for j in load_jsonl(gdir / "lake" / "event.jsonl"):
        raw = j.get("event_type")
        std, group = tax.std_event_type(raw)
        event_type_counts[std] += 1
        if std == "other" and raw:
            event_unmapped_raw[raw.strip().lower()] += 1
        event_rows.append(
            {
                "event_id": j.get("event_id"),
                "doc_id": j.get("doc_id"),
                "raw_event_type": raw,
                "std_event_type": std,
                "event_group": group,
                "series_id": j.get("series_id"),
            }
        )

    # ---- 4. causal edges ------------------------------------------------
    edge_rows_out = []
    mech_counts = collections.Counter()
    mech_unmapped_raw = collections.Counter()
    for j in edge_rows:
        raw_mech = j.get("mechanism")
        std_mech = tax.std_mechanism(raw_mech)
        mech_counts[std_mech] += 1
        if std_mech == "other" and raw_mech:
            mech_unmapped_raw[raw_mech.strip().lower()] += 1
        effect_entity = j.get("effect_entity")
        edge_rows_out.append(
            {
                "doc_id": j.get("doc_id"),
                "cause_entity": j.get("cause_entity"),
                "effect_entity": effect_entity,
                "raw_mechanism": raw_mech,
                "std_mechanism": std_mech,
                "effect_dir": j.get("effect_dir"),
                "std_sector_of_effect": sector_by_id.get(effect_entity),
            }
        )

    # ---- 5. write side-car files -----------------------------------------
    with open(out_dir / "entity_class.jsonl", "w") as f:
        for r in entity_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "event_class.jsonl", "w") as f:
        for r in event_rows:
            f.write(json.dumps(r) + "\n")
    with open(out_dir / "edge_class.jsonl", "w") as f:
        for r in edge_rows_out:
            f.write(json.dumps(r) + "\n")

    # ---- 6. coverage report -----------------------------------------------
    n_ent = len(entity_rows)
    n_ev = len(event_rows)
    n_edge = len(edge_rows_out)
    sector_source_counts = collections.Counter(r["sector_source"] for r in entity_rows)
    non_unk = sum(1 for r in entity_rows if r["std_sector"] != "UNK")
    sector_dist = collections.Counter(r["std_sector"] for r in entity_rows)
    resolution_counts = collections.Counter(r["resolution_status"] for r in entity_rows)
    ev_mapped = sum(v for k, v in event_type_counts.items() if k != "other")
    edge_mapped = sum(v for k, v in mech_counts.items() if k != "other")

    coverage = {
        "graph": label,
        "entity_source": entity_source_note,
        "n_entities": n_ent,
        "n_events": n_ev,
        "n_edges": n_edge,
        "entities_name_matched": matched_by_name,
        "pct_entities_non_unk_sector": round(100 * non_unk / n_ent, 2) if n_ent else None,
        "sector_source_breakdown": dict(sector_source_counts),
        "sector_distribution": dict(sector_dist),
        "resolution_status_counts": dict(resolution_counts),
        "pct_events_mapped_non_other": round(100 * ev_mapped / n_ev, 2) if n_ev else None,
        "pct_edges_mapped_non_other": round(100 * edge_mapped / n_edge, 2) if n_edge else None,
        "event_type_distribution": dict(event_type_counts),
        "mechanism_distribution": dict(mech_counts),
        "top_unmapped_event_types": event_unmapped_raw.most_common(15),
        "top_unmapped_mechanisms": mech_unmapped_raw.most_common(15),
    }
    with open(out_dir / "coverage.json", "w") as f:
        json.dump(coverage, f, indent=2)

    return coverage


ALL_GRAPHS = {
    "eg100k_graph": "eg100k_graph",
    "india2021": "india2021",
    # the live capture: same shape as eg100k (entity_symbol.jsonl at graph root),
    # so it needs no special handling — only to be named here.
    "eg_live2": "eg_live2",
}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", choices=sorted(ALL_GRAPHS), action="append",
                    help="repeatable; default = the two historical graphs")
    a = ap.parse_args()
    names = a.graph or ["eg100k_graph", "india2021"]
    graphs = [(DATA / n, n) for n in names]
    reports = []
    for gdir, label in graphs:
        print(f"=== {label} ===")
        cov = classify_graph(gdir, label)
        print(json.dumps(cov, indent=2)[:4000])
        reports.append(cov)
    return reports


if __name__ == "__main__":
    main()
