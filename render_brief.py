#!/usr/bin/env python3
"""market-color desk brief renderer — the last mile: fact subgraph -> the note.

For a desk and an as-of date, renders a grounded morning brief:
  - PRICE ANCHORS   significant movers from data/prices/prices.parquet (z-score
                    or |return| threshold), each explained ONLY by facts.
  - DRIVERS         per mover: seed facts (vector search over the fact index,
                    desk + date-window + as-of scoped) -> 1-hop entity expansion
                    -> causal chains (cause_entities -> subject entity), with
                    per-section source corroboration and citations.
  - NO CLEAR DRIVER movers with no fact above the score floor are listed as
                    exactly that (the anti-generalisation guard: abstain, never
                    reach for a macro prior).
  - WHAT'S NEW      facts first published in the trailing 24h are flagged NEW;
                    themes = top entities by fact count in the window.

Deterministic render (no LLM): every line in the brief traces to a stored fact
with provenance. Output: data/briefs/dt=<date>/<desk>.md (+ .json for the UI).

Run (needs Qdrant at :6333 with `market_color_facts`):
  uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed \
    --with numpy --with pyarrow python render_brief.py --desk energy
  ... --all-desks            # render every desk
  ... --date 2026-07-01      # as-of a specific day (point-in-time)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date as ddate, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import index_corpus as ic
from graph_experiment import FACTS_COLLECTION, norm_entity

HERE = Path(__file__).resolve().parent
DEFAULT_PRICES = HERE / "data" / "prices" / "prices.parquet"
DEFAULT_OUT = HERE / "data" / "briefs"
DESKS = list(ic.DESK_ANCHORS)

Z_THRESHOLD = 1.0
RET_FALLBACK = 0.0075  # |1d return| when no z-score history
RET_ABSOLUTE = 0.015   # a move this big anchors a section even if z is muted
SCORE_FLOOR = 0.40     # abstention floor for driver facts
WINDOW_DAYS = 3        # facts window feeding a day's brief
SEED_K = 10
EXPAND_K = 8
MAX_SECTION_FACTS = 6


# --------------------------------------------------------------------------- #
# Prices
# --------------------------------------------------------------------------- #
def load_movers(prices: Path, desk: str, day: str) -> dict[str, Any]:
    """{available, date, movers:[...], quiet:[...]} for the desk as of `day`."""
    if not prices.exists():
        return {"available": False, "date": None, "movers": [], "quiet": []}
    import duckdb
    con = duckdb.connect()
    rows = con.execute(
        f"""select symbol, name, date, close, ret_1d, zscore_20d
            from read_parquet('{prices}')
            where list_contains(desks, ?) and date <= ?
            qualify row_number() over (partition by symbol order by date desc) = 1""",
        [desk, day]).fetchall()
    con.close()
    movers, quiet = [], []
    for sym, name, dt, close, ret, z in rows:
        # a stale row (instrument stopped updating before the window) is not a mover
        stale = (ddate.fromisoformat(day) - ddate.fromisoformat(str(dt)[:10])).days > 4
        rec = {"symbol": sym, "name": name, "date": str(dt)[:10], "close": close,
               "ret_1d": ret, "zscore_20d": z}
        significant = (not stale and ret is not None and
                       ((z is not None and abs(z) >= Z_THRESHOLD) or
                        (z is None and abs(ret) >= RET_FALLBACK) or
                        abs(ret) >= RET_ABSOLUTE))
        (movers if significant else quiet).append(rec)
    movers.sort(key=lambda m: -abs(m.get("zscore_20d") or m.get("ret_1d") or 0))
    return {"available": True, "date": day, "movers": movers, "quiet": quiet}


# --------------------------------------------------------------------------- #
# Fact retrieval (desk + window + as-of scoped; abstaining)
# --------------------------------------------------------------------------- #
def _fact_filter(desk: str | None, day: str, window_days: int) -> Any:
    from qdrant_client.http import models
    end_ord = ddate.fromisoformat(day).toordinal()
    must = [models.FieldCondition(key="published_ordinal",
                                  range=models.Range(gte=end_ord - window_days, lte=end_ord))]
    if desk:
        must.append(models.FieldCondition(key="desks", match=models.MatchAny(any=[desk])))
    return models.Filter(must=must)


def seek_driver_facts(client: Any, provider: str, model: str, query: str,
                      desk: str, day: str, floor: float) -> list[dict[str, Any]]:
    from qdrant_client.http import models
    qv = ic.embed(provider, model, [query], ic.DEFAULT_OLLAMA_URL, None)[0]
    seeds = client.query_points(
        collection_name=FACTS_COLLECTION, query=qv, limit=SEED_K, with_payload=True,
        query_filter=_fact_filter(desk, day, WINDOW_DAYS),
        search_params=models.SearchParams(
            hnsw_ef=128,
            quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)),
    ).points
    seeds = [p for p in seeds if p.score >= floor]
    if not seeds:
        return []
    seed_ents: set[str] = set()
    for p in seeds:
        seed_ents.update((p.payload or {}).get("entities") or [])
    out = {p.id: dict((p.payload or {}), _score=p.score) for p in seeds}
    # 1-hop expansion via shared entities — deliberately NOT desk-filtered, so
    # cross-desk transmission (energy shock -> fx -> rates) can enter the section.
    if seed_ents:
        neigh = client.query_points(
            collection_name=FACTS_COLLECTION, query=qv, limit=EXPAND_K * 3, with_payload=True,
            query_filter=_fact_filter(None, day, WINDOW_DAYS),
            search_params=models.SearchParams(hnsw_ef=128,
                quantization=models.QuantizationSearchParams(rescore=True, oversampling=2.0)),
        ).points
        ranked = []
        for p in neigh:
            if p.id in out:
                continue
            shared = len(set((p.payload or {}).get("entities") or []) & seed_ents)
            if shared >= 2:
                ranked.append((shared, p.score, p))
        ranked.sort(key=lambda t: (-t[0], -t[1]))
        for _, _, p in ranked[:EXPAND_K]:
            out[p.id] = dict((p.payload or {}), _score=p.score)
    return sorted(out.values(), key=lambda f: -f["_score"])


def subject_entity(f: dict[str, Any]) -> str | None:
    """Map a fact's subject to one of its entity nodes for chain traversal."""
    subj = norm_entity(str(f.get("subject") or ""))
    if not subj:
        return None
    ents = f.get("entities") or []
    if subj in ents:
        return subj
    for e in ents:
        if e in subj or subj in e:
            return e
    return subj if len(subj.split()) <= 5 else None


def causal_chains(facts: list[dict[str, Any]]) -> list[list[str]]:
    """Entity->entity edges from cause_entities -> subject; joined into <=3-node
    chains when one fact's subject is another's cause. Ordered by corroboration."""
    edges: dict[tuple[str, str], dict[str, Any]] = {}
    for f in facts:
        subj = subject_entity(f)
        if not subj:
            continue
        for ce in (f.get("cause_entities") or []):
            if ce == subj:
                continue
            e = edges.setdefault((ce, subj), {"dirs": set(), "sources": set()})
            e["dirs"].add(str(f.get("direction") or "na"))
            e["sources"].add(str(f.get("source_name") or ""))
    chains: list[tuple[int, list[str]]] = []
    for (a, b), meta in edges.items():
        direc = next(iter(meta["dirs"])) if len(meta["dirs"]) == 1 else "mixed"
        chains.append((len(meta["sources"]), [a, b, direc]))
        for (c, d), meta2 in edges.items():
            if c == b and d not in (a, b):  # a -> b -> d
                d2 = next(iter(meta2["dirs"])) if len(meta2["dirs"]) == 1 else "mixed"
                chains.append((len(meta["sources"]) + len(meta2["sources"]),
                               [a, b, d, f"{direc}/{d2}"]))
    chains.sort(key=lambda t: -t[0])
    seen, out = set(), []
    for _, ch in chains:
        key = tuple(ch[:-1])
        if key not in seen:
            seen.add(key)
            out.append(ch)
    return out[:8]


def fact_view(f: dict[str, Any], new_after_epoch: int) -> dict[str, Any]:
    return {
        "claim": f.get("claim"), "source_name": f.get("source_name"),
        "published_date": f.get("published_date"), "url": f.get("url"),
        "predicate": f.get("predicate"), "direction": f.get("direction"),
        "magnitude": f.get("magnitude"), "cause": f.get("cause"),
        "confidence": f.get("confidence"), "score": round(f.get("_score", 0.0), 3),
        "is_new": int(f.get("published_epoch") or 0) >= new_after_epoch,
    }


# --------------------------------------------------------------------------- #
# Brief assembly
# --------------------------------------------------------------------------- #
def build_brief(desk: str, day: str, qdrant_url: str, provider: str, model: str,
                prices_path: Path, floor: float) -> dict[str, Any]:
    client = ic._client(qdrant_url, None)
    px = load_movers(prices_path, desk, day)
    day_epoch = int(datetime.fromisoformat(day + "T23:59:59+00:00").timestamp())
    new_after = day_epoch - 24 * 3600

    sections: list[dict[str, Any]] = []
    seen_claims: set[str] = set()

    for m in px["movers"]:
        direction = "rose" if (m["ret_1d"] or 0) > 0 else "fell"
        query = f"why {m['name']} {direction} — drivers behind the move in {m['name']}"
        facts = seek_driver_facts(client, provider, model, query, desk, day, floor)
        facts = [f for f in facts if f.get("claim") not in seen_claims][:MAX_SECTION_FACTS]
        if not facts:
            sections.append({"kind": "no_driver", "anchor": m, "title":
                             f"{m['name']}: no fact-backed driver found", "chains": [],
                             "facts": []})
            continue
        seen_claims.update(str(f.get("claim")) for f in facts)
        sources = sorted({str(f.get("source_name")) for f in facts})
        sections.append({
            "kind": "driver", "anchor": m,
            "title": f"{m['name']} {direction} {abs(m['ret_1d']) * 100:.1f}%",
            "chains": causal_chains(facts),
            "corroboration": len(sources), "sources": sources,
            "facts": [fact_view(f, new_after) for f in facts],
        })

    # Desk theme section — the day's dominant stories, price-anchored or not.
    theme_query = ic.DESK_ANCHORS[desk]
    theme_facts = seek_driver_facts(client, provider, model, theme_query, desk, day, floor)
    theme_facts = [f for f in theme_facts if f.get("claim") not in seen_claims][:MAX_SECTION_FACTS]
    if theme_facts:
        sources = sorted({str(f.get("source_name")) for f in theme_facts})
        sections.append({
            "kind": "driver", "anchor": None, "title": "Elsewhere on the desk",
            "chains": causal_chains(theme_facts),
            "corroboration": len(sources), "sources": sources,
            "facts": [fact_view(f, new_after) for f in theme_facts],
        })

    all_views = [f for s in sections for f in s["facts"]]
    return {
        "desk": desk, "date": day,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "prices_available": px["available"],
        "movers": px["movers"], "quiet": px["quiet"],
        "sections": sections,
        "new_fact_count": sum(1 for f in all_views if f["is_new"]),
        "abstained": all(s["kind"] == "no_driver" for s in sections) if sections else True,
    }


# --------------------------------------------------------------------------- #
# Markdown render
# --------------------------------------------------------------------------- #
def _pct(x: float | None) -> str:
    return "—" if x is None else f"{x * 100:+.1f}%"


def render_markdown(b: dict[str, Any]) -> str:
    L: list[str] = []
    L.append(f"# {b['desk'].capitalize()} desk — market color, {b['date']}")
    L.append("")
    L.append(f"*Generated {b['generated_utc']} · every line below traces to a cited "
             f"article; movers with no supporting facts are listed as exactly that.*")
    L.append("")
    if b["movers"]:
        L.append("## Price anchors")
        L.append("")
        L.append("| Instrument | Close | 1d | 20d z |")
        L.append("|---|---:|---:|---:|")
        for m in b["movers"]:
            z = f"{m['zscore_20d']:+.1f}" if m.get("zscore_20d") is not None else "—"
            L.append(f"| **{m['name']}** | {m['close']:g} | {_pct(m['ret_1d'])} | {z} |")
        L.append("")
    elif b["prices_available"]:
        L.append("_No significant price moves on this desk today._")
        L.append("")
    else:
        L.append("_Price layer unavailable — themes below are fact-frequency ranked._")
        L.append("")

    for s in b["sections"]:
        if s["kind"] == "no_driver":
            m = s["anchor"]
            L.append(f"## {m['name']} {_pct(m['ret_1d'])} — **no clear driver**")
            L.append("")
            L.append(f"_No fact in the corpus window clears the evidence floor for this move. "
                     f"That is the honest answer, not a retrieval failure to paper over._")
            L.append("")
            continue
        anchor = s.get("anchor")
        head = s["title"] if anchor else s["title"]
        L.append(f"## {head}")
        L.append("")
        if s.get("chains"):
            L.append("**Transmission:** " + " · ".join(
                " → ".join(ch[:-1]) + f" ({ch[-1]})" for ch in s["chains"][:4]))
            L.append("")
        L.append(f"*{s['corroboration']} source(s): {', '.join(s['sources'])}*")
        L.append("")
        for f in s["facts"]:
            new = " **NEW**" if f["is_new"] else ""
            mag = f" [{f['magnitude']}]" if f.get("magnitude") else ""
            cause = f" — driver: {f['cause']}" if f.get("cause") else ""
            L.append(f"- {f['claim']}{mag}{cause}{new}  ")
            L.append(f"  — [{f['source_name']}, {f['published_date']}]({f['url']})")
        L.append("")

    L.append(f"---")
    L.append(f"*{b['new_fact_count']} fact(s) new in the last 24h. Quiet: " +
             (", ".join(f"{q['name']} {_pct(q['ret_1d'])}" for q in b["quiet"][:10]) or "—") + "*")
    L.append("")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--desk", choices=DESKS)
    ap.add_argument("--all-desks", action="store_true")
    ap.add_argument("--date", default=ddate.today().isoformat())
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--embedding-provider", default=ic.DEFAULT_PROVIDER)
    ap.add_argument("--embedding-model", default=ic.DEFAULT_FASTEMBED_MODEL)
    ap.add_argument("--prices", type=Path, default=DEFAULT_PRICES)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-score", type=float, default=SCORE_FLOOR)
    ap.add_argument("--stdout", action="store_true", help="print markdown instead of writing files")
    args = ap.parse_args()

    desks = DESKS if args.all_desks else ([args.desk] if args.desk else None)
    if not desks:
        ap.error("--desk or --all-desks required")

    written = []
    for desk in desks:
        brief = build_brief(desk, args.date, args.qdrant_url, args.embedding_provider,
                            args.embedding_model, args.prices, args.min_score)
        md = render_markdown(brief)
        if args.stdout:
            print(md)
            continue
        out = args.out_dir / f"dt={args.date}"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{desk}.md").write_text(md)
        (out / f"{desk}.json").write_text(json.dumps(brief, indent=2, default=str))
        n_driver = sum(1 for s in brief["sections"] if s["kind"] == "driver")
        n_abstain = sum(1 for s in brief["sections"] if s["kind"] == "no_driver")
        written.append({"desk": desk, "drivers": n_driver, "no_driver": n_abstain,
                        "movers": len(brief["movers"]), "new_facts": brief["new_fact_count"]})
        print(f"[brief] {desk}: {n_driver} driver section(s), {n_abstain} abstention(s) "
              f"-> {out / (desk + '.md')}", file=sys.stderr)
    if written:
        print(json.dumps({"date": args.date, "briefs": written}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
