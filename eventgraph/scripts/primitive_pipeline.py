# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
THE PRIMITIVE PIPELINE — chunk, mention, resolve, place, classify, role, emit.
Design + rationale in docs/PRIMITIVE-PIPELINE.md.

WHY IT EXISTS. The current graph only creates a cell for a name if the extractor
produced a CAUSAL EDGE touching it. Measured on 574 takeover headlines, 48.1%
produced no causal edge at all and two thirds of those produced no event row
either — so a third of unambiguous, dateable takeover news simply vanishes before
anything can be measured. Causality is a hard extraction task; "which listed
company is this about, and what kind of event is it" is an easy one.

This is the ZERO-LLM baseline of that idea, deliberately. Establish what a purely
lexical pipeline reaches before paying a model to do better — if a dumb pipeline
beats 4.5% reach, that is a finding about the architecture, not the corpus.

    1 CHUNK     lake/chunk.jsonl (text already chunked)
    2 MENTION   alias match over headline+chunk, restricted to entities that
                already have a VERIFIED ticker (recall-first, precision later)
    3 RESOLVE   verified resolved_ticker only; `suggested` is never a merge key
    4 PLACE     ticker -> GICS sector + priceability on the day
    5 CLASSIFY  HEADLINE -> taxonomy class, skipping the lossy
                text -> LLM free-text event_type -> taxonomy hop
    6 ROLE      named in the headline = subject; body-only = mentioned.
                This is what replaces F39's doc-union inheritance.
    7 EMIT      (ticker, date, class, role, quote, chunk_id)
    8 FUNNEL    reach measured at every stage, always printed

Usage:
    uv run scripts/primitive_pipeline.py --probe takeover     # head-to-head vs F41
    uv run scripts/primitive_pipeline.py --limit 20000        # full-corpus funnel
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import taxonomy as tax  # noqa: E402
from panel import load  # noqa: E402

MC = Path(__file__).resolve().parent.parent.parent
ROOT = MC / "data" / "eg_runs"
ALIAS_RE = re.compile(r"INSERT INTO entity_alias \(alias,entity_id\) VALUES "
                      r"\('(.+?)','(.+?)'\)")
TAKEOVER = re.compile(r"\b(takeover|to be acquired|agrees? to buy|acquisition of|"
                      r"tender offer|buyout|hostile bid|bid for|merger with|to merge with)\b", re.I)

# Aliases that are junk by construction. The table contains extraction artefacts
# like "Inco's Net Soars" (a headline fragment registered as an entity name), and
# short/generic tokens match everything.
BAD_ALIAS = re.compile(r"^(the|a|an|inc|plc|corp|ltd|sa|ag|co|group|holdings?)$", re.I)


def load_aliases(graph: Path, tickered: set[str]) -> dict[str, str]:
    """alias -> entity_id, restricted to entities that already resolve to a
    verified ticker. Restricting first keeps this small and high-precision."""
    out = {}
    sql = graph / "pg_upsert.sql"
    if not sql.exists():
        return out
    for line in sql.read_text(errors="ignore").splitlines():
        m = ALIAS_RE.search(line)
        if not m:
            continue
        alias, eid = m.group(1).replace("''", "'"), m.group(2)
        if eid not in tickered:
            continue
        if len(alias) < 4 or BAD_ALIAS.match(alias) or len(alias) > 60:
            continue
        if not re.search(r"[A-Za-z]{3}", alias):
            continue
        out.setdefault(alias, eid)
    return out


def classify_headline(h: str) -> str:
    """HEADLINE -> controlled-vocab class, via the same substring rules the
    event_type mapper uses. One hop instead of two, and no dependency on the
    extractor having chosen to name an event at all."""
    s = " " + h.lower() + " "
    for needles, canon in tax.EVENT_TYPE_PATTERNS:
        if any(n in s for n in needles):
            return canon
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="eg100k_graph")
    ap.add_argument("--probe", choices=["takeover", "all"], default="takeover")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    G = ROOT / a.graph
    P = load("us")

    # ---- 3 RESOLVE (built first, because it bounds stage 2) ------------------
    tick, sector = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t and j.get("resolution_status", "").startswith("resolved"):
            tick[j["entity_id"]] = t
            if j.get("std_sector") not in (None, "UNK"):
                sector[t] = j["std_sector"]
    alias = load_aliases(G, set(tick))
    # longest-first so "Bank of America Corp" wins over "Bank of America"
    alias_order = sorted(alias, key=len, reverse=True)
    print(f"resolve: {len(tick)} entities with a verified ticker | "
          f"{len(alias)} usable aliases")

    # ---- 1 CHUNK -------------------------------------------------------------
    docday, headline = {}, {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
            headline[j["doc_id"]] = (j.get("headline") or "").strip()

    body = collections.defaultdict(list)
    for l in open(G / "lake" / "chunk.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("text"):
            body[j["doc_id"]].append(j["text"])

    docs = [d for d in docday if d in headline]
    if a.probe == "takeover":
        docs = [d for d in docs if TAKEOVER.search(headline[d])]
    if a.limit:
        docs = docs[:a.limit]
    print(f"corpus: {len(docs)} documents ({a.probe} probe)\n")

    # ---- 2/4/5/6/7 -----------------------------------------------------------
    funnel = collections.Counter()
    cells = collections.defaultdict(lambda: {"classes": set(), "role": "mentioned"})
    for doc in docs:
        h = headline[doc]
        text = " ".join(body.get(doc, []))[:6000]
        klass = classify_headline(h)
        funnel["documents"] += 1
        if klass == "other":
            funnel["headline yields no class"] += 1

        hay_h, hay_b = " " + h + " ", " " + text + " "
        found = {}
        for al in alias_order:
            if al in hay_h:
                found[tick[alias[al]]] = "subject"
            elif al in hay_b and tick[alias[al]] not in found:
                found[tick[alias[al]]] = "mentioned"
        if not found:
            funnel["no verified entity mentioned"] += 1
            continue
        funnel[">=1 verified entity mentioned"] += 1
        if any(r == "subject" for r in found.values()):
            funnel["  of which >=1 SUBJECT (headline)"] += 1

        day = docday[doc]
        priced_any = False
        for tk, role in found.items():
            if P.sigma_at(tk, day) is None:
                continue
            priced_any = True
            c = cells[(tk, day)]
            c["classes"].add(klass)
            if role == "subject":
                c["role"] = "subject"
        if priced_any:
            funnel["PRICEABLE (>=1 name with a sigma)"] += 1
        else:
            funnel["mentioned but no price that day"] += 1

    print("=== REACH FUNNEL (primitive pipeline)")
    n = funnel["documents"]
    for k in ["documents", "headline yields no class", "no verified entity mentioned",
              ">=1 verified entity mentioned", "  of which >=1 SUBJECT (headline)",
              "mentioned but no price that day", "PRICEABLE (>=1 name with a sigma)"]:
        v = funnel[k]
        print(f"  {v:>6}  ({v/n:6.1%})  {k}")

    # ---- measured lift, split by role ---------------------------------------
    ctrl = 0.753
    by_role = collections.defaultdict(list)
    by_class = collections.defaultdict(list)
    for (tk, day), c in cells.items():
        s = P.sigma_at(tk, day)
        if s is None:
            continue
        by_role[c["role"]].append(float(s))
        for k in c["classes"]:
            by_class[(k, c["role"])].append(float(s))

    print(f"\n=== CELLS EMITTED: {len(cells)}   (control sigma {ctrl})")
    for role in ("subject", "mentioned"):
        v = by_role.get(role, [])
        if v:
            arr = np.array(v)
            print(f"  {role:10} n={len(arr):>5}  mean sigma {arr.mean():.3f}  "
                  f"lift {arr.mean()/ctrl:.2f}x  |s|>2 in {(arr>2).mean():.0%}")

    rows = [(k, r, len(v), np.mean(v) / ctrl) for (k, r), v in by_class.items() if len(v) >= 8]
    if rows:
        print(f"\n  {'class':18} {'role':10} {'n':>5} {'lift':>6}")
        for k, r, nn, lift in sorted(rows, key=lambda x: -x[3])[:14]:
            print(f"  {k:18} {r:10} {nn:>5} {lift:>5.2f}x")


if __name__ == "__main__":
    main()
