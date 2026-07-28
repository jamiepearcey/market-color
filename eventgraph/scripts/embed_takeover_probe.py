# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
DOES THE CORPUS CONTAIN TAKEOVER EVENTS AT ALL? An embedding probe of F40.

F40 concluded that `m_and_a`'s flat 1.21x is not dilution but ABSENCE — takeover
targets get acquired and delisted, so the names that would show a 20-40% gap are
missing from a universe defined by current-day resolvability. That conclusion
rests on a lexical scan of 320 headlines, which could simply have missed
takeover language phrased differently.

This tests it in embedding space instead. `receptors/data_bloomberg` holds 11,100
all-MiniLM chunk vectors whose chunk_ids are `bbg_<doc>:<n>` — the SAME doc_id
namespace as the eventgraph lake, so the two join directly.

NO ENCODER IS INSTALLED, so the query is built by example rather than by encoding
a string: seed on chunks whose text carries unambiguous takeover language, take
the centroid of their vectors as the "takeover direction", and rank every chunk
by cosine to it. Query-by-example is the honest option here and arguably the
better one — it searches for what takeover text actually looks like in THIS
corpus rather than what the phrase embeds to in the abstract.

Then the question that matters: of the documents the embedding says are about
takeovers, how many reach a PRICEABLE ticker? If takeover news is plentiful but
lands on entities with no resolved ticker or no price history, F40 survives and
the ceiling is structural. If it lands on priceable names that were simply
mislabelled, F40 is wrong and the fix is classification after all.

Usage:
    uv run scripts/embed_takeover_probe.py --top 300
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
EMB = MC / "receptors" / "data_bloomberg"
ROOT = MC / "data" / "eg_runs"

# Unambiguous takeover language. Deliberately narrow: these seed the direction,
# and a loose seed would blur it into generic corporate-finance text.
SEED = re.compile(
    r"\b(takeover (bid|offer|approach|battle)|tender offer|agreed to be acquired|"
    r"agrees to be acquired|to be acquired by|acquisition of \w+ (Corp|Inc|Plc|AG|SA)|"
    r"buyout offer|hostile bid|unsolicited (bid|offer)|bid for the company|"
    r"sweetened (its )?(bid|offer)|rival bid|counterbid|"
    r"agreed to buy .{0,40} for \$|offer to buy all)\b", re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="eg100k_graph")
    ap.add_argument("--top", type=int, default=300)
    a = ap.parse_args()

    vecs = np.load(EMB / "chunks.npy")
    chunks = [json.loads(l) for l in open(EMB / "chunks.jsonl")]
    texts = [json.loads(l).get("t", "") for l in open(EMB / "chunk_texts.jsonl")]
    n = min(len(vecs), len(chunks), len(texts))
    print(f"chunk vectors {vecs.shape} | chunks {len(chunks)} | texts {len(texts)} -> using {n}")

    seed_ix = [i for i in range(n) if SEED.search(texts[i])]
    if not seed_ix:
        raise SystemExit("no seed chunks matched — widen SEED")
    V = vecs[:n].astype(np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    q = V[seed_ix].mean(axis=0)
    q /= np.linalg.norm(q) + 1e-9
    sim = V @ q
    order = np.argsort(-sim)[:a.top]
    print(f"seeded on {len(seed_ix)} chunks with explicit takeover language; "
          f"ranking {n} chunks by cosine to their centroid\n")

    print("top 12 nearest chunks (is the direction real?):")
    for i in order[:12]:
        print(f"   {sim[i]:.3f}  {texts[i][:118].strip()}")

    # ---- join the retrieved docs back to the graph ------------------------
    G = ROOT / a.graph
    tick, resolved = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        resolved[j["entity_id"]] = j.get("resolution_status")
        t = (j.get("resolved_ticker") or "").upper()
        if t:
            tick[j["entity_id"]] = t

    docday, docmeta = {}, {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
            docmeta[j["doc_id"]] = j.get("headline") or ""

    ev_by_doc = collections.defaultdict(set)
    issuer_by_doc = collections.defaultdict(set)
    for l in open(G / "lake" / "event.jsonl"):
        j = json.loads(l)
        std, _ = tax.std_event_type(j.get("event_type"))
        if j.get("doc_id"):
            ev_by_doc[j["doc_id"]].add(std)
            if j.get("issuer_entity"):
                issuer_by_doc[j["doc_id"]].add((j["issuer_entity"], std))

    eff_by_doc = collections.defaultdict(set)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("effect_entity"):
            eff_by_doc[j["doc_id"]].add(j["effect_entity"])

    P = load("us")
    docs = list(dict.fromkeys(chunks[i]["doc_id"] for i in order))
    in_graph = [d for d in docs if d in docday]
    labels = collections.Counter()
    reach = collections.Counter()
    sigmas, examples = [], []
    for d in in_graph:
        for c in ev_by_doc.get(d, {"<no event row>"}):
            labels[c] += 1
        ents = eff_by_doc.get(d, set())
        if not ents:
            reach["no causal edge"] += 1
            continue
        tk = {tick[e] for e in ents if e in tick}
        if not tk:
            reach["no entity resolves to a ticker"] += 1
            continue
        priced = [t for t in tk if P.sigma_at(t, docday[d]) is not None]
        if not priced:
            reach["ticker but no price on the day"] += 1
            continue
        reach["PRICEABLE"] += 1
        for t in priced:
            s = P.sigma_at(t, docday[d])
            sigmas.append(float(s))
            examples.append((float(s), t, docday[d], docmeta.get(d, "")[:70]))

    print(f"\n=== {len(docs)} distinct docs retrieved; {len(in_graph)} present in {a.graph}")
    print("\nwhat the taxonomy CALLS these takeover-like docs:")
    for c, k in labels.most_common(10):
        print(f"   {k:>5}  {c}")
    print("\ndoes the takeover news REACH a priceable name?")
    tot = sum(reach.values())
    for k, v in reach.most_common():
        print(f"   {v:>5}  ({v/tot:5.1%})  {k}")

    if sigmas:
        s = np.array(sigmas)
        print(f"\npriceable takeover-like cells: n={len(s)}  mean sigma {s.mean():.3f}  "
              f"median {np.median(s):.3f}  (panel control ~0.753)")
        print(f"   lift {s.mean()/0.753:.2f}x   |sigma|>2 in {(s>2).mean():.0%} of cells")
        print("\n   largest moves among them:")
        for sg, t, d, h in sorted(examples, reverse=True)[:10]:
            print(f"     {sg:5.2f}s  {t:6} {d}  {h}")


if __name__ == "__main__":
    main()
