# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Live retrieval tool — the substrate for the Stage-3 adaptive investigation loop.

An analyst agent calls this on demand: pose a targeted query for a thinly-evidenced
claim, get back the most relevant passages (with source tier / date / doc_id), then
decide whether to dig further. Two modes:

  topical      passage-level cosine over the full corpus; returns ranked evidence.
  corroborate  same retrieval, but reports how many DISTINCT sources/domains carry
               supporting evidence above a threshold — the input to the epistemic
               ledger's corroborated/supported call.

Flags:
  -k N            number of results (default 8)
  --before DATE   point-in-time: only evidence published strictly before DATE
                  (YYYY-MM-DD). Prevents look-ahead; makes runs reproducible.
  --exclude IDS   comma-separated doc_ids to suppress, so a dig-loop iteration
                  surfaces NEW evidence rather than what the agent already has.
  --tier T        keep only sources at tier <= T (1=highest quality).
  --json          machine-readable output (for verify/orchestration scripts).

Usage:
  uv run scripts/retrieve.py "india crude import costs current account deficit"
  uv run scripts/retrieve.py --corroborate "russia imports jet fuel from japan"
  uv run scripts/retrieve.py -k 6 --before 2026-07-01 --exclude a1b2,c3d4 "..."
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data"


def load():
    ch = np.load(D / "chunks.npy").astype(np.float32)
    ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
    texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
    dmeta = json.loads((D / "doc_meta.json").read_text())
    return ch, meta, texts, dmeta


def load_novelty():
    """doc_id -> first-story novelty (1 - max cos to strictly-earlier chunk); high=new.
    Doc novelty = max over its chunks (its most novel content)."""
    p = D / "novelty.jsonl"
    if not p.exists():
        return {}
    nov = {}
    for l in p.read_text().splitlines():
        if not l.strip():
            continue
        o = json.loads(l)
        nov[o["doc_id"]] = max(nov.get(o["doc_id"], 0.0), float(o.get("novelty", 0.0)))
    return nov


def date_to_epoch(s):
    import datetime
    y, m, d = map(int, s.split("-"))
    return int(datetime.datetime(y, m, d, tzinfo=datetime.timezone.utc).timestamp())


# NOTE: the symbolic cause_entities graph walk that briefly lived here was removed
# by design: the live search surface must be RAW CHUNKED SOURCE only. Decomposed
# artifacts (facts/entities) are training supervision for the receptors, never an
# inference-time index. Causal reach is provided by the transport receptor
# (scripts/receptors_tools.py causal_find) — pure embedding traversal.


_STOP = set("""the a an and or of to in on for with from as at by is are was were be been being
that this these those it its their his her they them we you your our not no than then over under
into out up down off about after before amid over more most less least such some any all each other
has have had will would could should may might can into per its it's about across between among""".split())

def salient_terms(claim):
    toks = "".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in claim).split()
    return {t for t in toks if len(t) >= 4 and t not in _STOP}

def term_coverage(claim_terms, passage):
    if not claim_terms:
        return 0.0
    ptoks = set("".join(ch.lower() if (ch.isalnum() or ch.isspace()) else " " for ch in passage).split())
    return len(claim_terms & ptoks) / len(claim_terms)


def retrieve(query, k=8, before=None, exclude=None, tier=None, pool=400,
             diverse=False, lam=0.5, novel=False, nov_w=0.5):
    ch, meta, texts, dmeta = load()
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    q = np.array(list(model.embed([query]))[0], dtype=np.float32)
    q /= (np.linalg.norm(q) + 1e-9)
    scores = ch @ q

    exclude = set((exclude or "").split(",")) - {""}
    cutoff = date_to_epoch(before) if before else None

    # rank chunks, then fold to best-chunk-per-doc, applying filters
    order = np.argsort(-scores)[: pool * 4]
    best = {}  # doc_id -> (score, chunk_idx)
    for ci in order:
        m = meta[ci]
        did = m["doc_id"]
        if did in exclude:
            continue
        if cutoff is not None and m.get("epoch", 0) >= cutoff:
            continue
        dm = dmeta.get(did, {})
        if tier is not None and dm.get("tier", 9) > tier:
            continue
        s = float(scores[ci])
        if did not in best or s > best[did][0]:
            best[did] = (s, int(ci))
        if len(best) >= pool:
            pass
    cand = sorted(best.items(), key=lambda kv: -kv[1][0])
    nov = load_novelty() if novel else {}
    if novel:
        # surface NEW developments over restatements: blend relevance with first-story
        # novelty (dedups echoes that are individually relevant). Take a relevance pool
        # first so we don't promote novel-but-irrelevant docs.
        cpool = cand[: max(k * 6, 40)]
        cand = sorted(cpool, key=lambda kv: -((1 - nov_w) * kv[1][0] + nov_w * nov.get(kv[0], 0.0)))
    if diverse:
        # MMR: cover the question's FACETS, not just its single strongest facet.
        # Greedily pick the doc maximising  lam*relevance - (1-lam)*max-similarity
        # to the already-picked set (doc = its best-matching chunk vector). Once a
        # facet is represented, its near-duplicates are penalised and a NEW facet
        # surfaces. Redundancy is measured in embedding space, so it spreads across
        # sub-topics rather than restating the top one.
        pooln = min(len(cand), max(k * 6, 40))
        cpool = cand[:pooln]
        V = np.stack([ch[ci] for _, (_, ci) in cpool])      # (pooln, d) unit vectors
        rel = np.array([s for _, (s, _) in cpool])
        picked, remaining = [], list(range(len(cpool)))
        while remaining and len(picked) < k:
            if not picked:
                j = int(np.argmax(rel[remaining]))
            else:
                sub = V[remaining] @ V[[p for p in picked]].T   # sims to picked
                red = sub.max(axis=1)
                mmr = lam * rel[remaining] - (1 - lam) * red
                j = int(np.argmax(mmr))
            picked.append(remaining.pop(j))
        ranked = [cpool[p] for p in picked]
    else:
        ranked = cand[:k]
    out = []
    for did, (s, ci) in ranked:
        dm = dmeta.get(did, {})
        out.append({
            "doc_id": did, "score": round(s, 4), "src": dm.get("src", "?"),
            "dom": dm.get("dom", "?"), "tier": dm.get("tier", 9),
            "date": dm.get("date", "?"), "title": dm.get("title", "?"),
            "snippet": texts[ci], "cidx": ci,
            "novelty": round(nov.get(did, float("nan")), 3) if novel else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--before")
    ap.add_argument("--exclude")
    ap.add_argument("--tier", type=int)
    ap.add_argument("--corroborate", action="store_true")
    ap.add_argument("--min-score", type=float, default=0.45,
                    help="corroborate: cosine support threshold")
    ap.add_argument("--min-cover", type=float, default=0.30,
                    help="corroborate: fraction of the claim's salient terms the "
                         "passage must contain (claim-specific, not just topical)")
    ap.add_argument("--diverse", action="store_true",
                    help="facet-coverage retrieval (MMR): spread results across the "
                         "question's sub-topics instead of clustering on the top one")
    ap.add_argument("--lam", type=float, default=0.5,
                    help="diverse: relevance vs diversity trade-off (1=pure relevance)")
    ap.add_argument("--novel", action="store_true",
                    help="coverage: surface NEW developments (first-story novelty) over echoes")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    # corroborate needs a wider net to count distinct sources
    k = max(a.k, 20) if a.corroborate else a.k
    res = retrieve(a.query, k=k, before=a.before, exclude=a.exclude, tier=a.tier,
                   diverse=a.diverse, lam=a.lam, novel=a.novel)

    if a.corroborate:
        terms = salient_terms(a.query)
        # a passage CONFIRMS only if it is both cosine-near AND lexically covers the
        # claim's salient terms — this is what separates independent confirmation of
        # the SPECIFIC proposition from a topically-adjacent neighbour.
        confirm, topical = [], []
        for r in res:
            if r["score"] < a.min_score:
                continue
            cov = term_coverage(terms, r["snippet"])
            r = {**r, "cover": round(cov, 2)}
            (confirm if cov >= a.min_cover else topical).append(r)
        by_dom = {}
        for r in confirm:
            by_dom.setdefault(r["dom"], []).append(r)
        n_src = len(by_dom)
        verdict = ("corroborated (>=2 independent sources confirm the specific claim)" if n_src >= 2
                   else "single-source" if n_src == 1 else "no source confirms the specific claim")
        payload = {
            "query": a.query, "min_score": a.min_score, "min_cover": a.min_cover,
            "distinct_sources": n_src, "confirming_docs": len(confirm),
            "topical_only_docs": len(topical), "verdict": verdict,
            "sources": [{"dom": d, "n": len(v), "cover": v[0]["cover"], "top": v[0]["title"][:70],
                         "doc_id": v[0]["doc_id"], "date": v[0]["date"], "tier": v[0]["tier"]}
                        for d, v in sorted(by_dom.items(), key=lambda kv: -len(kv[1]))],
            "topical_only": [{"dom": r["dom"], "cover": r["cover"], "title": r["title"][:60],
                              "doc_id": r["doc_id"]} for r in topical[:6]],
        }
        if a.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"CORROBORATION — {a.query!r}  (cosine>={a.min_score}, cover>={a.min_cover})")
            print(f"  {verdict}")
            print(f"  {n_src} confirming source(s), {len(confirm)} passage(s); "
                  f"{len(topical)} topical-only (cosine-near but claim not covered)")
            for s in payload["sources"]:
                print(f"  CONFIRM [{s['dom']}] t{s['tier']} cover={s['cover']} x{s['n']}  {s['date']}  {s['top']}  ({s['doc_id'][:8]})")
            for r in payload["topical_only"]:
                print(f"  topical [{r['dom']}] cover={r['cover']}  {r['title']}  ({r['doc_id'][:8]})")
        return

    # intra-list redundancy: mean pairwise cosine among returned docs (lower = more
    # facet spread). Makes the coverage objective measurable, not asserted.
    redun = None
    if len(res) > 1:
        V = np.stack([load()[0][r["cidx"]] for r in res])
        S = V @ V.T
        iu = np.triu_indices(len(res), k=1)
        redun = float(S[iu].mean())

    if a.json:
        print(json.dumps({"redundancy": redun, "results": res}, indent=2))
    else:
        mode = ("NOVEL (first-story coverage)" if a.novel else
                "DIVERSE (facet-coverage)" if a.diverse else "TOPICAL")
        meannov = np.nanmean([r["novelty"] for r in res]) if a.novel else None
        print(f"{mode} — {a.query!r}"
              + (f"  (before {a.before})" if a.before else "")
              + (f"  (tier<={a.tier})" if a.tier else "")
              + (f"   intra-list redundancy={redun:.3f}" if redun is not None else "")
              + (f"   mean novelty={meannov:.3f}" if meannov is not None else ""))
        for i, r in enumerate(res, 1):
            nv = f"  nov={r['novelty']:.2f}" if a.novel and r["novelty"] == r["novelty"] else ""
            print(f"\n[{i}] {r['score']:.3f}{nv}  [{r['src']}] t{r['tier']}  {r['date']}  ({r['doc_id'][:8]})")
            print(f"    {r['title'][:90]}")
            print(f"    {' '.join(r['snippet'].split()[:50])}")


if __name__ == "__main__":
    main()
