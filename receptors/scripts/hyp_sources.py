# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Hypothesis SOURCES for the driver-discovery eval.

The consolidated retrieval finding (RETRIEVAL_FINDINGS.md) is that driver discovery
is a *process* — decompose the effect into upstream causal hypotheses, cosine each,
union the novel drivers, gate-verify — and that the LLM step "forming the right
upstream questions" is what creates the value. But the tracked pipeline only ever
seeds those hypotheses from the cosine top-5 of the effect phrase (gen_hyde_input.py),
i.e. the very baseline neighbourhood it is meant to beat. That biases hypothesis
generation back toward the topical prior — the original v3 failure mode.

This module supplies the four hypothesis-generation CONTEXTS the discovery eval
compares. Every arm goes through the same downstream machinery (LLM distils queries
-> cosine each -> union -> rerank -> gate); the ONLY thing that varies is the context
the generator is allowed to condition on:

  A  prior    effect phrase only                         (world-knowledge floor)
  B  cosine5  effect + cosine top-5 snippets             (the CURRENT pipeline)
  C  graph    effect + cause_entities graph drivers       (endogenous traversal)
  D  hybrid   union of B and C                             (both)

Arm C is the one the recent findings never fairly tested: it uses the free
cause_entities graph (a doc is ABOUT entities, CAUSED-BY cause_entities) to name
drivers whose vocabulary does NOT overlap the effect — exactly what cosine-on-effect
is structurally blind to. See causal_hypotheses.py for the standalone CLI version of
the same traversal.

Loads the fact/doc-level substrate shared by the other receptor scripts:
  data/docs.jsonl        per-doc records: doc_id, entities, cause_entities,
                         published_epoch (+ title/predicate/confidence)
  data/embeddings.npy    doc embeddings aligned to docs.jsonl
Both are honoured under RECEPTORS_DATA_DIR so an alternative substrate (e.g.
data_bge_base) can be swapped in unchanged.
"""
import os, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
D = _ROOT / os.environ.get("RECEPTORS_DATA_DIR", "data")

_state = {}


def _load():
    """Lazy-load docs + normalised embeddings once."""
    if _state:
        return _state
    docs = [json.loads(l) for l in (D / "docs.jsonl").read_text().splitlines() if l.strip()]
    E = np.load(D / "embeddings.npy").astype(np.float32)
    E /= (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    epoch = np.array([d.get("published_epoch", 0) for d in docs])
    ents = [set(d.get("entities") or []) for d in docs]
    cents = [set(d.get("cause_entities") or []) for d in docs]
    N = len(docs)
    df = Counter()
    for s in cents:
        for e in s:
            df[e] += 1
    idf = {e: np.log(N / c) for e, c in df.items()}
    about = defaultdict(list)
    for i, s in enumerate(ents):
        for e in s:
            about[e].append(i)
    _state.update(docs=docs, E=E, epoch=epoch, ents=ents, cents=cents,
                  idf=idf, about=about, N=N)
    return _state


def _embed(texts):
    from fastembed import TextEmbedding
    model_name = os.environ.get("RECEPTORS_EMBED_MODEL",
                                "sentence-transformers/all-MiniLM-L6-v2")
    prefix = os.environ.get("RECEPTORS_QUERY_PREFIX", "")
    if prefix:
        texts = [prefix + t for t in texts]
    if "_model" not in _state:
        _state["_model"] = TextEmbedding(model_name=model_name)
    V = np.array(list(_state["_model"].embed(texts)), dtype=np.float32)
    V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
    return V


def effect_docs(effect, top=25, before_epoch=None):
    """Top cosine docs for the effect phrase — the shared 'baseline neighbourhood'."""
    s = _load()
    qv = _embed([effect])[0]
    rel = s["E"] @ qv
    order = np.argsort(-rel)
    if before_epoch is not None:
        order = [i for i in order if s["epoch"][i] < before_epoch]
    return list(order[:top]), rel


def graph_drivers(effect, top_effect=25, k=10, min_idf=0.5, before_epoch=None):
    """Arm C substrate: traverse the cause_entities graph to name UPSTREAM DRIVERS
    of the effect, ranked by (effect-doc relevance x IDF) mass. Mirrors
    causal_hypotheses.py. Returns [{entity, mass, upstream:[{doc_id,title}]}], the
    PRIMARY driver first, RIVALS/confounds after — the raw material an LLM (or the
    template composer) turns into upstream hypothesis queries.
    """
    s = _load()
    top, rel = effect_docs(effect, top=top_effect, before_epoch=before_epoch)
    eff_epoch = float(np.median(s["epoch"][top])) if top else 0.0

    mass = defaultdict(float)
    for i in top:
        r = max(float(rel[i]), 0.0)
        for e in s["cents"][i]:
            mass[e] += r * s["idf"].get(e, 0.0)
    drivers = sorted(mass.items(), key=lambda kv: -kv[1])
    drivers = [(e, m) for e, m in drivers if s["idf"].get(e, 0) > min_idf][:k]

    docs = s["docs"]
    out = []
    tot = sum(m for _, m in drivers) or 1.0
    for e, m in drivers:
        cand = [i for i in s["about"].get(e, [])
                if s["epoch"][i] < (eff_epoch or s["epoch"][i] + 1)]
        cand.sort(key=lambda i: -float(rel[i]))
        up = [{"doc_id": docs[i]["doc_id"], "title": docs[i].get("title", "?")}
              for i in cand[:3]]
        out.append({"entity": e, "mass": round(m / tot, 3), "upstream": up})
    return out


def cosine_snippets(effect, k=5, before_epoch=None):
    """Arm B substrate: the current pipeline's context — cosine top-k docs of the
    effect phrase, with the salient entities each carries (used by the template
    composer; the LLM composer just reads the titles/snippets)."""
    s = _load()
    top, rel = effect_docs(effect, top=k, before_epoch=before_epoch)
    docs = s["docs"]
    return [{"doc_id": docs[i]["doc_id"], "title": docs[i].get("title", "?"),
             "entities": sorted(s["ents"][i]), "rel": round(float(rel[i]), 3)}
            for i in top]


# ---- arm context assembly ------------------------------------------------------

ARMS = {
    "A": "prior",     # effect phrase only
    "B": "cosine5",   # + cosine top-5 snippets (current pipeline)
    "C": "graph",     # + cause_entities graph drivers (endogenous traversal)
    "D": "hybrid",    # union of B and C
}


def assemble_context(arm, effect, before_epoch=None):
    """Build the JSON context object handed to the hypothesis generator for `arm`.
    Identical downstream; only this differs between arms."""
    arm = arm.upper()
    ctx = {"effect": effect, "arm": arm, "source": ARMS[arm]}
    if arm in ("B", "D"):
        ctx["cosine_snippets"] = cosine_snippets(effect, k=5, before_epoch=before_epoch)
    if arm in ("C", "D"):
        ctx["graph_drivers"] = graph_drivers(effect, before_epoch=before_epoch)
    return ctx


if __name__ == "__main__":
    import sys
    eff = sys.argv[2] if len(sys.argv) > 2 else "heatwave lifts EM appliance makers"
    arm = sys.argv[1] if len(sys.argv) > 1 else "C"
    print(json.dumps(assemble_context(arm, eff), indent=2))
