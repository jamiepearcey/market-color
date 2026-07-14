#!/usr/bin/env python3
"""Three-arm retrieval experiment: dense vs uniform PPR vs query-conditioned PPR.

Builds a sparse fact–entity graph from facts.parquet (membership edges from
`entities`, causal edges from `cause_entities`, confidence as base weight) and
runs Personalized PageRank seeded by dense vector search over fact claims
(Qdrant `market_color_facts`, built by graph_experiment.py build).

Arms:
  dense  vector search only (baseline)
  ppr    dense-seeded PPR, uniform edge weights   (~HippoRAG)
  qppr   dense-seeded PPR, query-conditioned typed edge weights

Run (same runtime as run_pipeline.sh):
  uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed \
      --with numpy --with pyarrow python ppr_experiment.py search --query '...'
  ... python ppr_experiment.py eval --cases eval/ppr_cases.jsonl --k 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

import index_corpus as ic
from graph_experiment import FACTS_COLLECTION, _token_jaccard, norm_entity

HERE = Path(__file__).resolve().parent
DEFAULT_PARQUET = HERE / "facts_work" / "facts.parquet"

# ---- query-conditioned weighting ------------------------------------------ #
CAUSAL_MARKERS = ("why", "despite", "driv", "because", "cause", "behind",
                  "impact", "effect", "explain", "reason", "spill", "knock-on",
                  "response to", "leading to", "consequence")

# per-intent multipliers: (causal-edge mult, per-predicate mult)
INTENT_WEIGHTS: dict[str, tuple[float, dict[str, float]]] = {
    "causal_explanation": (3.0, {
        "policy_action": 1.6, "supply_change": 1.6, "demand_change": 1.6,
        "production_change": 1.5, "sanction": 1.6, "event": 1.4,
        "price_move": 1.2, "deal_or_contract": 1.0, "corporate_action": 1.0,
        "forecast": 0.7, "statement": 0.6, "other": 0.8,
    }),
    "lookup": (1.0, {}),
}


def classify_intent(query: str) -> str:
    q = query.lower()
    return "causal_explanation" if any(m in q for m in CAUSAL_MARKERS) else "lookup"


# ---- graph ------------------------------------------------------------------ #
class FactGraph:
    """Fact + entity nodes; membership and causal edges with typed base weights.

    hub_gamma damps edges into high-frequency entity hubs: edge weight is scaled
    by freq(entity)^-gamma, so mass routed via 'united states' (372 facts) does
    not drown a 2-fact entity like 'kiku'. gamma=0 disables.
    """

    def __init__(self, facts: list[dict[str, Any]], hub_gamma: float = 0.0):
        self.facts = facts
        ents: set[str] = set()
        for f in facts:
            ents.update(f["entities"] or [])
            ents.update(f["cause_entities"] or [])
        self.entities = sorted(ents)
        self.ent_idx = {e: i for i, e in enumerate(self.entities)}
        self.n_facts = len(facts)
        self.n = self.n_facts + len(self.entities)
        self.fact_by_id = {f["fact_id"]: i for i, f in enumerate(facts)}

        freq = {e: 0 for e in self.entities}
        for f in facts:
            for e in set((f["entities"] or []) + (f["cause_entities"] or [])):
                freq[e] += 1

        # edge lists (both directions added at weighting time)
        src, dst, base, kind, pred = [], [], [], [], []
        for i, f in enumerate(facts):
            conf = float(f.get("confidence") or 0.5)
            p = f.get("predicate") or "other"
            for e in set(f["entities"] or []):
                j = self.n_facts + self.ent_idx[e]
                w = conf * (freq[e] ** -hub_gamma if hub_gamma else 1.0)
                src.append(i); dst.append(j); base.append(w); kind.append(0); pred.append(p)
            for e in set(f["cause_entities"] or []):
                j = self.n_facts + self.ent_idx[e]
                w = conf * (freq[e] ** -hub_gamma if hub_gamma else 1.0)
                src.append(i); dst.append(j); base.append(w); kind.append(1); pred.append(p)
        self.src = np.asarray(src, dtype=np.int32)
        self.dst = np.asarray(dst, dtype=np.int32)
        self.base = np.asarray(base, dtype=np.float32)
        self.kind = np.asarray(kind, dtype=np.int8)  # 0 membership, 1 causal
        self.pred = pred

    def edge_weights(self, intent: str | None) -> np.ndarray:
        w = self.base.copy()
        if intent is not None:
            causal_mult, pred_mult = INTENT_WEIGHTS[intent]
            w[self.kind == 1] *= causal_mult
            if pred_mult:
                mult = np.asarray([pred_mult.get(p, 1.0) for p in self.pred],
                                  dtype=np.float32)
                w *= mult
        return w

    def ppr(self, seed: dict[int, float], intent: str | None,
            alpha: float = 0.4, iters: int = 40) -> np.ndarray:
        """Personalized PageRank; seed = {fact index: mass}. alpha = restart prob."""
        w = self.edge_weights(intent)
        # symmetrise
        s_all = np.concatenate([self.src, self.dst])
        d_all = np.concatenate([self.dst, self.src])
        w_all = np.concatenate([w, w])
        out = np.zeros(self.n, dtype=np.float32)
        np.add.at(out, s_all, w_all)
        w_norm = w_all / np.maximum(out[s_all], 1e-12)

        s = np.zeros(self.n, dtype=np.float32)
        for i, m in seed.items():
            s[i] = m
        s /= max(s.sum(), 1e-12)
        r = s.copy()
        for _ in range(iters):
            r_new = alpha * s
            np.add.at(r_new, d_all, (1.0 - alpha) * w_norm * r[s_all])
            if np.abs(r_new - r).sum() < 1e-9:
                r = r_new
                break
            r = r_new
        return r


def load_facts(parquet: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    rows = pq.read_table(parquet).to_pylist()
    for f in rows:
        f["entities"] = [norm_entity(e) for e in (f["entities"] or [])]
        f["cause_entities"] = [norm_entity(e) for e in (f["cause_entities"] or [])]
    return rows


# ---- retrieval arms --------------------------------------------------------- #
class Retriever:
    def __init__(self, args: argparse.Namespace):
        self.graph = FactGraph(load_facts(Path(args.parquet)),
                               hub_gamma=getattr(args, "hub_gamma", 0.0))
        self.client = ic._client(args.qdrant_url, None)
        self.provider, self.model = args.embedding_provider, args.embedding_model
        self.seed_k = args.seed_k

    def _story_clusters(self) -> dict[str, int]:
        """Lazy fact_id -> cluster id. Cross-source paraphrases of the same story
        (claim cosine >= 0.87 AND >= 1 shared entity) collapse into one cluster,
        so deduped retrieval spends top-k slots on evidence diversity."""
        if not hasattr(self, "_cluster_of"):
            facts = self.graph.facts
            vecs = []
            for chunk in ic._chunked([f["claim"] or "" for f in facts], 256):
                vecs.extend(ic.embed(self.provider, self.model, chunk,
                                     ic.DEFAULT_OLLAMA_URL, None))
            mat = np.asarray(vecs, dtype="float32")
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            parent = list(range(len(facts)))

            def find(i: int) -> int:
                while parent[i] != i:
                    parent[i] = parent[parent[i]]
                    i = parent[i]
                return i

            ents = [set(f["entities"] or []) for f in facts]
            step = 512
            for s in range(0, len(facts), step):
                sims = mat[s:s + step] @ mat.T
                rows, cols = np.nonzero(sims >= 0.87)
                for r, c in zip(rows.tolist(), cols.tolist()):
                    i, j = s + r, c
                    if i < j and (ents[i] & ents[j]):
                        ri, rj = find(i), find(j)
                        if ri != rj:
                            parent[rj] = ri
            self._cluster_of = {facts[i]["fact_id"]: find(i) for i in range(len(facts))}
            n_clusters = len(set(self._cluster_of.values()))
            print(f"[dedup] {len(facts)} facts -> {n_clusters} story clusters",
                  file=sys.stderr)
        return self._cluster_of

    def dense(self, query: str, k: int, dedup: bool = False) -> list[tuple[float, dict]]:
        from qdrant_client.http import models
        qv = ic.embed(self.provider, self.model, [query], ic.DEFAULT_OLLAMA_URL, None)[0]
        limit = max(k, self.seed_k) * (4 if dedup else 1)
        pts = self.client.query_points(
            collection_name=FACTS_COLLECTION, query=qv, limit=limit,
            with_payload=True,
            search_params=models.SearchParams(
                hnsw_ef=128, quantization=models.QuantizationSearchParams(
                    rescore=True, oversampling=2.0))).points
        hits = [(p.score, p.payload or {}) for p in pts]
        if not dedup:
            return hits
        clusters = self._story_clusters()
        out, seen = [], set()
        for s, pl in hits:
            cid = clusters.get(pl.get("fact_id"))
            if cid in seen:
                continue
            seen.add(cid)
            out.append((s, pl))
            if len(out) >= max(k, self.seed_k):
                break
        return out

    def _ppr_rank(self, query: str, k: int, intent: str | None) -> list[tuple[float, dict]]:
        hits = self.dense(query, self.seed_k)
        seed: dict[int, float] = {}
        for score, pl in hits[: self.seed_k]:
            i = self.graph.fact_by_id.get(pl.get("fact_id"))
            if i is not None:
                seed[i] = max(float(score), 0.0)
        if not seed:
            return hits[:k]
        r = self.graph.ppr(seed, intent)
        order = np.argsort(-r[: self.graph.n_facts])[:k]
        return [(float(r[i]), self.graph.facts[i]) for i in order if r[i] > 0]

    def _cause_index(self):
        """Lazy: embeddings of every causal fact's `cause` text, for the
        downstream (effect-direction) hop — facts citing a similar cause."""
        if not hasattr(self, "_cause_mat"):
            idx = [i for i, f in enumerate(self.graph.facts)
                   if f.get("cause") and str(f["cause"]).lower() not in ("none", "null")]
            vecs = []
            for chunk in ic._chunked([str(self.graph.facts[i]["cause"]) for i in idx], 256):
                vecs.extend(ic.embed(self.provider, self.model, chunk,
                                     ic.DEFAULT_OLLAMA_URL, None))
            mat = np.asarray(vecs, dtype="float32")
            mat /= np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9
            self._cause_mat, self._cause_idx = mat, idx
        return self._cause_mat, self._cause_idx

    def iterative(self, query: str, k: int, keep: int = 4, per_hop: int = 5,
                  depth: int = 2, decay: float = 0.8,
                  downstream: bool = True) -> list[tuple[float, dict]]:
        """Cause-guided iterative retrieval: walk the causal chain in QUERY space.
        Each hop embeds the frontier facts' raw `cause` text as a fresh dense
        query, so hop-2 facts are scored semantically against their own hop
        (the signal one-shot PPR lacks). Path score = parent * decay * hop sim.
        Final list: top `keep` dense facts + best walked facts, dedup, to k."""
        from qdrant_client.http import models
        dense_hits = self.dense(query, k)
        walked: dict[str, tuple[float, dict]] = {}
        frontier = [(float(s), pl) for s, pl in dense_hits[:keep]]
        for _hop in range(depth):
            subqs = [(ps, pl, str(pl["cause"])) for ps, pl in frontier
                     if pl.get("cause") and str(pl.get("cause")).lower() not in ("none", "null")]
            new_frontier: list[tuple[float, dict]] = []

            if downstream and frontier:  # effect direction: facts citing a similar cause
                cmat, cidx = self._cause_index()
                cl_vecs = ic.embed(self.provider, self.model,
                                   [str(pl.get("claim") or "") for _, pl in frontier],
                                   ic.DEFAULT_OLLAMA_URL, None)
                q = np.asarray(cl_vecs, dtype="float32")
                q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
                sims = q @ cmat.T
                for (ps, parent), row in zip(frontier, sims, strict=True):
                    p_ents = set(parent.get("entities") or [])
                    for j in np.argsort(-row)[:per_hop * 2]:
                        if row[j] < 0.4:
                            break
                        f = self.graph.facts[cidx[int(j)]]
                        if f.get("fact_id") == parent.get("fact_id"):
                            continue
                        # entity overlap between parent and the citing fact
                        # corroborates that its cause refers to the parent event
                        ov = len(p_ents & set((f.get("entities") or [])
                                              + (f.get("cause_entities") or [])))
                        sim = float(row[j]) * (1.0 + 0.3 * min(ov, 3))
                        if ov == 0 and float(row[j]) < 0.5:
                            continue
                        sc = ps * decay * sim
                        if sc > walked.get(f["fact_id"], (0.0,))[0]:
                            walked[f["fact_id"]] = (sc, f)
                            new_frontier.append((sc, f))

            if not subqs and not new_frontier:
                break
            vecs = ic.embed(self.provider, self.model, [c for _, _, c in subqs],
                            ic.DEFAULT_OLLAMA_URL, None) if subqs else []
            for (ps, parent, _), qv in zip(subqs, vecs, strict=True):
                pts = self.client.query_points(
                    collection_name=FACTS_COLLECTION, query=qv, limit=per_hop,
                    with_payload=True,
                    search_params=models.SearchParams(
                        hnsw_ef=128, quantization=models.QuantizationSearchParams(
                            rescore=True, oversampling=2.0))).points
                for p in pts:
                    f = p.payload or {}
                    if f.get("fact_id") == parent.get("fact_id"):
                        continue
                    # novelty guard: a hop that lands on a paraphrase of the
                    # parent (generic cause text) adds no new chain evidence
                    if _token_jaccard((f.get("claim") or "").lower(),
                                      (parent.get("claim") or "").lower()) > 0.5:
                        continue
                    sc = ps * decay * max(float(p.score), 0.0)
                    if sc > walked.get(f["fact_id"], (0.0,))[0]:
                        walked[f["fact_id"]] = (sc, f)
                        new_frontier.append((sc, f))
            new_frontier.sort(key=lambda t: -t[0])
            frontier = new_frontier[:keep]

        out = list(dense_hits[:keep])
        seen = {pl.get("fact_id") for _, pl in out}
        for sc, f in sorted(walked.values(), key=lambda t: -t[0]):
            if f["fact_id"] not in seen:
                out.append((sc, f))
                seen.add(f["fact_id"])
            if len(out) >= k:
                break
        for s, pl in dense_hits[keep:]:  # backfill if the walk found little
            if len(out) >= k:
                break
            if pl.get("fact_id") not in seen:
                out.append((float(s), pl))
                seen.add(pl.get("fact_id"))
        return out[:k]

    def siblings(self, query: str, k: int, keep: int = 4, per_doc: int = 3,
                 decay: float = 0.7, combine_iter: bool = False) -> list[tuple[float, dict]]:
        """Small-to-big over facts: expand top dense hits with other facts from
        the SAME source document. Articles carry their own background context;
        the decomposer turns it into sibling facts whose claims share no
        vocabulary with the query — dense skips them, doc structure keeps them.
        Causal siblings first (background = the connective tissue)."""
        if not hasattr(self, "_by_doc"):
            self._by_doc = {}
            for f in self.graph.facts:
                self._by_doc.setdefault(f.get("doc_id"), []).append(f)
        dense_hits = self.dense(query, k)
        out = list(dense_hits[:keep])
        seen = {pl.get("fact_id") for _, pl in out}
        pool: list[tuple[float, dict]] = []
        for ps, parent in dense_hits[:keep]:
            sibs = [f for f in self._by_doc.get(parent.get("doc_id"), [])
                    if f["fact_id"] not in seen]
            # causal siblings first, then by confidence
            sibs.sort(key=lambda f: (not f.get("cause_entities"),
                                     -(float(f.get("confidence") or 0.0))))
            for f in sibs[:per_doc]:
                pool.append((float(ps) * decay * float(f.get("confidence") or 0.5), f))
        if combine_iter:
            walked = self.iterative(query, k)
            pool.extend((sc, f) for sc, f in walked[len(dense_hits[:4]):])
        for sc, f in sorted(pool, key=lambda t: -t[0]):
            if f["fact_id"] in seen:
                continue
            out.append((sc, f))
            seen.add(f["fact_id"])
            if len(out) >= k:
                break
        for s, pl in dense_hits[keep:]:
            if len(out) >= k:
                break
            if pl.get("fact_id") not in seen:
                out.append((float(s), pl))
                seen.add(pl.get("fact_id"))
        return out[:k]

    def assemble(self, query: str, k: int, keep: int = 4,
                 per_primary: int = 2, dedup: bool = False) -> list[tuple[float, dict]]:
        """Assembly-time attachment (the router made concrete): dense primaries
        keep their slots; a primary whose cause points OUTSIDE its own fact
        (unresolved upstream hop) gets attached context — the best cross-doc
        walked fact for its cause + the best causal sibling covering it.
        Total budget still k facts; dense tail backfills unused slots."""
        from qdrant_client.http import models
        if not hasattr(self, "_by_doc"):
            self._by_doc = {}
            for f in self.graph.facts:
                self._by_doc.setdefault(f.get("doc_id"), []).append(f)
        dense_hits = self.dense(query, k, dedup=dedup)
        primaries = dense_hits[:keep]
        out = list(primaries)
        seen = {pl.get("fact_id") for _, pl in out}

        # router trigger: which primaries carry an unresolved cause?
        triggered = []
        for ps, pl in primaries:
            cause_ents = set(pl.get("cause_entities") or []) - set(pl.get("entities") or [])
            if pl.get("cause") and str(pl["cause"]).lower() not in ("none", "null") and cause_ents:
                triggered.append((ps, pl, cause_ents))

        attachments: list[tuple[float, dict]] = []
        if triggered:
            vecs = ic.embed(self.provider, self.model,
                            [str(pl["cause"]) for _, pl, _ in triggered],
                            ic.DEFAULT_OLLAMA_URL, None)
            for (ps, parent, cause_ents), qv in zip(triggered, vecs, strict=True):
                got = 0
                # (a) best cross-doc walked fact for the cause
                pts = self.client.query_points(
                    collection_name=FACTS_COLLECTION, query=qv, limit=6,
                    with_payload=True).points
                for p in pts:
                    f = p.payload or {}
                    if (f.get("fact_id") in seen or f.get("doc_id") == parent.get("doc_id")
                            or _token_jaccard((f.get("claim") or "").lower(),
                                              (parent.get("claim") or "").lower()) > 0.5):
                        continue
                    attachments.append((ps * 0.8 * float(p.score), f))
                    seen.add(f["fact_id"])
                    got += 1
                    break
                # (b) best causal sibling covering the unresolved cause entities
                if got < per_primary:
                    sibs = [f for f in self._by_doc.get(parent.get("doc_id"), [])
                            if f["fact_id"] not in seen
                            and (cause_ents & set(f.get("entities") or []))]
                    sibs.sort(key=lambda f: -(float(f.get("confidence") or 0.0)))
                    if sibs:
                        f = sibs[0]
                        attachments.append((ps * 0.7, f))
                        seen.add(f["fact_id"])

        out.extend(attachments[: k - len(out)])
        for s, pl in dense_hits[keep:]:
            if len(out) >= k:
                break
            if pl.get("fact_id") not in seen:
                out.append((float(s), pl))
                seen.add(pl.get("fact_id"))
        return out[:k]

    def run(self, arm: str, query: str, k: int) -> list[tuple[float, dict]]:
        if arm.endswith("_dd"):  # deduped variant of any dense-based arm
            base = arm[:-3]
            if base == "dense":
                return self.dense(query, k, dedup=True)[:k]
            if base == "asm":
                return self.assemble(query, k, dedup=True)
            raise ValueError(arm)
        if arm == "dense":
            return self.dense(query, k)[:k]
        if arm == "ppr":
            return self._ppr_rank(query, k, intent=None)
        if arm == "qppr":
            return self._ppr_rank(query, k, intent=classify_intent(query))
        if arm == "iter":
            return self.iterative(query, k)
        if arm == "sib":
            return self.siblings(query, k)
        if arm == "sibiter":
            return self.siblings(query, k, combine_iter=True)
        if arm == "asm":
            return self.assemble(query, k)
        raise ValueError(arm)


# ---- commands --------------------------------------------------------------- #
def cmd_search(args: argparse.Namespace) -> int:
    rt = Retriever(args)
    for arm in args.arms.split(","):
        print(f"\n=== {arm} (intent={classify_intent(args.query) if arm == 'qppr' else '-'}) ===")
        for score, f in rt.run(arm, args.query, args.k):
            cause = f" ⟵ {f.get('cause')}" if f.get("cause") else ""
            print(f"  {score:.4f} ({f.get('predicate')}/{f.get('direction')}) "
                  f"{(f.get('claim') or '')[:100]}{cause}")
    return 0


def _group_hits(group: dict, top: list[dict]) -> list[int]:
    """All ranks (0-based) in top-k facts satisfying an evidence group."""
    ents = {norm_entity(e) for e in group.get("any_entities", [])}
    subs = [s.lower() for s in group.get("any_claim_contains", [])]
    hits = []
    for rank, f in enumerate(top):
        fe = set(f.get("entities") or [])
        text = f"{f.get('claim') or ''} {f.get('cause') or ''}".lower()
        if (ents and (fe & ents)) or (subs and any(s in text for s in subs)):
            hits.append(rank)
    return hits


def _match_case(case: dict, top: list[dict]) -> list[int | None]:
    """Assign one distinct fact per evidence group (greedy, scarcest group first).
    With case["distinct_docs"], assigned facts must also come from distinct doc_ids —
    the multi-hop condition: no single document may cover two hops."""
    hits = [_group_hits(g, top) for g in case["evidence"]]
    order = sorted(range(len(hits)), key=lambda i: len(hits[i]))
    used_facts: set[int] = set()
    used_docs: set[str] = set()
    ranks: list[int | None] = [None] * len(hits)
    for gi in order:
        for r in hits[gi]:
            doc = top[r].get("doc_id")
            if r in used_facts or (case.get("distinct_docs") and doc in used_docs):
                continue
            ranks[gi] = r
            used_facts.add(r)
            used_docs.add(doc)
            break
    return ranks


def cmd_eval(args: argparse.Namespace) -> int:
    rt = Retriever(args)
    cases = [json.loads(l) for l in open(args.cases) if l.strip()]
    arms = args.arms.split(",")
    per_arm: dict[str, list[dict]] = {a: [] for a in arms}
    for case in cases:
        k = int(case.get("k", args.k))
        for arm in arms:
            top = [f for _, f in rt.run(arm, case["query"], k)]
            ranks = _match_case(case, top)
            found = [r for r in ranks if r is not None]
            per_arm[arm].append({
                "id": case["id"],
                "recall": len(found) / len(case["evidence"]),
                "full": len(found) == len(case["evidence"]),
                "mrr": (1.0 / (1 + min(found))) if found else 0.0,
            })
        got = {a: per_arm[a][-1] for a in arms}
        line = "  ".join(f"{a}: R={got[a]['recall']:.2f}" for a in arms)
        print(f"[{case['id']:<36}] {line}", file=sys.stderr)

    summary = {}
    for arm in arms:
        rs = per_arm[arm]
        summary[arm] = {
            "evidence_recall@k": round(sum(r["recall"] for r in rs) / len(rs), 4),
            "full_coverage_rate": round(sum(r["full"] for r in rs) / len(rs), 4),
            "mrr": round(sum(r["mrr"] for r in rs) / len(rs), 4),
        }
    print(json.dumps({"cases": len(cases), "k": args.k, "arms": summary}, indent=2))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("search", "eval"):
        sp = sub.add_parser(name)
        sp.add_argument("--qdrant-url", default="http://localhost:6333")
        sp.add_argument("--embedding-provider", default=ic.DEFAULT_PROVIDER)
        sp.add_argument("--embedding-model", default=ic.DEFAULT_FASTEMBED_MODEL)
        sp.add_argument("--parquet", default=str(DEFAULT_PARQUET))
        sp.add_argument("--seed-k", type=int, default=8)
        sp.add_argument("--hub-gamma", type=float, default=0.0)
        sp.add_argument("--k", type=int, default=10)
        sp.add_argument("--arms", default="dense,ppr,qppr")
    sub.choices["search"].add_argument("--query", required=True)
    sub.choices["search"].set_defaults(func=cmd_search)
    sub.choices["eval"].add_argument("--cases", default=str(HERE / "eval" / "ppr_cases.jsonl"))
    sub.choices["eval"].set_defaults(func=cmd_eval)
    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
