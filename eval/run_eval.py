#!/usr/bin/env python3
"""market-color retrieval eval/backtest harness.

Replays the frozen cases in `eval/cases.jsonl` against the live Qdrant
`market_color` collection using *exactly* the same embedding + filter +
score-floor semantics as `index_corpus.py search` (it imports index_corpus
and calls its `embed`/`_client`/`_filter` directly — no shelling out).

Case kinds (one JSON object per line in cases.jsonl):
  retrieval  a desk-scoped query MUST surface a known in-corpus doc in the
             top `expect.in_top` hits at/above `min_score` (grounding).
  abstain    a query about a topic absent from the corpus window MUST return
             no hits at/above `min_score` => no_strong_match (abstention
             calibration — the anti-generalisation guard, README north star).
  asof       point-in-time correctness: the same query run with an `as_of`
             before the target doc's publish time must NOT surface it
             (expect.present=false, no look-ahead leakage), and with a later
             `as_of` MUST surface it (expect.present=true).

Run (from the repo root or anywhere):
  uv run --with 'qdrant-client>=1.15' --with fastembed --with 'duckdb>=1.0' \
    python eval/run_eval.py

Options:
  --only <id>        run a single case (repeatable)
  --k <n>            retrieval depth (default 8; never below a case's in_top)
  --qdrant-url URL   Qdrant endpoint (default http://localhost:6333)
  --cases PATH       cases file (default: cases.jsonl next to this script)
  --verbose          print top hits for every case, not just failures

Output: one PASS/FAIL line per case, then a JSON summary
  {total, passed, failed, by_kind, failures:[ids]}
Exit code 1 if any case fails.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # import the production retrieval code

import index_corpus as ic  # noqa: E402

DEFAULT_CASES = HERE / "cases.jsonl"
DEFAULT_K = 8


# --------------------------------------------------------------------------- #
# Case loading / validation
# --------------------------------------------------------------------------- #
VALID_KINDS = {"retrieval", "abstain", "asof"}
MATCHER_KEYS = {"any_url_contains", "any_source", "any_title_regex"}


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        case = json.loads(raw)
        cid, kind = case.get("id"), case.get("kind")
        if not cid or kind not in VALID_KINDS:
            raise ValueError(f"{path}:{lineno}: bad id/kind: {raw[:80]}")
        if cid in seen:
            raise ValueError(f"{path}:{lineno}: duplicate case id {cid!r}")
        seen.add(cid)
        if kind != "abstain":
            expect = case.get("expect") or {}
            if not (MATCHER_KEYS & set(expect)):
                raise ValueError(f"{path}:{lineno}: case {cid} has no matcher in expect")
            if kind == "asof":
                if "as_of" not in case or "present" not in expect:
                    raise ValueError(f"{path}:{lineno}: asof case {cid} needs as_of + expect.present")
        cases.append(case)
    return cases


# --------------------------------------------------------------------------- #
# Hit matching (union across the matchers present in expect)
# --------------------------------------------------------------------------- #
def hit_matches(hit: dict[str, Any], expect: dict[str, Any]) -> bool:
    url = (hit.get("url") or "").lower()
    for frag in expect.get("any_url_contains", []):
        if frag.lower() in url:
            return True
    source = hit.get("source_name") or ""
    for src in expect.get("any_source", []):
        if source == src:
            return True
    pattern = expect.get("any_title_regex")
    if pattern and re.search(pattern, hit.get("title") or ""):
        return True
    return False


# --------------------------------------------------------------------------- #
# Search — same semantics as index_corpus.run_search
# --------------------------------------------------------------------------- #
def search(client: Any, collection: str, qvec: list[float], desks: list[str],
           as_of: str | None, min_score: float, limit: int) -> list[dict[str, Any]]:
    from qdrant_client.http import models

    as_of_epoch = ic._epoch(as_of) if as_of else None
    resp = client.query_points(
        collection_name=collection,
        query=qvec,
        query_filter=ic._filter(None, None, desks, [], [], as_of_epoch),
        limit=limit,
        with_payload=True,
        with_vectors=False,
        search_params=models.SearchParams(
            hnsw_ef=ic.DEFAULT_HNSW_EF,
            quantization=models.QuantizationSearchParams(
                rescore=True, oversampling=ic.DEFAULT_OVERSAMPLING),
        ),
    )
    return [{
        "score": p.score,
        "published_date": (p.payload or {}).get("published_date"),
        "source_name": (p.payload or {}).get("source_name"),
        "title": (p.payload or {}).get("title"),
        "url": (p.payload or {}).get("url"),
    } for p in resp.points if p.score >= min_score]


def _desks(case: dict[str, Any]) -> list[str]:
    desk = case.get("desk")
    if not desk:
        return []
    return desk if isinstance(desk, list) else [desk]


def _fmt_hits(hits: list[dict[str, Any]], n: int = 5) -> str:
    if not hits:
        return "      (no hits above min_score => no_strong_match)"
    return "\n".join(
        f"      #{i + 1} {h['score']:.3f} [{h['published_date']}] {h['source_name']}: {h['title']}"
        for i, h in enumerate(hits[:n]))


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_case(case: dict[str, Any], hits: list[dict[str, Any]]) -> tuple[bool, str]:
    kind = case["kind"]
    if kind == "abstain":
        if not hits:
            return True, "no_strong_match (abstained correctly)"
        top = hits[0]
        return False, (f"expected no_strong_match but got {len(hits)} hit(s); "
                       f"top: {top['score']:.3f} {top['source_name']}: {top['title']}")

    expect = case.get("expect") or {}
    in_top = int(expect.get("in_top", 5))
    window = hits[:in_top]
    matched = [i + 1 for i, h in enumerate(window) if hit_matches(h, expect)]

    if kind == "retrieval":
        if matched:
            return True, f"matched at rank {matched[0]}"
        return False, f"expected doc not in top {in_top} ({len(hits)} hits above floor)"

    # asof: presence/absence of the target doc under the as_of cutoff
    present = bool(expect.get("present"))
    if present:
        if matched:
            return True, f"doc present at rank {matched[0]} (as_of {case['as_of']})"
        return False, f"doc expected present under as_of {case['as_of']} but not in top {in_top}"
    if not matched:
        return True, f"doc correctly excluded by as_of {case['as_of']}"
    return False, (f"look-ahead leak: doc published after as_of {case['as_of']} "
                   f"surfaced at rank {matched[0]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(DEFAULT_CASES))
    ap.add_argument("--only", action="append", default=[],
                    help="run only the case(s) with this id (repeatable)")
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="retrieval depth (never below a case's expect.in_top)")
    ap.add_argument("--qdrant-url", default="http://localhost:6333")
    ap.add_argument("--qdrant-api-key")
    ap.add_argument("--collection", default=ic.DEFAULT_COLLECTION)
    ap.add_argument("--embedding-provider", default=ic.DEFAULT_PROVIDER)
    ap.add_argument("--embedding-model", default=ic.DEFAULT_FASTEMBED_MODEL)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = load_cases(Path(args.cases))
    if args.only:
        missing = set(args.only) - {c["id"] for c in cases}
        if missing:
            print(f"error: unknown case id(s): {sorted(missing)}", file=sys.stderr)
            return 2
        cases = [c for c in cases if c["id"] in args.only]
    if not cases:
        print("error: no cases to run", file=sys.stderr)
        return 2

    client = ic._client(args.qdrant_url, args.qdrant_api_key)

    # Embed all queries in one batch (fastembed model loads once).
    queries = [c["query"] for c in cases]
    vecs = ic.embed(args.embedding_provider, args.embedding_model, queries,
                    ic.DEFAULT_OLLAMA_URL, None)

    passed, failed = 0, 0
    by_kind: dict[str, dict[str, int]] = {}
    failures: list[str] = []
    for case, qvec in zip(cases, vecs, strict=True):
        expect = case.get("expect") or {}
        limit = max(args.k, int(expect.get("in_top", 5)))
        min_score = float(case.get("min_score", 0.45))
        hits = search(client, args.collection, qvec, _desks(case),
                      case.get("as_of"), min_score, limit)
        ok, why = evaluate_case(case, hits)
        bk = by_kind.setdefault(case["kind"], {"passed": 0, "failed": 0})
        if ok:
            passed += 1
            bk["passed"] += 1
        else:
            failed += 1
            bk["failed"] += 1
            failures.append(case["id"])
        status = "PASS" if ok else "FAIL"
        desk = ",".join(_desks(case)) or "-"
        print(f"{status} {case['id']:<28} [{case['kind']}/{desk}] {why}")
        if args.verbose or not ok:
            print(f"      query: {case['query']!r}"
                  + (f" as_of={case['as_of']}" if case.get("as_of") else ""))
            print(_fmt_hits(hits))

    summary = {"total": passed + failed, "passed": passed, "failed": failed,
               "by_kind": by_kind, "failures": failures}
    print(json.dumps(summary, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
