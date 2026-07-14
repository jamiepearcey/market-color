# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp", "numpy", "fastembed>=0.3"]
# ///
"""MCP server exposing the receptors toolkit + the report-process tools.

Register (project scope): receptors/.mcp.json — new sessions attach automatically.
All tools are pure embedding-space ops (BLAS) + the epistemic gate. The semantic
causal graph is PRECOMPUTED offline (build_semantic_graph.py) — at inference
causal_chain only runs a sparse power iteration; no per-call graph construction,
no per-call LLM inference.
"""
import json, subprocess, sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import receptors_tools as rt

mcp = FastMCP("receptors")


@mcp.tool()
def causal_find(query: str, k: int = 8, before: str = "", exclude: str = "") -> str:
    """Find candidate UPSTREAM CAUSES of an effect/phenomenon via the learned
    transport operator W (embedding traversal, no graph). Best first-stage cause
    finder (beats cosine +34-53% recall). Results are CANDIDATES — verify with
    corroborate/the gate before asserting. before=YYYY-MM-DD point-in-time;
    exclude=comma-separated doc_ids already seen."""
    return json.dumps(rt.causal_find(query, k=k, before=before or None,
                                     exclude=exclude or None), indent=2)


@mcp.tool()
def causal_chain(query: str, k: int = 8, before: str = "", exclude: str = "") -> str:
    """MULTI-HOP cause finder: like causal_find but propagates through the
    PRECOMPUTED semantic graph (abductive fusion of 1-hop + PPR chain + support)
    so it surfaces causes-of-causes a single hop misses. On held-out 2-hop chain
    gold it beats 1-hop W (+29% R@10) while staying hub-clean; runs in <0.5s, zero
    inference-time graph build. Each result is tagged via=2-hop-chain when it came
    from the multi-hop reach. Use for driver/transmission/second-order questions;
    results are CANDIDATES — verify with corroborate/the gate. before=YYYY-MM-DD."""
    return json.dumps(rt.causal_chain(query, k=k, before=before or None,
                                      exclude=exclude or None), indent=2)


@mcp.tool()
def causal_direction(text_a: str, text_b: str) -> str:
    """Which of two claims/texts is the cause and which the effect? Asymmetric
    operator score (0.80-0.83 held-out accuracy vs 0.50 cosine floor). Returns
    candidate arrow + margin; treat as a lead, not proof."""
    return json.dumps(rt.causal_direction(text_a, text_b), indent=2)


def _run(script, *args):
    r = subprocess.run(["uv", "run", "--quiet", str(ROOT / "scripts" / script), *args],
                       capture_output=True, text=True, cwd=ROOT)
    return r.stdout or r.stderr


@mcp.tool()
def retrieve(query: str, k: int = 8, mode: str = "topical", before: str = "",
             exclude: str = "", tier: int = 0) -> str:
    """Corpus retrieval. mode: topical | diverse (facet coverage, MMR) |
    novel (first-story developments over echoes). Point-in-time via before."""
    args = ["-k", str(k), query]
    if mode == "diverse": args = ["--diverse"] + args
    if mode == "novel": args = ["--novel"] + args
    if before: args = ["--before", before] + args
    if exclude: args = ["--exclude", exclude] + args
    if tier: args = ["--tier", str(tier)] + args
    return _run("retrieve.py", *args)


@mcp.tool()
def corroborate(claim: str) -> str:
    """Claim-specific corroboration: how many DISTINCT sources confirm this exact
    claim (term-coverage gated; topical neighbours reported separately)."""
    return _run("retrieve.py", "--corroborate", claim)


@mcp.tool()
def verify_ledger(ledger_path: str) -> str:
    """Run the epistemic gate over a ledger.json: re-derives (status, confidence)
    per claim from cited passages — citation alignment, distinct-source
    corroboration, numeric verification, modality cap, direction advisory."""
    return _run("verify_citations.py", ledger_path)


if __name__ == "__main__":
    mcp.run()
