#!/usr/bin/env python3
"""Combine the 20 naive-arm (plain chunk-RAG baseline) A/B answers into one md.

The naive arm is data/ab/q{0..19}_base.md — topical cosine only, <=4 calls, no
protocol (see data/ab/RESULTS.md). Output: data/ab/COMBINED_NAIVE.md with a
table of contents; each question keeps its own H1 as the section heading.
"""
from __future__ import annotations

import re
from pathlib import Path

AB = Path(__file__).resolve().parent.parent / "data" / "ab"
N = 20
OUT = AB / "COMBINED_NAIVE.md"


def anchor(text: str) -> str:
    """GitHub-style heading anchor."""
    a = text.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def main() -> None:
    toc, sections = [], []
    for i in range(N):
        body = (AB / f"q{i}_base.md").read_text().strip()
        first, rest = body.split("\n", 1)
        title = first.lstrip("# ").strip()
        # A few files carry bare-question titles; normalize to "Q{i}: ...".
        if not re.match(rf"Q{i}\b", title):
            title = f"Q{i}: {title}"
        toc.append(f"- [{title}](#{anchor(title)})")
        sections.append(f"# {title}\n{rest}")

    header = (
        "# Naive baseline (plain chunk-RAG) — all 20 questions\n\n"
        "Combined from `q{0..19}_base.md` (A/B run 2026-07-12). Baseline arm: "
        "topical cosine retrieval only, ≤4 search calls, no protocol — the "
        "comparison arm for the full report methodology (see `RESULTS.md`).\n\n"
        "## Contents\n\n" + "\n".join(toc) + "\n"
    )
    OUT.write_text(header + "\n\n---\n\n".join([""] + sections).lstrip() + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {N} questions)")


if __name__ == "__main__":
    main()
