#!/usr/bin/env python3
"""Combine the 20 full-methodology A/B answers + rankings into one md.

Sources (data/ab/):
  q{0..19}_full.md    — the full-protocol answers
  judges_n20.json     — blind pairwise judge scores per question (full vs base)
  gate2_q{i}_full.json — mechanical citation-gate claim rows per question
  RESULTS.md          — run header (quoted at the top)

Output: data/ab/COMBINED_FULL_WITH_RANKING.md — run summary, a ranking table
(questions sorted by judge-score gap, with gate metrics), then all 20 answers,
each prefaced by its per-question scorecard.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

AB = Path(__file__).resolve().parent.parent / "data" / "ab"
N = 20
OUT = AB / "COMBINED_FULL_WITH_RANKING.md"


def anchor(text: str) -> str:
    a = text.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def gate_stats(i: int) -> dict:
    rows = json.loads((AB / f"gate2_q{i}_full.json").read_text())
    n = len(rows)
    corr = sum(r["status"] == "corroborated" for r in rows)
    counter = sum(r.get("counter_ok", 0) > 0 for r in rows)
    cites = sum(r.get("ok_sources", 0) for r in rows)
    return {
        "claims": n,
        "corroborated": corr,
        "counter": counter,
        "cites_per_claim": cites / n if n else 0.0,
    }


def main() -> None:
    judges = json.loads((AB / "judges_n20.json").read_text())
    full_s, base_s = judges["full"], judges["base"]

    titles, bodies = {}, {}
    for i in range(N):
        body = (AB / f"q{i}_full.md").read_text().strip()
        bodies[i] = body
        t = body.splitlines()[0].lstrip("# ").strip()
        titles[i] = re.sub(r"\s*\(FULL methodology\)\s*$", "", t)

    stats = {i: gate_stats(i) for i in range(N)}

    # Ranking: decisiveness of the full arm's win, biggest gap first.
    order = sorted(range(N), key=lambda i: (-(full_s[str(i)] - base_s[str(i)]), i))
    rank_rows = [
        "| rank | q | question | judge full | judge base | gap | claims | corroborated | counter-evidenced | cites/claim |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for rank, i in enumerate(order, 1):
        f, b, st = full_s[str(i)], base_s[str(i)], stats[i]
        short = titles[i].split(" — ")[-1] if " — " in titles[i] else titles[i]
        short = short if len(short) <= 80 else short[:77] + "..."
        rank_rows.append(
            f"| {rank} | Q{i} | [{short}](#{anchor(titles[i])}) | {f} | {b} | "
            f"+{f - b} | {st['claims']} | {st['corroborated']} "
            f"({st['corroborated'] / st['claims']:.0%}) | {st['counter']} | "
            f"{st['cites_per_claim']:.2f} |"
        )

    results_head = (AB / "RESULTS.md").read_text().split("\n## ", 1)[0].strip()

    parts = [
        "# Full methodology — all 20 questions, with rankings\n",
        "Combined from `q{0..19}_full.md` + `judges_n20.json` (blind pairwise "
        "judges, 12-point scale) + `gate2_q{i}_full.json` (mechanical citation "
        "gate). Run 2026-07-12.\n",
        "## Run summary (from RESULTS.md)\n",
        "> " + results_head.replace("\n", "\n> ") + "\n",
        f"**Blind judges: full wins {judges['wins']}/20** — average "
        f"{judges['avg_full']} vs {judges['avg_base']} (of 12).\n",
        "## Ranking (by judge-score gap, most decisive first)\n",
        "\n".join(rank_rows) + "\n",
        "---\n",
    ]

    for i in range(N):
        f, b, st = full_s[str(i)], base_s[str(i)], stats[i]
        head, rest = bodies[i].split("\n", 1)
        scorecard = (
            f"\n> **Scorecard** — judges: full **{f}**/12 vs base {b}/12 "
            f"(gap +{f - b}, rank {order.index(i) + 1}/20) · gate: "
            f"{st['claims']} claims, {st['corroborated']} corroborated "
            f"({st['corroborated'] / st['claims']:.0%}), {st['counter']} "
            f"counter-evidenced, {st['cites_per_claim']:.2f} OK cites/claim\n"
        )
        parts.append(head + "\n" + scorecard + rest.lstrip("\n") + "\n\n---\n")

    OUT.write_text("\n".join(parts).rstrip("-\n ") + "\n")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes, {N} questions)")


if __name__ == "__main__":
    main()
