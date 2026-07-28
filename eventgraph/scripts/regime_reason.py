# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "httpx"]
# ///
"""
REGIME REASONING — what changed around a correlation transition, with a placebo.

pair_regimes.py measures WHEN a correlation started or ended. It says nothing
about why. This asks an LLM to assess whether the news around the transition is
CONSISTENT with the observed change — under the same discipline as the
single-name gate:

  - it must not claim the news CAUSED the change
  - it must state plainly when the news does not plausibly relate
  - a correlation change is not evidence for any particular narrative

THE PLACEBO IS THE POINT. A model handed news and told "a correlation changed
here" will almost always produce a fluent story. So every real transition is run
TWICE: once with the news actually surrounding the transition, and once with news
from a randomly chosen window for the same pair where NO transition occurred. The
model is not told which is which. If the placebo answers are as confident and as
specific as the real ones, the narratives carry no information — and that is a
measurable outcome, not a matter of taste.

Each answer must open with a verdict token so the two arms can be scored:
    CONSISTENT / PARTIAL / UNRELATED

Usage:
    source ../data/tmp/groq.env && uv run scripts/regime_reason.py
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

import httpx
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
OUT = G / "regime_reason.json"
GROQ = "https://api.groq.com/openai/v1/chat/completions"
KEY = os.environ.get("GROQ_API_KEY")
MODEL = "openai/gpt-oss-120b"
HALF = 130          # trading days either side of a transition
MIN_ARTICLES = 4    # a window with no news cannot be judged, only defaulted
N_CASES = 14
RNG = np.random.default_rng(7)

SYS = (
    "You assess whether a body of news is CONSISTENT with a measured change in the "
    "correlation between two companies' idiosyncratic (company-specific) returns.\n"
    "The correlation measurement is a FACT. Your job is only to judge the news against it.\n"
    "RULES:\n"
    "1. Begin your answer with exactly one token: CONSISTENT, PARTIAL, or UNRELATED.\n"
    "   CONSISTENT = the news describes something that would plausibly change how these two "
    "names co-move (shared customer/supplier, merger, regulation hitting both, one exiting a "
    "business, a shared shock ending).\n"
    "   PARTIAL = some connection, but thin or indirect.\n"
    "   UNRELATED = the news is ordinary company flow with no bearing on the relationship. "
    "USE THIS FREELY. Most news is unrelated to a correlation change, and saying so is the "
    "correct answer, not a failure.\n"
    "2. NEVER say the news CAUSED the change. You are judging consistency, not causation. "
    "A correlation change is not evidence for any particular story.\n"
    "3. Do not invent a mechanism the articles do not describe. If you cannot point to "
    "specific text, answer UNRELATED.\n"
    "4. Do not use the words 'likely', 'investors saw', or 'suggests that' to bridge a gap.\n"
    "Then at most 60 words of justification, quoting or naming the specific article content "
    "you relied on. If UNRELATED, say what the news actually was instead.")


def llm(user: str, max_tokens: int = 2200) -> str:
    """gpt-oss-120b is a REASONING model: reasoning tokens come from the same
    budget, so a small max_tokens silently returns EMPTY content. Do not lower."""
    if not KEY:
        return "(no GROQ_API_KEY — source ../data/tmp/groq.env)"
    try:
        r = httpx.post(GROQ, headers={"Authorization": f"Bearer {KEY}"},
                       json={"model": MODEL, "temperature": 0.2, "max_tokens": max_tokens,
                             "messages": [{"role": "system", "content": SYS},
                                          {"role": "user", "content": user}]},
                       timeout=90)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:
        return f"(LLM error: {exc})"


def verdict_of(txt: str) -> str:
    m = re.match(r"\s*\**\s*(CONSISTENT|PARTIAL|UNRELATED)", txt.upper())
    return m.group(1) if m else "UNPARSED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=int, default=N_CASES)
    a = ap.parse_args()

    C = json.loads((G / "mcp_cache.json").read_text())
    R = json.loads((G / "pair_regimes.json").read_text())
    days = C["days"]; di = {d: i for i, d in enumerate(days)}
    wdates = R["window_dates"]

    # news for a ticker within a date range, grouped by class
    def news(tk: str, lo: str, hi: str, cap: int = 6) -> list[dict]:
        out = []
        for k, v in C["events"].items():
            t, d = k.split("|")
            if t != tk or not (lo <= d <= hi):
                continue
            for doc in v["docs"][:2]:
                doc_o = C["docs"].get(doc)
                if doc_o:
                    out.append({"date": d, "cls": ",".join(v["types"][:2]),
                                "head": doc_o["headline"],
                                "text": " ".join(doc_o["chunks"])[:450]})
        out.sort(key=lambda x: x["date"])
        return out[:cap]

    def window_around(idx: int) -> tuple[str, str]:
        c = di.get(wdates[idx], 0)
        return days[max(0, c - HALF)], days[min(len(days) - 1, c + HALF)]

    def render(pair, lo, hi) -> str:
        blocks = []
        for tk in (pair["a"], pair["b"]):
            n = news(tk, lo, hi)
            body = "\n".join(f"  [{x['date']} · {x['cls']}] {x['head']}\n    {x['text']}"
                             for x in n) or "  (no articles captured for this name in this window)"
            blocks.append(f"{tk} ({C['names'].get(tk, tk)} — {pair[('a_ind' if tk == pair['a'] else 'b_ind')]}):\n{body}")
        return "\n\n".join(blocks)

    def n_articles(pair, idx) -> int:
        lo, hi = window_around(idx)
        return len(news(pair["a"], lo, hi)) + len(news(pair["b"], lo, hi))

    cases = [p for p in R["pairs"] if p["state"] in ("ENDED", "EMERGING")
             and p["transition_index"] is not None]
    # REQUIRE NEWS IN THE TRANSITION WINDOW. First cut ignored this: 9 of 14 windows
    # held ZERO articles (median 0), because most transitions land in 2013-2014 while
    # deep news coverage runs 2010-04 to 2012-07. The model was being asked to reason
    # about nothing, and "UNRELATED" was forced rather than judged.
    cases = [p for p in cases if n_articles(p, p["transition_index"]) >= MIN_ARTICLES]
    print(f"{len(cases)} transitions have >= {MIN_ARTICLES} articles in their window")
    # prefer the largest actual change in correlation across the transition
    def magnitude(p):
        c = [x for x in p["corr"] if x is not None]
        t = p["transition_index"]
        pre = [x for x in p["corr"][:t] if x is not None][-4:]
        post = [x for x in p["corr"][t:] if x is not None][:4]
        return abs(np.mean(post) - np.mean(pre)) if pre and post else 0
    cases.sort(key=magnitude, reverse=True)
    cases = cases[:a.cases]
    print(f"{len(cases)} transitions selected\n")

    results = []
    for p in cases:
        t = p["transition_index"]
        lo, hi = window_around(t)
        # PLACEBO, MATCHED ON ARTICLE COUNT. Same pair, same window width, far
        # enough away to contain no measured change — but ALSO carrying a similar
        # volume of news. An empty placebo makes UNRELATED trivial and turns the
        # control into a test of "news vs no news" rather than "transition news vs
        # ordinary news", which is not the question.
        n_real = n_articles(p, t)
        far = [i for i in range(len(wdates))
               if abs(di.get(wdates[i], 0) - di.get(wdates[t], 0)) > 2 * HALF]
        far = [(abs(n_articles(p, i) - n_real), i) for i in far]
        far = [(d_, i) for d_, i in far if n_articles(p, i) >= MIN_ARTICLES]
        if not far:
            print(f"  {p['a']}~{p['b']}: no article-matched placebo window, skipped")
            continue
        far.sort()
        pi = int(far[0][1])
        plo, phi = window_around(pi)

        head = (f"Pair: {p['a']} ({p['a_ind']}) and {p['b']} ({p['b_ind']}).\n"
                f"Measured: correlation of their idiosyncratic returns "
                f"{'FELL BELOW' if p['state'] == 'ENDED' else 'ROSE ABOVE'} the level "
                f"explainable by chance for this pair (empirical band |r| = {p['band']}), "
                f"and stayed there.\n")
        real = llm(head + f"Change detected around {wdates[t]}.\n\n"
                   f"News for both names in the surrounding window ({lo} to {hi}):\n\n"
                   + render(p, lo, hi) +
                   "\n\nIs this news consistent with a change in how these two co-move?")
        plac = llm(head + f"Change detected around {wdates[pi]}.\n\n"
                   f"News for both names in the surrounding window ({plo} to {phi}):\n\n"
                   + render(p, plo, phi) +
                   "\n\nIs this news consistent with a change in how these two co-move?")
        rv, pv = verdict_of(real), verdict_of(plac)
        results.append({
            "a": p["a"], "b": p["b"], "state": p["state"], "band": p["band"],
            "a_ind": p["a_ind"], "b_ind": p["b_ind"],
            "cross_industry": p["cross_industry"],
            "transition_date": p["transition_date"], "last_corr": p["last_corr"],
            "magnitude": round(float(magnitude(p)), 3),
            "real": {"window": [lo, hi], "verdict": rv, "answer": real,
                     "n_articles": len(news(p["a"], lo, hi)) + len(news(p["b"], lo, hi))},
            "placebo": {"window": [plo, phi], "verdict": pv, "answer": plac,
                        "n_articles": len(news(p["a"], plo, phi)) + len(news(p["b"], plo, phi))},
        })
        print(f"  {p['a']:6}~{p['b']:6} {p['state']:9} Δr={magnitude(p):.2f} "
              f"@{p['transition_date']}   real={rv:11} placebo={pv}")

    rc = collections.Counter(r["real"]["verdict"] for r in results)
    pc = collections.Counter(r["placebo"]["verdict"] for r in results)
    print(f"\nreal    verdicts: {dict(rc)}")
    print(f"placebo verdicts: {dict(pc)}")
    n = len(results) or 1
    rr = (rc["CONSISTENT"] + 0.5 * rc["PARTIAL"]) / n
    pp = (pc["CONSISTENT"] + 0.5 * pc["PARTIAL"]) / n
    print(f"\nweighted 'related' rate — real {rr:.0%}  vs  placebo {pp:.0%}")
    print("If these are close, the narratives carry no information about the transition:")
    print("the model tells an equally good story about a window where nothing happened.")

    OUT.write_text(json.dumps({"cases": results,
                               "real_verdicts": dict(rc), "placebo_verdicts": dict(pc),
                               "real_rate": round(rr, 3), "placebo_rate": round(pp, 3)}))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
