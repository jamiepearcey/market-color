# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2.0", "numpy"]
# ///
"""
DAILY NEWS EXPLANATION MCP — the gate enforced server-side.

WHY SERVER-SIDE. A prompt that says "don't over-claim" is a suggestion. This
server makes it structural: when the narrative-confidence gate fails, the tool
does not hand back a tidy story for the model to embellish. It returns
`do_not_narrate: true`, a `refusal` string stating what may and may not be said,
and the evidence marked as INSUFFICIENT. The calling model is told, in the
payload itself, that composing a causal explanation from that evidence would be
confabulation.

WHAT THE GATE IS. A NARRATIVE CONFIDENCE TEST, not a test of truth or causality:
is a candidate explanation strong enough to be the PRIMARY account of a move?
  - It never asserts what DID cause a move. Rejecting one explanation is not
    evidence for another.
  - A failing verdict does not mean the news is false or its impact zero. The
    reaction simply falls within the normal historical range for its class, so
    the narrative is CONTRIBUTORY RATHER THAN EXPLANATORY.
  - Priors are TRAILING as of the date, never pooled: class priors drift twice as
    much as the baseline and reorder (legal_regulatory spans rank 1 to 6 across
    the sample). A pooled prior mis-gates 3.6% of events in both directions.
  - SUFFICIENT is graded, not binary. A move clearing its prior by 0.03 of the
    class's own standard deviation returns confidence WEAK, and the payload says
    so, because otherwise a marginal clearance reads exactly like a decisive one.

TOOLS
    explain_move(ticker, date)      decomposition + gated verdict + evidence
    screen_day(date, limit)         the day's largest idiosyncratic moves, gated
    get_class_priors(as_of)         trailing priors and control in force that day
    get_risk_profile(ticker)        news-weight breakdown + measured risk by class
    get_neighbours(ticker)          idiosyncratic-correlation neighbours

Run:
    uv run scripts/mcp_server.py                    # stdio
Register:
    claude mcp add eventgraph -- uv run <abs path>/scripts/mcp_server.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate import Gate  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
CACHE = G / "mcp_cache.json"

if not CACHE.exists():
    raise SystemExit(f"{CACHE} missing — run: uv run scripts/mcp_precompute.py")

C = json.loads(CACHE.read_text())
GATE = Gate()
DAYS: list[str] = C["days"]
DI = {d: i for i, d in enumerate(DAYS)}

# DOCUMENTS PER DAY — the single most important thing to condition on. The corpus
# was sampled deliberately unevenly: some days were extracted deeply (up to a
# 500-document cap) and the rest given only token coverage. A pooled coverage
# figure therefore measures the sampling plan rather than the graph, so every
# tool reports the tier of the day it is answering about, and callers can filter
# to the days that were actually captured.
import collections as _c  # noqa: E402

_DPD: _c.Counter = _c.Counter()
for _l in open(G / "lake" / "document.jsonl"):
    _d = (json.loads(_l).get("published_at") or "")[:10]
    if _d:
        _DPD[_d] += 1

TIERS = [(100, "deep"), (20, "partial"), (5, "thin"), (1, "token")]


def coverage_tier(date: str) -> dict:
    """How well was this day sampled? Everything else is conditional on it."""
    n = _DPD.get(date, 0)
    tier = next((t for lo, t in TIERS if n >= lo), "none")
    return {
        "documents": n, "tier": tier,
        "interpretation": {
            "deep": ("This day was extracted deeply. On such days ~27% of >4-sigma, ~20% of "
                     ">3-sigma and ~12% of >2-sigma idiosyncratic moves have linked news, and "
                     "essentially all of those clear the gate. has_news=false here is "
                     "meaningful evidence that no article was captured for that name."),
            "partial": "Partially sampled. Absence of news is weak evidence.",
            "thin": ("Only token coverage (5-19 documents). Roughly 0.5-1% of even the largest "
                     "moves have linked news. has_news=false says almost nothing."),
            "token": ("1-4 documents only. This day was not meaningfully sampled. Do not draw "
                      "any conclusion from missing news."),
            "none": "No documents at all for this day.",
        }[tier],
    }


mcp = FastMCP("eventgraph")

REFUSAL = (
    "INSUFFICIENT. Do not present this news as the explanation for the move, and do not "
    "assert what did cause it — rejecting one explanation is not evidence for another. "
    "The news is not shown to be false, nor its impact zero; the observed reaction simply "
    "falls within the normal historical range for its class as of this date, so the "
    "narrative is CONTRIBUTORY RATHER THAN EXPLANATORY. State that, give the numbers, and "
    "stop. Composing a causal story from the attached evidence would be confabulation."
)


def _pick_class(ticker: str, date: str, types: list[str]) -> str:
    """The class this day's news most strongly implies, ranked by the prior IN
    FORCE ON THAT DATE — not by a pooled constant."""
    return max(types, key=lambda t: (GATE.as_of(date, t)["prior"] or 0.0))


def _evidence(doc_ids: list[str], limit: int = 3) -> list[dict]:
    out = []
    for d in doc_ids[:limit]:
        doc = C["docs"].get(d)
        if doc:
            out.append({"doc_id": d, "date": doc["date"], "headline": doc["headline"],
                        "text": " ".join(doc["chunks"])[:1200]})
    return out


@mcp.tool()
def explain_move(ticker: str, date: str) -> dict:
    """Decompose one name's move on one day and gate any news narrative for it.

    Returns the market/sector/idiosyncratic split, the trailing class prior in
    force on that date, a graded verdict, and the source articles. If the gate
    fails, `do_not_narrate` is true and `refusal` states what may be said.
    """
    tk = ticker.upper()
    if tk not in C["sigma"]:
        return {"error": f"{tk} not in the tradeable universe (547 names). It may have been "
                         "dropped as stale/illiquid, or never resolved in the graph."}
    if date not in DI:
        return {"error": f"{date} is not a trading day in range "
                         f"({DAYS[0]} .. {DAYS[-1]})"}
    i = str(DI[date])
    if i not in C["sigma"][tk]:
        return {"error": f"no usable measurement for {tk} on {date} "
                         "(insufficient trailing history, or a stale window)"}

    sg = C["sigma"][tk][i]
    dec = C["decomp"][tk].get(i)
    sec = C["sectors"].get(tk, {})
    cell = C["events"].get(f"{tk}|{date}")

    base = {
        "ticker": tk, "name": C["names"].get(tk, tk), "date": date,
        "sector": sec,
        "idiosyncratic_sigma": sg,
        "decomposition": (
            {"total": dec[0], "market": dec[1], "sector": dec[2], "idiosyncratic": dec[3],
             "idiosyncratic_share": round(abs(dec[3]) / max(abs(dec[0]), 1e-9), 3)}
            if dec else None),
        "note": ("Company news is expected to explain primarily the IDIOSYNCRATIC "
                 "component, which is usually the smallest of the three. Attribution "
                 "based on articles alone frequently overstates the role of company news."),
    }

    cov = coverage_tier(date)
    base["coverage"] = cov

    if not cell:
        return {**base, "has_news": False, "do_not_narrate": True,
                "refusal": (
                    "No classified news is linked to this name on this date. Do not invent "
                    "one, and do not assert what caused the move. "
                    + ("This day was sampled deeply, so the absence is informative: no article "
                       "for this name was captured."
                       if cov["tier"] == "deep" else
                       f"This day has only {cov['documents']} documents ({cov['tier']} "
                       "coverage), so the absence means NOT SAMPLED and carries almost no "
                       "information — do not treat it as evidence that no news existed."))}

    cls = _pick_class(tk, date, cell["types"])
    v = GATE.evaluate(date, cls, sg)
    out = {
        **base, "has_news": True,
        "event_class": cls, "all_classes": cell["types"],
        "gate": {
            "status": v["status"], "confidence": v.get("confidence"),
            "prior_as_of": v["asof"],
            "class_prior_sigma": v["prior"], "class_prior_sd": v.get("prior_sd"),
            "control_sigma": v["control"],
            "surprise_sigma": v["surprise"], "surprise_in_class_sds": v.get("surprise_z"),
            "reasons": v["reasons"], "strength": v.get("strength_note"),
            "prior_is_trailing": True,
            "pooled_prior_would_be": GATE.pooled.get(cls),
        },
        "evidence": _evidence(cell["docs"]),
        "do_not_narrate": v["do_not_narrate"],
    }
    if v["do_not_narrate"]:
        out["refusal"] = REFUSAL
    else:
        out["permitted"] = (
            f"SUFFICIENT at confidence {v['confidence']}. {v.get('strength_note','')} "
            "Explain using only the attached evidence, state how far the reaction exceeded "
            "the historical norm for this class, and match your confidence to the tier — "
            "SUFFICIENT is not permission to sound certain.")
    return out


@mcp.tool()
def screen_day(date: str, limit: int = 10) -> dict:
    """Rank a day's largest idiosyncratic moves and gate each one.

    Use this to find what is worth explaining on a given day. Every row carries
    its own verdict; rows with do_not_narrate must not be written up as caused
    by their news.
    """
    if date not in DI:
        return {"error": f"{date} is not a trading day in range ({DAYS[0]} .. {DAYS[-1]})"}
    i = str(DI[date])
    rows = []
    for tk, s in C["sigma"].items():
        if i not in s:
            continue
        cell = C["events"].get(f"{tk}|{date}")
        r = {"ticker": tk, "name": C["names"].get(tk, tk),
             "sector": C["sectors"].get(tk, {}).get("major_group"),
             "idiosyncratic_sigma": s[i], "has_news": bool(cell)}
        if cell:
            cls = _pick_class(tk, date, cell["types"])
            v = GATE.evaluate(date, cls, s[i])
            r |= {"event_class": cls, "gate": v["status"],
                  "confidence": v.get("confidence"),
                  "surprise_sigma": v["surprise"],
                  "do_not_narrate": v["do_not_narrate"]}
        rows.append(r)
    rows.sort(key=lambda r: -r["idiosyncratic_sigma"])
    top = rows[:max(1, min(limit, 50))]
    ctx = GATE.as_of(date, "earnings")
    n_ok = sum(1 for r in top if r.get("gate") == "SUFFICIENT")
    n_news = sum(1 for r in top if r["has_news"])
    return {
        "date": date, "universe": len(rows),
        "coverage": coverage_tier(date),
        "control_sigma_as_of": ctx["control"], "priors_as_of": ctx["asof"],
        "rows": top,
        "summary": (f"{n_news} of {len(top)} shown rows have any linked news; {n_ok} clear "
                    "the gate. Rows with do_not_narrate=true must not be attributed to their "
                    "news; rows with has_news=false have no linked article at all."),
        # The single most important limitation of this dataset, returned on every
        # call so it cannot be forgotten by a caller that only reads `rows`.
        "coverage_warning": (
            "COVERAGE IS BY DESIGN UNEVEN — read this before interpreting has_news=false. "
            "The corpus was sampled to give some days deep coverage and others only token "
            "coverage, so a POOLED coverage figure measures the sampling plan, not the graph. "
            "Conditioned on coverage: on the 96 well-covered days (>=100 documents, avg 453) "
            "27.4% of >4-sigma idiosyncratic moves have linked news, 20.0% of >3-sigma and "
            "12.5% of >2-sigma; on the 810 thinly-sampled days (1-19 documents) the same "
            "figures are ~0.1-1.0%. Of well-covered large moves that DO have news, essentially "
            "all clear the gate (100% at >3 sigma), so on those days reach is the only binding "
            "constraint. At small moves the gate rejects ~71% as insufficient, which is "
            "correct and is the point. "
            "CONSEQUENCE FOR CALLERS: has_news=false means NOT-SAMPLED far more often than "
            "'no news existed', especially on a thin day. Never fill that gap by borrowing a "
            "neighbouring name's news or a sector narrative. Do not narrate a whole trading "
            "day. Note also that even well-covered days were capped at 500 documents, so "
            "27.4% is a floor for those days rather than a ceiling."),
    }


@mcp.tool()
def list_covered_days(min_documents: int = 100, limit: int = 200) -> dict:
    """The trading days that were actually captured, so you can filter to them.

    Coverage is deliberately uneven — some days were extracted deeply, most were
    given only token coverage. Any statistic pooled across both measures the
    sampling plan rather than the data. Use this to restrict analysis to days
    where an absence of news is actually informative.
    """
    rows = [{"date": d, "documents": _DPD.get(d, 0),
             "tier": coverage_tier(d)["tier"]}
            for d in DAYS if _DPD.get(d, 0) >= max(0, min_documents)]
    rows.sort(key=lambda r: -r["documents"])
    hist = _c.Counter(coverage_tier(d)["tier"] for d in DAYS)
    return {
        "min_documents": min_documents,
        "n_matching": len(rows),
        "n_trading_days": len(DAYS),
        "days": rows[:max(1, min(limit, 2000))],
        "tier_histogram": dict(hist),
        "note": ("Even deeply-covered days were capped at 500 documents during extraction, so "
                 "their coverage figures are a floor rather than a ceiling. Days with tier "
                 "'thin' or 'token' should generally be EXCLUDED from any analysis that "
                 "interprets the absence of news."),
    }


@mcp.tool()
def get_class_priors(as_of: str) -> dict:
    """The trailing class priors and control in force on a date.

    Priors move: classes drift twice as much as the baseline and REORDER, so the
    prior for a class on one date is not the prior on another. Always read them
    as of the date you are explaining.
    """
    if not as_of:
        return {"error": "as_of date required, e.g. 2011-12-07"}
    i = GATE._idx(as_of)
    d = GATE.dates[i]
    ctrl = GATE.control[i]
    out = []
    for c, series in GATE.priors.items():
        v = series.get(d)
        if v:
            out.append({"event_class": c, "prior_sigma": v[0], "prior_sd": v[1],
                        "multiple_of_control": round(v[0] / ctrl, 2),
                        "pooled_would_be": GATE.pooled.get(c)})
    out.sort(key=lambda r: -r["prior_sigma"])
    return {"requested": as_of, "estimates_as_of": d, "control_sigma": ctrl,
            "classes": out,
            "note": ("Trailing 250-day estimates, never pooled. multiple_of_control is the "
                     "meaningful figure: 1.0 means that class is indistinguishable from an "
                     "ordinary non-event day in this regime.")}


@mcp.tool()
def get_risk_profile(ticker: str) -> dict:
    """What kinds of news attach to a name, and how much risk each class carries.

    `news_weight` is how often each class appears for this name — the breakdown of
    what the name's news flow is actually made of. It is a COVERAGE measure, not a
    forecast.
    """
    tk = ticker.upper()
    if tk not in C["sigma"]:
        return {"error": f"{tk} not in the tradeable universe"}
    counts = C["class_counts"].get(tk, {})
    tot = sum(counts.values()) or 1
    rows = [{"event_class": c, "n": n, "share": round(n / tot, 3),
             "pooled_prior_sigma": GATE.pooled.get(c),
             "multiple_of_pooled_control": (round(GATE.pooled[c] / GATE.pooled_control, 2)
                                            if c in GATE.pooled else None)}
            for c, n in sorted(counts.items(), key=lambda x: -x[1])]
    return {"ticker": tk, "name": C["names"].get(tk, tk),
            "sector": C["sectors"].get(tk, {}),
            "n_event_days": tot, "news_weight": rows,
            "note": ("Priors here are POOLED, shown only to characterise the name's news mix. "
                     "For any dated judgement use get_class_priors(as_of) or explain_move, "
                     "which use trailing values.")}


@mcp.tool()
def get_neighbours(ticker: str) -> dict:
    """Names whose IDIOSYNCRATIC residuals co-move with this one.

    These are correlations AFTER removing market and sector, so they are the
    relationships a sector map does not show. About 29% of the strongest links
    cross a 4-digit SIC industry boundary (only 6% cross a broad SIC division —
    the figure depends heavily on how coarse the buckets are).
    """
    tk = ticker.upper()
    if tk not in C["neighbours"]:
        return {"error": f"{tk} not in the tradeable universe"}
    rows = [{"ticker": t, "name": C["names"].get(t, t), "correlation": c,
             "same_industry": same,
             "industry": C["sectors"].get(t, {}).get("industry")}
            for t, c, same in C["neighbours"][tk]]
    return {"ticker": tk, "industry": C["sectors"].get(tk, {}).get("industry"),
            "neighbours": rows,
            "note": ("Correlation of idiosyncratic residuals, not of returns. A high value "
                     "means the two names surprise together after market and sector are "
                     "removed. This is contemporaneous association, not a lead-lag signal.")}


if __name__ == "__main__":
    mcp.run()
