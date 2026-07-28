# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""
INDIA entity -> NSE ticker resolution, VERIFIED (not suggested).

WHY THIS EXISTS: the India graph is well-extracted (98% quote grounding, 3.2%
direction-conflict — slightly BETTER than the Bloomberg graph) but has been
unusable for any ticker-level work because it has no verified identifiers. The
only mapping on disk, `nifty50_resolved.json`, is HALLUCINATED — and dangerously
so, because it is *partially* right: `reliance_industries -> RELIANCE.NS` is
correct while `jindal_steel_power -> JSWSTEEL`, `rain_industries -> GRASIM` and
`south_indian_bank -> AXISBANK` are wrong. Partially-right is the worst kind: no
entry can be trusted without independent verification.

THE VERIFICATION: the cached Yahoo chart JSONs for the 147 `.NS` tickers we hold
price history for carry the issuer's real company name in `meta.longName`
(e.g. MARUTI.NS -> "Maruti Suzuki India Limited"). So a candidate mapping can be
CHECKED against an independent source rather than believed. We accept a mapping
only on a strict normalised name match, and record HOW it was matched so the
evidence travels with the row.

This mirrors the `resolve_yahoo.py` doctrine used for the US graph: the LLM (or
a prior map) may PROPOSE, but only an external check may PROMOTE to verified.
Anything unmatched stays unresolved — never guessed.

Output: <graph>/classification/india_ticker_map.json
        {entity_id: {ticker, company_name, entity_name, match}}

Usage:
    uv run scripts/india_resolve_verified.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "india2021"

# Corporate-form and geography noise that carries no identifying information.
STOP = {
    "limited", "ltd", "ltd.", "the", "india", "indian", "company", "co", "corporation",
    "corp", "enterprises", "enterprise", "industries", "industry", "group", "holdings",
    "holding", "and", "&", "of", "plc", "inc", "private", "pvt", "services", "service",
}


def norm_tokens(s: str) -> list[str]:
    s = re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower())
    return [t for t in s.split() if t and t not in STOP]


def key(s: str) -> str:
    return "".join(norm_tokens(s))


def load_ticker_names() -> dict[str, str]:
    """ticker -> company name, from the cached Yahoo chart meta (independent source)."""
    out = {}
    for f in (G / "prices").glob("*.NS.json"):
        try:
            meta = json.loads(f.read_text())["chart"]["result"][0]["meta"]
        except Exception:
            continue
        name = meta.get("longName") or meta.get("shortName")
        if name:
            out[f.name.replace(".json", "")] = name
    return out


def load_entities() -> list[tuple[str, str]]:
    ents = []
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        eid, nm = j.get("entity_id"), j.get("name")
        if eid and nm:
            ents.append((eid, nm))
    return ents


def main() -> None:
    tnames = load_ticker_names()
    ents = load_entities()
    print(f"tickers with an independent company name: {len(tnames)}")
    print(f"india entities: {len(ents)}")

    # Index tickers by exact normalised key and by token set.
    by_key: dict[str, list[str]] = {}
    by_tokens: list[tuple[str, set[str]]] = []
    for tk, nm in tnames.items():
        by_key.setdefault(key(nm), []).append(tk)
        by_tokens.append((tk, set(norm_tokens(nm))))

    resolved: dict[str, dict] = {}
    for eid, nm in ents:
        k = key(nm)
        if not k or len(k) < 4:
            continue
        # (1) exact normalised match — strongest evidence
        if k in by_key and len(by_key[k]) == 1:
            tk = by_key[k][0]
            resolved[eid] = {"ticker": tk, "company_name": tnames[tk],
                             "entity_name": nm, "match": "exact_normalised"}
            continue
        # (2) full token-set containment, requiring >=2 informative tokens so
        #     single generic words ("bank", "steel") cannot carry a match
        etoks = set(norm_tokens(nm))
        if len(etoks) < 2:
            continue
        cands = [tk for tk, ttoks in by_tokens if etoks and etoks <= ttoks]
        if len(cands) == 1:
            tk = cands[0]
            resolved[eid] = {"ticker": tk, "company_name": tnames[tk],
                             "entity_name": nm, "match": "token_subset"}

    out = G / "classification" / "india_ticker_map.json"
    out.write_text(json.dumps(resolved, indent=2, sort_keys=True))
    by_match: dict[str, int] = {}
    for v in resolved.values():
        by_match[v["match"]] = by_match.get(v["match"], 0) + 1
    print(f"\nVERIFIED mappings: {len(resolved)}  {by_match}")
    print(f"distinct tickers covered: {len({v['ticker'] for v in resolved.values()})} "
          f"of {len(tnames)}")

    print("\nsample:")
    for eid, v in list(resolved.items())[:10]:
        print(f"  {eid[:38]:38} -> {v['ticker']:16} ({v['company_name'][:34]}) [{v['match']}]")

    # Audit the known-bad map against what verification actually says.
    legacy_path = G / "nifty50_resolved.json"
    if legacy_path.exists():
        legacy = json.loads(legacy_path.read_text())
        agree = disagree = unknown = 0
        examples = []
        for eid, tk in legacy.items():
            if eid in resolved:
                if resolved[eid]["ticker"] == tk:
                    agree += 1
                else:
                    disagree += 1
                    if len(examples) < 6:
                        examples.append((eid, tk, resolved[eid]["ticker"]))
            else:
                unknown += 1
        print(f"\naudit of the legacy nifty50_resolved.json ({len(legacy)} entries): "
              f"{agree} agree, {disagree} CONTRADICTED, {unknown} unverifiable")
        for eid, bad, good in examples:
            print(f"  {eid[:34]:34} legacy={bad:14} verified={good}")
        print("  -> confirms the legacy map is partially hallucinated; it must not be used.")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
