# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "fastembed>=0.3"]
# ///
"""Citation-integrity gate — closes the two prototype breaks:
  (1) validator miscalibration (confirmed facts flagged unsupported), and
  (2) mismapped/hallucinated citations.

Input  ledger.json : [{
    "claim": str,
    "status": "supported" | "contradicted" | "insufficient" | "speculative",
              (legacy corroborated/partial/unsupported are mapped in)
    "cite":         [doc_id, ...],   # supporting evidence
    "counter_cite": [doc_id, ...],   # disconfirming evidence (for contradicted)
    "searched":     bool             # was a targeted (dis)confirming search run?
  }]

TWO AXES, derived from the ACTUAL evidence (not the analyst's say-so):
  status     what the evidence does: supported / contradicted / insufficient / speculative
  confidence how strongly:            high / medium / low / n/a
Key: `insufficient` splits the old `unsupported` into "searched, not found" (HIGH
confidence it's absent) vs "coverage gap" (LOW). The gate RECLASSIFIES when the tag
disagrees with the evidence — e.g. supported-with-no-OK-cite -> insufficient;
contradicted-with-no-counter-evidence -> insufficient; insufficient-but-counter-
evidence-found -> contradicted. So "supported/high" now MEANS >=2 distinct
well-aligned sources, mechanically, and cannot be asserted into existence.

For each cited doc: re-fetch its best-matching passage, score claim<->passage
cosine alignment, tag OK / WEAK (< --min, likely mis-cite) / MISSING (not in corpus).

Output: verify_report.json (machine) + a two-axis summary to stdout + verify_bundles.txt.

Usage: uv run scripts/verify_citations.py data/ledger.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

D = Path(__file__).resolve().parents[1] / "data"

import re
# quantitative figures a report can get specifically, dangerously wrong: percentages
# and magnitudes with units (barrels, tonnes, capacity, mb/d, million/billion).
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*(?:%|per\s?cent|percent)", re.I)
_UNIT = re.compile(r"(\d[\d,\.]*)\s*(barrel|bbl|tonne|ton|mb/?d|bpd|b/d|"
                   r"million|billion|thousand|mmbbl|kb/?d)s?\b", re.I)
# currency / price magnitudes (the $4,525-gold class the pct/unit checks missed):
# a currency symbol/code immediately before a number — $4,525, US$28.5, €360, ₹92, Rs 15,157.
_CUR = re.compile(r"(?:US\$|A\$|R\$|S\$|\$|€|£|¥|₹|Rs\.?|SAR|AED|CNY|INR)\s?(\d[\d,\.]*)", re.I)
# basis points — 50 bps / 25 basis points (a rate-move figure a desk acts on).
_BPS = re.compile(r"(\d+(?:\.\d+)?)\s*(?:bps|basis\s+points)\b", re.I)

def _norm_num(s):
    s = s.replace(",", "").rstrip(".")
    try:
        return f"{float(s):g}"
    except ValueError:
        return s

def extract_figures(text):
    """Return the set of {value+kind} figures a claim commits to (pct / quantity)."""
    figs = set()
    for m in _PCT.finditer(text):
        figs.add(("pct", _norm_num(m.group(1))))
    for m in _UNIT.finditer(text):
        unit = m.group(2).lower().rstrip("s")
        unit = {"tonne": "ton", "bbl": "barrel", "b/d": "mb/d", "bpd": "mb/d"}.get(unit, unit)
        figs.add((unit, _norm_num(m.group(1))))
    for m in _CUR.finditer(text):
        figs.add(("cur", _norm_num(m.group(1))))
    for m in _BPS.finditer(text):
        figs.add(("bps", _norm_num(m.group(1))))
    return figs

def figure_in_text(kind, val, text):
    """Is this specific figure present in the cited passage (unit-aware, comma-agnostic)?
    Boundaried match only — a bare-substring test would let 500000 match inside
    1,500,000 and wave through fabricated figures."""
    if (kind, val) in extract_figures(text):
        return True
    # the same value may appear with different unit wording; match on a digit boundary
    # so it is not a fragment of a larger number.
    return bool(re.search(rf"(?<!\d){re.escape(val)}(?!\d)", text.replace(",", "")))


def load():
    ch = np.load(D / "chunks.npy").astype(np.float32)
    ch /= (np.linalg.norm(ch, axis=1, keepdims=True) + 1e-9)
    meta = [json.loads(l) for l in (D / "chunks.jsonl").read_text().splitlines() if l.strip()]
    texts = [json.loads(l)["t"] for l in (D / "chunk_texts.jsonl").read_text().splitlines() if l.strip()]
    dmeta = json.loads((D / "doc_meta.json").read_text())
    by_doc = {}
    for i, m in enumerate(meta):
        by_doc.setdefault(m["doc_id"], []).append(i)
    return ch, texts, dmeta, by_doc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", nargs="?", default=str(D / "ledger.json"))
    ap.add_argument("--min", type=float, default=0.40,
                    help="claim<->cited-passage cosine below this = WEAK cite")
    ap.add_argument("--enforce-direction", action="store_true",
                    help="hard-demote direction-flipped cites (default: advisory flag "
                         "only; the polarity probe mis-fires on crisis-framed passages)")
    a = ap.parse_args()

    ledger = json.loads(Path(a.ledger).read_text())
    if isinstance(ledger, dict):  # accept {"claims": [...]} wrapper shape
        ledger = ledger.get("claims", ledger.get("ledger", []))
    ch, texts, dmeta, by_doc = load()
    from fastembed import TextEmbedding
    model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    claims = [c["claim"] for c in ledger]
    cvec = np.array(list(model.embed(claims)), dtype=np.float32)
    cvec /= (np.linalg.norm(cvec, axis=1, keepdims=True) + 1e-9)

    # POLARITY (direction) axis — the cosine gate is directionally blind ("rose" vs
    # "fell" about the same subject cosine ~0.6). A cite that expresses the OPPOSITE
    # direction to the claim cannot support it, however well it aligns topically.
    wdir = None
    wpath = D / "polarity_w.npy"
    if wpath.exists():
        wdir = np.load(wpath).astype(np.float32)
    DIR_THR = 0.5  # |z| above which a text expresses a *clear* direction
    claim_dir = (cvec @ wdir) if wdir is not None else np.zeros(len(claims))

    # MODALITY (evidence-type) axis — a factual claim "corroborated" only by forecast/
    # opinion passages is not the same as one backed by event/data. Type each cite.
    wmod = None; mclasses = []
    if (D / "modality_w.npy").exists():
        wmod = np.load(D / "modality_w.npy").astype(np.float32)
        mclasses = json.load(open(D / "modality_classes.json"))
    SOFT = {"forecast", "statement"}

    def feats_of(snippets):
        """Return (polarity z, modality-class-name) per snippet, embedding once."""
        if not snippets:
            return [], []
        V = np.array(list(model.embed(snippets)), dtype=np.float32)
        V /= (np.linalg.norm(V, axis=1, keepdims=True) + 1e-9)
        dirs = list(V @ wdir) if wdir is not None else [0.0] * len(snippets)
        mods = [mclasses[i] for i in (V @ wmod).argmax(1)] if wmod is not None else [""] * len(snippets)
        return dirs, mods

    # agents often copy the 8-char short id retrieve.py prints; resolve unique prefixes
    full_ids = list(by_doc)
    def resolve(did):
        if did in by_doc:
            return did
        hits = [f for f in full_ids if f.startswith(did)]
        return hits[0] if len(hits) == 1 else None

    # legacy single-axis tags -> the two-axis vocabulary (status is re-derived below
    # from the actual evidence, so this is only the starting intent).
    MAP = {"corroborated": "supported", "supported": "supported", "partial": "supported",
           "unsupported": "insufficient", "insufficient": "insufficient",
           "contradicted": "contradicted", "speculative": "speculative"}

    def verify_cites(cite_list, cv):
        """Return (cite records, #distinct OK sources, bundle lines)."""
        recs, lines = [], []
        for did_raw in cite_list:
            did = resolve(did_raw)
            if did is None:
                recs.append({"doc_id": did_raw, "status": "MISSING", "align": None})
                lines.append(f"  - [{did_raw[:8]}] !! NOT IN CORPUS (or ambiguous prefix)")
                continue
            idxs = by_doc[did]
            al = float(np.max(ch[idxs] @ cv))
            best = idxs[int(np.argmax(ch[idxs] @ cv))]
            flag = "WEAK" if al < a.min else "OK"
            dm = dmeta.get(did, {})
            doc_text = " ".join(texts[i] for i in idxs)  # all chunks, for numeric scan
            recs.append({"doc_id": did, "status": flag, "align": round(al, 3),
                         "src": dm.get("src", "?"), "tier": dm.get("tier", 9),
                         "date": dm.get("date", "?"), "dom": dm.get("dom", "?"),
                         "text": doc_text, "snippet": texts[best]})
            lines.append(f"  - [{did[:8]}] align={al:.3f} {flag} [{dm.get('src','?')}] "
                         f"{dm.get('date','?')}\n      {' '.join(texts[best].split()[:55])}")
        ok_src = {r["dom"] for r in recs if r["status"] == "OK"}
        return recs, len(ok_src), lines

    report, bundles = [], []
    n_weak = n_missing = n_ok = 0
    for ci_idx, (c, cv) in enumerate(zip(ledger, cvec)):
        raw = c.get("status", "?")
        status = MAP.get(raw, raw)
        searched = bool(c.get("searched", False))
        cites, ok_sources, blines = verify_cites(c.get("cite", []), cv)
        counters, counter_ok, clines = verify_cites(c.get("counter_cite", []), cv)

        # ---- DIRECTION CONSISTENCY: if the claim expresses a clear direction, an OK
        # ---- cite whose passage expresses the OPPOSITE direction is topically right
        # ---- but evidentially wrong — demote it to DIRFLIP so it cannot support.
        cdir = float(claim_dir[ci_idx]) if wdir is not None else 0.0
        n_dirflip = 0
        ok_snips = [r for r in cites if r["status"] == "OK"]
        odirs, omods = feats_of([r["snippet"] for r in ok_snips])
        # APPLICABILITY GUARD: the polarity probe is only reliable on genuinely
        # DIRECTIONAL claims (a price/quantity move). It conflates crisis sentiment
        # with price direction and assigns spurious directions to event/fee/deal
        # claims — so only run the direction check when the CLAIM itself types as a
        # quantitative-move modality.
        MOVE = {"price_move", "supply_change", "demand_change", "production_change"}
        claim_mod = feats_of([c["claim"]])[1][0] if mclasses else ""
        dir_applies = wdir is not None and claim_mod in MOVE and abs(cdir) >= DIR_THR
        for r, dv, mv in zip(ok_snips, odirs, omods):
            r["dir"] = round(float(dv), 2); r["modality"] = mv
            # opposite-direction evidence can't support — but the polarity probe
            # conflates crisis sentiment with price direction on real passages, so this
            # is ADVISORY (flag for review) by default. Hard enforcement (demote the
            # cite) only under --enforce-direction, pending a sentiment-robust probe.
            if dir_applies and abs(dv) >= DIR_THR and (dv > 0) != (cdir > 0):
                r["dir_flag"] = True; n_dirflip += 1
                if a.enforce_direction:
                    r["status"] = "DIRFLIP"
        if n_dirflip:
            ok_sources = len({r["dom"] for r in cites if r["status"] == "OK"})
        # modality: is any surviving OK support a HARD (event/data) source?
        surv = [r for r in cites if r["status"] == "OK"]
        hard_support = any(r.get("modality") and r["modality"] not in SOFT for r in surv)
        claim_is_forecast = mclasses and feats_of([c["claim"]])[1][0] in SOFT
        for r in cites + counters:
            n_ok += r["status"] == "OK"; n_weak += r["status"] == "WEAK"; n_missing += r["status"] == "MISSING"

        # ---- NUMERIC VERIFICATION: every specific figure the claim commits to must
        # ---- actually appear in an OK cited passage, or the figure is unverified and
        # ---- the claim cannot be promoted to corroborated on the strength of it.
        claim_figs = extract_figures(c["claim"])
        ok_text = " ".join(r["text"] for r in cites if r["status"] == "OK")
        unverified_figs = [f"{v}{'%' if k=='pct' else ' '+k}" for (k, v) in claim_figs
                           if not figure_in_text(k, v, ok_text)]
        nums_ok = not unverified_figs

        # ---- derive STATUS (corroborated / supported / contradicted / no_evidence_found)
        # ---- and a CONFIDENCE qualifier, from the ACTUAL evidence — reclassifying when
        # ---- the analyst's tag disagrees with what verifies.
        final, conf, reason = "no_evidence_found", "low", ""
        if status == "speculative":
            final, conf, reason = "speculative", "n/a", "model-inferred hypothesis; not an evidentiary claim"
        elif counter_ok >= 1:                        # counter-evidence wins regardless of tag
            final, conf = "contradicted", ("high" if counter_ok >= 2 else "medium")
            reason = f"{counter_ok} aligned counter-source(s)"
        elif status == "contradicted" and counter_ok == 0:
            final, conf = "no_evidence_found", ("high" if searched else "low")
            reason = "tagged contradicted but NO aligned counter-evidence; " + (
                "searched, none found" if searched else "coverage thin")
        elif ok_sources >= 2:
            if not nums_ok:
                final, conf = "supported", "medium"
                reason = f"{ok_sources} sources BUT figure(s) unverified in cited text: {unverified_figs}"
            elif wmod is not None and not hard_support and not claim_is_forecast:
                # factual claim 'corroborated' only by forecast/opinion passages
                final, conf = "supported", "low"
                reason = f"{ok_sources} sources but ALL are forecast/opinion (soft modality); no event/data backing"
            else:
                final, conf = "corroborated", "high"
                reason = f"{ok_sources} distinct OK sources; figures verified" + (
                    "; hard evidence" if wmod is not None else "")
        elif ok_sources == 1:
            final, conf = "supported", "medium" if nums_ok else "low"
            reason = "single OK source" + ("" if nums_ok else f"; unverified figure(s): {unverified_figs}")
        else:
            final, conf = "no_evidence_found", ("high" if searched else "low")
            reason = "no OK supporting cite; " + (
                "searched, not found" if searched else "coverage gap (not thoroughly searched)")

        if n_dirflip:  # advisory unless --enforce-direction
            kind = "DIRECTION-FLIPPED" if a.enforce_direction else "⚠ possible direction mismatch (review)"
            reason += f"; {n_dirflip} cite(s) {kind} (claim dir {cdir:+.1f})"
        clean = [{k: v for k, v in r.items() if k not in ("text", "snippet")} for r in cites]
        cclean = [{k: v for k, v in r.items() if k not in ("text", "snippet")} for r in counters]
        report.append({"claim": c["claim"], "raw_status": raw,
                       "status": final, "confidence": conf, "reason": reason,
                       "ok_sources": ok_sources, "counter_ok": counter_ok,
                       "searched": searched, "claim_dir": round(cdir, 2), "dirflips": n_dirflip,
                       "figures": sorted(f"{v}{'%' if k=='pct' else ' '+k}"
                                         for k, v in claim_figs),
                       "unverified_figures": unverified_figs,
                       "cites": clean, "counter_cites": cclean})
        fig_note = f"  figs!={unverified_figs}" if unverified_figs else ""
        bundles.append(f"CLAIM ({raw} -> {final}/{conf}){fig_note}: {c['claim']}\nCITED:\n"
                       + "\n".join(blines) + ("\nCOUNTER:\n" + "\n".join(clines) if clines else ""))

    (D / "verify_report.json").write_text(json.dumps(report, indent=2))
    (D / "verify_bundles.txt").write_text("\n\n" + ("\n" + "=" * 80 + "\n").join(bundles))

    fam = {"corroborated": "affirmed", "supported": "affirmed", "contradicted": "contradicted",
           "insufficient": "none", "unsupported": "none", "no_evidence_found": "none",
           "speculative": "speculative", "partial": "affirmed"}
    reclass = [r for r in report if fam.get(r["status"]) != fam.get(r["raw_status"], r["raw_status"])]
    numflag = [r for r in report if r["unverified_figures"]]
    dirflag = [r for r in report if r["dirflips"]]
    from collections import Counter
    tally = Counter(r["status"] for r in report)
    print(f"claims={len(ledger)}  cites: OK={n_ok} WEAK={n_weak} MISSING={n_missing}  "
          f"reclassified={len(reclass)}  numeric-flags={len(numflag)}  dir-flips={len(dirflag)}")
    print("  " + "  ".join(f"{s}={n}" for s, n in tally.most_common()))
    print(f"\n{'STATUS':<17}{'CONF':<7}{'raw':<13} claim")
    for r in report:
        flag = "  <-- reclassified" if r in reclass else ""
        print(f"{r['status']:<17}{r['confidence']:<7}{r['raw_status']:<13} {r['claim'][:56]}{flag}")
        if r in reclass or r["unverified_figures"] or any(c['status'] != 'OK' for c in r['cites']):
            print(f"                 ({r['reason']})")
    print("\nwrote verify_report.json + verify_bundles.txt")


if __name__ == "__main__":
    main()
