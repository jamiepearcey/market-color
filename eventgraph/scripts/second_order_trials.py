# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
SECOND-ORDER EXPOSURE: do drug-trial events move UNMENTIONED pharma names?

THE ANGLE. Everything measured in this project so far runs document -> named firm
-> did it move. That direction keeps failing on recall (F41: 95.5% of takeover news
never reaches a priceable name) and on efficiency (first-order names are priced in
minutes). This inverts it: take an event, then ask which firms the event should
touch WITHOUT being named in it. For a clinical trial the economics are crisp — a
readout moves the sponsor, but it also moves same-indication rivals (a competitor's
failure is your win), licensees, and the CRO/CDMO chain.

WHY TRIALS. `clinical_trial` only exists as a class as of F38, created because
Health Care's `other` bucket ran 2.40x the sector control while every other sector
sat near 1.0x. Inspecting it showed FDA reviews and study readouts with no home in
the vocabulary. It measured 3.09x in Health Care — the highest cell in the sector —
on n=9, i.e. suggestive and underpowered.

THE DESIGN, and the control that makes it falsifiable:

    named        firms named in the trial document        -> sanity check, must be high
    2nd-order    UNMENTIONED Health Care firms            -> THE TEST
    control      UNMENTIONED non-Health-Care firms        -> must be ~1.0x or the
                                                            result is just "eventful day"

Each group is measured against ITS OWN non-event baseline, so a sector with higher
idiosyncratic vol cannot masquerade as a second-order effect. A circular-shift
placebo over event dates gives the chance band, preserving the clustering of trial
news in time.

READ IT LIKE THIS:
    2nd-order >> control, placebo p<0.05  -> genuine second-order propagation
    2nd-order ~ control                   -> trial days are just noisy days
    2nd-order ~ 1.0                       -> no propagation; the thesis dies here

Usage:
    uv run scripts/second_order_trials.py
    uv run scripts/second_order_trials.py --window 1 --placebo 3000
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gics  # noqa: E402
import taxonomy as tax  # noqa: E402
from panel import load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
RNG = np.random.default_rng(20260726)

# Headline-level trial language, for recall on top of the extractor's event_type.
# Deliberately narrow -- these must be trial/regulatory readouts, not pharma news
# in general, or the "event" set becomes "any healthcare story".
TRIAL_RE = re.compile(
    r"\b(phase (i|ii|iii|1|2|3)\b|clinical trial|late-stage (trial|stud)|"
    r"study (show|found|fail|met|miss)|trial (show|found|fail|met|miss|result)|"
    r"fda (approv|reject|panel|advisers|advisory)|regulatory approval|"
    r"drug (approv|fail|reject)|met (its )?primary endpoint|missed (its )?primary endpoint)",
    re.I)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", default="eg100k_graph")
    ap.add_argument("--window", type=int, default=0, help="+/- sessions around the event day")
    ap.add_argument("--placebo", type=int, default=2000)
    a = ap.parse_args()
    G = ROOT / a.graph
    P = load("us")

    # ---- entity -> ticker, and the sector map --------------------------------
    # gics.py is a small hand-curated map (0 of 548 tickers come back Health Care),
    # so the sector source is entity_class.std_sector -- the same one the
    # sector x class risk table uses -- with gics as a fallback.
    tick, std_sec = {}, {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        t = (j.get("resolved_ticker") or "").upper()
        if t and str(j.get("resolution_status", "")).startswith("resolved"):
            tick[j["entity_id"]] = t
            if j.get("std_sector") not in (None, "UNK"):
                std_sec.setdefault(t, j["std_sector"])

    tickers = [t for t in P.sigma]
    def sector_of(t):
        s = std_sec.get(t)
        if s:
            return s
        g = gics.sector(t)
        return None if g in (None, "UNK") else g
    sec = {t: sector_of(t) for t in tickers}
    hc = {t for t in tickers if sec.get(t) == "Health Care"}
    other = {t for t in tickers if sec.get(t) not in (None, "Health Care")}
    print(f"universe {len(tickers)} priceable | Health Care {len(hc)} | other sectors {len(other)}")

    ALIAS = re.compile(r"INSERT INTO entity_alias \(alias,entity_id\) VALUES \('(.+?)','(.+?)'\)")
    BAD = re.compile(r"^(the|a|an|inc|plc|corp|ltd|sa|ag|co|group|holdings?)$", re.I)
    alias = {}
    for line in open(G / "pg_upsert.sql", errors="ignore"):
        m = ALIAS.search(line)
        if not m:
            continue
        al, eid = m.group(1).replace("''", "'"), m.group(2)
        if eid in tick and 4 <= len(al) <= 60 and not BAD.match(al) and re.search(r"[A-Za-z]{3}", al):
            alias.setdefault(al, eid)
    alias_order = sorted(alias, key=len, reverse=True)

    # ---- trial documents ------------------------------------------------------
    trial_docs = set()
    for l in open(G / "lake" / "event.jsonl"):
        j = json.loads(l)
        if tax.std_event_type(j.get("event_type"))[0] == "clinical_trial" and j.get("doc_id"):
            trial_docs.add(j["doc_id"])
    by_extractor = len(trial_docs)

    docday, headline = {}, {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if not d:
            continue
        docday[j["doc_id"]] = d
        headline[j["doc_id"]] = (j.get("headline") or "").strip()
        if TRIAL_RE.search(headline[j["doc_id"]]):
            trial_docs.add(j["doc_id"])
    print(f"trial documents: {by_extractor} from the extractor's class, "
          f"{len(trial_docs)} after adding headline matches")

    # ---- named firms per event day -------------------------------------------
    eff = collections.defaultdict(set)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") in trial_docs and j.get("effect_entity") in tick:
            eff[j["doc_id"]].add(tick[j["effect_entity"]])

    named_by_day = collections.defaultdict(set)
    ev_days = set()
    for doc in trial_docs:
        d = docday.get(doc)
        if not d or d not in P.day_index:
            continue
        h = " " + headline.get(doc, "") + " "
        named = set(eff.get(doc, ()))
        for al in alias_order:
            if al in h:
                named.add(tick[alias[al]])
        if not named:
            continue
        named_by_day[d] |= named
        ev_days.add(d)
    ev_days = sorted(ev_days)
    print(f"event days with >=1 named priceable firm: {len(ev_days)}")
    hc_named = sum(1 for d in ev_days if named_by_day[d] & hc)
    print(f"  of which name >=1 Health Care firm: {hc_named}")

    # ---- measurement ----------------------------------------------------------
    days = P.days
    di = P.day_index
    ev_idx = sorted({di[d] for d in ev_days})
    evset = set()
    for i in ev_idx:
        for k in range(-a.window, a.window + 1):
            if 0 <= i + k < len(days):
                evset.add(i + k)

    def sig(t, i):
        v = P.sigma.get(t, {}).get(str(i))
        return None if v is None else float(v)

    def group_stats(members, idxs, exclude_named):
        """mean |sigma| over (member, day) pairs, skipping firms named that day."""
        out = []
        for i in idxs:
            d = days[i]
            nm = set()
            if exclude_named:
                for k in range(-a.window, a.window + 1):
                    j = i + k
                    if 0 <= j < len(days):
                        nm |= named_by_day.get(days[j], set())
            for t in members:
                if t in nm:
                    continue
                s = sig(t, i)
                if s is not None:
                    out.append(s)
        return np.array(out)

    base_idx = [i for i in range(len(days)) if i not in evset]
    rows = []
    for label, members, excl in (("named (first-order)", None, False),
                                 ("2nd-order: unmentioned Health Care", hc, True),
                                 ("control: unmentioned other sectors", other, True)):
        if members is None:
            ev = []
            for i in ev_idx:
                for t in named_by_day.get(days[i], ()):
                    s = sig(t, i)
                    if s is not None:
                        ev.append(s)
            ev = np.array(ev)
            bs = group_stats(set(t for dd in named_by_day.values() for t in dd), base_idx, False)
        else:
            ev = group_stats(members, sorted(evset), True)
            bs = group_stats(members, base_idx, False)
        if len(ev) < 5 or len(bs) < 50:
            print(f"  {label}: too thin (event n={len(ev)}, base n={len(bs)})")
            continue
        rows.append((label, len(ev), ev.mean(), bs.mean(), ev.mean() / bs.mean()))

    print(f"\n=== TRIAL-DAY |sigma| BY GROUP  (window +/-{a.window})")
    print(f"  {'group':38} {'n':>7} {'event':>7} {'base':>7} {'lift':>6}")
    for label, n, e, b, lift in rows:
        print(f"  {label:38} {n:>7} {e:>7.3f} {b:>7.3f} {lift:>5.2f}x")

    # ---- circular-shift placebo on the second-order group ---------------------
    if any(r[0].startswith("2nd-order") for r in rows):
        obs = [r for r in rows if r[0].startswith("2nd-order")][0][2]
        null = []
        n_days = len(days)
        for _ in range(a.placebo):
            sh = int(RNG.integers(0, n_days))
            idxs = [(i + sh) % n_days for i in ev_idx]
            fake = set()
            for i in idxs:
                for k in range(-a.window, a.window + 1):
                    if 0 <= i + k < n_days:
                        fake.add(i + k)
            v = group_stats(hc, sorted(fake), False)
            if len(v) >= 5:
                null.append(v.mean())
        null = np.array(null)
        p = (np.sum(null >= obs) + 1) / (len(null) + 1)
        print(f"\n=== CIRCULAR-SHIFT PLACEBO on the second-order group ({len(null)} draws)")
        print(f"  observed {obs:.3f}   null mean {null.mean():.3f}   null p95 "
              f"{np.percentile(null, 95):.3f}   p = {p:.3f}"
              f"{'  *' if p < 0.05 else '   (not significant)'}")


if __name__ == "__main__":
    main()
