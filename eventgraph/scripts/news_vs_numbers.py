# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
NEWS vs NUMBERS — does the price follow the results, the narrative, or neither?

THE QUESTION. Everything measured here so far conflates two things: an earnings day
moves the price 3.2x an ordinary day, but is that the RESULTS moving it, or the
NEWS COVERAGE moving it? Those are indistinguishable until you have the numbers
themselves. With SUE (from XBRL) and the extracted news direction (from the graph)
they separate, and the interesting case is where the two DISAGREE.

THREE TESTS, in order of what they establish.

  1. VALIDATION — does |price move| scale with |SUE|?
     If the market does not respond more to bigger surprises, SUE is not being
     priced and nothing after this means anything.

  2. DIRECTION — does the signed move follow the sign of the surprise?
     Measured as mean(sign(SUE) x signed idiosyncratic move). Positive = the price
     goes the way the numbers went.

  3. DISCREPANCY — for releases the news graph ALSO covered, does the extracted
     narrative direction (effect_dir) agree with the numbers? Where they conflict,
     which one does the price follow? That is the only test in this project that
     can say whether news carries information BEYOND the fundamentals it reports.

Everything is block-bootstrapped over dates, and MDE is reported, because a null
from an underpowered comparison is not a null.

Usage:
    uv run scripts/news_vs_numbers.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
G = ROOT / "eg100k_graph"
N_BOOT = 2000
RNG = np.random.default_rng(404)


def boot(bd):
    d = np.array(sorted(bd), dtype=object)
    if len(d) < 5:
        return None
    ms = []
    for _ in range(N_BOOT):
        pick = d[RNG.integers(0, len(d), len(d))]
        v = [x for k in pick for x in bd.get(k, ())]
        if v:
            ms.append(float(np.mean(v)))
    if not ms:
        return None
    m = np.array(ms)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return {"mean": float(m.mean()), "lo": float(lo), "hi": float(hi),
            "mde": float(2.80 * (hi - lo) / 3.92)}


def qend(key: str) -> dt.date:
    y, q = int(key[2:6]), int(key[-1])
    return {1: dt.date(y, 3, 31), 2: dt.date(y, 6, 30),
            3: dt.date(y, 9, 30), 4: dt.date(y, 12, 31)}[q]


def main() -> None:
    P = panel.load("us")
    days, di = P.days, P.day_index
    sue = json.loads((ROOT / "earnings_sue.json").read_text())
    cal = P.__dict__.get("_cal") or json.loads((G / "mcp_cache.json").read_text()).get(
        "earnings_calendar", {})
    print(f"SUE: {len(sue)} tickers   calendar: {len(cal)} tickers")

    # news direction per (ticker, date), from the graph's extracted effect_dir
    ent = {}
    for l in open(G / "classification" / "entity_class.jsonl"):
        j = json.loads(l)
        if j.get("resolution_status") == "resolved_security" and j.get("resolved_ticker"):
            ent[j["entity_id"]] = j["resolved_ticker"].upper()
    docday = {}
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            docday[j["doc_id"]] = d
    dirs = collections.defaultdict(list)
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        ed = j.get("effect_dir")
        if e in ent and doc in docday and ed in ("up", "down"):
            dirs[(ent[e], docday[doc])].append(1 if ed == "up" else -1)
    news_dir = {k: (1 if sum(v) > 0 else -1 if sum(v) < 0 else 0)
                for k, v in dirs.items()}
    print(f"news direction available for {len(news_dir)} (ticker, date) cells\n")

    # join: SUE quarter -> the first 8-K release after that quarter end
    rows = []
    for tk, quarters in sue.items():
        recs = cal.get(tk)
        if not recs or tk not in P.sigma:
            continue
        filed = sorted((r["filed"], r["session"]) for r in recs)
        for q in quarters:
            qe = qend(q["quarter"])
            hit = None
            for f, sess in filed:
                fd = dt.date.fromisoformat(f)
                if 0 < (fd - qe).days <= 100:
                    hit = (f, sess)
                    break
            if not hit:
                continue
            i = di.get(hit[1])
            if i is None:
                continue
            dec = P.decomp.get(tk, {}).get(str(i))
            sg = P.sigma.get(tk, {}).get(str(i))
            if dec is None or sg is None:
                continue
            # news direction on the filing date or the session
            nd = news_dir.get((tk, hit[0]), news_dir.get((tk, hit[1]), 0))
            rows.append({"tk": tk, "d": hit[1], "sue": float(np.clip(q["sue"], -5, 5)), "sig": float(sg),
                         "idio": float(dec[3]), "news_dir": nd})
    print(f"{len(rows)} earnings releases with BOTH a SUE and a measured reaction")
    print(f"  of which {sum(1 for r in rows if r['news_dir'])} also have an extracted "
          f"news direction\n")
    if len(rows) < 300:
        raise SystemExit("too few to test")

    # ---- 1. does the reaction scale with the surprise? ----
    a = np.array([abs(r["sue"]) for r in rows])
    qs = np.percentile(a, [20, 40, 60, 80])
    print("1) DOES THE REACTION SCALE WITH THE SURPRISE?")
    print(f"{'|SUE| quintile':>16}{'n':>7}{'mean |SUE|':>12}{'move (sigma)':>14}{'95% CI':>20}")
    for k in range(5):
        lo = -np.inf if k == 0 else qs[k - 1]
        hi = np.inf if k == 4 else qs[k]
        sel = [r for r in rows if lo <= abs(r["sue"]) < hi]
        bd = collections.defaultdict(list)
        for r in sel:
            bd[r["d"]].append(r["sig"])
        b = boot(bd)
        if b:
            print(f"{'Q'+str(k+1):>16}{len(sel):7}"
                  f"{np.mean([abs(r['sue']) for r in sel]):12.2f}"
                  f"{b['mean']:14.3f}  [{b['lo']:.3f}, {b['hi']:.3f}]")
    # correlation
    x = np.array([abs(r["sue"]) for r in rows]); y = np.array([r["sig"] for r in rows])
    x = np.minimum(x, 10)
    print(f"   corr(|SUE|, |move| in sigma) = {np.corrcoef(x, y)[0,1]:+.3f}\n")

    # ---- 2. does the price follow the SIGN of the numbers? ----
    print("2) DOES THE PRICE FOLLOW THE SIGN OF THE NUMBERS?")
    bd = collections.defaultdict(list)
    for r in rows:
        if r["sue"] != 0:
            bd[r["d"]].append(np.sign(r["sue"]) * r["idio"] * 100)
    b = boot(bd)
    print(f"   mean sign(SUE) x idiosyncratic return = {b['mean']:+.4f}%  "
          f"[{b['lo']:+.4f}, {b['hi']:+.4f}]  MDE {b['mde']:.4f}")
    print(f"   -> {'the price follows the numbers' if b['lo']>0 else 'no directional response'}\n")

    # ---- 3. THE DISCREPANCY ----
    print("3) WHERE THE NARRATIVE AND THE NUMBERS DISAGREE, WHICH WINS?")
    have = [r for r in rows if r["news_dir"] != 0 and r["sue"] != 0]
    agree = [r for r in have if np.sign(r["sue"]) == r["news_dir"]]
    clash = [r for r in have if np.sign(r["sue"]) != r["news_dir"]]
    print(f"   {len(have)} releases with both signals: "
          f"{len(agree)} agree ({len(agree)/max(len(have),1):.0%}), {len(clash)} disagree")
    for label, sel in (("AGREE", agree), ("DISAGREE", clash)):
        if len(sel) < 30:
            print(f"   {label}: n={len(sel)} — too few to test")
            continue
        for follow, name in ((lambda r: np.sign(r["sue"]), "numbers"),
                             (lambda r: r["news_dir"], "narrative")):
            bd = collections.defaultdict(list)
            for r in sel:
                bd[r["d"]].append(follow(r) * r["idio"] * 100)
            b = boot(bd)
            if b:
                tag = "*" if b["lo"] > 0 else (" " if b["hi"] > 0 else "-")
                print(f"   {label:9} follows {name:10} {b['mean']:+.4f}%  "
                      f"[{b['lo']:+.4f}, {b['hi']:+.4f}]  MDE {b['mde']:.4f} {tag}")
        bd = collections.defaultdict(list)
        for r in sel:
            bd[r["d"]].append(r["sig"])
        b = boot(bd)
        print(f"   {label:9} reaction SIZE  {b['mean']:.3f}σ  [{b['lo']:.3f}, {b['hi']:.3f}]")
    # persist the full detail so the showcase reports the same numbers
    quint = []
    for k in range(5):
        lo = -np.inf if k == 0 else qs[k - 1]
        hi = np.inf if k == 4 else qs[k]
        sel = [r for r in rows if lo <= abs(r["sue"]) < hi]
        bs = collections.defaultdict(list); bdir = collections.defaultdict(list)
        for r in sel:
            bs[r["d"]].append(r["sig"])
            if r["sue"] != 0:
                bdir[r["d"]].append(np.sign(r["sue"]) * r["idio"] * 100)
        b1, b2 = boot(bs), boot(bdir)
        quint.append({"q": k + 1, "n": len(sel),
                      "mean_abs_sue": round(float(np.mean([abs(r["sue"]) for r in sel])), 3),
                      "size": b1, "dir": b2})
    xx = np.abs([r["sue"] for r in rows]); yy = np.array([r["sig"] for r in rows])
    out = G / "news_vs_numbers.json"
    out.write_text(json.dumps({
        "n_rows": len(rows), "n_with_news": len(have),
        "n_agree": len(agree), "n_clash": len(clash),
        "corr_abs_sue_reaction": round(float(np.corrcoef(xx, yy)[0, 1]), 4),
        "overall_dir": b, "quintiles": quint}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
