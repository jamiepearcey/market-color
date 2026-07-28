# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
RE-TEST THE NULLS ON WELL-COVERED DAYS ONLY.

WHY THIS IS NECESSARY. Every predictive null in this project was run over the full
panel. But coverage is deliberately uneven: 96 days were extracted deeply (up to a
500-document cap) and ~810 were given only token coverage (1-19 documents). On a
token day, a name labelled "no news" may well HAVE had news that was simply never
sampled.

That contamination sits in the NEGATIVE class. Label noise in the negatives makes a
predictive test LESS able to detect an effect — it biases toward the null. So every
"news does not predict X" result is confounded with "we did not sample the news".

Note the asymmetry: the same contamination biases the CONTEMPORANEOUS positive
result the other way. Unsampled news days landing in the random-control pool inflate
the control, which UNDERSTATES the measured effect. The positive is conservative;
the nulls are not.

WHAT THIS DOES. Restricts to deep-coverage days, where a "no news" label is
meaningful, and re-runs the sharp test:

    target    |idiosyncratic move| at t+1, standardised by the name's own trailing vol
    baseline  the name's own recent realised volatility (the thing news must beat)
    +news     baseline plus event-class indicators known at t
    control   the SAME model with the news columns row-shuffled

A gain only counts if it clearly exceeds its own permutation control. Reported
alongside the identical test on thin days and on the full panel, so the effect of
conditioning is visible rather than asserted.

Usage:
    uv run scripts/coverage_conditioned_retest.py
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
# India is UNIFORMLY deep where it is sampled at all (p25=122 docs/day, no thin days),
# so its null was already coverage-conditioned by construction — and it doubles as the
# second-market replication of the NULL, which matters: a replication that only ever
# tests the positive result is not much of a test.
GRAPHS = {"us": "eg100k_graph", "india": "india2021"}
G = ROOT / GRAPHS["us"]
OUT = G / "coverage_conditioned_retest.json"
RNG = np.random.default_rng(17)
SPLIT = 0.65


def ols_fit(X, y):
    XtX = X.T @ X + 1e-8 * np.eye(X.shape[1])
    return np.linalg.solve(XtX, X.T @ y)


def oos_r2(Xtr, ytr, Xte, yte):
    b = ols_fit(Xtr, ytr)
    p = Xte @ b
    ss = float(((yte - p) ** 2).sum())
    tot = float(((yte - ytr.mean()) ** 2).sum())
    return 1.0 - ss / tot if tot > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--market", choices=sorted(GRAPHS), default="us")
    a = ap.parse_args()
    global G, OUT
    G = ROOT / GRAPHS[a.market]
    OUT = G / "coverage_conditioned_retest.json"
    print(f"market={a.market}  graph={G.name}\n")
    C = json.loads((G / "mcp_cache.json").read_text())
    days: list[str] = C["days"]
    di = {d: i for i, d in enumerate(days)}

    dpd = collections.Counter()
    for l in open(G / "lake" / "document.jsonl"):
        j = json.loads(l)
        d = (j.get("published_at") or "")[:10]
        if d:
            dpd[d] += 1
    tier = {d: ("deep" if dpd.get(d, 0) >= 100 else
                "thin" if dpd.get(d, 0) >= 1 else "none") for d in days}
    n_deep = sum(1 for d in days if tier[d] == "deep")
    print(f"{n_deep} deep days, {sum(1 for d in days if tier[d]=='thin')} thin, "
          f"{sum(1 for d in days if tier[d]=='none')} with no documents\n")

    # event classes present, as indicator columns
    ev = collections.defaultdict(set)
    for k, v in C["events"].items():
        tk, d = k.split("|")
        ev[(tk, d)] |= set(v["types"])
    classes = sorted({t for s in ev.values() for t in s})
    ci = {c: k for k, c in enumerate(classes)}
    print(f"{len(classes)} event classes as indicators")

    # SYSTEMATIC risk series per name: |market + sector| standardised by that name's
    # own trailing systematic volatility, so it is on the same footing as the
    # idiosyncratic sigma. Needed for the taxonomy-alignment test.
    sys_sig = {}
    for tk, dec in C["decomp"].items():
        ks = sorted(int(k) for k in dec)
        if len(ks) < 120:
            continue
        arr = {k: abs(dec[str(k)][1] + dec[str(k)][2]) for k in ks}
        vals = np.array([arr[k] for k in ks])
        out_s = {}
        for p, k in enumerate(ks):
            if p < 60:
                continue
            w = vals[p - 60:p]
            sd_ = float(w.std())
            if sd_ > 1e-6:
                out_s[k] = float(arr[k] / sd_)
        sys_sig[tk] = out_s

    # build the panel: for each (name, day) with a measurable sigma at t and t+1
    rows = []
    for tk, sd in C["sigma"].items():
        idx = sorted(int(k) for k in sd)
        have = {int(k): v for k, v in sd.items()}
        hsys = sys_sig.get(tk, {})
        for i in idx:
            if (i + 1) not in have:
                continue
            d = days[i]
            if tier[d] == "none":
                continue
            # baseline features: the name's own recent realised risk
            prev = [have[j] for j in (i - 1, i - 2, i - 3, i - 5) if j in have]
            if len(prev) < 4:
                continue
            news = np.zeros(len(classes))
            types = ev.get((tk, d))
            if types:
                for t in types:
                    if t in ci:
                        news[ci[t]] = 1.0
            rows.append({"tk": tk, "d": d, "tier": tier[d], "y": have[i + 1],
                         "y_sys": hsys.get(i + 1),
                         "base": [1.0, have[i], *prev], "news": news,
                         "has_news": bool(types)})
    print(f"{len(rows)} usable (name, day) rows\n")

    def run(sel, label):
        if len(sel) < 400:
            return None
        sel = sorted(sel, key=lambda r: r["d"])
        cut = int(len(sel) * SPLIT)
        tr, te = sel[:cut], sel[cut:]
        Xb_tr = np.array([r["base"] for r in tr]); Xb_te = np.array([r["base"] for r in te])
        Xn_tr = np.array([np.concatenate([r["base"], r["news"]]) for r in tr])
        Xn_te = np.array([np.concatenate([r["base"], r["news"]]) for r in te])
        y_tr = np.array([r["y"] for r in tr]); y_te = np.array([r["y"] for r in te])
        r2b = oos_r2(Xb_tr, y_tr, Xb_te, y_te)
        r2n = oos_r2(Xn_tr, y_tr, Xn_te, y_te)
        # permutation control: shuffle the news block across rows, keep everything else
        perms = []
        for _ in range(12):
            p_tr = RNG.permutation(len(tr)); p_te = RNG.permutation(len(te))
            Xp_tr = np.array([np.concatenate([tr[k]["base"], tr[p_tr[k]]["news"]])
                              for k in range(len(tr))])
            Xp_te = np.array([np.concatenate([te[k]["base"], te[p_te[k]]["news"]])
                              for k in range(len(te))])
            perms.append(oos_r2(Xp_tr, y_tr, Xp_te, y_te) - r2b)
        perm = float(np.mean(perms)); perm_sd = float(np.std(perms))
        inc = r2n - r2b
        return {"set": label, "n": len(sel), "n_news": sum(1 for r in sel if r["has_news"]),
                "r2_base": round(r2b, 4), "r2_news": round(r2n, 4),
                "incremental": round(inc, 5),
                "perm_mean": round(perm, 5), "perm_sd": round(perm_sd, 5),
                "z_vs_perm": round((inc - perm) / perm_sd, 2) if perm_sd > 0 else None,
                "verdict": ("SIGNAL" if perm_sd > 0 and (inc - perm) / perm_sd > 3
                            and inc > 0 else "null")}

    out = []
    for label, fn in [
        ("DEEP days only (news labels trustworthy)", lambda r: r["tier"] == "deep"),
        ("THIN days only (absence uninformative)", lambda r: r["tier"] == "thin"),
        ("full panel (as originally run)", lambda r: True),
    ]:
        res = run([r for r in rows if fn(r)], label)
        if res:
            out.append(res)

    print(f"{'sample':44}{'n':>8}{'news':>7}{'base R²':>9}{'+news':>9}"
          f"{'incr':>9}{'perm':>9}{'z':>6}  verdict")
    for r in out:
        print(f"{r['set']:44}{r['n']:8}{r['n_news']:7}{r['r2_base']:9.4f}{r['r2_news']:9.4f}"
              f"{r['incremental']:9.5f}{r['perm_mean']:9.5f}"
              f"{(r['z_vs_perm'] if r['z_vs_perm'] is not None else 0):6.1f}  {r['verdict']}")

    print("\nincr = OOS R² gain from adding news columns. perm = the same gain when the")
    print("news block is row-shuffled. A gain only counts if it clearly exceeds perm.")
    deep = next((r for r in out if r["set"].startswith("DEEP")), None)
    full = next((r for r in out if r["set"].startswith("full")), None)
    if deep and full:
        print(f"\nCONDITIONING EFFECT: incremental on deep days {deep['incremental']:+.5f} "
              f"vs full panel {full['incremental']:+.5f}")
        if deep["verdict"] == "null" and full["verdict"] == "null":
            print("Both null. The original nulls were NOT an artifact of unsampled news in the")
            print("negative class — restricting to days where 'no news' is meaningful does not")
            print("recover a signal.")
        elif deep["verdict"] == "SIGNAL":
            print("The deep-day test finds signal the full-panel test missed. The original nulls")
            print("WERE confounded by coverage and must be retracted.")
    # ------------------------------------------------------------------
    # NON-LINEAR ARM. A gradient-boosted model has the most to gain from cleaner
    # negatives, so if coverage contamination were hiding a signal this is where it
    # should appear. Both arms are non-linear, and the control is the SAME model with
    # the news block row-shuffled — comparing GBM(news) against a LINEAR baseline
    # would confound information with functional form and manufacture a win.
    # ------------------------------------------------------------------
    from sklearn.ensemble import HistGradientBoostingRegressor as HGB

    def gbm_r2(Xtr, ytr, Xte, yte):
        m = HGB(max_iter=180, learning_rate=0.06, max_depth=5,
                early_stopping=False, random_state=0).fit(Xtr, ytr)
        p = m.predict(Xte)
        ss = float(((yte - p) ** 2).sum()); tot = float(((yte - ytr.mean()) ** 2).sum())
        return 1.0 - ss / tot if tot > 0 else float("nan")

    def run_gbm(sel, label, n_perm=5):
        if len(sel) < 2000:
            return None
        sel = sorted(sel, key=lambda r: r["d"])
        cut = int(len(sel) * SPLIT); tr, te = sel[:cut], sel[cut:]
        Xb_tr = np.array([r["base"] for r in tr]); Xb_te = np.array([r["base"] for r in te])
        Xn_tr = np.array([np.concatenate([r["base"], r["news"]]) for r in tr])
        Xn_te = np.array([np.concatenate([r["base"], r["news"]]) for r in te])
        y_tr = np.array([r["y"] for r in tr]); y_te = np.array([r["y"] for r in te])
        rb = gbm_r2(Xb_tr, y_tr, Xb_te, y_te)
        rn = gbm_r2(Xn_tr, y_tr, Xn_te, y_te)
        perms = []
        for _ in range(n_perm):
            p_tr = RNG.permutation(len(tr)); p_te = RNG.permutation(len(te))
            Xp_tr = np.array([np.concatenate([tr[k]["base"], tr[p_tr[k]]["news"]])
                              for k in range(len(tr))])
            Xp_te = np.array([np.concatenate([te[k]["base"], te[p_te[k]]["news"]])
                              for k in range(len(te))])
            perms.append(gbm_r2(Xp_tr, y_tr, Xp_te, y_te) - rb)
        pm, ps = float(np.mean(perms)), float(np.std(perms))
        inc = rn - rb
        return {"set": label, "model": "GBM", "n": len(sel),
                "n_news": sum(1 for r in sel if r["has_news"]),
                "r2_base": round(rb, 4), "r2_news": round(rn, 4),
                "incremental": round(inc, 5), "perm_mean": round(pm, 5),
                "perm_sd": round(ps, 5),
                "z_vs_perm": round((inc - pm) / ps, 2) if ps > 0 else None,
                "verdict": ("SIGNAL" if ps > 0 and (inc - pm) / ps > 3 and inc > 0 else "null")}

    print("\n=== NON-LINEAR (GBM) ARM — where cleaner negatives should help most ===")
    gout = []
    for label, fn in [("DEEP days only", lambda r: r["tier"] == "deep"),
                      ("full panel", lambda r: True)]:
        g = run_gbm([r for r in rows if fn(r)], label)
        if g:
            gout.append(g)
            print(f"{g['set']:24}{g['n']:8}{g['n_news']:7}{g['r2_base']:9.4f}{g['r2_news']:9.4f}"
                  f"{g['incremental']:9.5f}{g['perm_mean']:9.5f}"
                  f"{(g['z_vs_perm'] or 0):6.1f}  {g['verdict']}")

    # ------------------------------------------------------------------
    # TAXONOMY ALIGNMENT, coverage-conditioned. Corporate news should help the
    # IDIOSYNCRATIC target more than the SYSTEMATIC one, and macro news the reverse.
    # ------------------------------------------------------------------
    MACRO = {"monetary_policy", "employment", "econ_indicator", "election", "growth",
             "fiscal_policy", "trade_policy", "commodity", "geopolitical", "inflation"}
    corp_i = [ci[c] for c in classes if c not in MACRO]
    macro_i = [ci[c] for c in classes if c in MACRO]
    print("\n=== TAXONOMY ALIGNMENT on deep days (corporate->idio vs macro->systematic) ===")
    tax = []
    deep_rows = [r for r in rows if r["tier"] == "deep"]
    for tgt in ("idio", "systematic"):
        for grp, idxs in (("CORPORATE", corp_i), ("MACRO", macro_i)):
            yk = "y" if tgt == "idio" else "y_sys"
            sel = sorted([r for r in deep_rows if r.get(yk) is not None],
                         key=lambda r: r["d"])
            if len(sel) < 2000:
                continue
            cut = int(len(sel) * SPLIT); tr, te = sel[:cut], sel[cut:]
            Xb_tr = np.array([r["base"] for r in tr]); Xb_te = np.array([r["base"] for r in te])
            Xn_tr = np.array([np.concatenate([r["base"], r["news"][idxs]]) for r in tr])
            Xn_te = np.array([np.concatenate([r["base"], r["news"][idxs]]) for r in te])
            y_tr = np.array([r[yk] for r in tr]); y_te = np.array([r[yk] for r in te])
            rb = oos_r2(Xb_tr, y_tr, Xb_te, y_te); rn = oos_r2(Xn_tr, y_tr, Xn_te, y_te)
            tax.append({"target": tgt, "group": grp, "incremental": round(rn - rb, 5)})
            print(f"  {grp:10} -> {tgt:11} incremental {rn - rb:+.5f}")
    if len(tax) == 4:
        d = {(t["group"], t["target"]): t["incremental"] for t in tax}
        aligned = (d[("CORPORATE", "idio")] > d[("CORPORATE", "systematic")]
                   and d[("MACRO", "systematic")] > d[("MACRO", "idio")])
        print(f"  expected pattern holds: {aligned}"
              + ("" if aligned else "  -> the labels do not align with the risk taxonomy"))

    OUT.write_text(json.dumps({"results": out, "gbm": gout, "taxonomy": tax}, indent=1))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
