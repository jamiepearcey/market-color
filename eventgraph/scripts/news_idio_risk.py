# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
NEWS -> DECOMPOSED RISK: idiosyncratic vs systematic.

THE GAP THIS CLOSES. `news_risk.py` targeted TOTAL volatility. But total vol is
dominated by market/factor moves, and company news does not move the market
factor — it moves the RESIDUAL. Testing company news against total vol therefore
dilutes the very signal it should produce, by roughly the systematic share of
variance. That is the same dilution error as running a news regression over a
panel that is 99% news-free, in a different disguise.

So decompose first, then test:

    r_it = alpha_i + beta_i * f_t + eps_it        (f = equal-weighted market)

    idio_vol       = std(eps)   over the next h days   <- company-specific risk
    systematic_vol = |beta_i| * std(f)  over the next h days

TWO QUESTIONS, not one:

  Q1  Does news predict IDIOSYNCRATIC vol, incrementally over a trailing-idio-vol
      baseline? This is where company news should work if it works anywhere.

  Q2  Does the news TAXONOMY align with the RISK taxonomy? i.e. do MACRO event
      groups (monetary_policy, growth, inflation, employment...) predict
      SYSTEMATIC risk while CORPORATE groups (earnings, m_and_a, legal,
      rating_action...) predict IDIOSYNCRATIC risk? That cross-mapping is the
      classification layer's whole reason for existing, and it has never been
      tested. A taxonomy that carries no differential risk information is
      decoration.

Baselines stay deliberately strong (volatility is persistent): trailing
realised idio (or systematic) vol at 5 lags + weekday. Incremental R2 is
measured out of sample on a strictly later temporal split, conditional on news
days — identical discipline to every other test in this series.

Usage:
    uv run scripts/news_idio_risk.py
"""
from __future__ import annotations

import collections
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_volume import ols_fit, oos_r2  # noqa: E402
from liquidity import tradeable, usable_window  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
SPLIT = "2011-12-15"
HORIZONS = [5, 21]        # h=1 is near-pure noise for a vol target; excluded deliberately
BETA_WIN = 250            # trailing window for the market-model beta
MACRO_GROUPS = {"macro", "market", "commodity"}   # event_group -> systematic-ish
CORP_GROUPS = {"corporate"}                       # event_group -> idiosyncratic-ish


def returns_series(sym: str, cache: Path) -> dict[str, float]:
    p = cache / f"{sym}.json"
    if not p.exists():
        return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]
        ind = res["indicators"]
        cl = (ind.get("adjclose", [{}])[0].get("adjclose")
              if "adjclose" in ind else None) or ind["quote"][0]["close"]
    except Exception:
        return {}
    days, px = [], []
    for t, c in zip(ts, cl):
        if c and c > 0:
            days.append(dt.datetime.fromtimestamp(t, dt.UTC).strftime("%Y-%m-%d"))
            px.append(float(c))
    return {days[i]: math.log(px[i] / px[i - 1]) for i in range(1, len(px))}


def load_news_grouped() -> dict[str, dict[str, collections.Counter]]:
    """ticker -> {day: Counter(event_group)} using the classification layer."""
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
    docgrp = collections.defaultdict(set)
    for l in open(G / "classification" / "event_class.jsonl"):
        j = json.loads(l)
        if j.get("doc_id") and j.get("event_group"):
            docgrp[j["doc_id"]].add(j["event_group"])
    news = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, doc = j.get("effect_entity"), j.get("doc_id")
        d = docday.get(doc)
        if e not in ent or not d:
            continue
        for g in (docgrp.get(doc) or {"other"}):
            news[ent[e]][d][g] += 1
    return news


def main() -> None:
    news = load_news_grouped()
    cache = G / "prices"
    rets = {}
    for tk in news:
        r = returns_series(tk.replace("/", "-"), cache)
        if len(r) > 500:
            rets[tk] = r
    print(f"tickers with news + returns: {len(rets)}")

    # Equal-weighted market factor over the common trading days.
    daycount = collections.Counter()
    for r in rets.values():
        daycount.update(r.keys())  # count days, not sum returns
    thresh = max(30, int(0.30 * len(rets)))
    alldays = sorted([d for d, c in daycount.items() if c >= thresh])
    dayidx = {d: i for i, d in enumerate(alldays)}
    F = np.array([np.mean([r[d] for r in rets.values() if d in r]) for d in alldays])
    print(f"market factor: {len(alldays)} days with >={thresh} tickers reporting "
          f"({alldays[0]}..{alldays[-1]})")

    # Per ticker: aligned returns, rolling beta, residuals.
    resid, sysvol = {}, {}
    for tk, r in rets.items():
        y = np.array([r.get(d, np.nan) for d in alldays])
        ok = np.isfinite(y)
        if ok.sum() < 600:
            continue
        e = np.full(len(alldays), np.nan)
        bser = np.full(len(alldays), np.nan)
        for i in range(BETA_WIN, len(alldays)):
            sl = slice(i - BETA_WIN, i)
            m = ok[sl]
            if m.sum() < 150:
                continue
            yy, ff = y[sl][m], F[sl][m]
            fc = ff - ff.mean()
            den = float(fc @ fc)
            if den <= 0:
                continue
            b = float((yy - yy.mean()) @ fc / den)
            bser[i] = b
            if ok[i]:
                e[i] = y[i] - b * F[i]
        resid[tk] = e
        sysvol[tk] = bser
    print(f"tickers with residual series: {len(resid)}")

    def build(target, group_filter=None):
        rows = []
        for tk, e in resid.items():
            b = sysvol[tk]
            for d, groups in news.get(tk, {}).items():
                i = dayidx.get(d)
                if i is None or i < BETA_WIN + 120 or i + 1 + max(HORIZONS) >= len(alldays):
                    continue
                if group_filter is not None:
                    n = sum(v for g, v in groups.items() if g in group_filter)
                    if n == 0:
                        continue
                else:
                    n = sum(groups.values())
                rows.append((tk, d, i, float(n)))
        return rows

    def evaluate(rows, h, target):
        X_tr, y_tr, X_te, y_te = [], [], [], []
        for tk, d, i, n in rows:
            e = resid[tk]; b = sysvol[tk]
            lags = []
            ok = True
            for k in range(5):
                w = e[i - 20 * (k + 1):i - 20 * k]
                w = w[np.isfinite(w)]
                if not usable_window(w, min_len=10):
                    ok = False
                    break
                lags.append(math.log(w.std()))
            if not ok:
                continue
            fut_e = e[i + 1:i + 1 + h]
            fut_e = fut_e[np.isfinite(fut_e)]
            fut_f = F[i + 1:i + 1 + h]
            if len(fut_e) < h or not np.isfinite(b[i]):
                continue
            if target == "idio":
                y = math.log(max(fut_e.std(), 1e-8))
            else:  # systematic: |beta| * std(market) over the window
                y = math.log(max(abs(b[i]) * fut_f.std(), 1e-8))
            if not math.isfinite(y):
                continue
            wd = [0.0] * 4
            w = int(np.datetime64(d, "D").astype(int) + 4) % 7
            if w < 4:
                wd[w] = 1.0
            base = [1.0] + lags + wd
            (X_tr if d < SPLIT else X_te).append((base, [n, math.log1p(n)]))
            (y_tr if d < SPLIT else y_te).append(y)
        if len(X_tr) < 100 or len(X_te) < 60:
            return None
        def mat(X, wn):
            return np.array([b + (nf if wn else []) for b, nf in X])
        ytr, yte = np.array(y_tr), np.array(y_te)
        out = []
        for wn in (False, True):
            beta = ols_fit(mat(X_tr, wn), ytr)
            out.append(oos_r2(yte, mat(X_te, wn) @ beta, ytr.mean()))
        return out[0], out[1], out[1] - out[0], len(X_te)

    print("\n=== Q1. Does news predict IDIOSYNCRATIC vol? (all news) ===")
    print(f"  {'h':>3} {'base R2':>9} {'+news R2':>9} {'incremental':>12} {'n_test':>7}")
    rows_all = build("idio")
    for h in HORIZONS:
        r = evaluate(rows_all, h, "idio")
        print(f"  {h:3d} " + (f"{r[0]:9.4f} {r[1]:9.4f} {r[2]:12.4f} {r[3]:7d}"
                              if r else "(insufficient)"))

    print("\n=== Q2. Does the news TAXONOMY align with the RISK taxonomy? ===")
    print("    (corporate news should hit IDIO; macro news should hit SYSTEMATIC)")
    print(f"  {'news group':11} {'risk target':12} {'h':>3} {'base':>8} {'+news':>8} "
          f"{'incremental':>12} {'n_test':>7}")
    for gname, gset in (("CORPORATE", CORP_GROUPS), ("MACRO", MACRO_GROUPS)):
        rows = build("idio", gset)
        for target in ("idio", "systematic"):
            for h in HORIZONS:
                r = evaluate(rows, h, target)
                if r:
                    print(f"  {gname:11} {target:12} {h:3d} {r[0]:8.4f} {r[1]:8.4f} "
                          f"{r[2]:12.4f} {r[3]:7d}")

    print("\n  A taxonomy that carries real information should show CORPORATE news\n"
          "  helping the IDIO target more than the SYSTEMATIC one, and MACRO news the\n"
          "  reverse. Equal (or zero) everywhere means the labels are decoration.")


if __name__ == "__main__":
    main()
