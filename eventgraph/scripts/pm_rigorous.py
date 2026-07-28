# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn"]
# ///
"""
PREDICTION MARKETS under the refined methodology.

WHY REDO THIS. The earlier PM work found a Kalshi macro YES-overpricing that
survived a pre-registered temporal split (+0.164/contract, cluster-robust
t=3.22) while the Polymarket cross-regime check was weak (+0.047, t=1.24, sign
flipping by year). Those tests lacked the two things the news work eventually
forced us to adopt:

  1. AN EMPIRICALLY-CALIBRATED NULL. Every "small positive" in this project
     turned out to be inside a noise floor once one was measured. PM P&L needs
     the same treatment.
  2. A NON-LINEAR CHECK. All prior PM analysis was price-bucket calibration,
     i.e. linear-in-price. Structure could live in interactions (price x
     category x time-to-resolution) that bucketing cannot see.

PM data is a BETTER test bed than anything in the news work: the target is a
settled binary outcome — unambiguous ground truth — not a noisy volatility proxy.

THE RIGHT NULL (this is the crux). Shuffling outcomes would destroy the
price-outcome relationship entirely, testing "prices carry no information",
which is not the question and is trivially rejected. The question is whether
prices are MISCALIBRATED. So the null is a PERFECTLY CALIBRATED MARKET:

    outcome_i ~ Bernoulli(price_i)

Simulate that B times, recompute the strategy statistic each time, and compare
the observed value against that distribution. This preserves the actual price
distribution, the sample size, and the base rate, while imposing exactly the
hypothesis being tested. A p-value from this is a real p-value.

TESTS
  A. buy-NO edge in the pre-registered [0.50, 0.90) band, per venue and pooled,
     with the calibrated-null p-value and a fee/spread-aware net.
  B. non-linear: can GBM on (price, category, days-to-resolution, volume, year)
     beat price-alone at predicting the outcome? Scored by log-loss, with the
     same calibrated null, and reported against a permutation control.

Usage:
    uv run scripts/pm_rigorous.py
"""
from __future__ import annotations

import collections
import glob
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KALSHI_DIR = ROOT / "data" / "eg_live" / "kalshi_markets"
POLY_PATH = Path("/tmp/pm_run/polymarket_macro/markets.jsonl")
BAND = (0.50, 0.90)          # the pre-registered buy-NO band
B_SIM = 4000
RNG = np.random.default_rng(20260726)

MACRO_KW = ("cpi", "inflation", "fed", "fomc", "rate", "gdp", "jobs", "payroll",
            "unemploy", "jobless", "economic", "pce", "recession")


def load_polymarket():
    if not POLY_PATH.exists():
        return []
    out = []
    for l in open(POLY_PATH):
        j = json.loads(l)
        try:
            p = float(j["price_yes"])
        except Exception:
            continue
        o = j.get("outcome")
        if o not in ("yes", "no") or not (0 < p < 1):
            continue
        out.append({
            "venue": "polymarket", "price": p, "y": 1.0 if o == "yes" else 0.0,
            "cat": j.get("subcat") or "other", "year": str(j.get("resolution_year")),
            "date": str(j.get("resolution_date")),
            "days": float(j.get("actual_days_before_resolution") or 7),
            "vol": float(j.get("volume") or 0.0),
            "series": (j.get("question") or "")[:40],
        })
    return out


def load_kalshi():
    out = []
    for f in glob.glob(str(KALSHI_DIR / "*.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        markets = d if isinstance(d, list) else [d]
        for m in markets:
            if not isinstance(m, dict):
                continue
            res = str(m.get("expiration_value") or m.get("result") or "").lower()
            if res not in ("yes", "no"):
                continue
            price = m.get("last_price_dollars") or m.get("last_price")
            if price in (None, "", "0.0000"):
                continue
            p = float(price)
            if p > 1.0:
                p /= 100.0
            if not (0 < p < 1):
                continue
            title = " ".join(str(m.get(k) or "") for k in ("title","ticker","event_ticker","no_sub_title"))
            low = title.lower()
            out.append({
                "venue": "kalshi", "price": p, "y": 1.0 if res == "yes" else 0.0,
                "cat": "macro" if any(k in low for k in MACRO_KW) else "other",
                "year": str(m.get("close_time") or "")[:4],
                "date": str(m.get("close_time") or "")[:10],
                "days": 7.0, "vol": float(m.get("liquidity_dollars") or m.get("volume") or 0.0),
                "series": str(m.get("event_ticker") or m.get("ticker") or "")[:24],
            })
    return out


def buy_no_pnl(price: np.ndarray, y: np.ndarray, fee: float = 0.0) -> float:
    """Mean P&L per contract of buying NO. Cost is (1 - price); it pays $1 when
    the outcome is NO, so profit = +price; when YES the (1 - price) stake is
    lost. NOTE a perfectly calibrated market gives EXACTLY zero expected P&L:
    (1-p)*p + p*(-(1-p)) = 0 — which is why the Bernoulli(price) null is the
    correct one and should centre on 0."""
    return float(np.mean(np.where(y < 0.5, price, -(1.0 - price))) - fee)


def calibrated_null(price: np.ndarray, stat_fn, b: int = B_SIM) -> np.ndarray:
    """Null distribution of a statistic under a PERFECTLY CALIBRATED market:
    outcome_i ~ Bernoulli(price_i). Preserves prices, n, and base rate."""
    draws = (RNG.random((b, len(price))) < price[None, :]).astype(float)
    return np.array([stat_fn(price, draws[i]) for i in range(b)])


def report_band(rows, label):
    p = np.array([r["price"] for r in rows])
    y = np.array([r["y"] for r in rows])
    m = (p >= BAND[0]) & (p < BAND[1])
    if m.sum() < 25:
        print(f"  {label:26} (n={m.sum()} in band — too few)")
        return
    pb, yb = p[m], y[m]
    obs = buy_no_pnl(pb, yb)
    null = calibrated_null(pb, buy_no_pnl)
    pval = float(np.mean(null >= obs))
    # Kalshi-style fee on the traded price, applied pessimistically.
    fee = float(np.mean(np.ceil(0.07 * pb * (1 - pb) * 100) / 100))
    print(f"  {label:26} n={m.sum():5d}  realised_YES={yb.mean():.3f} vs price={pb.mean():.3f}"
          f"  gross={obs:+.4f}  net_of_fee={obs - fee:+.4f}"
          f"  null_mean={null.mean():+.4f} null_p95={np.percentile(null, 95):+.4f}"
          f"  p={pval:.4f}")


def main() -> None:
    poly, kal = load_polymarket(), load_kalshi()
    print(f"polymarket rows: {len(poly)}   kalshi rows: {len(kal)}")
    if kal:
        print("  kalshi cat:", dict(collections.Counter(r['cat'] for r in kal)))

    print(f"\n=== A. buy-NO in [{BAND[0]}, {BAND[1]}) vs a PERFECTLY-CALIBRATED null "
          f"(outcome ~ Bernoulli(price), {B_SIM} sims) ===")
    report_band(poly, "polymarket macro")
    for yr in sorted({r["year"] for r in poly}):
        report_band([r for r in poly if r["year"] == yr], f"  polymarket {yr}")
    if kal:
        report_band(kal, "kalshi all")
        report_band([r for r in kal if r["cat"] == "macro"], "kalshi macro")
        report_band([r for r in kal if r["cat"] == "other"], "kalshi other")
    report_band(poly + kal, "POOLED")

    # ---- B. non-linear: is there structure beyond the price? ----
    allr = poly + kal
    if len(allr) < 500:
        return
    from sklearn.ensemble import HistGradientBoostingRegressor

    allr.sort(key=lambda r: r["date"])
    cut = int(len(allr) * 0.65)
    tr, te = allr[:cut], allr[cut:]
    cats = sorted({r["cat"] for r in allr})
    ven = sorted({r["venue"] for r in allr})

    def feats(rs):
        return np.array([[r["price"], math.log1p(max(r["vol"], 0)), r["days"]]
                         + [1.0 if r["cat"] == c else 0.0 for c in cats]
                         + [1.0 if r["venue"] == v else 0.0 for v in ven] for r in rs])

    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    ptr = np.array([r["price"] for r in tr]); pte = np.array([r["price"] for r in te])

    def logloss(pred, y):
        q = np.clip(pred, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    base_ll = logloss(pte, yte)                       # the market price itself
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                      early_stopping=True, random_state=0)
    m.fit(feats(tr), ytr)
    gbm_ll = logloss(m.predict(feats(te)), yte)

    # Permutation control: shuffle the non-price features against the targets.
    Xtr, Xte = feats(tr), feats(te)
    perm_tr = RNG.permutation(len(tr)); perm_te = RNG.permutation(len(te))
    Xtr_p = Xtr.copy(); Xte_p = Xte.copy()
    Xtr_p[:, 1:] = Xtr[perm_tr, 1:]; Xte_p[:, 1:] = Xte[perm_te, 1:]
    m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                       early_stopping=True, random_state=0)
    m2.fit(Xtr_p, ytr)
    perm_ll = logloss(m2.predict(Xte_p), yte)

    print(f"\n=== B. NON-LINEAR: can features beat the price at predicting the outcome? "
          f"(train={len(tr)} test={len(te)}) ===")
    print(f"  {'model':38} {'log-loss':>10}  (lower is better)")
    print(f"  {'market price alone':38} {base_ll:10.4f}")
    print(f"  {'GBM(price + cat + days + vol + venue)':38} {gbm_ll:10.4f}   "
          f"improvement {base_ll - gbm_ll:+.4f}")
    print(f"  {'GBM with SHUFFLED non-price features':38} {perm_ll:10.4f}   "
          f"improvement {base_ll - perm_ll:+.4f}   <- null control")
    print("\n  The GBM improvement must clearly exceed the shuffled control to count.")
    print("  Null in A is a CALIBRATED market (outcome ~ Bernoulli(price)) — not shuffled\n"
          "  outcomes, which would test 'prices carry no information' and is not the\n"
          "  question. p is the fraction of calibrated-null draws at or above observed.")


if __name__ == "__main__":
    main()
