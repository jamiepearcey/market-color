# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "duckdb"]
# ///
"""
NEWS -> VOLUME, the three cuts the first study could not rule out.

The first study (news_volume.py) established, on the ~679 most liquid US names
at daily resolution: news is INFORMATIVE (news-only OOS R2 = 0.036) but
REDUNDANT — same-day volume vol_z(T) is a sufficient statistic, so news adds
-0.002 incremental R2 for T+1. That is a real result, but it was run in the
hardest possible setting. Three cuts with genuinely different priors were left
untested, and all three are answerable with data already on disk:

  A. LIQUIDITY. The universe was the top names by dollar volume — the most
     analyst-covered, most efficiently priced securities there are. News should
     matter MOST where coverage is thin. Measured first: the corpus DOES cover
     small caps, but ~4x more thinly (median 2 edges/name vs 9 for large caps
     over 2010-12), so this is a pooled test, not a per-name one — and that
     coverage gradient is itself a confound to report, not hide.

  B. HORIZON. T+1 was the wrong horizon by the first study's own timing
     evidence (the volume spike lands ON the news day and is >half decayed by
     T+1). Drift-type effects (post-earnings drift) live at weeks. Sweep
     h = 1, 2, 3, 5, 10, 21.

  C. CROSS-SECTION. The first study asked "how much volume" per name. The
     different question is "WHICH names" — on a given day, does news intensity
     RANK the cross-section of abnormal volume? Rank correlation is a much
     weaker, and therefore more attainable, claim than level prediction.

Not answerable here (data ceilings, stated so they are not mistaken for nulls):
intraday lead/lag (published_at is date-resolution), options implied-vs-realised
(history still accruing), India (ticker-unresolved).

Method is unchanged from the first study: news on day T, predict vol_z(T+h),
baseline = volume-AR (vol_z at T..T-4) + weekday, incremental R2 measured OUT OF
SAMPLE on a strictly later temporal split, conditional on news days.

Usage:
    uv run scripts/news_volume_extended.py
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from news_volume import load_jsonl, ols_fit, oos_r2, volume_z_series  # noqa: E402
from liquidity import tradeable, usable_window  # noqa: E402

G = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs" / "eg100k_graph"
PARQUET = ("/Users/jamiepearcey/projects/finance/quant-algos/data/parquet/daily/"
           "market=us/*/*.parquet")
HORIZONS = [1, 2, 3, 5, 10, 21]
SPLIT = "2011-12-15"  # temporal train/test boundary (same as the first study)


def load_news() -> tuple[dict, dict]:
    """(ticker -> {day: n_edges}), (ticker -> entity_id)."""
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
    news = collections.defaultdict(lambda: collections.Counter())
    for l in open(G / "lake" / "causal_event_edge.jsonl"):
        j = json.loads(l)
        e, d = j.get("effect_entity"), docday.get(j.get("doc_id"))
        if e in ent and d:
            news[ent[e]][d] += 1
    return news, ent


def liquidity(tickers: list[str]) -> dict[str, float]:
    c = duckdb.connect()
    syms = ",".join("'" + t + ".US'" for t in tickers)
    rows = c.execute(
        f"""SELECT symbol, median(close*vol) FROM read_parquet('{PARQUET}', hive_partitioning=1)
            WHERE date BETWEEN 20100101 AND 20121231 AND close > 1 AND vol > 0
              AND symbol IN ({syms}) GROUP BY 1"""
    ).fetchall()
    return {r[0].replace(".US", ""): r[1] for r in rows}


def build_rows(news, vz_by_ticker, horizon):
    """One row per (ticker, news-day): AR lags + news features + target vol_z(T+h)."""
    rows = []
    for tk, vz in vz_by_ticker.items():
        days = sorted(vz)
        pos = {d: i for i, d in enumerate(days)}
        for d, n_edges in news.get(tk, {}).items():
            i = pos.get(d)
            if i is None or i < 5 or i + horizon >= len(days):
                continue
            lags = [vz[days[i - k]] for k in range(5)]
            if not all(math.isfinite(v) for v in lags):
                continue
            y = vz[days[i + horizon]]
            if not math.isfinite(y):
                continue
            wd = [0.0] * 4
            w = int(np.datetime64(d, "D").astype("datetime64[D]").astype(int) + 4) % 7
            if w < 4:
                wd[w] = 1.0
            rows.append({"tk": tk, "day": d, "lags": lags, "wd": wd,
                         "news": [float(n_edges), math.log1p(n_edges)], "y": y})
    return rows


def incremental_r2(rows, split=SPLIT):
    """OOS incremental R2 of news over the volume-AR baseline. (base, with, incr, n_test)."""
    tr = [r for r in rows if r["day"] < split]
    te = [r for r in rows if r["day"] >= split]
    if len(tr) < 60 or len(te) < 40:
        return None
    def design(rs, with_news):
        return np.array([[1.0] + r["lags"] + r["wd"] + (r["news"] if with_news else []) for r in rs])
    ytr = np.array([r["y"] for r in tr]); yte = np.array([r["y"] for r in te])
    sst = ytr.mean()
    out = []
    for wn in (False, True):
        b = ols_fit(design(tr, wn), ytr)
        out.append(oos_r2(yte, design(te, wn) @ b, sst))
    return out[0], out[1], out[1] - out[0], len(te)


def rank_ic(news, vz_by_ticker, horizon=1, split=SPLIT):
    """Cross-sectional: on each day, does news intensity rank abnormal volume?"""
    by_day = collections.defaultdict(list)
    for tk, vz in vz_by_ticker.items():
        days = sorted(vz); pos = {d: i for i, d in enumerate(days)}
        for d, n in news.get(tk, {}).items():
            i = pos.get(d)
            if i is None or i + horizon >= len(days):
                continue
            y = vz[days[i + horizon]]
            if math.isfinite(y):
                by_day[d].append((n, y))
    ics = []
    for d, obs in by_day.items():
        if d < split or len(obs) < 5:
            continue
        a = np.array([o[0] for o in obs], float); b = np.array([o[1] for o in obs], float)
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        if ra.std() > 0 and rb.std() > 0:
            ics.append(float(np.corrcoef(ra, rb)[0, 1]))
    if not ics:
        return None
    ics = np.array(ics)
    return float(ics.mean()), float(ics.std(ddof=1) / math.sqrt(len(ics))), len(ics)


def main() -> None:
    news, ent = load_news()
    cache = G / "prices"
    tickers = sorted(news)
    print(f"tickers with news: {len(tickers)}")

    vz_by_ticker = {}
    for tk in tickers:
        vz = volume_z_series(tk.replace("/", "-"), cache)
        if vz and len(vz) > 300:
            vz_by_ticker[tk] = vz
    print(f"with usable volume series: {len(vz_by_ticker)}")

    adv = liquidity(list(vz_by_ticker))
    common = [t for t in vz_by_ticker if t in adv]
    common.sort(key=lambda t: adv[t])
    k = len(common) // 3
    terciles = {"SMALL": common[:k], "MID": common[k:2 * k], "LARGE": common[2 * k:]}
    print(f"liquidity terciles from {len(common)} tickers "
          f"(median ADV: " + ", ".join(
              f"{g}=${np.median([adv[t] for t in ts])/1e6:.0f}m" for g, ts in terciles.items()) + ")")

    # ---- B. HORIZON SWEEP (all names) ----
    print(f"\n=== B. HORIZON: incremental OOS R2 of news over volume-AR (test from {SPLIT}) ===")
    print(f"  {'h':>3} {'base R2':>9} {'+news R2':>9} {'incremental':>12} {'n_test':>7}")
    for h in HORIZONS:
        rows = build_rows(news, vz_by_ticker, h)
        r = incremental_r2(rows)
        if r:
            print(f"  {h:3d} {r[0]:9.4f} {r[1]:9.4f} {r[2]:12.4f} {r[3]:7d}")

    # ---- A. LIQUIDITY TERCILES (h=1) ----
    print("\n=== A. LIQUIDITY: same test within each tercile (h=1) ===")
    print(f"  {'tercile':8} {'base R2':>9} {'+news R2':>9} {'incremental':>12} {'n_test':>7}")
    for g, ts in terciles.items():
        sub = {t: vz_by_ticker[t] for t in ts}
        rows = build_rows(news, sub, 1)
        r = incremental_r2(rows)
        print(f"  {g:8} " + (f"{r[0]:9.4f} {r[1]:9.4f} {r[2]:12.4f} {r[3]:7d}"
                             if r else "      (insufficient observations)"))

    # ---- C. CROSS-SECTION ----
    print("\n=== C. CROSS-SECTION: rank IC of news intensity vs abnormal volume ===")
    print(f"  {'h':>3} {'mean rank IC':>13} {'se':>9} {'t':>7} {'n_days':>7}")
    for h in (0, 1, 5):
        r = rank_ic(news, vz_by_ticker, h)
        if r:
            t = r[0] / r[1] if r[1] > 0 else 0.0
            tag = "  <- contemporaneous (not a prediction)" if h == 0 else ""
            print(f"  {h:3d} {r[0]:13.4f} {r[1]:9.4f} {t:7.2f} {r[2]:7d}{tag}")

    print("\n  NOTE the liquidity confound: small caps carry ~4x less news per name, so a "
          "weaker\n  small-cap result may be thinner coverage rather than absent structure.")


if __name__ == "__main__":
    main()
