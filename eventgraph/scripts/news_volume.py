# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy"]
# ///
"""
NEWS -> NEXT-DAY ABNORMAL VOLUME: incremental-predictive-power study.

Hypothesis under test (the ONLY one that counts): volume is highly persistent, so the
question is not "does news correlate with volume" (trivial/near-tautological same-day)
but does news dated day T add INCREMENTAL out-of-sample predictive power for abnormal
volume on day T+1, over and above a volume-autoregression baseline (5 lags + weekday
dummies)?

Pipeline (real data only, no network):
  universe   = entity_class.jsonl resolution_status=="resolved_security" AND a usable
               price/volume series in <graph>/prices/ (reuse activation_split.volume_z_series
               conventions: log-volume z vs trailing 60-obs mean/sd).
  news       = causal_event_edge.jsonl joined doc_id->published_at (document.jsonl) and
               effect_entity->entity_class.entity_id->resolved_ticker (the tradable
               universe only). Per (ticker, day) we count edges and distinct
               std_mechanisms (from edge_class.jsonl), and per (ticker,day) we attach the
               std_event_type set from event.jsonl/event_class.jsonl matched by
               (doc_id, issuer_entity==effect_entity) -- events don't carry event_id
               links into causal_event_edge (always null in this graph), so the
               doc_id+issuer_entity join is the correct path.
  leakage    = news on day T predicts volume z on day T+1 ONLY. Same-day (T vs T) is
               reported SEPARATELY as a sanity/contemporaneous check, never fed to the
               T+1 model. Weekend/holiday gaps: T+1 means the next TRADING day present
               in that ticker's own volume_z_series (calendar-day gaps collapse
               naturally since the series is keyed by actual trading days).
  baseline   = OLS: vol_z(T+1) ~ vol_z(T..T-4) + weekday(T+1) dummies.
  model      = OLS: vol_z(T+1) ~ [baseline features] + news features on day T.
  split      = temporal by date: first ~65% of TEST-eligible days per ticker pooled
               chronologically (global date cut, not per-ticker) = train, remainder =
               test. Fit baseline-only and baseline+news on TRAIN, evaluate BOTH on
               TEST (held out, never touched during feature/model selection).
  metrics    = OOS R^2 (1 - SSE/SST using TEST mean as SST reference), incremental
               R^2 (baseline+news minus baseline, both OOS), IC = corr(pred, actual)
               on TEST for the news model.
  cuts       = mean next-day abnormal volume by std_event_type (N); scheduled vs
               unscheduled subset incremental R^2; winsorised-y (+-5sd) robustness re-run.

Usage: uv run scripts/news_volume.py --graph-dir ../data/eg_runs/eg100k_graph
"""
import argparse, json, collections, sys, math, datetime as dt
from pathlib import Path
import sys
from pathlib import Path

import numpy as np
from liquidity import tradeable, usable_window  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))


def volume_z_series(symn, cache):
    """day -> log-volume z vs trailing 60 obs, from the cached Yahoo chart JSON.
    (inlined from activation_split.py to avoid its httpx import chain -- this
    study is disk-only, no network calls.)"""
    p = cache / f"{symn}.json"
    if not p.exists():
        return {}
    try:
        res = json.loads(p.read_text())["chart"]["result"][0]
        ts = res["timestamp"]
        vol = (res["indicators"]["quote"][0].get("volume") or [])
    except Exception:
        return {}
    days, lv = [], []
    for t, v in zip(ts, vol):
        if v:
            days.append(dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d"))
            lv.append(math.log(v))
    out = {}
    for i in range(60, len(days)):
        w = np.array(lv[i - 60:i])
        sd = w.std()
        if sd > 0:
            out[days[i]] = (lv[i] - w.mean()) / sd
    return out

SCHEDULED = {"earnings", "guidance", "monetary_policy", "inflation", "employment",
             "growth", "econ_indicator"}


def load_jsonl(path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def ols_fit(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def oos_r2(y_true, y_pred, sst_mean):
    sse = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - sst_mean) ** 2)
    return 1.0 - sse / sst if sst > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph-dir", default="../data/eg_runs/eg100k_graph")
    a = ap.parse_args()
    gd = Path(a.graph_dir)
    lake = gd / "lake"
    cls = gd / "classification"
    cache = gd / "prices"
    out = {}

    # ---- 1. universe: resolved_security entities with usable volume series ----
    ent_rows = list(load_jsonl(cls / "entity_class.jsonl"))
    resolved = [r for r in ent_rows if r["resolution_status"] == "resolved_security"]
    ticker_of_entity = {}   # entity_id -> ticker (only resolved_security)
    for r in resolved:
        ticker_of_entity[r["entity_id"]] = r["resolved_ticker"]

    print(f"resolved_security entities: {len(resolved)}")

    vz_by_ticker = {}   # ticker -> {day: z}
    n_usable = 0
    seen_tickers = set()
    for r in resolved:
        t = r["resolved_ticker"]
        if t in seen_tickers:
            continue
        seen_tickers.add(t)
        symn = t.replace("/", "-")  # yahoo cache uses raw ticker string as filename stem
        vz = volume_z_series(symn, cache)
        # restrict to 2010-2012 window as per graph coverage
        vz = {d: z for d, z in vz.items() if d[:4] in ("2010", "2011", "2012")}
        if len(vz) >= 300:
            vz_by_ticker[t] = vz
            n_usable += 1

    print(f"tickers with >=300 usable 2010-2012 vol-z obs: {n_usable}")

    # ---- 2. news joins ----
    docdate = {}
    for j in load_jsonl(lake / "document.jsonl"):
        pa = j.get("published_at")
        if pa:
            docdate[j["doc_id"]] = pa[:10]

    # edge_class: doc_id, cause_entity, effect_entity, std_mechanism, effect_dir
    # (edge_class rows are positional-parallel to causal_event_edge.jsonl; join by
    # doc_id+effect_entity since there's no edge_key kept in edge_class.jsonl)
    edge_class_rows = list(load_jsonl(cls / "edge_class.jsonl"))

    # per (ticker, day): set of std_mechanisms, edge count
    news_edges = collections.defaultdict(lambda: {"n": 0, "mechs": set()})
    n_edge_join = 0
    for j in edge_class_rows:
        eff = j["effect_entity"]
        t = ticker_of_entity.get(eff)
        if not t or t not in vz_by_ticker:
            continue
        d = docdate.get(j["doc_id"])
        if not d:
            continue
        key = (t, d)
        news_edges[key]["n"] += 1
        news_edges[key]["mechs"].add(j["std_mechanism"])
        n_edge_join += 1

    # event.jsonl + event_class.jsonl (parallel by event_id) -> issuer_entity std_event_type
    event_rows = list(load_jsonl(lake / "event.jsonl"))
    event_class_by_id = {j["event_id"]: j for j in load_jsonl(cls / "event_class.jsonl")}
    news_events = collections.defaultdict(set)  # (ticker, day) -> {std_event_type}
    n_event_join = 0
    for ev in event_rows:
        t = ticker_of_entity.get(ev.get("issuer_entity"))
        if not t or t not in vz_by_ticker:
            continue
        d = docdate.get(ev.get("doc_id"))
        if not d:
            continue
        ec = event_class_by_id.get(ev["event_id"])
        if not ec:
            continue
        news_events[(t, d)].add(ec["std_event_type"])
        n_event_join += 1

    all_news_keys = set(news_edges) | set(news_events)
    print(f"edge-news joins: {n_edge_join}, event-news joins: {n_event_join}, "
          f"distinct (ticker,day) news keys: {len(all_news_keys)}")

    std_event_types = sorted({et for s in news_events.values() for et in s})
    et_index = {et: i for i, et in enumerate(std_event_types)}

    # ---- 3. build panel: for each ticker, each day T with T and T+1 both present
    # in that ticker's own trading-day-indexed vol_z series (so "T+1" = next trading
    # day for THAT ticker, no cross-ticker calendar assumptions) ----
    rows = []  # dict per ticker-day-T+1 obs
    for t, vz in vz_by_ticker.items():
        days = sorted(vz)
        day_idx = {d: i for i, d in enumerate(days)}
        for i in range(4, len(days) - 1):
            d_t = days[i]        # day T
            d_t1 = days[i + 1]   # day T+1 (next trading day for this ticker)
            lags = [vz[days[i - k]] for k in range(0, 5)]  # T, T-1..T-4
            y = vz[d_t1]
            key = (t, d_t)
            ne = news_edges.get(key)
            evs = news_events.get(key, set())
            n_edges = ne["n"] if ne else 0
            n_mechs = len(ne["mechs"]) if ne else 0
            has_news = 1 if (ne or evs) else 0
            weekday_t1 = dt.date.fromisoformat(d_t1).weekday()
            rows.append(dict(ticker=t, day_t=d_t, day_t1=d_t1, y=y, lags=lags,
                              n_edges=n_edges, n_mechs=n_mechs, n_types=len(evs),
                              has_news=has_news, event_types=evs, weekday=weekday_t1))

    n_total = len(rows)
    n_news = sum(1 for r in rows if r["has_news"])
    print(f"ticker-days (T+1 obs): {n_total}, with any news on T: {n_news}")

    # ---- contemporaneous sanity check: news(T) vs vol_z(T), NOT a prediction ----
    same_day_news = np.array([1 if r["has_news"] else 0 for r in rows])
    same_day_vz = np.array([r["lags"][0] for r in rows])  # vol_z at T itself
    if same_day_news.std() > 0:
        contemp_corr = float(np.corrcoef(same_day_news, same_day_vz)[0, 1])
    else:
        contemp_corr = float("nan")
    print(f"[sanity, NOT a prediction] corr(has_news(T), vol_z(T)) = {contemp_corr:+.3f}  n={n_total}")

    # ---- feature matrices ----
    dates_sorted = sorted(set(r["day_t1"] for r in rows))
    cut_idx = int(len(dates_sorted) * 0.65)
    cut_date = dates_sorted[cut_idx]
    print(f"temporal split: train < {cut_date}, test >= {cut_date} "
          f"({cut_idx}/{len(dates_sorted)} dates)")

    def build_X(rows_subset, with_news):
        n = len(rows_subset)
        wd = np.zeros((n, 6))  # 6 weekday dummies (drop Sunday as baseline; rare anyway)
        for i, r in enumerate(rows_subset):
            if r["weekday"] < 6:
                wd[i, r["weekday"]] = 1.0
        lags = np.array([r["lags"] for r in rows_subset])
        base = np.column_stack([np.ones(n), lags, wd])
        if not with_news:
            return base
        et_mat = np.zeros((n, len(std_event_types)))
        for i, r in enumerate(rows_subset):
            for et in r["event_types"]:
                et_mat[i, et_index[et]] = 1.0
        news_feat = np.column_stack([
            [r["n_edges"] for r in rows_subset],
            [r["n_mechs"] for r in rows_subset],
            [r["n_types"] for r in rows_subset],
            et_mat,
        ])
        return np.column_stack([base, news_feat])

    def run_split(rows_subset, winsorize=False, label=""):
        train = [r for r in rows_subset if r["day_t1"] < cut_date]
        test = [r for r in rows_subset if r["day_t1"] >= cut_date]
        if len(train) < 50 or len(test) < 50:
            return None
        y_train = np.array([r["y"] for r in train])
        y_test = np.array([r["y"] for r in test])
        if winsorize:
            y_train = np.clip(y_train, -5, 5)
            y_test_eval = np.clip(y_test, -5, 5)
        else:
            y_test_eval = y_test

        Xb_train, Xb_test = build_X(train, False), build_X(test, False)
        Xn_train, Xn_test = build_X(train, True), build_X(test, True)

        beta_b = ols_fit(Xb_train, y_train)
        beta_n = ols_fit(Xn_train, y_train)

        pred_b = Xb_test @ beta_b
        pred_n = Xn_test @ beta_n

        sst_mean = y_train.mean()  # reference mean from TRAIN, applied to TEST SST
        r2_b = oos_r2(y_test_eval, pred_b, sst_mean)
        r2_n = oos_r2(y_test_eval, pred_n, sst_mean)
        ic = float(np.corrcoef(pred_n, y_test_eval)[0, 1]) if np.std(pred_n) > 0 else float("nan")

        return dict(n_train=len(train), n_test=len(test),
                     r2_baseline=r2_b, r2_news=r2_n, incremental_r2=r2_n - r2_b, ic=ic)

    main_result = run_split(rows, winsorize=False, label="all")
    wins_result = run_split(rows, winsorize=True, label="winsorized")

    unscheduled_rows = [r for r in rows if r["event_types"] and
                         not (r["event_types"] & SCHEDULED) or
                         (not r["event_types"] and False)]
    # unscheduled subset = news days where event types are exclusively non-scheduled
    unsched_rows = [r for r in rows if r["has_news"] and r["event_types"] and
                    r["event_types"].isdisjoint(SCHEDULED)]
    # include also n_edges-only news (edge without event) as "unscheduled-ish"? keep strict:
    # only rows where we actually have a std_event_type set that is unscheduled.
    unsched_result = run_split(unsched_rows, winsorize=False) if len(unsched_rows) >= 150 else None

    # ---- event-type cut: mean next-day abnormal volume by std_event_type ----
    et_stats = collections.defaultdict(list)
    for r in rows:
        for et in r["event_types"]:
            et_stats[et].append(r["y"])
    et_summary = sorted(
        [(et, float(np.mean(v)), len(v)) for et, v in et_stats.items() if len(v) >= 10],
        key=lambda x: -x[1],
    )

    # ==================================================================
    # SHARP TEST: condition on NEWS DAYS ONLY (the full-panel result above
    # is diluted by construction -- 99.1% of rows have all-zero news
    # features, so incremental R^2 over the full panel is mechanically
    # ~0 regardless of how informative news is on the days it fires).
    # ==================================================================
    news_rows = [r for r in rows if r["has_news"]]
    news_dates_sorted = sorted(set(r["day_t1"] for r in news_rows))
    news_cut_idx = int(len(news_dates_sorted) * 0.65)
    news_cut_date = news_dates_sorted[news_cut_idx] if news_dates_sorted else None
    print(f"\n[SHARP TEST] news-day sample: n={len(news_rows)}, "
          f"temporal split train < {news_cut_date}, test >= {news_cut_date} "
          f"({news_cut_idx}/{len(news_dates_sorted)} dates)")

    def build_X_lag0(rows_subset, with_news, with_event_types):
        """AR-with-lag-0 design: vol_z(T), vol_z(T-1..T-4) already in r['lags']
        (index 0 = T), + weekday dummies, optionally + news features."""
        n = len(rows_subset)
        wd = np.zeros((n, 6))
        for i, r in enumerate(rows_subset):
            if r["weekday"] < 6:
                wd[i, r["weekday"]] = 1.0
        lags = np.array([r["lags"] for r in rows_subset])
        base = np.column_stack([np.ones(n), lags, wd])
        if not with_news:
            return base
        cols = [base, np.array([[r["n_edges"]] for r in rows_subset])]
        if with_event_types:
            et_mat = np.zeros((n, len(std_event_types)))
            for i, r in enumerate(rows_subset):
                for et in r["event_types"]:
                    et_mat[i, et_index[et]] = 1.0
            cols.append(np.array([[r["n_types"]] for r in rows_subset]))
            cols.append(et_mat)
        return np.column_stack(cols)

    def build_X_news_only(rows_subset):
        """news features only, no volume lags (still + weekday, since weekday
        is a news-independent calendar control, not volume information)."""
        n = len(rows_subset)
        wd = np.zeros((n, 6))
        for i, r in enumerate(rows_subset):
            if r["weekday"] < 6:
                wd[i, r["weekday"]] = 1.0
        et_mat = np.zeros((n, len(std_event_types)))
        for i, r in enumerate(rows_subset):
            for et in r["event_types"]:
                et_mat[i, et_index[et]] = 1.0
        return np.column_stack([np.ones(n), wd,
                                 np.array([[r["n_edges"]] for r in rows_subset]),
                                 np.array([[r["n_types"]] for r in rows_subset]),
                                 et_mat])

    def fit_eval(Xtr, ytr, Xte, yte, sst_mean):
        beta = ols_fit(Xtr, ytr)
        pred = Xte @ beta
        r2 = oos_r2(yte, pred, sst_mean)
        ic = float(np.corrcoef(pred, yte)[0, 1]) if np.std(pred) > 0 else float("nan")
        return r2, ic, pred

    def news_day_test2(rows_subset, cut_date_local, label):
        train = [r for r in rows_subset if r["day_t1"] < cut_date_local]
        test = [r for r in rows_subset if r["day_t1"] >= cut_date_local]
        if len(train) < 40 or len(test) < 40:
            return dict(note=f"too small for a stable OOS split (train={len(train)}, test={len(test)})",
                         n_train=len(train), n_test=len(test))
        y_train = np.array([r["y"] for r in train])
        y_test = np.array([r["y"] for r in test])
        sst_mean = y_train.mean()

        Xa_tr, Xa_te = build_X_lag0(train, False, False), build_X_lag0(test, False, False)
        Xb_tr, Xb_te = build_X_lag0(train, True, True), build_X_lag0(test, True, True)
        Xc_tr, Xc_te = build_X_news_only(train), build_X_news_only(test)

        r2_a, ic_a, _ = fit_eval(Xa_tr, y_train, Xa_te, y_test, sst_mean)
        r2_b, ic_b, _ = fit_eval(Xb_tr, y_train, Xb_te, y_test, sst_mean)
        r2_c, ic_c, _ = fit_eval(Xc_tr, y_train, Xc_te, y_test, sst_mean)

        return dict(n_train=len(train), n_test=len(test),
                    r2_ar_lag0=r2_a, r2_ar_lag0_plus_news=r2_b, r2_news_only=r2_c,
                    incremental_r2_news_over_ar=r2_b - r2_a,
                    ic_ar_lag0=ic_a, ic_ar_lag0_plus_news=ic_b, ic_news_only=ic_c)

    sharp_test2 = news_day_test2(news_rows, news_cut_date, "news-day AR-lag0 vs +news vs news-only")

    # ---- test 3: event-type content beyond raw news arrival ----
    def event_type_content_test(rows_subset, cut_date_local):
        train = [r for r in rows_subset if r["day_t1"] < cut_date_local]
        test = [r for r in rows_subset if r["day_t1"] >= cut_date_local]
        if len(train) < 40 or len(test) < 40:
            return dict(note=f"too small (train={len(train)}, test={len(test)})",
                         n_train=len(train), n_test=len(test))
        y_train = np.array([r["y"] for r in train])
        y_test = np.array([r["y"] for r in test])
        sst_mean = y_train.mean()
        # (a)+has_news_count: AR-lag0 + n_edges only, no event-type indicators
        Xarrival_tr, Xarrival_te = build_X_lag0(train, True, False), build_X_lag0(test, True, False)
        # (a)+has_news_count+event_type_indicators
        Xtyped_tr, Xtyped_te = build_X_lag0(train, True, True), build_X_lag0(test, True, True)
        r2_arrival, ic_arrival, _ = fit_eval(Xarrival_tr, y_train, Xarrival_te, y_test, sst_mean)
        r2_typed, ic_typed, _ = fit_eval(Xtyped_tr, y_train, Xtyped_te, y_test, sst_mean)
        return dict(n_train=len(train), n_test=len(test),
                    r2_arrival_only=r2_arrival, r2_arrival_plus_type=r2_typed,
                    incremental_r2_from_typing=r2_typed - r2_arrival)

    event_type_content = event_type_content_test(news_rows, news_cut_date)

    # ---- test 4: unscheduled-only version of test 2 ----
    unsched_news_rows = [r for r in news_rows if r["event_types"] and
                          r["event_types"].isdisjoint(SCHEDULED)]
    unsched_dates = sorted(set(r["day_t1"] for r in unsched_news_rows))
    if len(unsched_news_rows) >= 150 and len(unsched_dates) >= 10:
        u_cut = unsched_dates[int(len(unsched_dates) * 0.65)]
        sharp_test4 = news_day_test2(unsched_news_rows, u_cut, "unscheduled-only")
    else:
        sharp_test4 = dict(note=f"N too small for a stable OOS split (n={len(unsched_news_rows)}, "
                                 f"{len(unsched_dates)} distinct dates) -- not reporting a regression here",
                            n=len(unsched_news_rows))

    # ---- test 5: timing -- mean vol_z(T) vs vol_z(T+1) on news days ----
    mean_vz_T = float(np.mean([r["lags"][0] for r in news_rows]))
    mean_vz_T1 = float(np.mean([r["y"] for r in news_rows]))
    print(f"\n[timing] on news days (n={len(news_rows)}): mean vol_z(T) = {mean_vz_T:+.3f}, "
          f"mean vol_z(T+1) = {mean_vz_T1:+.3f}")

    out = dict(
        n_tickers=n_usable,
        n_ticker_days=n_total,
        n_ticker_days_with_news=n_news,
        date_span=[dates_sorted[0], dates_sorted[-1]],
        contemporaneous_sanity_corr=contemp_corr,
        headline_full_panel_diluted_by_construction=main_result,
        winsorized_full_panel=wins_result,
        unscheduled_subset_full_panel=dict(n=len(unsched_rows), result=unsched_result),
        event_type_summary=[dict(std_event_type=et, mean_next_day_vz=m, n=n) for et, m, n in et_summary],
        std_event_types_seen=std_event_types,
        cut_date=cut_date,
        sharp_test_news_day_cut_date=news_cut_date,
        sharp_test2_ar_lag0_vs_news=sharp_test2,
        sharp_test3_event_type_content=event_type_content,
        sharp_test4_unscheduled=sharp_test4,
        sharp_test5_timing=dict(n=len(news_rows), mean_vz_T=mean_vz_T, mean_vz_T1=mean_vz_T1),
    )

    out_path = gd / "news_volume.json"
    out_path.write_text(json.dumps(out, indent=2))

    print("\n=== HEADLINE (T+1 OOS) ===")
    print(main_result)
    print("\n=== WINSORIZED (+-5sd) ===")
    print(wins_result)
    print(f"\n=== UNSCHEDULED subset n={len(unsched_rows)} ===")
    print(unsched_result)
    print("\n=== top event types by mean next-day abnormal vol-z ===")
    for et, m, n in et_summary[:10]:
        print(f"  {et:20s} mean_vz={m:+.3f}  n={n}")

    print("\n=== SHARP TEST 2 (news-day sample, AR-lag0 vs +news vs news-only) ===")
    print(sharp_test2)
    print("\n=== SHARP TEST 3 (event-type content beyond raw arrival) ===")
    print(event_type_content)
    print("\n=== SHARP TEST 4 (unscheduled-only) ===")
    print(sharp_test4)
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
