# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
Formal-plane calendar ingestion: populate eg.calendar_event (+ event_series dims)
from FREE, AUTHORITATIVE, TIMESTAMP-PRECISE sources -- deliberately not scraped
consensus calendars (Investing.com / ForexFactory), whose timestamps and ToS are
both unreliable.

  alfred  US macro releases via FRED/ALFRED. output_type=4 gives each observation
          AS FIRST PUBLISHED with realtime_start = the publication date, so
          actual = first print (no lookahead through revisions) and
          prior  = the previous period's first print.  Free key: fred.stlouisfed.org.
          Release *times* are the agency's standard ET slot (e.g. BLS 08:30),
          converted ET->UTC DST-correctly via zoneinfo -- override per series with
          --release-time SERIES=HH:MM if needed.
          NOTE actuals are first-print LEVELS/INDICES (CPI/PPI index, NFP level in
          thousands, GDP $bn), except UNRATE (rate) and UMCSENT (index) which ARE
          the headline. So v_event_surprise's `actual - prior` is a first-print MoM
          CHANGE vs a random-walk baseline, NOT a consensus surprise (no free
          consensus feed) and won't exactly equal the BLS headline change (which
          revises the prior within the same release). Derive %/annualized transforms
          downstream if you need the reported figure.
  ics     Generic agency release-calendar ICS ingester (BLS/BEA/Fed all publish
          .ics). Parses VEVENT DTSTART/SUMMARY; series_id = slug(summary) unless
          mapped with --map 'REGEX=SERIES_ID'.  No expected/actual -- schedule only
          (the alfred pass fills actuals for the same series_id).
  edgar   Corporate events from SEC EDGAR submissions JSON: 8-K (+ 10-Q/10-K)
          acceptanceDateTime per ticker -> exact moment the market could know.
          series_id = '{TICKER}-8K'.  Tickers resolved via company_tickers.json.

All modes append eg.calendar_event rows to <graph-dir>/lake/calendar_event.jsonl
(series_id, event_time UTC, expected, actual, prior, unit, source) and upsert-style
dim rows to <graph-dir>/lake/event_series.jsonl (loaded to Postgres event_series
by the existing pg upsert path; dedup key = series_id).

Usage:
  uv run scripts/formal_calendar.py alfred --graph-dir /tmp/eg_6k --api-key $FRED_KEY \
      --start 2009-01-01 --end 2014-01-01
  uv run scripts/formal_calendar.py ics --graph-dir /tmp/eg_6k \
      --url https://www.bls.gov/schedule/news_release/bls.ics --agency bls
  uv run scripts/formal_calendar.py edgar --graph-dir /tmp/eg_6k \
      --tickers AAPL,MSFT,XOM --start 2009-01-01 --end 2014-01-01
"""
import argparse, json, re, time, datetime as dt
from pathlib import Path
import httpx

UA = {"User-Agent": "eventgraph-research jamiep.tigereye@gmail.com"}

# tracked ALFRED series: fred_id -> (series_id, event_type, unit, agency, default ET time)
ALFRED_SERIES = {
    "CPIAUCSL": ("US-CPI",          "cpi",           "index",  "bls", "08:30"),
    "PPIACO":   ("US-PPI",          "ppi",           "index",  "bls", "08:30"),
    "PAYEMS":   ("US-NFP",          "employment",    "thous",  "bls", "08:30"),
    "UNRATE":   ("US-UNEMPLOYMENT", "employment",    "pct",    "bls", "08:30"),
    "GDPC1":    ("US-GDP",          "gdp",           "bil_09", "bea", "08:30"),
    "RSAFS":    ("US-RETAIL-SALES", "retail_sales",  "mil",    "census", "08:30"),
    "BOPGSTB":  ("US-TRADE-BAL",    "trade_balance", "mil",    "bea", "08:30"),
    "INDPRO":   ("US-INDPRO",       "other",         "index",  "frb", "09:15"),
    "HOUST":    ("US-HOUSING-STARTS","other",        "thous",  "census", "08:30"),
    "UMCSENT":  ("US-UMICH-SENT",   "other",         "index",  "umich", "10:00"),
}

def et_to_utc(date_str, hhmm):
    # release date + ET wall-clock time -> UTC ISO, DST-correct via zoneinfo.
    from zoneinfo import ZoneInfo
    hh, mm = hhmm.split(":")
    y, mo, d = date_str.split("-")
    t = dt.datetime(int(y), int(mo), int(d), int(hh), int(mm),
                    tzinfo=ZoneInfo("America/New_York"))
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"  +{len(rows)} -> {path}")


def emit(gd, cal_rows, series_rows):
    lake = Path(gd) / "lake"
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for r in cal_rows:
        r["ingested_at"] = now
    if cal_rows:
        append_jsonl(lake / "calendar_event.jsonl", cal_rows)
    if series_rows:  # dedup by series_id against what's already there
        p = lake / "event_series.jsonl"
        seen = {json.loads(l)["series_id"] for l in open(p)} if p.exists() else set()
        fresh = [r for r in series_rows if r["series_id"] not in seen]
        if fresh:
            append_jsonl(p, fresh)


def do_alfred(a):
    cal, dims = [], []
    for fred_id, (sid, etype, unit, agency, et) in ALFRED_SERIES.items():
        et = dict(kv.split("=") for kv in (a.release_time or [])).get(sid, et)
        r = httpx.get("https://api.stlouisfed.org/fred/series/observations",
                      params={"series_id": fred_id, "api_key": a.api_key, "file_type": "json",
                              "output_type": 4,  # initial release only: first prints + realtime_start = pub date
                              "realtime_start": a.start, "realtime_end": a.end},
                      headers=UA, timeout=30)
        r.raise_for_status()
        obs = [o for o in r.json()["observations"] if o["value"] != "."]
        obs.sort(key=lambda o: o["date"])
        prior = None
        for o in obs:
            t = et_to_utc(o["realtime_start"], et)
            if a.start <= o["realtime_start"] <= a.end:
                cal.append({"series_id": sid, "event_time": t, "expected": None,
                            "actual": float(o["value"]), "prior": prior, "unit": unit,
                            "source": "alfred", "source_ref": f"fred:{fred_id}"})
            prior = float(o["value"])
        dims.append({"series_id": sid, "event_type": etype, "issuer_entity": None,
                     "region": "US", "cadence": "monthly" if fred_id != "GDPC1" else "quarterly",
                     "calendar_ref": f"fred:{fred_id}"})
        print(f"{sid}: {len(obs)} first prints"); time.sleep(0.4)
    emit(a.graph_dir, cal, dims)


def parse_ics(text):
    # minimal RFC5545: unfold continuation lines, walk VEVENTs
    lines, out, cur = [], [], None
    for l in text.splitlines():
        if l[:1] in (" ", "\t") and lines: lines[-1] += l[1:]
        else: lines.append(l.rstrip("\r"))
    for l in lines:
        if l == "BEGIN:VEVENT": cur = {}
        elif l == "END:VEVENT" and cur is not None: out.append(cur); cur = None
        elif cur is not None and ":" in l:
            k, v = l.split(":", 1)
            name, params = k.split(";")[0], dict(p.split("=", 1) for p in k.split(";")[1:] if "=" in p)
            cur[name] = v
            if name == "DTSTART" and "TZID" in params: cur["_TZID"] = params["TZID"]
    return out


TZ_ALIAS = {"US-Eastern": "America/New_York", "US-Central": "America/Chicago"}


def ics_to_utc(raw, tzid):
    m = re.match(r"(\d{4})(\d{2})(\d{2})(?:T(\d{2})(\d{2})(\d{2})?(Z?))?", raw)
    if not m: return None
    y, mo, d, hh, mm, ss, z = m.groups()
    if hh is None:  # all-day -> default 08:30 ET
        hh, mm, ss, tzid = "08", "30", "00", tzid or "America/New_York"
    if z:  # already UTC
        return f"{y}-{mo}-{d}T{hh}:{mm}:{ss or '00'}Z"
    from zoneinfo import ZoneInfo
    t = dt.datetime(int(y), int(mo), int(d), int(hh), int(mm), int(ss or 0),
                    tzinfo=ZoneInfo(TZ_ALIAS.get(tzid, tzid or "America/New_York")))
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def do_ics(a):
    text = httpx.get(a.url, headers=UA, timeout=30, follow_redirects=True).text if a.url.startswith("http") else Path(a.url).read_text()
    maps = [(re.compile(p, re.I), s) for p, s in (m.split("=", 1) for m in a.map or [])]
    cal, dims, sids = [], [], set()
    for ev in parse_ics(text):
        raw, summ = ev.get("DTSTART", ""), ev.get("SUMMARY", "unknown").replace("\\,", ",")
        t = ics_to_utc(raw, ev.get("_TZID"))
        if t is None: continue
        sid = next((s for rx, s in maps if rx.search(summ)),
                   re.sub(r"[^a-z0-9]+", "_", summ.lower()).strip("_"))
        cal.append({"series_id": sid, "event_time": t, "expected": None, "actual": None,
                    "prior": None, "unit": None, "source": f"ics:{a.agency}", "source_ref": a.url})
        if sid not in sids:
            sids.add(sid)
            dims.append({"series_id": sid, "event_type": "other", "issuer_entity": None,
                         "region": a.region, "cadence": None, "calendar_ref": f"ics:{a.agency}:{summ}"})
    emit(a.graph_dir, cal, dims)


def do_edgar(a):
    tk2cik = {v["ticker"]: f"{v['cik_str']:010d}"
              for v in httpx.get("https://www.sec.gov/files/company_tickers.json",
                                 headers=UA, timeout=30).json().values()}
    forms = set(a.forms.split(","))
    cal, dims = [], []
    for tk in a.tickers.split(","):
        cik = tk2cik.get(tk.upper())
        if not cik:
            print(f"{tk}: no CIK, skipped"); continue
        j = httpx.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=UA, timeout=30).json()
        recent = j["filings"]["recent"]
        rows = list(zip(recent["form"], recent["acceptanceDateTime"], recent["accessionNumber"]))
        for extra in j["filings"].get("files", []):  # older filings live in paged files
            e = httpx.get(f"https://data.sec.gov/submissions/{extra['name']}", headers=UA, timeout=30).json()
            rows += list(zip(e["form"], e["acceptanceDateTime"], e["accessionNumber"]))
            time.sleep(0.15)
        n = 0
        for form, acc_t, accno in rows:
            if form not in forms or not (a.start <= acc_t[:10] <= a.end): continue
            sid = f"{tk.upper()}-{form.replace('/', '')}"
            cal.append({"series_id": sid, "event_time": acc_t, "expected": None,
                        "actual": None, "prior": None, "unit": None, "source": "edgar",
                        "source_ref": f"edgar:{accno}"})
            n += 1
        for form in forms:
            dims.append({"series_id": f"{tk.upper()}-{form.replace('/', '')}",
                         "event_type": "earnings" if form.startswith("10-") else "other",
                         "issuer_entity": None, "region": "US", "cadence": None,
                         "calendar_ref": f"edgar:CIK{cik}:{form}"})
        print(f"{tk}: {n} filings"); time.sleep(0.15)
    emit(a.graph_dir, cal, dims)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("alfred", "ics", "edgar"):
        s = sub.add_parser(name)
        s.add_argument("--graph-dir", required=True)
    sub.choices["alfred"].add_argument("--api-key", required=True)
    sub.choices["alfred"].add_argument("--start", default="2000-01-01")
    sub.choices["alfred"].add_argument("--end", default="2100-01-01")
    sub.choices["alfred"].add_argument("--release-time", action="append", metavar="SERIES=HH:MM")
    sub.choices["ics"].add_argument("--url", required=True, help="http(s) URL or local .ics path")
    sub.choices["ics"].add_argument("--agency", default="agency")
    sub.choices["ics"].add_argument("--region", default="US")
    sub.choices["ics"].add_argument("--map", action="append", metavar="REGEX=SERIES_ID")
    sub.choices["edgar"].add_argument("--tickers", required=True)
    sub.choices["edgar"].add_argument("--forms", default="8-K,10-Q,10-K")
    sub.choices["edgar"].add_argument("--start", default="2000-01-01")
    sub.choices["edgar"].add_argument("--end", default="2100-01-01")
    a = ap.parse_args()
    {"alfred": do_alfred, "ics": do_ics, "edgar": do_edgar}[a.cmd](a)
