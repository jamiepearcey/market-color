# /// script
# requires-python = ">=3.10"
# dependencies = ["httpx"]
# ///
"""
EM/FRONTIER EVENT CALENDAR — the snapshotter and its parsers (Phase 0 of
docs/EM-CALENDAR-STRATEGY.md).

WHAT THIS IS FOR. Vendors publish a calendar date and overwrite it in place when
it moves. A revision history therefore exists only if somebody snapshots, and it
CANNOT be bought or backfilled -- the Wayback Machine covers a fraction of EM
statistics-office pages. So the primary job here is not parsing: it is starting
the clock. Every fetch is content-addressed and logged whether or not a parser
exists for it, because silent upstream mutation IS the phenomenon we are
capturing. Parsing catches up later; the snapshots cannot.

WHAT IT EMITS.
  eg.source_snapshot        one row per fetch (+ a `changed` tripwire vs the
                            previous fetch of the same source), and the raw bytes
                            content-addressed under <graph-dir>/raw/.
  eg.calendar_observation   one row per CLAIM read out of a snapshot -- the
                            bitemporal primitive from lake/0012. Identity is the
                            OCCURRENCE (series_id, period_ref); event_time is a
                            belief attribute, so a reschedule shows up as a
                            second observation rather than destroying the first.

HONESTY ABOUT knowable_at. For a live snapshot, knowable_at = fetch time. We may
well be learning something the source published days earlier, but claiming an
earlier knowable_at would be inventing point-in-time knowledge we did not have.
Archival sources that carry their own publication date should set it from that
date instead (the discipline ALFRED's realtime_start already gives us).

SOURCES. config/em_calendar_sources.json, every entry live-probed. Entries with
mode='parse' have a parser below; mode='snapshot' are fetched and stored raw with
the blocker documented (SPA, WAF, PDF-only). That backlog is deliberately visible
rather than silently dropped.

Usage:
  uv run scripts/em_calendar.py snapshot --graph-dir data/eg_runs/em_cal
  uv run scripts/em_calendar.py parse    --graph-dir data/eg_runs/em_cal
  uv run scripts/em_calendar.py report   --graph-dir data/eg_runs/em_cal
  uv run scripts/em_calendar.py revisions --graph-dir data/eg_runs/em_cal
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "config" / "em_calendar_sources.json"

# A real browser UA is required by several of these hosts (BLS taught us the same
# lesson in formal_calendar.py); it is not evasion, it is what they serve HTML to.
UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/pdf,text/calendar,*/*",
    "Accept-Language": "en,fr;q=0.8,tr;q=0.7,ar;q=0.6",
}

EXT = {"text/html": "html", "application/pdf": "pdf", "text/calendar": "ics",
       "application/json": "json", "text/plain": "txt", "application/xml": "xml"}


def now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def iso(t: dt.datetime | None) -> str | None:
    return None if t is None else t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")


def load_registry() -> dict:
    return json.loads(REGISTRY.read_text())


def jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def append(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for r in rows:
            fh.write(json.dumps(r, default=str) + "\n")


# ---------------------------------------------------------------- snapshotting

async def fetch(client: httpx.AsyncClient, src: dict) -> dict:
    t0 = now_utc()
    try:
        r = await client.get(src["url"], headers=UA, follow_redirects=True, timeout=45.0)
        body = r.content
        return dict(source_id=src["source_id"], url=src["url"], final_url=str(r.url),
                    status_code=r.status_code,
                    content_type=r.headers.get("content-type", "").split(";")[0],
                    bytes=len(body), error=None, fetched_at=iso(t0), _body=body)
    except Exception as e:
        return dict(source_id=src["source_id"], url=src["url"], final_url=None,
                    status_code=None, content_type=None, bytes=0,
                    error=f"{type(e).__name__}: {e}"[:200], fetched_at=iso(t0), _body=None)


async def snapshot(graph_dir: Path, only: set[str] | None, concurrency: int) -> None:
    reg = load_registry()
    sources = [s for s in reg["sources"] if not only or s["source_id"] in only]
    raw_dir, lake = graph_dir / "raw", graph_dir / "lake"
    ledger_path = lake / "source_snapshot.jsonl"
    prev_sha = {}
    for row in jsonl(ledger_path):                       # last hash seen per source
        if row.get("sha256"):
            prev_sha[row["source_id"]] = row["sha256"]

    async with httpx.AsyncClient(verify=False) as client:
        sem = asyncio.Semaphore(concurrency)

        async def run(s):
            async with sem:
                return await fetch(client, s)

        results = await asyncio.gather(*[run(s) for s in sources])

    rows, n_ok, n_changed, n_new = [], 0, 0, 0
    for res in results:
        body = res.pop("_body")
        sha = artifact = None
        if body:
            sha = hashlib.sha256(body).hexdigest()
            ext = EXT.get(res["content_type"] or "", "bin")
            p = raw_dir / sha[:2] / f"{sha}.{ext}"
            if not p.exists():                            # content-addressed: write once
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
            artifact = str(p.relative_to(graph_dir))
        known = prev_sha.get(res["source_id"])
        changed = bool(sha and known and sha != known)
        if sha and not known:
            n_new += 1
        n_changed += changed
        n_ok += res["status_code"] == 200
        rows.append({**res, "sha256": sha, "artifact_path": artifact, "changed": changed})

    append(ledger_path, rows)
    print(f"snapshot: {len(rows)} sources | {n_ok} returned 200 | {n_new} first-seen | "
          f"{n_changed} CHANGED since last fetch")
    for r in rows:
        flag = "CHANGED" if r["changed"] else ("new" if r["sha256"] and r["source_id"] not in prev_sha else "same")
        st = r["status_code"] if r["status_code"] is not None else "ERR"
        print(f"  {r['source_id']:24} {str(st):>4} {str(r['bytes']):>8}B  {flag:8} {r['error'] or ''}")


# -------------------------------------------------------------------- parsers
#
# Each parser takes (raw_bytes, source, snapshot_row, venues) and returns
# observation dicts. Parsers transcribe; they do not infer. Where a source gives
# a date without a time we degrade precision to 'date' rather than inventing an
# hour -- the strategy's rule is degrade, never guess.

MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], 1)}
TAGS = re.compile(r"<[^>]+>")


def clean(s: str) -> str:
    s = TAGS.sub(" ", s)
    s = (s.replace("&amp;", "&").replace("&#038;", "&").replace("&nbsp;", " ")
          .replace("&#8211;", "-").replace("&rsquo;", "'"))
    return re.sub(r"\s+", " ", s).strip()


def to_utc(d: dt.date, tz: str) -> dt.datetime:
    """Local calendar date -> UTC instant at local midnight. Precision stays 'date':
    this is a placement on the timeline, not a claim about the release hour."""
    try:
        local = dt.datetime(d.year, d.month, d.day, tzinfo=ZoneInfo(tz))
    except Exception:
        local = dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc)
    return local.astimezone(dt.timezone.utc)


def parse_dmy(s: str) -> dt.date | None:
    """'31-Jul-2025' / '31 Jul 2025' / '31/07/2025'."""
    s = s.strip()
    m = re.match(r"^(\d{1,2})[-/ ]([A-Za-z]{3,9})[-/ ](\d{4})$", s)
    if m and m.group(2)[:3].lower() in MONTHS:
        return dt.date(int(m.group(3)), MONTHS[m.group(2)[:3].lower()], int(m.group(1)))
    m = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", s)
    if m:
        return dt.date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
    return None


def table_rows(html: str) -> list[list[str]]:
    html = re.sub(r"<script.*?</script>", "", html, flags=re.S | re.I)
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = [clean(c) for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)]
        if cells:
            out.append(cells)
    return out


# KNBS product name -> canonical series. Anything unmapped keeps a slugged id so
# the tail stays visible (same doctrine as series_alias's --unmapped dump).
KNBS_SERIES = {
    "consumer price index": "KE-CPI",
    "gross domestic product": "KE-GDP",
    "trade statistics": "KE-TRADE",
    "monetary & financial statistics": "KE-MONETARY",
    "balance of payments": "KE-BOP",
    "tourism statistics": "KE-TOURISM",
    "transport statistics": "KE-TRANSPORT",
    "ict statistics": "KE-ICT",
    "electricity statistics": "KE-ELECTRICITY",
    "agriculture statistics": "KE-AGRI",
    "producer price index": "KE-PPI",
}
CLASS_OF = {"KE-CPI": "cpi", "KE-GDP": "gdp", "KE-TRADE": "trade",
            "KE-MONETARY": "monetary", "KE-BOP": "bop", "KE-PPI": "ppi"}


def parse_knbs_arc(body: bytes, src: dict, snap: dict, venues: dict) -> list[dict]:
    """KNBS Advance Release Calendar: Month | Publication | Product | Final day of
    reference period | Frequency | Expected Release Date. Month/Publication are
    blank on continuation rows, so both are forward-filled."""
    html = body.decode("utf-8", "ignore")
    tz = venues.get(src["venue"], {}).get("tz", "UTC")
    out, month, pub = [], "", ""
    for cells in table_rows(html):
        if len(cells) < 6 or cells[0].lower() == "month":
            continue
        month = cells[0] or month
        pub = cells[1] or pub
        product, ref_end, freq, expected = cells[2], cells[3], cells[4], cells[5]
        d = parse_dmy(expected)
        if not d or not product:
            continue
        key = product.lower().strip()
        series = next((v for k, v in KNBS_SERIES.items() if k in key),
                      "KE-" + re.sub(r"[^a-z0-9]+", "-", key).strip("-").upper()[:28])
        # Occurrence identity must survive two DIFFERENT publications carrying the
        # same product for the same reference month (e.g. Leading Economic
        # Indicators (December 2025) and Economic Survey 2026 both deliver trade
        # statistics for 2025-12). Keying on the reference month alone collides,
        # and a collision silently drops one release from current belief. The
        # publication slug disambiguates; parentheticals are stripped so the slug
        # is stable month to month ("Consumer Price Index (July, 2025)" -> cpi).
        ref = parse_dmy(ref_end)
        base = f"{ref:%Y-%m}" if ref else f"{d:%Y-%m}"
        pub_slug = re.sub(r"\(.*?\)", "", pub or product)
        pub_slug = re.sub(r"[^a-z0-9]+", "-", pub_slug.lower()).strip("-")[:24]
        period_ref = f"{base}:{pub_slug}" if pub_slug else base
        out.append(observation(
            series_id=series, period_ref=period_ref, event_date_local=d, tz=tz,
            event_time_local=expected, time_precision="date", status="scheduled",
            venue=src["venue"], event_class=CLASS_OF.get(series, "statistics"),
            title=f"{pub} — {product}".strip(" —") or product,
            src=src, snap=snap, method="html_table:knbs_arc",
            extra={"frequency": freq}))
    return out


ISO2 = {"nigeria": "NG", "kenya": "KE", "ghana": "GH", "south africa": "ZA", "egypt": "EG",
        "turkey": "TR", "türkiye": "TR", "pakistan": "PK", "zambia": "ZM", "angola": "AO",
        "côte d'ivoire": "CI", "cote d'ivoire": "CI", "senegal": "SN", "morocco": "MA",
        "tunisia": "TN", "jordan": "JO", "lebanon": "LB", "sri lanka": "LK",
        "bangladesh": "BD", "argentina": "AR", "brazil": "BR", "mexico": "MX",
        "colombia": "CO", "chile": "CL", "peru": "PE", "ethiopia": "ET", "uganda": "UG",
        "tanzania": "TZ", "mozambique": "MZ", "zimbabwe": "ZW", "iraq": "IQ",
        "saudi arabia": "SA", "united arab emirates": "AE", "qatar": "QA", "oman": "OM"}


def parse_electionguide(body: bytes, src: dict, snap: dict, venues: dict) -> list[dict]:
    """IFES ElectionGuide upcoming elections. Rows carry a native (d)=definitive /
    (t)=tentative flag, which maps straight onto our status axis -- a rare source
    that already distinguishes confirmed from merely scheduled. The election id in
    the href is the natural period_ref."""
    html = re.sub(r"<script.*?</script>", "", body.decode("utf-8", "ignore"), flags=re.S | re.I)
    out = []
    for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        m = re.search(r"<strong>\s*([A-Za-z]{3})\s+(\d{1,2})\s+(\d{4})\s*</strong>\s*"
                      r"<small>\s*\((d|t)\)\s*</small>", tr, re.S | re.I)
        if not m:
            continue
        mon = MONTHS.get(m.group(1)[:3].lower())
        if not mon:
            continue
        d = dt.date(int(m.group(3)), mon, int(m.group(2)))
        definitive = m.group(4).lower() == "d"
        elec = re.search(r'href="/elections/id/(\d+)/"[^>]*>(.*?)</a>', tr, re.S | re.I)
        ctry = re.search(r'href="/countries/id/\d+/"[^>]*>(.*?)</a>', tr, re.S | re.I)
        if not elec:
            continue
        name = clean(elec.group(2))
        country = clean(ctry.group(1)) if ctry else ""
        # ISO2 for the markets we trade; a stable slug for the rest, so an
        # unmapped country never leaks whitespace into a series_id.
        venue = ISO2.get(country.lower()) or (
            re.sub(r"[^A-Za-z0-9]+", "-", country).strip("-").upper()[:20] or "GLOBAL")
        tz = venues.get(venue, {}).get("tz", "UTC")
        out.append(observation(
            series_id=f"{venue}-ELECTION", period_ref=f"EG-{elec.group(1)}",
            event_date_local=d, tz=tz,
            event_time_local=f"{m.group(1)} {m.group(2)} {m.group(3)}",
            time_precision="date",
            status="confirmed" if definitive else "scheduled",
            venue=venue, event_class="election",
            title=f"{name} — {country}".strip(" —"),
            src=src, snap=snap, method="html_table:electionguide",
            extra={"definitive": definitive}))
    return out


PARSERS = {"knbs_arc": parse_knbs_arc, "electionguide": parse_electionguide}


def observation(*, series_id, period_ref, event_date_local, tz, event_time_local,
                time_precision, status, venue, event_class, title, src, snap, method,
                extra=None) -> dict:
    """Assemble one eg.calendar_observation row.

    event_date_local is the authoritative field for date-precision claims; the UTC
    event_time is derived from it purely for ordering. Storing only the UTC instant
    would move the calendar date back a day for every venue east of Greenwich (see
    the note in lake/0012). knowable_at = fetch time for a live snapshot: we cannot
    honestly claim to have known it earlier."""
    fetched = snap["fetched_at"]
    event_time = to_utc(event_date_local, tz) if event_date_local else None
    return {
        "series_id": series_id, "period_ref": period_ref,
        "event_time": iso(event_time) if event_time else None,
        "event_date_local": event_date_local.isoformat() if event_date_local else None,
        "event_time_local": event_time_local, "time_precision": time_precision,
        "status": status, "venue": venue, "event_class": event_class, "title": title,
        "expected": None, "actual": None, "prior": None, "unit": None,
        "source": src["source_id"], "source_ref": snap.get("final_url") or src["url"],
        "source_tier": src["tier"], "method": method,
        "artifact_sha256": snap.get("sha256"),
        "observed_at": None,          # page carries no publication timestamp
        "knowable_at": fetched, "ingested_at": fetched,
        "doc_id": None, "chunk_id": None, "quote": None,
        # source-specific extras stay in attrs so a new parser never widens the
        # table -- the schema is the contract, not the union of sources.
        "attrs": json.dumps(extra, default=str) if extra else None,
    }


def parse(graph_dir: Path, only: set[str] | None) -> None:
    reg = load_registry()
    venues = reg["venues"]
    by_id = {s["source_id"]: s for s in reg["sources"]}
    lake = graph_dir / "lake"
    snaps = jsonl(lake / "source_snapshot.jsonl")
    if not snaps:
        raise SystemExit("no snapshots yet — run `snapshot` first")

    latest = {}                                  # newest successful snapshot per source
    for s in snaps:
        if s.get("sha256") and s.get("status_code") == 200:
            cur = latest.get(s["source_id"])
            if not cur or s["fetched_at"] >= cur["fetched_at"]:
                latest[s["source_id"]] = s

    seen = {(o["source"], o["artifact_sha256"]) for o in jsonl(lake / "calendar_observation.jsonl")}
    rows, skipped = [], []
    for sid, snap in sorted(latest.items()):
        src = by_id.get(sid)
        if not src or (only and sid not in only):
            continue
        p = PARSERS.get(src.get("parser") or "")
        if not p:
            skipped.append(sid)
            continue
        if (sid, snap["sha256"]) in seen:        # already parsed this exact artifact
            print(f"  {sid:24} unchanged artifact — skip")
            continue
        body = (graph_dir / snap["artifact_path"]).read_bytes()
        got = p(body, src, snap, venues)
        rows.extend(got)
        print(f"  {sid:24} {len(got):>4} observations")

    if rows:
        append(lake / "calendar_observation.jsonl", rows)
    print(f"parse: +{len(rows)} observations | {len(skipped)} sources snapshot-only "
          f"({', '.join(skipped) if skipped else '-'})")


# --------------------------------------------------------------------- reports

def report(graph_dir: Path) -> None:
    lake = graph_dir / "lake"
    obs = jsonl(lake / "calendar_observation.jsonl")
    snaps = jsonl(lake / "source_snapshot.jsonl")
    today = now_utc().date().isoformat()   # compare LOCAL dates, not UTC instants

    print(f"\n=== SNAPSHOT LEDGER ({len(snaps)} fetches)")
    per = {}
    for s in snaps:
        d = per.setdefault(s["source_id"], {"n": 0, "ok": 0, "changed": 0, "last": None})
        d["n"] += 1
        d["ok"] += s.get("status_code") == 200
        d["changed"] += bool(s.get("changed"))
        d["last"] = s["fetched_at"]
    print(f"  {'source':26} {'fetches':>7} {'ok':>4} {'changed':>8}  last")
    for sid, d in sorted(per.items()):
        print(f"  {sid:26} {d['n']:>7} {d['ok']:>4} {d['changed']:>8}  {d['last']}")

    print(f"\n=== OBSERVATIONS ({len(obs)} rows)")
    # current belief = best tier, then most recent, per occurrence
    rank = {"primary": 1, "secondary": 2, "narrative": 3, "model": 4}
    cur = {}
    for o in obs:
        k = (o["series_id"], o["period_ref"])
        c = cur.get(k)
        if not c or (rank.get(o["source_tier"], 9), o["knowable_at"] or "") < \
                    (rank.get(c["source_tier"], 9), c["knowable_at"] or ""):
            cur[k] = o
    def loc(o):
        return o.get("event_date_local") or (o["event_time"] or "")[:10]

    fwd = [o for o in cur.values() if loc(o) > today]
    print(f"  occurrences: {len(cur)}   forward-dated: {len(fwd)}   (as of {today})")
    by_venue = {}
    for o in cur.values():
        d = by_venue.setdefault((o["venue"], o["event_class"]),
                                {"n": 0, "fwd": 0, "prim": 0, "conf": 0, "last": ""})
        d["n"] += 1
        d["fwd"] += loc(o) > today
        d["prim"] += o["source_tier"] == "primary"
        d["conf"] += o["status"] == "confirmed"
        d["last"] = max(d["last"], loc(o))
    print(f"  {'venue':22} {'class':14} {'occ':>5} {'fwd':>5} {'primary':>8} {'conf':>6}  last")
    for (v, c), d in sorted(by_venue.items(), key=lambda x: (-x[1]["n"], x[0])):
        print(f"  {v:22} {c:14} {d['n']:>5} {d['fwd']:>5} {d['prim']:>8} {d['conf']:>6}  {d['last']}")

    print("\n=== NEXT 12 FORWARD EVENTS (current belief, venue-local dates)")
    for o in sorted(fwd, key=loc)[:12]:
        print(f"  {loc(o)}  {o['venue']:16} {o['series_id']:26} "
              f"{o['status']:10} {(o['title'] or '')[:46]}")

    # A calendar whose newest entry is in the past is not a calendar. Surface it.
    stale = [(v, c, d) for (v, c), d in by_venue.items() if d["fwd"] == 0]
    if stale:
        print("\n=== STALE — no forward-dated events (source needs re-check)")
        for v, c, d in sorted(stale):
            print(f"  {v:22} {c:14} {d['n']:>4} occurrences, newest {d['last']}")


def revisions(graph_dir: Path) -> None:
    """Consecutive observations of the same occurrence whose date or status moved.
    Empty on the first run by construction — this table only fills as the
    snapshotter accrues history, which is exactly the point."""
    obs = jsonl(graph_dir / "lake" / "calendar_observation.jsonl")
    by_occ = {}
    for o in obs:
        by_occ.setdefault((o["series_id"], o["period_ref"]), []).append(o)
    found = 0
    for (sid, pref), rows in sorted(by_occ.items()):
        rows.sort(key=lambda r: (r["knowable_at"] or "", r["ingested_at"] or ""))

        def loc(r):
            return r.get("event_date_local") or (r["event_time"] or "")[:10]

        for a, b in zip(rows, rows[1:]):
            if loc(a) == loc(b) and a["status"] == b["status"]:
                continue
            found += 1
            moved = ""
            if loc(a) and loc(b):
                moved = f"{(dt.date.fromisoformat(loc(b)) - dt.date.fromisoformat(loc(a))).days:+d}d"
            print(f"  {sid:22} {pref:12} {loc(a) or '-':<12} -> {loc(b) or '-':<12} {moved:>6}  "
                  f"{a['status']}->{b['status']}  learned {b['knowable_at']}")
    print(f"revisions: {found} schedule changes across {len(by_occ)} occurrences")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["snapshot", "parse", "report", "revisions", "sources"])
    ap.add_argument("--graph-dir", type=Path, required=True)
    ap.add_argument("--only", help="comma-separated source_ids")
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    only = set(a.only.split(",")) if a.only else None

    if a.mode == "snapshot":
        asyncio.run(snapshot(a.graph_dir, only, a.concurrency))
    elif a.mode == "parse":
        parse(a.graph_dir, only)
    elif a.mode == "report":
        report(a.graph_dir)
    elif a.mode == "revisions":
        revisions(a.graph_dir)
    elif a.mode == "sources":
        reg = load_registry()
        print(f"{'source_id':26} {'venue':7} {'tier':10} {'mode':9} classes")
        for s in reg["sources"]:
            print(f"  {s['source_id']:24} {s['venue']:7} {s['tier']:10} {s['mode']:9} "
                  f"{','.join(s['classes'])}")


if __name__ == "__main__":
    main()
