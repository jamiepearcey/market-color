#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "httpx",
#   "feedparser",
#   "trafilatura",
#   "pyarrow",
#   "python-dateutil",
#   "googlenewsdecoder",
# ]
# ///
"""Hardened market-color feed scraper (eventgraph ingest).

Ported from ../../crawl_corpus.py. Same parsing + same 22-column parquet output,
but split into two lanes to fix the "burst then get blocked by Google" problem:

  FAST lane (method=rss)            -- fetch -> extract -> parquet, parallel.
                                       Never touches news.google.com.
  SLOW lane (method=google_news_rss) -- opaque /articles/<id> resolution made
                                       INCREMENTAL + DURABLE + RATE-LIMITED:
      * durable resolve-cache (.state/gnews_cache.jsonl) keyed by article id;
        resolved ids are never re-resolved; failed ids wait out a backoff.
      * one shared limiter, concurrency 1, >= --gnews-delay-ms between calls.
      * per-run budget --gnews-max-per-run new resolutions, then stop.
      * circuit breaker: on a 429 or CAPTCHA body, trip -> stop ALL Google
        calls for the rest of the run (no powering through).

Run (fast lane only, safe -- zero Google calls):
  uv run eventgraph/ingest/scrape.py --no-gnews --feeds config/feeds_em.json \
      --limit 15 --out /tmp/eg_py_scrape
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import base64  # noqa: F401  (kept for parity / decode fallback path)
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

import feedparser  # type: ignore
import httpx  # type: ignore
from dateutil import parser as dateparser  # type: ignore

# -- state primitives (pure stdlib, unit-tested offline) --------------------- #
try:
    from . import gnews_state as gs  # type: ignore  (python -m eventgraph.ingest.scrape)
except Exception:  # run directly as a uv script
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import gnews_state as gs  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAVE_PARQUET = True
except Exception:
    _HAVE_PARQUET = False

try:
    import trafilatura  # type: ignore
    _HAVE_TRAFILATURA = True
except Exception:
    _HAVE_TRAFILATURA = False

try:
    from googlenewsdecoder import gnewsdecoder as _gnews_lib  # type: ignore
    _HAVE_GNEWS = True
except Exception:
    try:
        from googlenewsdecoder import new_decoderv1 as _gnews_lib  # type: ignore
        _HAVE_GNEWS = True
    except Exception:
        _gnews_lib = None
        _HAVE_GNEWS = False

UA = "Mozilla/5.0 (compatible; market-color-crawler/0.2; +research corpus builder)"
HERE = Path(__file__).resolve().parent
# market-color/ (two levels up from eventgraph/ingest/)
PROJECT_ROOT = HERE.parent.parent
DEFAULT_FEEDS = PROJECT_ROOT / "config" / "feeds.json"
# During dev the default output is a TEST dir, NOT the production corpus.
DEFAULT_OUT = Path("/tmp/eg_py_scrape")
DEFAULT_GNEWS_CACHE = HERE / ".state" / "gnews_cache.jsonl"

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "mc_cid", "mc_eid", "ref", "cmpid", "guccounter",
}


# --------------------------------------------------------------------------- #
@dataclass
class Article:
    doc_id: str
    source_name: str
    source_scope: str
    source_tier: int
    source_access: str
    source_method: str
    source_domain: str
    desks: list[str]
    discovery: str               # rss | google_news
    title: str
    body_text: str | None
    author: str | None
    url: str
    canonical_url: str
    lang: str | None
    word_count: int
    extraction_ok: bool
    published_utc: str | None
    published_date: str
    published_is_estimated: bool
    fetched_utc: str
    content_hash: str | None


# --------------------------------------------------------------------------- #
# URL / time helpers (verbatim from crawl_corpus.py)
# --------------------------------------------------------------------------- #
def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def domain_of(url: str) -> str:
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def site_root(url: str) -> str:
    parts = urlsplit(url)
    return f"{parts.scheme}://{parts.netloc}"


def canonicalize(url: str) -> str:
    p = urlsplit(url)
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False) if k.lower() not in TRACKING_PARAMS]
    path = p.path.rstrip("/") or "/"
    return urlunsplit((p.scheme, p.netloc.lower(), path, urlencode(q), ""))


def stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update((part or "").strip().encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def parse_dt(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        dt = dateparser.parse(str(value))
    except (ValueError, OverflowError, TypeError):
        return None
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


_TS_PATTERNS = [
    r'"datePublished"\s*:\s*"([^"]{8,40})"',
    r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']{8,40})["\']',
    r'<meta[^>]+content=["\']([^"\']{8,40})["\'][^>]+property=["\']article:published_time["\']',
    r'<meta[^>]+name=["\']parsely-pub-date["\'][^>]+content=["\']([^"\']{8,40})["\']',
    r'<meta[^>]+itemprop=["\']datePublished["\'][^>]+content=["\']([^"\']{8,40})["\']',
    r'<meta[^>]+name=["\'](?:sailthru\.date|pubdate|dc\.date\.issued)["\'][^>]+content=["\']([^"\']{8,40})["\']',
    r'<time[^>]+datetime=["\']([^"\']{8,40})["\']',
]


def extract_pub_ts(html: str) -> datetime | None:
    head = html[:200_000]
    best = None
    for pat in _TS_PATTERNS:
        for m in re.finditer(pat, head, re.I):
            dt = parse_dt(m.group(1))
            if dt is None or not (2000 <= dt.year <= 2100):
                continue
            has_time = (dt.hour, dt.minute, dt.second) != (0, 0, 0)
            if has_time:
                return dt
            best = best or dt
    return best


def clean_text(text: Any) -> str:
    if not text:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", s).strip()


# Re-exported for tests / parity; canonical impl lives in gnews_state.
decode_gnews_url = gs.decode_gnews_url


def resolve_gnews_sync(link: str) -> str | None:
    """Resolve a Google-News redirect to the publisher URL (crawl_corpus parity).

    Prefers the maintained `googlenewsdecoder` (opaque AU_yqL batchexecute form);
    falls back to the base64 CBMi heuristic. Returns None if it stays a
    news.google.com link. THIS MAKES A LIVE GOOGLE CALL -- only invoked from the
    slow lane, which is gated behind the rate limiter + breaker + budget.
    """
    if "news.google.com" not in link:
        return link
    if _HAVE_GNEWS:
        try:
            out = _gnews_lib(link, interval=0)
            if isinstance(out, dict) and out.get("decoded_url"):
                return out["decoded_url"]
            if isinstance(out, str) and out.startswith("http"):
                return out
        except Exception:
            pass
    decoded = decode_gnews_url(link)
    return decoded if decoded and "news.google.com" not in decoded else None


# --------------------------------------------------------------------------- #
# Per-domain polite throttle (verbatim)
# --------------------------------------------------------------------------- #
class DomainThrottle:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    def _lock(self, dom: str) -> asyncio.Lock:
        return self._locks.setdefault(dom, asyncio.Lock())

    async def acquire(self, dom: str) -> asyncio.Lock:
        lock = self._lock(dom)
        await lock.acquire()
        wait = self.delay - (time.monotonic() - self._last.get(dom, 0.0))
        if wait > 0:
            await asyncio.sleep(wait)
        return lock

    def release(self, dom: str, lock: asyncio.Lock) -> None:
        self._last[dom] = time.monotonic()
        lock.release()


async def fetch_bytes(client: httpx.AsyncClient, url: str) -> tuple[int | None, bytes | None]:
    try:
        r = await client.get(url, follow_redirects=True)
        return r.status_code, r.content
    except httpx.HTTPError:
        return None, None


# --------------------------------------------------------------------------- #
# Extraction (verbatim from crawl_corpus.py)
# --------------------------------------------------------------------------- #
def extract_with_trafilatura(html: str, url: str) -> dict[str, Any] | None:
    if not _HAVE_TRAFILATURA:
        return None
    data = trafilatura.extract(
        html, url=url, output_format="json", with_metadata=True,
        include_comments=False, favor_precision=True,
    )
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


HTML_CHAR_CAP = 3_000_000
EXTRACT_TIMEOUT = 20.0
_extract_pool: concurrent.futures.ThreadPoolExecutor | None = None
_extract_sem: asyncio.Semaphore | None = None


class _Progress:
    def __init__(self, total: int) -> None:
        self.total = total
        self.done = 0
        self.bodies = 0

    def tick(self, ok: bool) -> None:
        self.done += 1
        if ok:
            self.bodies += 1
        if self.done % 50 == 0 or self.done == self.total:
            pct = self.done * 100 // max(1, self.total)
            print(f"  [extract] {self.done}/{self.total} ({pct}%), {self.bodies} bodies",
                  file=sys.stderr)


_progress: _Progress | None = None


async def extract_capped(html: str, url: str) -> dict[str, Any] | None:
    if len(html) > HTML_CHAR_CAP:
        html = html[:HTML_CHAR_CAP]
    loop = asyncio.get_running_loop()
    if _extract_sem is None or _extract_pool is None:
        return await asyncio.to_thread(extract_with_trafilatura, html, url)
    async with _extract_sem:
        try:
            return await asyncio.wait_for(
                loop.run_in_executor(_extract_pool, extract_with_trafilatura, html, url),
                timeout=EXTRACT_TIMEOUT,
            )
        except (asyncio.TimeoutError, Exception):
            return None


async def fetch_and_extract(
    client: httpx.AsyncClient,
    url: str,
    source: dict[str, Any],
    discovery: str,
    hint_dt: datetime | None,
    throttle: DomainThrottle,
    sem: asyncio.Semaphore,
    now: datetime,
) -> Article | None:
    dom = domain_of(url)
    async with sem:
        lock = await throttle.acquire(dom)
        try:
            try:
                resp = await client.get(url, follow_redirects=True)
            except httpx.HTTPError:
                return None
        finally:
            throttle.release(dom, lock)

    if resp.status_code >= 400 or not resp.text:
        return None
    final_url = str(resp.url)
    rec = await extract_capped(resp.text, final_url)

    body = clean_text(rec.get("text")) if rec else ""
    title = clean_text(rec.get("title")) if rec else ""
    author = (clean_text(rec.get("author")) or None) if rec else None
    meta_dt = extract_pub_ts(resp.text)
    traf_dt = parse_dt(rec.get("date")) if rec else None
    pub_dt = meta_dt if meta_dt is not None else traf_dt
    if (pub_dt is not None and traf_dt is not None
            and pub_dt.date() != traf_dt.date()
            and (pub_dt.hour, pub_dt.minute) == (0, 0)):
        pub_dt = traf_dt
    pub_dt = pub_dt or hint_dt
    estimated = pub_dt is None
    if pub_dt is None:
        pub_dt = now
    extraction_ok = bool(body) and len(body) >= 300
    canon = canonicalize(final_url)
    content_hash = (
        hashlib.sha256(body[:2000].encode("utf-8", "ignore")).hexdigest()[:16]
        if extraction_ok else None
    )
    return Article(
        doc_id=stable_id(canon),
        source_name=source["name"],
        source_scope=source.get("scope", "unknown"),
        source_tier=int(source.get("tier", 0)),
        source_access=source.get("access", "open"),
        source_method=source.get("method", "rss"),
        source_domain=dom,
        desks=list(source.get("desks", [])),
        discovery=discovery,
        title=title,
        body_text=body or None,
        author=author,
        url=final_url,
        canonical_url=canon,
        lang=(rec.get("language") if rec else None),
        word_count=len(body.split()) if body else 0,
        extraction_ok=extraction_ok,
        published_utc=pub_dt.isoformat(),
        published_date=pub_dt.date().isoformat(),
        published_is_estimated=estimated,
        fetched_utc=now.isoformat(),
        content_hash=content_hash,
    )


# --------------------------------------------------------------------------- #
# Output (verbatim from crawl_corpus.py)
# --------------------------------------------------------------------------- #
def write_corpus(out_root: Path, articles: list[Article]) -> dict[str, int]:
    by_date: dict[str, dict[str, Article]] = {}
    for a in articles:
        by_date.setdefault(a.published_date, {})[a.doc_id] = a
    written: dict[str, int] = {}
    for date, docs in sorted(by_date.items()):
        part_dir = out_root / f"dt={date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        rows = [asdict(a) for a in docs.values()]
        path = part_dir / "part-000.parquet"
        if _HAVE_PARQUET and path.exists():
            try:
                prev = pq.ParquetFile(path).read().to_pylist()
                seen = {r["doc_id"] for r in rows}
                rows.extend(r for r in prev if r["doc_id"] not in seen)
            except Exception as exc:
                print(f"  [write] REFUSING to overwrite {path}: cannot read "
                      f"existing rows ({exc})", file=sys.stderr)
                written[date] = -1
                continue
        if _HAVE_PARQUET:
            pq.write_table(pa.Table.from_pylist(rows), path)
        else:
            with (part_dir / "part-000.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[date] = len(rows)
    return written


def load_existing_state(out_root: Path) -> tuple[set[str], set[str]]:
    doc_ids: set[str] = set()
    hashes: set[str] = set()
    if not _HAVE_PARQUET or not out_root.exists():
        return doc_ids, hashes
    for path in out_root.glob("dt=*/part-*.parquet"):
        try:
            t = pq.read_table(path, columns=["doc_id", "content_hash"])
            doc_ids.update(x for x in t.column("doc_id").to_pylist() if x)
            hashes.update(x for x in t.column("content_hash").to_pylist() if x)
        except Exception:
            continue
    return doc_ids, hashes


def corpus_size(out_root: Path) -> tuple[int, int]:
    if not _HAVE_PARQUET or not out_root.exists():
        return 0, 0
    files = list(out_root.glob("dt=*/part-*.parquet"))
    rows = 0
    for f in files:
        try:
            rows += pq.read_metadata(f).num_rows
        except Exception:
            pass
    return rows, len(files)


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def load_sources(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["feeds"] if isinstance(data, dict) else data


async def discover_rss_feed(
    client: httpx.AsyncClient, feed_url: str, since: datetime | None,
) -> list[tuple[str, datetime | None]]:
    """Fetch + parse a DIRECT RSS/Atom feed. Never touches news.google.com."""
    status, body = await fetch_bytes(client, feed_url)
    if not status or status >= 400 or not body:
        return []
    parsed = feedparser.parse(body)
    out: list[tuple[str, datetime | None]] = []
    for e in parsed.get("entries", []) or []:
        link = e.get("link") or ""
        if not link:
            continue
        pub = parse_dt(e.get("published") or e.get("updated"))
        if since is not None and pub is not None and pub < since:
            continue
        out.append((link, pub))
    return out


# --------------------------------------------------------------------------- #
# SLOW LANE: incremental, durable, rate-limited Google-News resolution.
# --------------------------------------------------------------------------- #
class SlowLaneStats:
    def __init__(self) -> None:
        self.attempted = 0          # live resolution attempts made this run
        self.resolved_new = 0       # newly resolved this run
        self.from_cache = 0         # served from resolve cache (no network)
        self.failed = 0
        self.skipped_backoff = 0
        self.budget_stopped = False
        self.breaker_tripped = False
        self.feed_fetch_calls = 0   # google RSS feed fetches (also google hits)


async def _google_get(
    client: httpx.AsyncClient, url: str, limiter: gs.RateLimiter,
) -> tuple[int | None, str | None]:
    """A single rate-limited GET against news.google.com (concurrency 1)."""
    wait = limiter.acquire()
    if wait > 0:
        await asyncio.sleep(wait)
    try:
        r = await client.get(url, follow_redirects=True)
        return r.status_code, r.text
    except httpx.HTTPError:
        return None, None


async def run_slow_lane(
    client: httpx.AsyncClient,
    gnews_sources: list[dict[str, Any]],
    since: datetime | None,
    cache: gs.ResolveCache,
    limiter: gs.RateLimiter,
    breaker: gs.CircuitBreaker,
    budget: int,
    stats: SlowLaneStats,
) -> list[tuple[dict[str, Any], str, datetime | None]]:
    """Resolve up to `budget` NEW Google-News ids, honouring cache + breaker.

    Returns [(source, publisher_url, hint_dt)] ready for the extraction pipeline.
    Every call to news.google.com goes through `limiter` (concurrency 1) and is
    checked by `breaker`; the moment a block is detected the breaker trips and
    NO further Google calls are made this run.
    """
    resolved_work: list[tuple[dict[str, Any], str, datetime | None]] = []

    # 1) Fetch each Google-News RSS feed (rate-limited; also a Google hit).
    entries: list[tuple[dict[str, Any], str, datetime | None]] = []
    for src in gnews_sources:
        if breaker.tripped:
            break
        status, body = await _google_get(client, src["url"], limiter)
        stats.feed_fetch_calls += 1
        if gs.detect_block(status, body):
            breaker.trip(f"block detected fetching feed {src['name']}")
            print(f"  [slow] BLOCK detected on feed fetch ({src['name']}); breaker tripped",
                  file=sys.stderr)
            break
        if not status or status >= 400 or not body:
            continue
        parsed = feedparser.parse(body.encode("utf-8", "ignore"))
        for e in parsed.get("entries", []) or []:
            link = e.get("link") or ""
            if not link or "news.google.com" not in link:
                continue
            pub = parse_dt(e.get("published") or e.get("updated"))
            if since is not None and pub is not None and pub < since:
                continue
            entries.append((src, link, pub))

    # 2) Resolve, obeying cache / budget / backoff / breaker.
    for src, link, pub in entries:
        gid = gs.gnews_article_id(link)
        if gid is None:
            continue
        # Already resolved -> reuse cached publisher URL, zero network.
        cached_url = cache.resolved_url(gid)
        if cached_url:
            stats.from_cache += 1
            resolved_work.append((src, cached_url, pub))
            continue
        if breaker.tripped:
            continue
        if cache.should_skip(gid):
            stats.skipped_backoff += 1
            continue
        if stats.resolved_new + stats.failed >= budget:
            stats.budget_stopped = True
            continue

        # -- live, rate-limited resolution attempt -------------------------- #
        cache.mark_pending(gid)
        stats.attempted += 1
        status, body = await _google_get(client, link, limiter)
        if gs.detect_block(status, body):
            cache.mark_failed(gid)
            breaker.trip(f"block detected resolving {gid}")
            stats.failed += 1
            print(f"  [slow] BLOCK detected resolving {gid}; breaker tripped, "
                  f"stopping all Google calls", file=sys.stderr)
            continue
        # Not blocked: decode via the maintained lib (may itself refetch).
        resolved = await asyncio.to_thread(resolve_gnews_sync, link)
        if resolved and "news.google.com" not in resolved:
            cache.mark_resolved(gid, resolved)
            stats.resolved_new += 1
            resolved_work.append((src, resolved, pub))
        else:
            cache.mark_failed(gid)
            stats.failed += 1

    stats.breaker_tripped = breaker.tripped
    cache.compact()
    return resolved_work


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
async def run(args: argparse.Namespace) -> int:
    sources = load_sources(args.feeds)
    if args.desk:
        sources = [s for s in sources if args.desk in s.get("desks", [])]
    if args.scope:
        sources = [s for s in sources if s.get("scope") == args.scope]
    if args.source:
        want = {s.lower() for s in args.source}
        sources = [s for s in sources if s["name"].lower() in want]
    if not sources:
        print("No sources match filters.", file=sys.stderr)
        return 2

    rss_sources = [s for s in sources if s.get("method", "rss") == "rss"]
    gnews_sources = [s for s in sources if s.get("method") == "google_news_rss"]
    if args.no_gnews:
        gnews_sources = []

    now = now_utc()
    since = now - timedelta(hours=args.since_hours) if args.since_hours else None

    seen_doc_ids: set[str] = set()
    seen_hashes: set[str] = set()
    if args.resume:
        seen_doc_ids, seen_hashes = load_existing_state(args.out)
        if seen_doc_ids:
            print(f"[resume] {len(seen_doc_ids)} docs already in corpus -> skipping those URLs",
                  file=sys.stderr)

    timeout = httpx.Timeout(args.timeout, connect=10.0)
    headers = {"User-Agent": UA, "Accept": "*/*"}

    # slow-lane primitives (durable + rate-limited + breaker)
    cache = gs.ResolveCache(args.gnews_cache)
    limiter = gs.RateLimiter(args.gnews_delay_ms, jitter_ms=args.gnews_jitter_ms,
                             rand=random.random)
    breaker = gs.CircuitBreaker()
    slow_stats = SlowLaneStats()

    print(
        f"[lanes] rss={len(rss_sources)} google_news={len(gnews_sources)} "
        f"| since={since.isoformat() if since else 'all'} "
        f"| gnews={'OFF' if args.no_gnews else 'ON'} "
        f"delay_ms={args.gnews_delay_ms} budget={args.gnews_max_per_run}",
        file=sys.stderr,
    )

    async with httpx.AsyncClient(timeout=timeout, headers=headers, http2=False) as client:
        # ---- FAST LANE: direct RSS discovery (parallel) ------------------- #
        disc_sem = asyncio.Semaphore(args.concurrency)

        async def _disc(src: dict[str, Any]):
            async with disc_sem:
                try:
                    urls = await discover_rss_feed(client, src["url"], since)
                    return src, urls
                except Exception as exc:  # noqa: BLE001
                    print(f"  discover ERR {src['name']}: {exc}", file=sys.stderr)
                    return src, []

        disc_results = await asyncio.gather(*(_disc(s) for s in rss_sources))

        per_source_counts: dict[str, int] = {}
        seen: set[str] = set()
        work: list[tuple[dict[str, Any], str, str, datetime | None]] = []
        skipped_resume = 0
        for src, urls in disc_results:
            per_source_counts[src["name"]] = len(urls)
            for link, dt in urls:
                canon = canonicalize(link)
                if canon in seen:
                    continue
                seen.add(canon)
                if args.resume and stable_id(canon) in seen_doc_ids:
                    skipped_resume += 1
                    continue
                work.append((src, canon, "rss", dt))

        # ---- SLOW LANE: incremental Google-News resolution ---------------- #
        if gnews_sources:
            print(f"[slow] resolving up to {args.gnews_max_per_run} new Google-News ids "
                  f"(cache={args.gnews_cache})", file=sys.stderr)
            resolved = await run_slow_lane(
                client, gnews_sources, since, cache, limiter, breaker,
                args.gnews_max_per_run, slow_stats,
            )
            for src, url, dt in resolved:
                per_source_counts[src["name"]] = per_source_counts.get(src["name"], 0) + 1
                canon = canonicalize(url)
                if canon in seen:
                    continue
                seen.add(canon)
                if args.resume and stable_id(canon) in seen_doc_ids:
                    skipped_resume += 1
                    continue
                work.append((src, canon, "google_news", dt))

        if skipped_resume:
            print(f"[resume] skipped {skipped_resume} already-collected URLs", file=sys.stderr)

        print("\n=== discovery report ===", file=sys.stderr)
        for name, n in sorted(per_source_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5}  {name}", file=sys.stderr)
        print(f"\nunique URLs to fetch: {len(work)}", file=sys.stderr)

        cap = args.limit if args.limit and args.limit > 0 else args.max_articles
        if cap and cap > 0 and len(work) > cap:
            print(f"(capping to {cap})", file=sys.stderr)
            work = work[:cap]

        if args.dry_run:
            args.out.mkdir(parents=True, exist_ok=True)
            report = {"run_utc": now.isoformat(), "sources": len(sources),
                      "unique_urls": len(work), "per_source": per_source_counts}
            (args.out / "discovery_report.json").write_text(json.dumps(report, indent=2))
            print(f"\ndry-run: wrote {args.out / 'discovery_report.json'}", file=sys.stderr)
            return 0

        if not _HAVE_TRAFILATURA:
            print("ERROR: trafilatura not installed.", file=sys.stderr)
            return 3

        # ---- EXTRACTION (both lanes' URLs, parallel) ---------------------- #
        global _extract_pool, _extract_sem, _progress
        throttle = DomainThrottle(args.per_domain_delay)
        fetch_sem = asyncio.Semaphore(args.concurrency)
        _extract_pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency + 16)
        _extract_sem = asyncio.Semaphore(args.concurrency)
        _progress = _Progress(len(work))
        window_cutoff = (since - timedelta(days=1)) if since is not None else None
        buffer: list[Article] = []
        written_total: dict[str, int] = {}
        flush_lock = asyncio.Lock()
        counters = {"fetched": 0, "kept": 0, "dropped_old": 0}
        print(f"\n[extraction] fetching {len(work)} URLs (flush every {args.flush_every})...",
              file=sys.stderr)
        t0 = time.monotonic()

        async def _flush() -> None:
            async with flush_lock:
                if not buffer:
                    return
                snapshot = list(buffer)
                buffer.clear()
            w = await asyncio.to_thread(write_corpus, args.out, snapshot)
            written_total.update(w)

        async def _extract(item):
            src, url, kind, dt = item
            art = await fetch_and_extract(client, url, src, kind, dt, throttle, fetch_sem, now)
            if art is not None:
                counters["fetched"] += 1
                keep = True
                if window_cutoff is not None and not art.published_is_estimated:
                    pdt = parse_dt(art.published_utc)
                    if pdt is not None and pdt < window_cutoff:
                        keep = False
                        counters["dropped_old"] += 1
                if keep and art.doc_id in seen_doc_ids:
                    keep = False
                if keep and art.content_hash and art.content_hash in seen_hashes:
                    keep = False
                if keep:
                    seen_doc_ids.add(art.doc_id)
                    if art.content_hash:
                        seen_hashes.add(art.content_hash)
                    buffer.append(art)
                    counters["kept"] += 1
                    if len(buffer) >= args.flush_every:
                        await _flush()
            if _progress is not None:
                _progress.tick(bool(art and art.extraction_ok))

        await asyncio.gather(*(_extract(w) for w in work))
        await _flush()
        _extract_pool.shutdown(wait=False, cancel_futures=True)

    elapsed = time.monotonic() - t0
    body_pct = _progress.bodies * 100 // max(1, counters["fetched"]) if _progress else 0
    print(
        f"\nfetched={counters['fetched']} kept(new, deduped)={counters['kept']} "
        f"full_body={_progress.bodies if _progress else 0} ({body_pct}% of fetched) "
        f"in {elapsed:.0f}s", file=sys.stderr,
    )

    total_rows, total_parts = corpus_size(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    cache_stats = cache.stats()
    report = {
        "run_utc": now.isoformat(),
        "sources": len(sources),
        "fetched": counters["fetched"],
        "kept_this_run": counters["kept"],
        "dropped_old": counters["dropped_old"],
        "full_body_this_run": _progress.bodies if _progress else 0,
        "partitions_written_this_run": written_total,
        "corpus_total_rows": total_rows,
        "corpus_partitions": total_parts,
        "per_source_discovered": per_source_counts,
        "gnews": {
            "enabled": not args.no_gnews,
            "gnews_attempted": slow_stats.attempted,
            "gnews_feed_fetch_calls": slow_stats.feed_fetch_calls,
            "resolved_new": slow_stats.resolved_new,
            "from_cache": slow_stats.from_cache,
            "failed_this_run": slow_stats.failed,
            "skipped_backoff": slow_stats.skipped_backoff,
            "budget": args.gnews_max_per_run,
            "budget_stopped": slow_stats.budget_stopped,
            "breaker_tripped": slow_stats.breaker_tripped,
            "cache_totals": cache_stats,
            "delay_ms": args.gnews_delay_ms,
        },
    }
    (args.out / "last_crawl_report.json").write_text(json.dumps(report, indent=2))
    print(f"\ncorpus now {total_rows} rows across {total_parts} partitions -> {args.out}",
          file=sys.stderr)
    print(f"[gnews] attempted={slow_stats.attempted} resolved_new={slow_stats.resolved_new} "
          f"from_cache={slow_stats.from_cache} failed={slow_stats.failed} "
          f"budget_stopped={slow_stats.budget_stopped} breaker_tripped={slow_stats.breaker_tripped}",
          file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="hardened market-color feed scraper (eventgraph)")
    p.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="output corpus dir (default a TEST dir, NOT production data/news_corpus)")
    p.add_argument("--since-hours", type=int, default=48, help="recency window (0=all)")
    p.add_argument("--max-urls-per-source", type=int, default=400)
    p.add_argument("--max-articles", type=int, default=0, help="global fetch cap (0=unlimited)")
    p.add_argument("--limit", type=int, default=0, help="alias for global fetch cap (0=unlimited)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--per-domain-delay", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--desk")
    p.add_argument("--scope", choices=["global", "focused"])
    p.add_argument("--source", nargs="*")
    p.add_argument("--flush-every", type=int, default=200)
    p.add_argument("--resume", dest="resume", action="store_true", default=True)
    p.add_argument("--no-resume", dest="resume", action="store_false")
    # -- slow-lane (Google-News) hardening flags -- #
    p.add_argument("--no-gnews", action="store_true",
                   help="direct-RSS only; never contact news.google.com")
    p.add_argument("--gnews-delay-ms", type=int, default=4000,
                   help="min delay between news.google.com calls")
    p.add_argument("--gnews-jitter-ms", type=int, default=750,
                   help="random extra delay added per Google call")
    p.add_argument("--gnews-max-per-run", type=int, default=60,
                   help="max NEW Google-News ids to resolve per run (budget)")
    p.add_argument("--gnews-cache", type=Path, default=DEFAULT_GNEWS_CACHE,
                   help="durable resolve-cache path (jsonl)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    try:
        rc = asyncio.run(run(args))
    except KeyboardInterrupt:
        rc = 130
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
