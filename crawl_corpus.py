#!/usr/bin/env python3
"""market-color full-text corpus crawler.

Two-stage open-web ingestion that produces real article BODIES, not headlines:

  1. DISCOVERY  -- gather candidate article URLs per source from:
                     * RSS/Atom feeds (config/feeds.json)
                     * news sitemaps (uncapped, unlike RSS' ~30-100 item limit),
                       discovered via robots.txt `Sitemap:` lines + common paths
                     * Google-News discovery feeds, with the obfuscated
                       /articles/CBMi... link decoded back to the publisher URL.
  2. EXTRACTION -- fetch each unique URL (robots.txt-respecting, per-domain rate
                   limited, globally concurrency capped) and extract the full
                   article body + metadata with trafilatura.

Output: date-partitioned parquet keyed on the article's published (UTC) date,
deduped by canonical URL and by content hash. Paywalled premium sources
(Bloomberg/WSJ/FT/Economist/Platts/Argus/Fastmarkets) will not yield body text
here -- extraction returns little and they are marked `extraction_ok=false`;
they stay headline-only until a licensed feed is wired in.

Run:
  uv run --with httpx --with feedparser --with trafilatura --with pyarrow \
      --with python-dateutil \
      python crawl_corpus.py --dry-run                 # discovery report only
  uv run ... python crawl_corpus.py --since-hours 48 --max-urls-per-source 400
"""
from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import os
import base64
import gzip
import hashlib
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode
from urllib.robotparser import RobotFileParser
from xml.etree import ElementTree as ET

import feedparser  # type: ignore
import httpx  # type: ignore
from dateutil import parser as dateparser  # type: ignore

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
DEFAULT_FEEDS = HERE / "config" / "feeds.json"
DEFAULT_OUT = HERE / "data" / "news_corpus"

# Common sitemap locations to probe when robots.txt advertises none.
COMMON_SITEMAPS = (
    "/sitemap.xml",
    "/sitemap_index.xml",
    "/sitemap-news.xml",
    "/news-sitemap.xml",
    "/news.xml",
    "/sitemap/news.xml",
    "/arc/outboundfeeds/sitemap-index/",
)
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
    discovery: str               # rss | sitemap | google_news
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
# URL / time helpers
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


def clean_text(text: Any) -> str:
    if not text:
        return ""
    s = re.sub(r"<[^>]+>", " ", str(text))
    return re.sub(r"\s+", " ", s).strip()


def decode_gnews_url(url: str) -> str | None:
    """Decode a news.google.com/rss/articles/CBMi... link to the publisher URL.

    Handles the common base64-protobuf form where the real URL is embedded as a
    length-delimited string. Newer opaque formats return None (skipped + counted).
    """
    if "news.google.com" not in url:
        return url
    m = re.search(r"/articles/([A-Za-z0-9_\-]+)", url)
    if not m:
        return None
    enc = m.group(1)
    enc += "=" * (-len(enc) % 4)
    try:
        raw = base64.urlsafe_b64decode(enc)
    except Exception:
        return None
    found = re.search(rb"https?://[^\x00-\x1f\"'\\<> ]+", raw)
    if not found:
        return None
    candidate = found.group(0).decode("utf-8", "ignore")
    # Trailing protobuf bytes occasionally bleed in; cut at obvious junk.
    candidate = re.split(r"[\x00-\x1f]", candidate)[0]
    return candidate if candidate.startswith("http") else None


def resolve_gnews_sync(link: str) -> str | None:
    """Resolve a Google-News redirect to the publisher URL.

    Prefers the maintained `googlenewsdecoder` (handles the opaque AU_yqL...
    batchexecute format); falls back to the base64 heuristic for the older
    CBMi... form. Returns None if it stays a news.google.com link.
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
# Per-domain polite throttle
# --------------------------------------------------------------------------- #
class DomainThrottle:
    def __init__(self, delay: float) -> None:
        self.delay = delay
        self._locks: dict[str, asyncio.Lock] = {}
        self._last: dict[str, float] = {}

    def _lock(self, dom: str) -> asyncio.Lock:
        return self._locks.setdefault(dom, asyncio.Lock())

    async def __aenter__(self):  # not used directly; see acquire()
        raise NotImplementedError

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


# --------------------------------------------------------------------------- #
# Robots
# --------------------------------------------------------------------------- #
@dataclass
class Robots:
    parser: RobotFileParser | None
    sitemaps: list[str] = field(default_factory=list)


async def fetch_bytes(client: httpx.AsyncClient, url: str) -> tuple[int | None, bytes | None]:
    try:
        r = await client.get(url, follow_redirects=True)
        return r.status_code, r.content
    except httpx.HTTPError:
        return None, None


async def get_robots(client: httpx.AsyncClient, root: str, cache: dict[str, Robots]) -> Robots:
    dom = domain_of(root)
    if dom in cache:
        return cache[dom]
    status, body = await fetch_bytes(client, root + "/robots.txt")
    rp = RobotFileParser()
    sitemaps: list[str] = []
    if status and status < 400 and body:
        text = body.decode("utf-8", "ignore")
        rp.parse(text.splitlines())
        sitemaps = re.findall(r"(?im)^\s*Sitemap:\s*(\S+)", text)
    else:
        rp = None  # type: ignore  # no robots => allow (be permissive but polite)
    robots = Robots(parser=rp, sitemaps=sitemaps)
    cache[dom] = robots
    return robots


def robots_allows(robots: Robots, url: str) -> bool:
    if robots.parser is None:
        return True
    try:
        return robots.parser.can_fetch(UA, url)
    except Exception:
        return True


# --------------------------------------------------------------------------- #
# Sitemap discovery
# --------------------------------------------------------------------------- #
def _strip_ns(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def parse_sitemap(content: bytes) -> tuple[list[str], list[tuple[str, datetime | None]]]:
    """Return (child_sitemap_urls, [(article_url, lastmod)]). Handles gzip + ns."""
    if content[:2] == b"\x1f\x8b":
        try:
            content = gzip.decompress(content)
        except Exception:
            return [], []
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return [], []
    tag = _strip_ns(root.tag)
    children: list[str] = []
    urls: list[tuple[str, datetime | None]] = []
    if tag == "sitemapindex":
        for sm in root:
            loc = next((c.text for c in sm if _strip_ns(c.tag) == "loc"), None)
            if loc:
                children.append(loc.strip())
    elif tag == "urlset":
        for u in root:
            loc = None
            lastmod = None
            for c in u:
                ct = _strip_ns(c.tag)
                if ct == "loc":
                    loc = (c.text or "").strip()
                elif ct == "lastmod":
                    lastmod = parse_dt(c.text)
                elif ct == "news":  # news:news block
                    for nc in c:
                        if _strip_ns(nc.tag) == "publication_date":
                            lastmod = parse_dt(nc.text) or lastmod
            if loc:
                urls.append((loc, lastmod))
    return children, urls


async def discover_sitemap_urls(
    client: httpx.AsyncClient,
    root: str,
    robots: Robots,
    since: datetime | None,
    max_urls: int,
    max_sitemaps: int = 25,
) -> list[str]:
    seeds = list(dict.fromkeys(robots.sitemaps + [root + p for p in COMMON_SITEMAPS]))
    seen_sitemaps: set[str] = set()
    queue = list(seeds)
    out: list[str] = []
    processed = 0
    while queue and len(out) < max_urls and processed < max_sitemaps:
        sm = queue.pop(0)
        if sm in seen_sitemaps:
            continue
        seen_sitemaps.add(sm)
        status, body = await fetch_bytes(client, sm)
        if not status or status >= 400 or not body:
            continue
        processed += 1
        children, urls = parse_sitemap(body)
        # Prefer news/recent sitemaps: queue children, newest-looking first.
        for child in children[:max_sitemaps]:
            if child not in seen_sitemaps:
                queue.append(child)
        for url, lastmod in urls:
            if since is not None and lastmod is not None and lastmod < since:
                continue
            out.append(url)
            if len(out) >= max_urls:
                break
    return out


# --------------------------------------------------------------------------- #
# Feed discovery
# --------------------------------------------------------------------------- #
async def discover_feed_urls(
    client: httpx.AsyncClient,
    feed_url: str,
    method: str,
    since: datetime | None,
    gnews_cap: int = 100,
    gnews_concurrency: int = 6,
) -> list[tuple[str, datetime | None]]:
    status, body = await fetch_bytes(client, feed_url)
    if not status or status >= 400 or not body:
        return []
    parsed = feedparser.parse(body)
    raw: list[tuple[str, datetime | None]] = []
    for e in parsed.get("entries", []) or []:
        link = e.get("link") or ""
        if not link:
            continue
        pub = parse_dt(e.get("published") or e.get("updated"))
        if since is not None and pub is not None and pub < since:
            continue
        raw.append((link, pub))

    if method != "google_news_rss":
        return raw

    # Resolve Google-News redirect links to publisher URLs, concurrently + capped.
    raw = raw[:gnews_cap]
    sem = asyncio.Semaphore(gnews_concurrency)

    async def _resolve(link: str, pub: datetime | None) -> tuple[str, datetime | None] | None:
        async with sem:
            resolved = await asyncio.to_thread(resolve_gnews_sync, link)
        return (resolved, pub) if resolved and "news.google.com" not in resolved else None

    resolved = await asyncio.gather(*(_resolve(l, p) for l, p in raw))
    return [r for r in resolved if r is not None]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def extract_with_trafilatura(html: str, url: str) -> dict[str, Any] | None:
    if not _HAVE_TRAFILATURA:
        return None
    data = trafilatura.extract(
        html,
        url=url,
        output_format="json",
        with_metadata=True,
        include_comments=False,
        favor_precision=True,
    )
    if not data:
        return None
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None


# Hardened extraction infra (configured in run()). trafilatura can hang on a few
# pathological pages; without a per-call timeout one stuck thread deadlocks the
# whole run. We cap HTML size, run extraction on a dedicated bounded pool, and
# abandon any extraction that exceeds the timeout.
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
        if self.done % 100 == 0 or self.done == self.total:
            pct = self.done * 100 // max(1, self.total)
            print(
                f"  [extract] {self.done}/{self.total} ({pct}%), {self.bodies} bodies",
                file=sys.stderr,
            )


_progress: _Progress | None = None


async def extract_capped(html: str, url: str) -> dict[str, Any] | None:
    if len(html) > HTML_CHAR_CAP:
        html = html[:HTML_CHAR_CAP]
    loop = asyncio.get_running_loop()
    if _extract_sem is None or _extract_pool is None:  # fallback: inline thread
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
    robots_cache: dict[str, Robots],
    sem: asyncio.Semaphore,
    now: datetime,
) -> Article | None:
    dom = domain_of(url)
    async with sem:
        robots = await get_robots(client, site_root(url), robots_cache)
        if not robots_allows(robots, url):
            return None
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
    pub_dt = parse_dt(rec.get("date")) if rec else None
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
# Output
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
                prev = pq.read_table(path).to_pylist()
                seen = {r["doc_id"] for r in rows}
                rows.extend(r for r in prev if r["doc_id"] not in seen)
            except Exception:
                pass
        if _HAVE_PARQUET:
            pq.write_table(pa.Table.from_pylist(rows), path)
        else:
            with (part_dir / "part-000.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[date] = len(rows)
    return written


def load_existing_state(out_root: Path) -> tuple[set[str], set[str]]:
    """Read doc_ids + content_hashes already in the corpus, for resume + dedup."""
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
    """(total_rows, partition_files) across the whole corpus on disk."""
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
# Orchestration
# --------------------------------------------------------------------------- #
def load_sources(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["feeds"] if isinstance(data, dict) else data


async def discover_source(
    client: httpx.AsyncClient,
    source: dict[str, Any],
    since: datetime | None,
    max_urls: int,
    use_sitemaps: bool,
    use_feeds: bool,
    robots_cache: dict[str, Robots],
) -> dict[str, tuple[str, datetime | None]]:
    """Return {canonical_url: (discovery_kind, hint_dt)} for one source."""
    found: dict[str, tuple[str, datetime | None]] = {}
    method = source.get("method", "rss")

    if use_feeds:
        kind = "google_news" if method == "google_news_rss" else "rss"
        for url, dt in await discover_feed_urls(client, source["url"], method, since):
            found.setdefault(canonicalize(url), (kind, dt))

    # Sitemaps only make sense for a source's own domain (not Google News proxies).
    if use_sitemaps and method == "rss":
        root = site_root(source["url"])
        robots = await get_robots(client, root, robots_cache)
        for url in await discover_sitemap_urls(client, root, robots, since, max_urls):
            found.setdefault(canonicalize(url), ("sitemap", None))
        # cap per source
    if max_urls > 0 and len(found) > max_urls:
        found = dict(list(found.items())[:max_urls])
    return found


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

    now = now_utc()
    since = now - timedelta(hours=args.since_hours) if args.since_hours else None

    seen_doc_ids: set[str] = set()
    seen_hashes: set[str] = set()
    if args.resume:
        seen_doc_ids, seen_hashes = load_existing_state(args.out)
        if seen_doc_ids:
            print(f"[resume] {len(seen_doc_ids)} docs already in corpus -> skipping those URLs",
                  file=sys.stderr)

    robots_cache: dict[str, Robots] = {}
    timeout = httpx.Timeout(args.timeout, connect=10.0)
    headers = {"User-Agent": UA, "Accept": "*/*"}
    disc_sem = asyncio.Semaphore(args.concurrency)

    print(
        f"[discovery] {len(sources)} sources | since="
        f"{since.isoformat() if since else 'all'} | sitemaps={not args.no_sitemaps} "
        f"feeds={not args.no_feeds}",
        file=sys.stderr,
    )

    async with httpx.AsyncClient(timeout=timeout, headers=headers, http2=False) as client:
        async def _disc(src: dict[str, Any]):
            async with disc_sem:
                try:
                    urls = await discover_source(
                        client, src, since, args.max_urls_per_source,
                        not args.no_sitemaps, not args.no_feeds, robots_cache,
                    )
                    return src, urls
                except Exception as exc:  # noqa: BLE001
                    print(f"  discover ERR {src['name']}: {exc}", file=sys.stderr)
                    return src, {}

        disc_results = await asyncio.gather(*(_disc(s) for s in sources))

        # Build global work list, dedup canonical URLs across sources.
        seen: set[str] = set()
        work: list[tuple[dict[str, Any], str, str, datetime | None]] = []
        per_source_counts: dict[str, int] = {}
        skipped_resume = 0
        for src, urls in disc_results:
            per_source_counts[src["name"]] = len(urls)
            for canon, (kind, dt) in urls.items():
                if canon in seen:
                    continue
                seen.add(canon)
                if args.resume and stable_id(canon) in seen_doc_ids:
                    skipped_resume += 1
                    continue
                work.append((src, canon, kind, dt))
        if skipped_resume:
            print(f"[resume] skipped {skipped_resume} already-collected URLs", file=sys.stderr)

        print("\n=== discovery report ===", file=sys.stderr)
        for name, n in sorted(per_source_counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>5}  {name}", file=sys.stderr)
        print(f"\nunique URLs to fetch: {len(work)}", file=sys.stderr)

        if args.max_articles > 0 and len(work) > args.max_articles:
            print(f"(capping to --max-articles={args.max_articles})", file=sys.stderr)
            work = work[: args.max_articles]

        if args.dry_run:
            args.out.mkdir(parents=True, exist_ok=True)
            report = {
                "run_utc": now.isoformat(),
                "sources": len(sources),
                "unique_urls": len(work),
                "per_source": per_source_counts,
            }
            (args.out / "discovery_report.json").write_text(json.dumps(report, indent=2))
            print(f"\ndry-run: wrote {args.out / 'discovery_report.json'}", file=sys.stderr)
            return 0

        if not _HAVE_TRAFILATURA:
            print("ERROR: trafilatura not installed; add `--with trafilatura`.", file=sys.stderr)
            return 3

        # Extraction with incremental durable writes (flush as partitions fill).
        global _extract_pool, _extract_sem, _progress
        throttle = DomainThrottle(args.per_domain_delay)
        fetch_sem = asyncio.Semaphore(args.concurrency)
        _extract_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency + 16
        )
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
            art = await fetch_and_extract(
                client, url, src, kind, dt, throttle, robots_cache, fetch_sem, now
            )
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
    if counters["dropped_old"]:
        print(f"dropped {counters['dropped_old']} out-of-window (stale-date) articles",
              file=sys.stderr)
    body_pct = _progress.bodies * 100 // max(1, counters["fetched"]) if _progress else 0
    print(
        f"\nfetched={counters['fetched']} kept(new, deduped)={counters['kept']} "
        f"full_body={_progress.bodies if _progress else 0} ({body_pct}% of fetched) "
        f"in {elapsed:.0f}s",
        file=sys.stderr,
    )

    total_rows, total_parts = corpus_size(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
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
    }
    (args.out / "last_crawl_report.json").write_text(json.dumps(report, indent=2))
    print(f"\ncorpus now {total_rows} rows across {total_parts} partitions -> {args.out}",
          file=sys.stderr)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="market-color full-text corpus crawler")
    p.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--since-hours", type=int, default=48, help="recency window (0=all)")
    p.add_argument("--max-urls-per-source", type=int, default=400)
    p.add_argument("--max-articles", type=int, default=0, help="global fetch cap (0=unlimited)")
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--per-domain-delay", type=float, default=1.0, help="seconds between hits to same domain")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--no-sitemaps", action="store_true")
    p.add_argument("--no-feeds", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="discovery only, no fetch/extract")
    p.add_argument("--desk")
    p.add_argument("--scope", choices=["global", "focused"])
    p.add_argument("--source", nargs="*", help="only these source names")
    p.add_argument("--flush-every", type=int, default=200,
                   help="write partitions to disk every N kept articles (durable progress)")
    p.add_argument("--resume", dest="resume", action="store_true", default=True,
                   help="skip URLs already in the corpus (default on)")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   help="ignore existing corpus; re-fetch everything")
    args = p.parse_args()
    try:
        rc = asyncio.run(run(args))
    except KeyboardInterrupt:
        rc = 130
    # A trafilatura extraction thread may still be hung; force-exit so it can't
    # block interpreter shutdown (we've already written the corpus + report).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    main()
