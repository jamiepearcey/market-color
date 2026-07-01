#!/usr/bin/env python3
"""v4 market-color news scraper.

RSS/Atom-first ingestion of the curated feed list into a point-in-time-correct
local corpus. Each item is normalized to a common schema, deduplicated by a
stable doc_id, and written to date-partitioned parquet keyed on the article's
*published* time (UTC), not fetch time -- so the corpus respects temporal
ordering for downstream as-of queries.

Body text is the grounding constraint for the whole project, so an optional
full-text extraction pass (--fetch-full-text) hydrates `body_text` from the
article URL when the feed only ships a summary.

Run:
    uv run --with feedparser --with httpx --with pyarrow --with python-dateutil \
        python v4/scrape_news_feeds.py --validate

    uv run --with feedparser --with httpx --with pyarrow --with python-dateutil \
        --with trafilatura \
        python v4/scrape_news_feeds.py --fetch-full-text --since-hours 48
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit

import feedparser  # type: ignore
import httpx  # type: ignore
from dateutil import parser as dateparser  # type: ignore

try:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore
    _HAVE_PARQUET = True
except Exception:  # pragma: no cover - optional at import time
    _HAVE_PARQUET = False

try:
    import trafilatura  # type: ignore
    _HAVE_TRAFILATURA = True
except Exception:
    _HAVE_TRAFILATURA = False


USER_AGENT = (
    "Mozilla/5.0 (compatible; v4-market-color/0.1; +research news-narrative-explainer)"
)
DEFAULT_FEEDS = Path(__file__).resolve().parent / "config" / "feeds.json"
DEFAULT_OUT = Path(__file__).resolve().parent / "data" / "news_corpus"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
@dataclass
class Article:
    doc_id: str
    source_name: str
    source_scope: str
    source_tier: int
    source_access: str              # open | headline | paywall | licensed
    source_method: str              # rss | google_news_rss | scrape
    source_domain: str
    desks: list[str]
    title: str
    summary: str
    body_text: str | None
    author: str | None
    url: str
    raw_guid: str | None
    lang: str | None
    published_utc: str | None       # ISO8601 UTC, None if undatable
    published_date: str             # YYYY-MM-DD partition key (UTC)
    published_is_estimated: bool     # True when we fell back to fetch time
    fetched_utc: str
    has_full_text: bool


@dataclass
class FeedResult:
    name: str
    url: str
    ok: bool
    status: int | None = None
    entries: int = 0
    kept: int = 0
    error: str | None = None
    elapsed_ms: int = 0
    bozo: bool = False
    articles: list[Article] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _domain(url: str) -> str:
    try:
        host = urlsplit(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        return ""


def _stable_id(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").strip().encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def _clean(text: Any) -> str:
    if not text:
        return ""
    s = str(text)
    # feedparser gives HTML in summaries; strip tags cheaply for the summary field.
    import re

    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_published(entry: Any) -> tuple[datetime | None, bool]:
    """Return (utc_datetime, estimated). estimated=True means we could not date it."""
    for key in ("published", "updated", "created"):
        val = entry.get(key)
        if val:
            try:
                dt = dateparser.parse(val)
                if dt is not None:
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt.astimezone(timezone.utc), False
            except (ValueError, OverflowError, TypeError):
                pass
    # struct_time fallbacks
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                return datetime(*st[:6], tzinfo=timezone.utc), False
            except (ValueError, TypeError):
                pass
    return None, True


def _best_summary(entry: Any) -> str:
    if entry.get("summary"):
        return _clean(entry.get("summary"))
    content = entry.get("content")
    if content and isinstance(content, list) and content:
        return _clean(content[0].get("value"))
    return ""


def _inline_body(entry: Any) -> str | None:
    """Body text that the feed itself shipped (content:encoded), if substantial."""
    content = entry.get("content")
    if content and isinstance(content, list) and content:
        body = _clean(content[0].get("value"))
        if len(body) > 400:  # heuristic: real article body, not a teaser
            return body
    return None


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #
async def fetch_feed(
    client: httpx.AsyncClient,
    feed: dict[str, Any],
    sem: asyncio.Semaphore,
    since: datetime | None,
    max_per_feed: int,
    now: datetime,
) -> FeedResult:
    name, url = feed["name"], feed["url"]
    res = FeedResult(name=name, url=url, ok=False)
    t0 = time.monotonic()
    async with sem:
        try:
            resp = await client.get(url, follow_redirects=True)
            res.status = resp.status_code
            resp.raise_for_status()
            parsed = feedparser.parse(resp.content)
            res.bozo = bool(parsed.get("bozo"))
            entries = parsed.get("entries", []) or []
            res.entries = len(entries)
            fetched_iso = now.isoformat()
            for entry in entries:
                link = entry.get("link") or ""
                guid = entry.get("id") or entry.get("guid")
                if not link and not guid:
                    continue
                pub_dt, estimated = _parse_published(entry)
                if pub_dt is None:
                    pub_dt = now  # date with fetch time, but flag it
                if since is not None and pub_dt < since:
                    continue
                title = _clean(entry.get("title"))
                inline = _inline_body(entry)
                art = Article(
                    doc_id=_stable_id(guid or link, name),
                    source_name=name,
                    source_scope=feed.get("scope", "unknown"),
                    source_tier=int(feed.get("tier", 0)),
                    source_access=feed.get("access", "open"),
                    source_method=feed.get("method", "rss"),
                    source_domain=_domain(link or url),
                    desks=list(feed.get("desks", [])),
                    title=title,
                    summary=_best_summary(entry),
                    body_text=inline,
                    author=_clean(entry.get("author")) or None,
                    url=link,
                    raw_guid=str(guid) if guid else None,
                    lang=(parsed.feed.get("language") if parsed.get("feed") else None),
                    published_utc=pub_dt.isoformat(),
                    published_date=pub_dt.date().isoformat(),
                    published_is_estimated=estimated,
                    fetched_utc=fetched_iso,
                    has_full_text=inline is not None,
                )
                res.articles.append(art)
            # newest first, cap
            res.articles.sort(key=lambda a: a.published_utc or "", reverse=True)
            if max_per_feed > 0:
                res.articles = res.articles[:max_per_feed]
            res.kept = len(res.articles)
            res.ok = True
        except httpx.HTTPStatusError as exc:
            res.error = f"http {exc.response.status_code}"
        except httpx.HTTPError as exc:
            res.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001 - report, never crash the batch
            res.error = f"{type(exc).__name__}: {exc}"
    res.elapsed_ms = int((time.monotonic() - t0) * 1000)
    return res


async def hydrate_full_text(
    client: httpx.AsyncClient, articles: list[Article], sem: asyncio.Semaphore
) -> int:
    if not _HAVE_TRAFILATURA:
        print(
            "  [full-text] trafilatura not installed; "
            "re-run with `--with trafilatura`. Skipping.",
            file=sys.stderr,
        )
        return 0
    targets = [a for a in articles if not a.has_full_text and a.url]
    hydrated = 0

    async def _one(art: Article) -> None:
        nonlocal hydrated
        async with sem:
            try:
                resp = await client.get(art.url, follow_redirects=True)
                resp.raise_for_status()
                html = resp.text
            except httpx.HTTPError:
                return
        extracted = await asyncio.to_thread(
            trafilatura.extract, html, include_comments=False, favor_precision=True
        )
        if extracted and len(extracted) > 400:
            art.body_text = _clean(extracted)
            art.has_full_text = True
            hydrated += 1

    await asyncio.gather(*(_one(a) for a in targets))
    return hydrated


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _slug(name: str) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def write_corpus(out_root: Path, articles: list[Article]) -> dict[str, int]:
    """Write date-partitioned parquet, deduped by doc_id within each partition."""
    by_date: dict[str, dict[str, Article]] = {}
    for a in articles:
        by_date.setdefault(a.published_date, {})[a.doc_id] = a

    written: dict[str, int] = {}
    for date, docs in sorted(by_date.items()):
        part_dir = out_root / f"dt={date}"
        part_dir.mkdir(parents=True, exist_ok=True)
        rows = [asdict(a) for a in docs.values()]

        # Merge with any existing partition so re-runs accumulate rather than clobber.
        existing_path = part_dir / "part-000.parquet"
        if _HAVE_PARQUET and existing_path.exists():
            try:
                prev = pq.read_table(existing_path).to_pylist()
                seen = {r["doc_id"] for r in rows}
                rows.extend(r for r in prev if r["doc_id"] not in seen)
            except Exception:
                pass

        if _HAVE_PARQUET:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, existing_path)
        else:  # graceful fallback: jsonl
            with (part_dir / "part-000.jsonl").open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written[date] = len(rows)
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_feeds(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    return data["feeds"] if isinstance(data, dict) else data


async def run(args: argparse.Namespace) -> int:
    feeds = load_feeds(args.feeds)
    if args.desk:
        feeds = [f for f in feeds if args.desk in f.get("desks", [])]
    if args.scope:
        feeds = [f for f in feeds if f.get("scope") == args.scope]
    if not feeds:
        print("No feeds match the given filters.", file=sys.stderr)
        return 2

    now = _now_utc()
    since = now - timedelta(hours=args.since_hours) if args.since_hours else None
    sem = asyncio.Semaphore(args.concurrency)
    timeout = httpx.Timeout(args.timeout, connect=min(10.0, args.timeout))
    headers = {"User-Agent": USER_AGENT, "Accept": "application/rss+xml, application/xml, text/xml, */*"}

    print(
        f"v4 scrape: {len(feeds)} feeds | since={since.isoformat() if since else 'all'} "
        f"| full_text={args.fetch_full_text}",
        file=sys.stderr,
    )

    async with httpx.AsyncClient(timeout=timeout, headers=headers) as client:
        results = await asyncio.gather(
            *(fetch_feed(client, f, sem, since, args.max_per_feed, now) for f in feeds)
        )

        all_articles = [a for r in results for a in r.articles]
        if args.fetch_full_text and all_articles:
            ft_sem = asyncio.Semaphore(max(2, args.concurrency // 2))
            print(f"  hydrating full text for up to {len(all_articles)} articles...", file=sys.stderr)
            n = await hydrate_full_text(client, all_articles, ft_sem)
            print(f"  full text hydrated: {n}", file=sys.stderr)

    # Report
    ok = [r for r in results if r.ok]
    bad = [r for r in results if not r.ok]
    print("\n=== feed report ===", file=sys.stderr)
    for r in sorted(results, key=lambda r: (r.ok, -r.kept)):
        flag = "OK " if r.ok else "ERR"
        detail = f"{r.kept}/{r.entries} kept" if r.ok else (r.error or "?")
        bozo = " [bozo]" if r.bozo and r.ok else ""
        print(f"  {flag} {r.name:<34} {detail:<18} {r.elapsed_ms:>5}ms{bozo}", file=sys.stderr)
    print(
        f"\nfeeds ok={len(ok)} failed={len(bad)} | articles={len(all_articles)} "
        f"| with_body={sum(1 for a in all_articles if a.has_full_text)}",
        file=sys.stderr,
    )

    report = {
        "run_utc": now.isoformat(),
        "feeds_total": len(feeds),
        "feeds_ok": len(ok),
        "feeds_failed": len(bad),
        "articles": len(all_articles),
        "articles_with_body": sum(1 for a in all_articles if a.has_full_text),
        "since_hours": args.since_hours,
        "full_text": args.fetch_full_text,
        "feed_status": [
            {k: v for k, v in asdict(r).items() if k != "articles"} for r in results
        ],
    }

    if args.validate:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "validate_report.json").write_text(json.dumps(report, indent=2))
        print(f"\nvalidate-only: wrote {args.out / 'validate_report.json'}", file=sys.stderr)
        return 0 if not bad else 1

    if all_articles:
        written = write_corpus(args.out, all_articles)
        report["partitions_written"] = written
        print(f"\nwrote {sum(written.values())} rows across {len(written)} day partitions -> {args.out}", file=sys.stderr)
    (args.out).mkdir(parents=True, exist_ok=True)
    (args.out / "last_run_report.json").write_text(json.dumps(report, indent=2))
    return 0 if not bad else 1


def main() -> int:
    p = argparse.ArgumentParser(description="v4 market-color RSS/Atom news scraper")
    p.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS, help="feeds config json")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output corpus root")
    p.add_argument("--since-hours", type=int, default=48, help="only keep items newer than N hours (0=all)")
    p.add_argument("--max-per-feed", type=int, default=200, help="cap items per feed (0=unlimited)")
    p.add_argument("--concurrency", type=int, default=12, help="max concurrent feed fetches")
    p.add_argument("--timeout", type=float, default=25.0, help="per-request timeout seconds")
    p.add_argument("--fetch-full-text", action="store_true", help="hydrate body_text from article URLs (needs trafilatura)")
    p.add_argument("--validate", action="store_true", help="fetch/parse only, write a feed health report, no corpus write")
    p.add_argument("--desk", help="only feeds tagged with this desk")
    p.add_argument("--scope", choices=["global", "focused"], help="only feeds with this scope")
    args = p.parse_args()

    if not _HAVE_PARQUET and not args.validate:
        print("warning: pyarrow not available; will write jsonl fallback. Add `--with pyarrow`.", file=sys.stderr)
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
