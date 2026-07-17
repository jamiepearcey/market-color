"""Pure-stdlib durability primitives for the Google-News slow lane.

No third-party deps, no network — everything here is unit-testable offline.
Provides:

  * ResolveCache     -- durable, crash-safe resolve cache (jsonl, append-per-update)
  * RateLimiter      -- single shared min-delay + jitter limiter (concurrency 1)
  * CircuitBreaker   -- one-shot "stop all Google calls this run" latch
  * detect_block     -- recognise Google's automated-query CAPTCHA / 429
  * gnews_article_id -- extract the /articles/<id> key from a Google-News link
  * decode_gnews_url -- base64 CBMi decode (ported verbatim from crawl_corpus.py)

Backoff schedule (by attempt count): 1m, 5m, 30m, 2h, 12h.
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

# --------------------------------------------------------------------------- #
# Block detection
# --------------------------------------------------------------------------- #
# Exact substrings (lower-cased) that mark Google's "you're being throttled"
# interstitial. Matching ANY of these (or an HTTP 429) trips the breaker.
BLOCK_STRINGS: tuple[str, ...] = (
    "sending automated queries",
    "can't process your request",
    "unusual traffic",
)


def detect_block(status: Optional[int], body: Optional[str]) -> bool:
    """True iff the response indicates Google is blocking automated queries."""
    if status == 429:
        return True
    if body:
        low = body.lower()
        if any(s in low for s in BLOCK_STRINGS):
            return True
    return False


# --------------------------------------------------------------------------- #
# Backoff schedule
# --------------------------------------------------------------------------- #
# Seconds to wait before retrying a `failed` id, indexed by (attempts - 1),
# clamped to the last bucket: 1m, 5m, 30m, 2h, 12h.
BACKOFF_SECONDS: tuple[int, ...] = (60, 300, 1800, 7200, 43200)


def backoff_for(attempts: int) -> int:
    if attempts <= 0:
        return 0
    return BACKOFF_SECONDS[min(attempts, len(BACKOFF_SECONDS)) - 1]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# Durable resolve cache
# --------------------------------------------------------------------------- #
# On-disk format: one JSON object per line (jsonl). Each line is a full snapshot
# of one id's entry; the LAST line for an id wins on load. Updates are appended
# (with flush + fsync) so a crash mid-run never corrupts prior state. compact()
# rewrites the file atomically to collapse superseded lines.
class ResolveCache:
    STATUS_RESOLVED = "resolved"
    STATUS_PENDING = "pending"
    STATUS_FAILED = "failed"

    def __init__(self, path: os.PathLike | str, *, clock: Callable[[], datetime] = _now_utc) -> None:
        self.path = Path(path)
        self._clock = clock
        self._entries: dict[str, dict] = {}
        self.load()

    # -- persistence ------------------------------------------------------- #
    def load(self) -> None:
        self._entries = {}
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line
                gid = rec.get("id")
                if gid:
                    self._entries[gid] = rec

    def _append(self, rec: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def compact(self) -> None:
        """Atomically rewrite the file with one line per id (drops history)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for rec in self._entries.values():
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.path)

    # -- queries ----------------------------------------------------------- #
    def get(self, gid: str) -> Optional[dict]:
        return self._entries.get(gid)

    def is_resolved(self, gid: str) -> bool:
        e = self._entries.get(gid)
        return bool(e and e.get("status") == self.STATUS_RESOLVED)

    def resolved_url(self, gid: str) -> Optional[str]:
        e = self._entries.get(gid)
        if e and e.get("status") == self.STATUS_RESOLVED:
            return e.get("url")
        return None

    def should_skip(self, gid: str, *, now: Optional[datetime] = None) -> bool:
        """True if we must NOT (re)resolve this id right now.

        - resolved ids: always skip (never re-resolve).
        - failed ids:   skip until their backoff window elapses.
        - pending/new:  do not skip.
        """
        e = self._entries.get(gid)
        if e is None:
            return False
        status = e.get("status")
        if status == self.STATUS_RESOLVED:
            return True
        if status == self.STATUS_FAILED:
            now = now or self._clock()
            last = _parse_iso(e.get("last_attempt_utc"))
            attempts = int(e.get("attempts", 0))
            if last is None:
                return False
            wait = backoff_for(attempts)
            return (now - last).total_seconds() < wait
        return False

    # -- mutations --------------------------------------------------------- #
    def mark_resolved(self, gid: str, url: str) -> dict:
        e = self._entries.get(gid, {"id": gid, "attempts": 0})
        rec = {
            "id": gid,
            "status": self.STATUS_RESOLVED,
            "url": url,
            "last_attempt_utc": self._clock().isoformat(),
            "attempts": int(e.get("attempts", 0)) + 1,
        }
        self._entries[gid] = rec
        self._append(rec)
        return rec

    def mark_pending(self, gid: str) -> dict:
        if gid in self._entries:
            return self._entries[gid]
        rec = {
            "id": gid,
            "status": self.STATUS_PENDING,
            "url": None,
            "last_attempt_utc": None,
            "attempts": 0,
        }
        self._entries[gid] = rec
        self._append(rec)
        return rec

    def mark_failed(self, gid: str) -> dict:
        e = self._entries.get(gid, {"id": gid, "attempts": 0})
        rec = {
            "id": gid,
            "status": self.STATUS_FAILED,
            "url": e.get("url"),
            "last_attempt_utc": self._clock().isoformat(),
            "attempts": int(e.get("attempts", 0)) + 1,
        }
        self._entries[gid] = rec
        self._append(rec)
        return rec

    # -- stats ------------------------------------------------------------- #
    def stats(self) -> dict[str, int]:
        out = {"resolved": 0, "pending": 0, "failed": 0}
        for e in self._entries.values():
            s = e.get("status")
            if s in out:
                out[s] += 1
        return out


# --------------------------------------------------------------------------- #
# Rate limiter (single shared, concurrency 1)
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Enforces a minimum delay (+ jitter) between successive acquisitions.

    `acquire()` returns the number of seconds the caller must sleep before its
    request; it reserves the slot immediately, so serial callers are guaranteed
    to be spaced >= min_delay apart. Pure/deterministic given injected clock +
    rand, so it is testable without ever sleeping.
    """

    def __init__(
        self,
        min_delay_ms: float,
        jitter_ms: float = 0.0,
        *,
        clock: Callable[[], float] = time.monotonic,
        rand: Callable[[], float] = None,
    ) -> None:
        self.min_delay = float(min_delay_ms) / 1000.0
        self.jitter = float(jitter_ms) / 1000.0
        self._clock = clock
        self._rand = rand if rand is not None else __import__("random").random
        self._next_free: float = -math.inf

    def acquire(self) -> float:
        now = self._clock()
        jitter = self._rand() * self.jitter if self.jitter else 0.0
        earliest = max(now, self._next_free) + jitter
        wait = max(0.0, earliest - now)
        # Reserve the slot: the *next* caller must be min_delay past this one.
        self._next_free = (now + wait) + self.min_delay
        return wait


# --------------------------------------------------------------------------- #
# Circuit breaker (one-shot latch)
# --------------------------------------------------------------------------- #
class CircuitBreaker:
    def __init__(self) -> None:
        self._tripped = False
        self.reason: Optional[str] = None

    def trip(self, reason: str = "") -> None:
        if not self._tripped:
            self._tripped = True
            self.reason = reason

    @property
    def tripped(self) -> bool:
        return self._tripped


# --------------------------------------------------------------------------- #
# Google-News URL helpers
# --------------------------------------------------------------------------- #
def gnews_article_id(url: str) -> Optional[str]:
    """Extract the opaque /articles/<id> key used as the cache key."""
    m = re.search(r"/articles/([A-Za-z0-9_\-]+)", url)
    return m.group(1) if m else None


def decode_gnews_url(url: str) -> Optional[str]:
    """Decode a news.google.com/rss/articles/CBMi... link to the publisher URL.

    Ported verbatim from crawl_corpus.py. Handles the common base64-protobuf
    form where the real URL is embedded as a length-delimited string. Newer
    opaque formats return None.
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
    candidate = re.split(r"[\x00-\x1f]", candidate)[0]
    return candidate if candidate.startswith("http") else None
