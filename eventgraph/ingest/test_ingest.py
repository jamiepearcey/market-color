#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Offline unit tests for the eventgraph ingest hardening.

NO network. Exercises: durable resolve-cache (round-trip, skip-resolved,
backoff-gating), the rate limiter (min-delay enforcement), the block detector /
circuit breaker (each CAPTCHA string + 429), and the base64 CBMi decode against
a hand-constructed known input.

Run:  uv run eventgraph/ingest/test_ingest.py
      (or)  python -m pytest eventgraph/ingest/test_ingest.py
"""
from __future__ import annotations

import base64
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnews_state as gs  # noqa: E402


def _fixed_clock(dt: datetime):
    return lambda: dt


# --------------------------------------------------------------------------- #
def test_cache_roundtrip_and_skip_resolved():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "gnews_cache.jsonl"
        c = gs.ResolveCache(p)
        c.mark_pending("A")
        c.mark_resolved("A", "https://example.com/a")
        c.mark_pending("B")
        # reload from disk -> state survives (crash-safe append)
        c2 = gs.ResolveCache(p)
        assert c2.is_resolved("A")
        assert c2.resolved_url("A") == "https://example.com/a"
        assert c2.should_skip("A") is True          # never re-resolve resolved
        assert c2.get("B")["status"] == "pending"
        assert c2.should_skip("B") is False          # pending is resolvable
        assert c2.should_skip("NEVER_SEEN") is False
        assert c2.stats() == {"resolved": 1, "pending": 1, "failed": 0}


def test_cache_failed_backoff_gating():
    t0 = datetime(2026, 7, 17, 12, 0, 0, tzinfo=timezone.utc)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "c.jsonl"
        c = gs.ResolveCache(p, clock=_fixed_clock(t0))
        c.mark_failed("X")  # attempts -> 1, backoff = 60s
        # 30s later: still inside 60s window -> skip
        assert c.should_skip("X", now=t0 + timedelta(seconds=30)) is True
        # 61s later: window elapsed -> resolvable again
        assert c.should_skip("X", now=t0 + timedelta(seconds=61)) is False
        # escalate: second failure -> 300s window
        c2 = gs.ResolveCache(p, clock=_fixed_clock(t0 + timedelta(seconds=61)))
        c2.mark_failed("X")
        base = t0 + timedelta(seconds=61)
        assert c2.get("X")["attempts"] == 2
        assert c2.should_skip("X", now=base + timedelta(seconds=120)) is True   # <300
        assert c2.should_skip("X", now=base + timedelta(seconds=301)) is False  # >300


def test_backoff_schedule():
    assert gs.backoff_for(1) == 60
    assert gs.backoff_for(2) == 300
    assert gs.backoff_for(3) == 1800
    assert gs.backoff_for(4) == 7200
    assert gs.backoff_for(5) == 43200
    assert gs.backoff_for(9) == 43200   # clamped to last bucket


def test_rate_limiter_min_delay():
    clk = [0.0]
    rl = gs.RateLimiter(min_delay_ms=4000, jitter_ms=0, clock=lambda: clk[0])
    assert rl.acquire() == 0.0                  # first call: no wait
    assert abs(rl.acquire() - 4.0) < 1e-9       # immediately after: full 4s wait
    # advance the clock past the reservation -> no wait
    clk[0] = 100.0
    assert rl.acquire() == 0.0
    # partial advance -> partial wait
    clk[0] = 101.0                              # next_free was 100+4=104
    assert abs(rl.acquire() - 3.0) < 1e-9


def test_rate_limiter_jitter_bounds():
    clk = [0.0]
    rl = gs.RateLimiter(min_delay_ms=1000, jitter_ms=500,
                        clock=lambda: clk[0], rand=lambda: 1.0)
    # fire time = call-clock + returned wait; spacing between successive fires
    # must be >= min_delay (1.0) + full jitter (0.5) = 1.5 when rand()==1.0.
    fire1 = clk[0] + rl.acquire()
    fire2 = clk[0] + rl.acquire()
    assert abs((fire2 - fire1) - 1.5) < 1e-9


def test_block_detector_all_strings_and_429():
    assert gs.detect_block(429, "") is True
    assert gs.detect_block(200, "Our systems have detected unusual traffic") is True
    assert gs.detect_block(200, "we're sending automated queries") is True
    assert gs.detect_block(503, "sorry, we can't process your request right now") is True
    assert gs.detect_block(200, "<html>normal article body</html>") is False
    assert gs.detect_block(200, None) is False
    assert gs.detect_block(None, None) is False
    # case-insensitivity
    assert gs.detect_block(200, "SENDING AUTOMATED QUERIES") is True


def test_circuit_breaker_latch():
    b = gs.CircuitBreaker()
    assert b.tripped is False
    b.trip("429 seen")
    assert b.tripped is True
    assert b.reason == "429 seen"
    b.trip("second reason")     # latch: first reason sticks
    assert b.reason == "429 seen"


def test_decode_gnews_url_known_cbmi():
    # Hand-construct a CBMi-style base64 payload embedding a real URL, matching
    # decode_gnews_url's contract (regex-find the http(s) bytes in the b64 blob).
    target = "https://www.example.com/markets/story-123"
    raw = b"\x08\x13\x22\x2c" + target.encode() + b"\x00\x30\x01"
    enc = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    link = f"https://news.google.com/rss/articles/{enc}?oc=5"
    assert gs.decode_gnews_url(link) == target
    # non-google URL passes through untouched
    assert gs.decode_gnews_url("https://foo.com/x") == "https://foo.com/x"
    # opaque garbage with no embedded URL -> None
    bad = base64.urlsafe_b64encode(b"\x08\x13nozero").decode().rstrip("=")
    assert gs.decode_gnews_url(f"https://news.google.com/rss/articles/{bad}") is None


def test_gnews_article_id():
    link = "https://news.google.com/rss/articles/CBMiABCDEF_-xyz?oc=5&hl=en"
    assert gs.gnews_article_id(link) == "CBMiABCDEF_-xyz"
    assert gs.gnews_article_id("https://example.com/no-articles") is None


# --------------------------------------------------------------------------- #
def _run_all() -> int:
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
