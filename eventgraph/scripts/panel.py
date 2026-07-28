"""
SHARED PANEL LOADER — one definition of "idiosyncratic", read from the cache.

WHY THIS EXISTS. Thirteen scripts each derived their own residuals. That was
survivable while they all did the same thing; it stopped being survivable when the
definition changed. `mcp_precompute` now removes the market factor AND a
hierarchical SIC sector factor before standardising, while the per-script copies
remove only the market factor. The same event therefore measured ~5% larger in a
research script than in the gate that judged it.

Rather than patch the same 60 lines thirteen times, everything should read the
already-computed series from mcp_cache.json. Anything that needs a quantity the
cache does not carry (abnormal volume, for instance) can load just that.

    days      trading-day axis
    sigma     ticker -> {day_index: |idiosyncratic move| / trailing vol}
    decomp    ticker -> {day_index: [total, market, sector, idiosyncratic]}   (signed)
    events    (ticker, date) -> {event classes}
    docs      doc_id -> {date, headline, chunks}

Usage:
    from panel import load
    P = load("us")
    P.sigma_at("AAPL", "2012-04-25")
"""
from __future__ import annotations

import collections
import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "eg_runs"
GRAPHS = {"us": "eg100k_graph", "india": "india2021"}


@dataclass
class Panel:
    market: str
    graph: Path
    days: list[str]
    day_index: dict[str, int]
    sigma: dict[str, dict[str, float]]
    decomp: dict[str, dict[str, list[float]]]
    events: dict[tuple[str, str], set[str]]
    docs: dict[str, dict]
    names: dict[str, str]
    sectors: dict[str, dict]
    neighbours: dict[str, list] = field(default_factory=dict)

    # -- convenience ------------------------------------------------------
    def sigma_at(self, ticker: str, date: str):
        i = self.day_index.get(date)
        if i is None:
            return None
        return self.sigma.get(ticker, {}).get(str(i))

    def idio_at(self, ticker: str, date: str):
        """Signed idiosyncratic return (market and sector already removed)."""
        i = self.day_index.get(date)
        if i is None:
            return None
        d = self.decomp.get(ticker, {}).get(str(i))
        return d[3] if d else None

    def tickers(self) -> list[str]:
        return sorted(self.sigma)

    def event_cells(self):
        """Yield (ticker, date, day_index, classes, sigma) for measurable events."""
        for (tk, d), types in sorted(self.events.items()):
            i = self.day_index.get(d)
            if i is None or tk not in self.sigma:
                continue
            m = self.sigma[tk].get(str(i))
            if m is not None:
                yield tk, d, i, types, float(m)

    def docs_per_day(self) -> collections.Counter:
        c = collections.Counter()
        for l in open(self.graph / "lake" / "document.jsonl"):
            d = (json.loads(l).get("published_at") or "")[:10]
            if d:
                c[d] += 1
        return c


def load(market: str = "us") -> Panel:
    g = ROOT / GRAPHS[market]
    p = g / "mcp_cache.json"
    if not p.exists():
        raise SystemExit(f"{p} missing — run: uv run scripts/mcp_precompute.py "
                         f"--market {market}")
    C = json.loads(p.read_text())
    ev: dict[tuple[str, str], set[str]] = {}
    for k, v in C["events"].items():
        tk, d = k.split("|")
        ev[(tk, d)] = set(v["types"])
    days = C["days"]
    return Panel(market=market, graph=g, days=days,
                 day_index={d: i for i, d in enumerate(days)},
                 sigma=C["sigma"], decomp=C["decomp"], events=ev,
                 docs=C["docs"], names=C["names"], sectors=C["sectors"],
                 neighbours=C.get("neighbours", {}))
