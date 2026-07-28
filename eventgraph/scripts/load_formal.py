# /// script
# requires-python = ">=3.10"
# dependencies = ["duckdb"]
# ///
"""
Load the Python-written FORMAL-plane outputs into DuckDB/DuckLake. The Rust
`eventgraph ingest` loader (src/lake.rs) writes+loads the NARRATIVE tables; the
formal feeds (formal_calendar / options_surface / implied_prob / resolve_series)
write JSONL+parquet that this script applies with the same convention:
`.read` the idempotent DDL, then INSERT ... BY NAME (json) / read_parquet.

Loaded from <graph-dir>/lake/ when present:
  calendar_event.jsonl      -> eg.calendar_event   (append; latest-fetch view dedup)
  market_quote.jsonl        -> eg.market_quote     (append)
  series_alias.jsonl        -> eg.series_alias     (append; latest-mapping view dedup)
  proposition_contract.jsonl-> eg.proposition_contract
  option_iv.parquet         -> eg.option_iv        (snapshot)
  option_surface.parquet    -> eg.option_surface   (snapshot)
event_series.jsonl is intentionally NOT loaded here -- it's the Postgres dim
(applied via pg.rs upsert), not a lake fact.

Idempotency mirrors the Rust path (append + view-level latest-wins dedup). Use
--recreate to DROP+rebuild the formal tables first for a clean deterministic
load (narrative tables untouched).

Usage:
  uv run scripts/load_formal.py --graph-dir /tmp/eg_formal --db /tmp/eg.duckdb
  uv run scripts/load_formal.py --graph-dir /tmp/eg_formal \
      --catalog 'dbname=eventgraph' --data-path /data/eg   # DuckLake attach
"""
import argparse, sys
from pathlib import Path
import duckdb

DDL = ["0002_facts.sql", "0003_realised.sql", "0004_analytics.sql", "0005_v2.sql",
       "0006_options.sql", "0007_lineage.sql", "0008_provenance.sql",
       "0009_surprise.sql", "0010_series_alias.sql", "0011_market_quote.sql",
       "0012_calendar_observation.sql"]

# formal tables this script owns (for --recreate); narrative tables are NOT here.
FORMAL_TABLES = ["calendar_event", "market_quote", "series_alias",
                 "proposition_contract", "option_iv", "option_surface"]
JSON_FILES  = ["calendar_event", "market_quote", "series_alias", "proposition_contract"]
PARQUET_FILES = ["option_iv", "option_surface"]


def main(a):
    ddl_dir = Path(__file__).resolve().parent.parent / "lake"
    lake = Path(a.graph_dir) / "lake"
    con = duckdb.connect(a.db) if a.db else duckdb.connect()
    if a.catalog:  # DuckLake attach (production), mirrors src/lake.rs
        con.execute("INSTALL ducklake; LOAD ducklake;")
        con.execute(f"ATTACH '{a.catalog}' AS eg (DATA_PATH '{a.data_path}');")
        con.execute("USE eg;")
    else:
        con.execute("CREATE SCHEMA IF NOT EXISTS eg;")

    for f in DDL:  # idempotent CREATE ... IF NOT EXISTS / CREATE OR REPLACE VIEW
        con.execute((ddl_dir / f).read_text())

    if a.recreate:
        for t in FORMAL_TABLES:
            con.execute(f"DELETE FROM eg.{t}")  # keep DDL/views; just clear rows
        print(f"recreate: cleared {len(FORMAL_TABLES)} formal tables")

    loaded = 0
    for name in JSON_FILES:
        p = lake / f"{name}.jsonl"
        if p.exists():
            n0 = con.execute(f"SELECT count(*) FROM eg.{name}").fetchone()[0]
            con.execute(f"INSERT INTO eg.{name} BY NAME SELECT * FROM read_json_auto('{p}')")
            n1 = con.execute(f"SELECT count(*) FROM eg.{name}").fetchone()[0]
            print(f"  {name:22s} +{n1 - n0:>7} (json)"); loaded += 1
    for name in PARQUET_FILES:
        p = lake / f"{name}.parquet"
        if p.exists():
            con.execute(f"INSERT INTO eg.{name} BY NAME SELECT * FROM read_parquet('{p}')")
            n = con.execute(f"SELECT count(*) FROM eg.{name}").fetchone()[0]
            print(f"  {name:22s} ={n:>8} (parquet snapshot)"); loaded += 1

    if not loaded:
        sys.exit(f"no formal-plane files found in {lake}")
    con.close()
    print(f"loaded {loaded} formal sources into {a.db or a.catalog or ':memory:'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--graph-dir", required=True)
    ap.add_argument("--db", help="DuckDB file path (research); omit for in-memory")
    ap.add_argument("--catalog", help="DuckLake catalog (production; e.g. 'dbname=eventgraph')")
    ap.add_argument("--data-path", help="DuckLake DATA_PATH (with --catalog)")
    ap.add_argument("--recreate", action="store_true", help="clear formal tables before load")
    a = ap.parse_args()
    if a.catalog and not a.data_path:
        ap.error("--catalog requires --data-path")
    main(a)
