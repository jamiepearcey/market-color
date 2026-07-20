# /// script
# requires-python = ">=3.10"
# dependencies = ["pandas","pyarrow"]
# ///
"""
eventgraph :: temporal feed sampler over the Bloomberg corpus (2006-2013).

Builds a feed.jsonl (doc_id, headline, article, source, published_at, url) with a
coverage plan that gives INTERVALS THROUGHOUT THE YEARS:
  * DENSE days  -- many days spread evenly across the span, each taken to high
    coverage (up to --dense-per-day). These are the "robust intervals".
  * SPARSE baseline -- a thin, CONTINUOUS layer (--sparse-per-day on every active
    day) so the whole period has signal; the gaps between dense days are
    backfillable later by just raising --sparse-per-day on a rerun.

Stable doc_id = 'bbg_'+sha1(Link)[:16]. Deterministic selection (stable sort by
doc_id) => reproducible and disjoint across reruns. --exclude skips doc_ids already
processed (a processed.jsonl manifest) so a run only ever adds NEW documents.

Usage (100-doc smoke):
  uv run sample_feed.py --mode spread --n 100 --out /tmp/eg100_feed.jsonl
Usage (100k coverage):
  uv run sample_feed.py --mode coverage --dense-days 300 --dense-per-day 250 \
      --sparse-per-day 8 --out /tmp/eg100k_feed.jsonl --exclude /tmp/eg100k/processed.jsonl
"""
import argparse, json, hashlib
from pathlib import Path
import pandas as pd

PARQUET = "/Users/jamiepearcey/Downloads/bloomberg_financial_data.parquet.gzip"

def doc_id(link):
    return "bbg_" + hashlib.sha1((link or "").encode()).hexdigest()[:16]

def load_exclude(paths):
    done = set()
    for p in paths or []:
        p = Path(p)
        if p.is_dir():
            p = p / "processed.jsonl"
        if p.exists():
            for l in open(p):
                try: done.add(json.loads(l)["doc_id"])
                except Exception: pass
    return done

def write_feed(rows, out):
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps({
                "doc_id": r["doc_id"],
                "headline": r["Headline"],
                "article": r["Article"],
                "source": "bloomberg",
                "published_at": r["published_at"],
                "url": r["Link"],
            }) + "\n")

def coverage_summary(df_sel):
    yr = pd.to_datetime(df_sel["published_at"]).dt.year.value_counts().sort_index()
    print("  per-year:", {int(k): int(v) for k, v in yr.items()})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["spread", "coverage"], default="spread")
    ap.add_argument("--n", type=int, default=100, help="spread mode: total docs")
    ap.add_argument("--dense-days", type=int, default=300)
    ap.add_argument("--dense-per-day", type=int, default=250)
    ap.add_argument("--sparse-per-day", type=int, default=8)
    ap.add_argument("--target", type=int, default=0, help="optional hard cap on total")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--parquet", default=PARQUET)
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet, columns=["Date", "Headline", "Article", "Link"])
    df = df.dropna(subset=["Link"]).drop_duplicates(subset=["Link"])
    df["published_at"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%dT%H:%M:%S")
    df["doc_id"] = df["Link"].map(doc_id)
    df["day"] = pd.to_datetime(df["Date"]).dt.floor("D")
    # stable deterministic order within a day
    df = df.sort_values(["day", "doc_id"])

    exclude = load_exclude(args.exclude)
    if exclude:
        df = df[~df["doc_id"].isin(exclude)]
    print(f"corpus available (after exclude {len(exclude)}): {len(df)} docs, "
          f"{df['day'].nunique()} active days")

    if args.mode == "spread":
        step = max(1, len(df) // args.n)
        sel = df.iloc[::step].head(args.n)
    else:
        days = sorted(df["day"].unique())
        # dense days: even spread across the active-day axis
        k = args.dense_days
        dense_idx = [int(round(i * (len(days) - 1) / max(1, k - 1))) for i in range(min(k, len(days)))]
        dense_days = {days[i] for i in dense_idx}
        picks = []
        seen = set()
        gb = df.groupby("day", sort=True)
        for day, g in gb:
            per = args.dense_per_day if day in dense_days else args.sparse_per_day
            take = g.head(per)
            for _, r in take.iterrows():
                if r["doc_id"] not in seen:
                    seen.add(r["doc_id"]); picks.append(r)
        sel = pd.DataFrame(picks)
        if args.target and len(sel) > args.target:
            # keep temporal spread: evenly thin to target
            step = len(sel) / args.target
            sel = sel.iloc[[int(i * step) for i in range(args.target)]]

    rows = sel.to_dict("records")
    write_feed(rows, args.out)
    print(f"wrote {len(rows)} docs -> {args.out}")
    coverage_summary(sel)

if __name__ == "__main__":
    main()
