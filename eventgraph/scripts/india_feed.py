# /// script
# requires-python = ">=3.10"
# ///
"""Build an UNBIASED complete-time-slice feed from the HF Indian markets news dataset
(mveen3/Six_Year_Indian_Stock_Market_Dataset). Streams processed_news_dataset.csv,
filters to a year (NOT to relevance — avoids the endogeneity skew of selecting on
single-name channels), dedups, emits the eventgraph feed shape. Pipe the CSV in:

  curl -sL <hf-processed_news_dataset.csv> | uv run scripts/india_feed.py 2021 out/feed.jsonl
"""
import csv, sys, json, hashlib, re
csv.field_size_limit(10**7)
year, outpath = sys.argv[1], sys.argv[2]
r=csv.reader(sys.stdin); next(r)
dom=lambda u:(re.search(r'https?://(?:www\.)?([^/]+)',u or '') or [None,'india'])[1].split('.')[0] if u else 'india'
seen=set(); n=kept=0
with open(outpath,"w") as out:
    for row in r:
        n+=1
        if len(row)<4 or not row[0].startswith(year) or len(row[2] or "")<250: continue
        h=hashlib.sha1((row[3] or row[1]).encode()).hexdigest()[:16]
        if h in seen: continue
        seen.add(h)
        out.write(json.dumps({"doc_id":"in_"+h,"headline":row[1],"article":row[2],
            "source":dom(row[3]),"published_at":row[0],"url":row[3]})+"\n"); kept+=1
sys.stderr.write(f"{year}: scanned {n} -> kept {kept}\n")
