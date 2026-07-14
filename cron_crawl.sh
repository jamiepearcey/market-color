#!/bin/zsh
# Recurring collection for the EM prospective benchmark. COLLECTION ONLY —
# no LLM extraction (that stays question-driven, per token-budget rule).
# Installed in crontab every 6h. Logs to facts_work/cron_crawl.log.
set -u
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"
STAMP=$(date -u +%Y-%m-%dT%H:%M)
LOG=facts_work/cron_crawl.log
DEPS=(--with httpx --with feedparser --with trafilatura --with pyarrow
      --with python-dateutil --with googlenewsdecoder)

{
  echo "=== $STAMP main feeds ==="
  uv run "${DEPS[@]}" python crawl_corpus.py --since-hours 8 --no-sitemaps 2>&1 | tail -3
  echo "=== $STAMP EM feeds ==="
  uv run "${DEPS[@]}" python crawl_corpus.py --feeds config/feeds_em.json \
      --no-sitemaps --since-hours 8 2>&1 | tail -3
  echo "=== $STAMP hourly prices ==="
  uv run --with yfinance --with pyarrow --with pandas \
      python fetch_hourly.py --days 30 2>&1 | tail -1
  echo "=== $STAMP receptor index ==="
  # local-embedding index rebuild, no LLM; no-ops unless the corpus grew
  /bin/zsh receptors/scripts/rebuild_index.sh 2>&1 | tail -3
} >> "$LOG" 2>&1
