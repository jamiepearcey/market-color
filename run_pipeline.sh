#!/bin/bash
# market-color live pipeline — one idempotent pass, safe to run on a schedule.
#
#   crawl (incremental, --resume) -> index (upsert) -> queue new fact batches
#   -> codex extraction (resume-safe) -> graph build -> bridge to market_facts
#   -> prices fetch -> render all desk briefs
#
# Usage:
#   ./run_pipeline.sh                 # full pass, last 24h of news
#   SINCE_HOURS=6 ./run_pipeline.sh   # tighter crawl window
#   SKIP_FACTS=1 ./run_pipeline.sh    # skip codex extraction (fast refresh:
#                                     # crawl+index+prices+briefs only)
#   MAXP=4 ./run_pipeline.sh          # codex parallelism
#
# Schedule (every 6h at :15):
#   crontab -e
#   15 */6 * * * cd /Users/jamiepearcey/projects/research/market-color && ./run_pipeline.sh >> data/pipeline_cron.log 2>&1
#
# Every stage is individually resumable; a failed stage aborts the pass except
# prices (network flake there shouldn't block the news side). Requires Qdrant
# at :6333 (docker compose up qdrant) and `codex` CLI auth for extraction.
set -uo pipefail
cd "$(dirname "$0")" || exit 1

SINCE_HOURS="${SINCE_HOURS:-24}"
MAXP="${MAXP:-3}"
RUN_TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="data/pipeline_runs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/$RUN_TS.log"
TODAY="$(date +%Y-%m-%d)"
# bound incremental indexing to the recent window (older docs are already in)
if date -v-4d +%Y-%m-%d >/dev/null 2>&1; then
  INDEX_FROM="$(date -v-4d +%Y-%m-%d)"           # macOS
else
  INDEX_FROM="$(date -d '4 days ago' +%Y-%m-%d)" # Linux
fi

say() { echo "[pipeline $(date -u +%H:%M:%SZ)] $*" | tee -a "$LOG"; }
run() { # run <label> <fatal:0|1> <cmd...>
  local label="$1" fatal="$2"; shift 2
  say "== $label =="
  if "$@" >> "$LOG" 2>&1; then
    say "$label: ok"
  else
    say "$label: FAILED (see $LOG)"
    [ "$fatal" = "1" ] && exit 1
  fi
}

PY_CRAWL=(uv run --with httpx --with feedparser --with trafilatura --with pyarrow --with python-dateutil python)
PY_INDEX=(uv run --with 'qdrant-client>=1.15' --with 'duckdb>=1.0' --with fastembed --with numpy --with pyarrow python)
PY_PRICES=(uv run --with httpx --with pyarrow --with 'duckdb>=1.0' python)

say "pass start (since_hours=$SINCE_HOURS skip_facts=${SKIP_FACTS:-0})"

run "crawl"  1 "${PY_CRAWL[@]}" crawl_corpus.py --since-hours "$SINCE_HOURS" --resume --flush-every 200
run "index"  1 "${PY_INDEX[@]}" index_corpus.py index --start-date "$INDEX_FROM"

if [ "${SKIP_FACTS:-0}" != "1" ]; then
  run "fact-batches" 1 "${PY_INDEX[@]}" make_fact_batches.py --prefix all --start-date "$INDEX_FROM"
  run "fact-extract" 1 env PREFIX=all MAXP="$MAXP" bash facts_work/run_codex_batches.sh
  run "graph-build"  1 "${PY_INDEX[@]}" graph_experiment.py build
  run "bridge"       1 "${PY_INDEX[@]}" bridge_facts_to_market_facts.py
fi

run "prices" 0 "${PY_PRICES[@]}" prices.py fetch
run "briefs" 1 "${PY_INDEX[@]}" render_brief.py --all-desks --date "$TODAY"

say "pass done -> briefs in data/briefs/dt=$TODAY/"
