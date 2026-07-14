#!/bin/zsh
# Rebuild the receptors search/index chain when the corpus has grown.
# NO LLM anywhere (local MiniLM embeddings + BLAS only) — consistent with the
# collection-only token-budget rule. Safe to run every crawl: the staleness
# guard exits immediately when the chunk store already covers the corpus.
set -u
cd "$(dirname "$0")/.." || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

CORPUS_N=$(uv run --quiet --with 'duckdb>=1.0' python3 -c "
import duckdb
print(duckdb.sql(\"select count(distinct doc_id) from read_parquet('../data/news_corpus/dt=*/part-*.parquet')\").fetchone()[0])" 2>/dev/null)
INDEX_N=$(uv run --quiet python3 -c "
import json
print(len(json.load(open('data/doc_meta.json'))))" 2>/dev/null || echo 0)

if [[ -z "$CORPUS_N" ]]; then
  echo "rebuild_index: could not read corpus count; skipping"; exit 0
fi
if [[ "$CORPUS_N" == "$INDEX_N" ]]; then
  echo "rebuild_index: up to date ($INDEX_N docs)"; exit 0
fi

echo "rebuild_index: corpus=$CORPUS_N index=$INDEX_N -> rebuilding"
uv run --quiet scripts/export_rag.py 2>&1 | tail -1
uv run --quiet scripts/export_for_receptors.py 2>&1 | tail -1
uv run --quiet scripts/build_snippet_store.py 2>&1 | tail -2
uv run --quiet scripts/fit_transport_indist.py 2>&1 | tail -1
uv run --quiet scripts/build_semantic_graph.py 2>&1 | tail -1
RECEPTORS_MODE=novelty /Users/jamiepearcey/tmp/codex-cargo-target/release/receptors novelty 2>&1 | tail -1
uv run --quiet scripts/export_polarity.py 2>&1 | tail -1
uv run --quiet scripts/export_modality.py 2>&1 | tail -1
uv run --quiet scripts/polarity_gate_probe.py 2>&1 | tail -1
uv run --quiet scripts/modality_probe.py 2>&1 | tail -1
echo "rebuild_index: done ($CORPUS_N docs)"
