#!/bin/bash
# Drain fact-extraction batches through `codex exec`, MAXP at a time.
#   PREFIX=all MAXP=4 bash facts_work/run_codex_batches.sh
# Resume-safe: batches with an existing output file are skipped.
cd "$(dirname "$0")/batches" || exit 1
MAXP="${MAXP:-3}"
PREFIX="${PREFIX:-energy}"
shopt -s nullglob
for f in "${PREFIX}"_batch_*.jsonl; do
  i="${f#${PREFIX}_batch_}"; i="${i%.jsonl}"
  out="${PREFIX}_facts_${i}.jsonl"
  [ -f "$out" ] && { echo "skip $i (done)"; continue; }
  sed "s|__INPUT__|$f|g; s|__OUTPUT__|$out|g" ../extract_template.md > ".prompt_${PREFIX}_$i.md"
  echo "launch batch ${PREFIX}/$i"
  ( codex exec -C "$(pwd)" -s workspace-write --skip-git-repo-check --ephemeral \
      -o ".last_${PREFIX}_$i.txt" - < ".prompt_${PREFIX}_$i.md" > ".log_${PREFIX}_$i.txt" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
done
wait
echo "ALL_BATCHES_DONE"
