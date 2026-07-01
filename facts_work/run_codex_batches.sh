#!/bin/bash
cd "$(dirname "$0")/batches" || exit 1
MAXP="${MAXP:-3}"
for f in energy_batch_*.jsonl; do
  i="${f#energy_batch_}"; i="${i%.jsonl}"
  out="energy_facts_${i}.jsonl"
  [ -f "$out" ] && { echo "skip $i (done)"; continue; }
  sed "s|__INPUT__|$f|g; s|__OUTPUT__|$out|g" ../extract_template.md > ".prompt_$i.md"
  echo "launch batch $i"
  ( codex exec -C "$(pwd)" -s workspace-write --skip-git-repo-check --ephemeral \
      -o ".last_$i.txt" - < ".prompt_$i.md" > ".log_$i.txt" 2>&1 ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 5; done
done
wait
echo "ALL_BATCHES_DONE"
