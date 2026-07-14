#!/bin/bash
# Drain pending fact-extraction batches through the claude CLI (Haiku),
# stdin -> stdout (no file-tool permissions needed). Resume-safe: batches
# with an existing output file are skipped. Outputs validated separately
# (validate_claude_batches.py deletes invalid outputs so re-runs retry them).
#   MAXP=4 bash facts_work/run_claude_batches.sh
cd "$(dirname "$0")/batches" || exit 1
CLAUDE=/opt/homebrew/bin/claude
MODEL="${MODEL:-claude-haiku-4-5}"
MAXP="${MAXP:-4}"
PREFIX="${PREFIX:-all}"
HEADER=$(cat <<'EOH'
You are a financial-news fact extractor. Below are documents, one JSON object per line: {doc_id, source_name, published_date, url, title, body_text}.

For EACH document, extract the atomic, market-relevant FACTS explicitly supported by its title+body_text. An atomic fact is a single checkable claim about an asset, commodity, company, country, central bank, or market — with direction/magnitude/time where stated, and a causal link where the text states one. Do NOT invent or infer beyond the text. If a document has no market-relevant facts (e.g. a job posting), use an empty facts list.

Each fact uses this shape:
{"claim":"<one self-contained sentence>","subject":"<primary entity>","predicate":"<one of: price_move, supply_change, demand_change, production_change, policy_action, deal_or_contract, corporate_action, forecast, sanction, statement, event, other>","object":"<counterparty/target or null>","direction":"<up|down|flat|na>","magnitude":"<value/percent as written or null>","time":"<date/period as written, else the doc published_date>","cause":"<the MOST SPECIFIC stated driver, naming the concrete event/actor (e.g. 'Iran fired on the tanker Kiku in the Strait of Hormuz', not 'the conflict') — when the text states both a general and a specific cause, use the specific one; null if none stated>","entities":["<canonical entity names>"],"confidence":<0.0-1.0>}

OUTPUT REQUIREMENTS: Output ONLY raw JSONL — exactly one line per input document, in input order:
{"doc_id":"<id>","title":"<title>","facts":[ <fact>, ... ]}
No markdown fences. No commentary. Nothing before or after the JSONL.

DOCUMENTS:
EOH
)
shopt -s nullglob
for f in "${PREFIX}"_batch_*.jsonl; do
  i="${f#${PREFIX}_batch_}"; i="${i%.jsonl}"
  out="${PREFIX}_facts_${i}.jsonl"
  [ -s "$out" ] && continue
  ( { printf '%s\n' "$HEADER"; cat "$f"; } \
      | "$CLAUDE" -p --model "$MODEL" > "$out" 2>".clog_${i}.txt" \
      || { echo "FAILED batch $i"; rm -f "$out"; } ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 3; done
done
wait
echo "ALL_CLAUDE_BATCHES_DONE"
