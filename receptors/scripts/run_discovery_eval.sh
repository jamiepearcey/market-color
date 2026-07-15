#!/usr/bin/env bash
# Driver-discovery eval — end-to-end. Implements the corrected experiment from
# RETRIEVAL_FINDINGS.md / EVAL_DISCOVERY_PROTOCOL.md: a head-to-head on the
# HYPOTHESIS SOURCE (A/B/C/D), scored on NOVEL LOAD-BEARING DRIVERS vs baseline,
# with the validated cos+ndir reranker layered on each arm.
#
# Run from the receptors/ dir. Requires the built substrate under $RECEPTORS_DATA_DIR
# (docs.jsonl, embeddings.npy, chunks.*, doc_meta.json) and data/eval/queries.json.
#
#   COMPOSE=template ./scripts/run_discovery_eval.sh     # no-LLM floor, fully auto
#   COMPOSE=llm      ./scripts/run_discovery_eval.sh     # assembles LLM prompts, pauses
#
# In llm mode the script stops after emitting data/eval/hypgen_<ARM>.jsonl; run your
# LLM over each row's "prompt", collect {qid,hypotheses} replies into
# data/eval/hypgen_<ARM>_out.jsonl, ingest, then re-run with STAGE=candidates.
set -euo pipefail
cd "$(dirname "$0")/.."

COMPOSE="${COMPOSE:-template}"
ARMS="${ARMS:-A,B,C,D}"
RERANK="${RERANK:-ndir}"      # ndir | none
K="${K:-10}"
STAGE="${STAGE:-all}"
run() { echo "+ $*"; uv run "$@"; }

if [[ "$STAGE" == "all" || "$STAGE" == "hypotheses" ]]; then
  echo "== [1] hypotheses (compose=$COMPOSE) =="
  run scripts/gen_hypotheses.py --compose "$COMPOSE" --arms "$ARMS"
  if [[ "$COMPOSE" == "llm" ]]; then
    echo
    echo ">> LLM prompts written to data/eval/hypgen_<ARM>.jsonl."
    echo ">> Run your LLM per row, save replies to data/eval/hypgen_<ARM>_out.jsonl, then:"
    for arm in ${ARMS//,/ }; do
      echo "     uv run scripts/gen_hypotheses.py ingest --arm $arm --replies data/eval/hypgen_${arm}_out.jsonl"
    done
    echo ">> Then re-run:  STAGE=candidates ./scripts/run_discovery_eval.sh"
    [[ "$STAGE" == "all" ]] && exit 0
  fi
fi

if [[ "$STAGE" == "all" || "$STAGE" == "candidates" ]]; then
  echo "== [2] candidates (baseline + arms, rerank=$RERANK) =="
  # baseline = cosine on the effect phrase alone (the novelty reference)
  run scripts/gen_hyp_candidates.py --arm baseline --hyps effect --k "$K"
  for arm in ${ARMS//,/ }; do
    run scripts/gen_hyp_candidates.py --arm "$arm" --hyps "hyps_${arm}" --k "$K" --rerank "$RERANK"
  done

  echo "== [3] novelty-aware judge packets =="
  run scripts/build_novelty_packets.py --baseline baseline --arms "$ARMS"
  echo
  echo ">> Blind content-review packets: data/eval/packet_disc_<qid>.json"
  echo ">> Grade each candidate 0/1/2 per EVAL_DISCOVERY_PROTOCOL.md into"
  echo "     data/eval/rating_disc_<qid>.json = {\"D01\": 2, \"D02\": 0, ...}"
  echo ">> Then:  STAGE=score ./scripts/run_discovery_eval.sh"
  [[ "$STAGE" == "all" ]] && exit 0
fi

if [[ "$STAGE" == "score" ]]; then
  echo "== [4] score (discovery@K, ref=B) =="
  run scripts/score_discovery.py --arms "$ARMS" --ref B --k "$K"
fi
