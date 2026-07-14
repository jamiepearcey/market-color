#!/bin/bash
# Drive generation/judging prompts in eval/ablation_out/ through claude -p.
# Resume-safe: outputs that already exist are skipped.
#   bash eval/run_ablation.sh gen     # gen_*.txt   -> ans_*.txt
#   bash eval/run_ablation.sh judge   # judge_*.txt -> verdict_*.txt
cd "$(dirname "$0")/${OUT:-ablation_out}" || exit 1
CLAUDE=/opt/homebrew/bin/claude
MODEL=claude-opus-4-8
MAXP=4
mode="$1"
case "$mode" in
  gen)   pre=gen_;   out_pre=ans_ ;;
  judge) pre=judge_; out_pre=verdict_ ;;
  *) echo "usage: $0 gen|judge"; exit 1 ;;
esac
for f in ${pre}*.txt; do
  base="${f#${pre}}"
  out="${out_pre}${base}"
  [ -s "$out" ] && continue
  ( "$CLAUDE" -p --model "$MODEL" < "$f" > "$out" 2>"err_${base}" \
      && rm -f "err_${base}" || echo "FAILED $f" ) &
  while [ "$(jobs -rp | wc -l)" -ge "$MAXP" ]; do sleep 2; done
done
wait
echo "DONE $mode: $(ls ${out_pre}*.txt 2>/dev/null | wc -l | tr -d ' ') outputs"
