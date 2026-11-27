#!/usr/bin/env bash
# Full pipeline; safe to re-run after a preemption (finished stages are skipped,
# unfinished training resumes from its last checkpoint).
#
#   PROFILE=short SEEDS="1" scripts/run_all.sh
#   PROFILE=full  SEEDS="1 2 3" EXPERIMENTS="grpo grpo_nsd_zero" scripts/run_all.sh
source "$(dirname "$0")/lib.sh"
SEEDS=${SEEDS:-1}
EXPERIMENTS=${EXPERIMENTS:-"grpo grpo_nsd_all grpo_nsd_failed grpo_nsd_zero"}

if [[ ! -f "$(cfg baseline run_dir)/eval.json" ]]; then
  eval_run baseline 0
fi
for seed in $SEEDS; do
  for exp in $EXPERIMENTS; do
    run_dir=$(cfg "$exp" run_dir --seed "$seed")
    if [[ ! -f "$run_dir/train_metrics.json" ]]; then
      train_run "$exp" "$seed"
    fi
    if [[ ! -f "$run_dir/eval.json" ]]; then
      eval_run "$exp" "$seed"
    fi
  done
done
python -m src.plots --profile "$PROFILE"
echo ">>> done: results/runs/$TAG/summary.md"
