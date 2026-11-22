#!/usr/bin/env bash
# Smoke test: same batch shapes as the target profile, 10 steps, with a simulated
# preemption at step 7 (checkpoint at step 5) to check that resume works.
#
#   scripts/smoke.sh                 # checks the `short` profile
#   BASE_PROFILE=full scripts/smoke.sh
BASE_PROFILE=${BASE_PROFILE:-short}
PROFILE="$BASE_PROFILE,smoke"
source "$(dirname "$0")/lib.sh"
EXP=${EXP:-grpo_nsd_all}   # most expensive mode -> upper bound on step time
SEED=1
run_dir=$(cfg "$EXP" run_dir --seed "$SEED")
if [[ -d "$run_dir" ]]; then
  echo "Removing previous smoke run $run_dir"; rm -rf "$run_dir"
fi
echo "=== phase 1: train until step 7, then 'preempt' ==="
train_run "$EXP" "$SEED" --set stop_after_steps=7
echo "=== phase 2: rerun the same command, must resume from checkpoint-5 ==="
train_run "$EXP" "$SEED"
echo "=== eval (small) ==="
eval_run "$EXP" "$SEED"
python -m src.plots --smoke "$run_dir"
