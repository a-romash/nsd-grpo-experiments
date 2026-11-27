#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
methods=(baseline grpo grpo_nsd_all grpo_nsd_failed grpo_nsd_zero)

usage() {
  echo 'Использование: ./run.sh [номер или название эксперимента ...]'
  echo 'Без аргументов — выбор из меню. По умолчанию PROFILE=full, SEEDS=1.'
  local i
  for i in "${!methods[@]}"; do
    printf '  %s) %s\n' "$((i + 1))" "${methods[i]}"
  done
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then
  usage
  exit 0
fi
choices=("$@")
if (( $# == 0 )); then
  usage
  printf 'Что запустить? Введи номера через пробел: '
  read -r -a choices
fi
if (( ${#choices[@]} == 0 )); then
  echo 'Эксперимент не выбран.' >&2
  exit 2
fi
selected=()
for choice in "${choices[@]}"; do
  case "$choice" in
    [1-5]) selected+=("${methods[choice-1]}") ;;
    baseline|grpo|grpo_nsd_all|grpo_nsd_failed|grpo_nsd_zero) selected+=("$choice") ;;
    *) echo "Неизвестный эксперимент: $choice" >&2; exit 2 ;;
  esac
done
export PROFILE=${PROFILE:-full}
echo "PROFILE=$PROFILE SEEDS=${SEEDS:-1}: ${selected[*]}"
source scripts/lib.sh
SEEDS=${SEEDS:-1}
for seed in $SEEDS; do
  for exp in "${selected[@]}"; do
    if [[ "$exp" == baseline ]]; then
      if [[ ! -f "$(cfg baseline run_dir)/eval.json" ]]; then
        eval_run baseline 0
      fi
      continue
    fi
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
