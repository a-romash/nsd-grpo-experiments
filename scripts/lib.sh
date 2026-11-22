# Shared helpers for run_all.sh / smoke.sh. Source, do not execute.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PROFILE=${PROFILE:-short}
TRAIN_GPU=${TRAIN_GPU:-0}
VLLM_GPU=${VLLM_GPU:-1}
TAG=${PROFILE//,/+}
LOG_DIR=results/logs/$TAG
mkdir -p "$LOG_DIR"
VLLM_PID=""

cfg() {  # cfg <experiment> <key> [extra args...]
  local exp=$1 key=$2; shift 2
  python -m src.config "configs/$exp.yaml" --profile "$PROFILE" "$@" --get "$key"
}

start_vllm() {  # start_vllm <log file>
  local log=$1
  local model port util len
  model=$(cfg grpo model); port=$(cfg grpo vllm_port)
  util=$(cfg grpo vllm_gpu_memory_utilization); len=$(cfg grpo vllm_max_model_len)
  if curl -sf "http://127.0.0.1:$port/health/" >/dev/null 2>&1; then
    echo "Port $port already serves something; stop it first (pkill -f vllm-serve)"; exit 1
  fi
  echo ">>> starting vLLM on GPU $VLLM_GPU (log: $log)"
  CUDA_VISIBLE_DEVICES=$VLLM_GPU setsid trl vllm-serve --model "$model" --port "$port" \
    --gpu_memory_utilization "$util" --max_model_len "$len" >"$log" 2>&1 &
  VLLM_PID=$!
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$port/health/" >/dev/null 2>&1; then echo ">>> vLLM ready"; return 0; fi
    if ! kill -0 "$VLLM_PID" 2>/dev/null; then echo "vLLM died, see $log"; tail -20 "$log"; exit 1; fi
    sleep 2
  done
  echo "vLLM did not become healthy in 10 min, see $log"; exit 1
}

stop_vllm() {
  if [[ -n "$VLLM_PID" ]] && kill -0 "$VLLM_PID" 2>/dev/null; then
    echo ">>> stopping vLLM"
    kill -TERM -- "-$VLLM_PID" 2>/dev/null || kill -TERM "$VLLM_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 1; done
    kill -KILL -- "-$VLLM_PID" 2>/dev/null || true
  fi
  VLLM_PID=""
}
trap stop_vllm EXIT INT TERM

train_run() {  # train_run <experiment> <seed> [extra args...]; vLLM is restarted so every run starts from base weights
  local exp=$1 seed=$2; shift 2
  local name="${exp}_s${seed}"
  start_vllm "$LOG_DIR/${name}_vllm.log"
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU python -m src.experiment --config "configs/$exp.yaml" \
    --profile "$PROFILE" --seed "$seed" --stage train "$@" 2>&1 | tee -a "$LOG_DIR/${name}_train.log"
  stop_vllm
}

eval_run() {  # eval_run <experiment> <seed> [extra args...]
  local exp=$1 seed=$2; shift 2
  CUDA_VISIBLE_DEVICES=$TRAIN_GPU python -m src.experiment --config "configs/$exp.yaml" \
    --profile "$PROFILE" --seed "$seed" --stage eval "$@" 2>&1 | tee -a "$LOG_DIR/${exp}_s${seed}_eval.log"
}
