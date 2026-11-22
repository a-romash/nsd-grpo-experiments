#!/usr/bin/env bash
# Run the existing smoke test and record whole-server resource usage on Linux.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
log_dir=$(mktemp -d "results/logs/resources-$(date +%Y%m%d-%H%M%S)-XXXXXX")
echo "Resource logs: $log_dir"
gpu_fields=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw
nvidia-smi --query-gpu="$gpu_fields" --format=csv > "$log_dir/gpu.csv"
nvidia-smi --query-gpu="$gpu_fields" --format=csv,noheader -l 2 >> "$log_dir/gpu.csv" &
gpu_pid=$!
host_pid=""
cleanup() {
  kill "$gpu_pid" 2>/dev/null || true
  if [[ -n "$host_pid" ]]; then kill "$host_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
vmstat -w -t 2 > "$log_dir/host.log" &
host_pid=$!
df -h . > "$log_dir/disk-before.txt"
bash scripts/smoke.sh 2>&1 | tee "$log_dir/smoke.log"
