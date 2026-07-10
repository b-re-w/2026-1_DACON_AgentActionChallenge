#!/usr/bin/env bash
# 유연한 증류 스윕 러너. 인자: <GPU> "name::<distill 추가플래그...>" ...
# 공통: --student 0.5B --bf16 --max-length 640 --serialize base --val-fold 0. 캐시(teacher forward 0).
# 예: scripts/kd_run.sh 1 "qw6_t4a15::--teacher-logits qw6 --temperature 4 --alpha 0.15"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf
GPU=$1; shift
STU=Qwen/Qwen2.5-0.5B; ML=640; LOG=runs/kd_run.log
for spec in "$@"; do
  name=${spec%%::*}; extra=${spec#*::}
  out=runs/kd_$name
  [ -f "$out/metrics.json" ] && { echo "[$name] 완료 스킵" | tee -a "$LOG"; continue; }
  echo "[$(date '+%m-%d %H:%M')] [$name] GPU$GPU :: $extra" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.distill \
    --student $STU --out "$out" --bf16 --max-length $ML --serialize base $extra \
    > "$out.log" 2>&1 || { echo "[$name] 실패 — $out.log" | tee -a "$LOG"; continue; }
  echo "[$name] OOF=$(grep -oE '"macro_f1": [0-9]+\.[0-9]+' $out/metrics.json | grep -oE '[0-9]+\.[0-9]+')" | tee -a "$LOG"
done
echo "[kd_run] GPU$GPU 배치 완료" | tee -a "$LOG"
