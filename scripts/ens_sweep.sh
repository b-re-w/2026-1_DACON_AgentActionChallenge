#!/usr/bin/env bash
# 앙상블 subset 증류 스윕 — 캐시된 teacher store 조합으로 0.5B 증류(teacher forward 0).
# 인자: <GPU> "name=tag1,tag2,..." ...   (tags 는 쉼표구분; 공백 회피)
# 예: scripts/ens_sweep.sh 1 "ladder=q3b,q14b,q32b" "w6_32b=qw6,q32b"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf
GPU=$1; shift
ML=640; STU=Qwen/Qwen2.5-0.5B; LOG=runs/ens_sweep.log
for spec in "$@"; do
  name=${spec%%=*}; tags=${spec#*=}
  out=runs/kd_ens_$name
  [ -f "$out/metrics.json" ] && { echo "[$name] 이미 완료 — 스킵" | tee -a "$LOG"; continue; }
  echo "[$(date '+%m-%d %H:%M')] [$name] teachers=$tags (GPU$GPU)" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.distill \
    --teacher-logits ${tags//,/ } --student $STU --out "$out" --bf16 \
    --temperature 3 --alpha 0.25 --max-length $ML > "$out.log" 2>&1 \
    || { echo "[$name] 실패 — $out.log" | tee -a "$LOG"; continue; }
  echo "[$name] OOF=$(grep -oE '"macro_f1": [0-9.]+' $out/metrics.json) teachers=$tags" | tee -a "$LOG"
done
echo "[ens_sweep] GPU$GPU 배치 완료" | tee -a "$LOG"
