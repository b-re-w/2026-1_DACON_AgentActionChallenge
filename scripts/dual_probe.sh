#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f "kd_evening2.sh" >/dev/null; do sleep 180; done
echo "[$(date +%H:%M:%S)] 듀얼헤드 fold0 시작"
uv run python -m ai_challenge.method.distill_dual \
  --teachers team_qgptoss team_qw6 qwen7b_base_f0__base --weights 1 1 1 \
  --fold 0 --epochs 2.86 --name kdd_f0 > logs/kdd_f0.log 2>&1 \
  && grep -E "result" logs/kdd_f0.log | tail -1 \
  || { echo "실패"; tail -4 logs/kdd_f0.log | tr '\r' '\n' | tail -4; }
echo "[$(date +%H:%M:%S)] 듀얼헤드 프로브 종료"
