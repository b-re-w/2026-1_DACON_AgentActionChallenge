#!/usr/bin/env bash
# 한 설정을 5-fold 로 순차 학습 → runs/<tag>_f{0..4} 생성(각 fold OOF 포함).
# 사용: scripts/run_cv.sh <tag> <model> <preset> [extra train args...]
#   예: scripts/run_cv.sh mdeberta_rich microsoft/mdeberta-v3-base rich --bs 64 --lr 2e-5 --epochs 3
set -euo pipefail
cd "$(dirname "$0")/.."

TAG="$1"; MODEL="$2"; PRESET="$3"; shift 3
EXTRA=("$@")
mkdir -p logs
for f in 0 1 2 3 4; do
  NAME="${TAG}_f${f}"
  echo "===== [$(date +%H:%M:%S)] training ${NAME} ====="
  uv run python -m ai_challenge.method.train \
      --model "$MODEL" --preset "$PRESET" --fold "$f" \
      --name "$NAME" "${EXTRA[@]}" > "logs/${NAME}.log" 2>&1
  grep -E "result|done" "logs/${NAME}.log" | tail -2 || true
done
echo "===== [$(date +%H:%M:%S)] CV done: ${TAG} ====="
uv run python -m ai_challenge.method.analyze_oof --runs runs/${TAG}_f0 runs/${TAG}_f1 runs/${TAG}_f2 runs/${TAG}_f3 runs/${TAG}_f4 | head -30
