#!/usr/bin/env bash
# DeBERTa-v3-large 개별 성능 향상 재학습.
# 근거: 기존 5ep/lr1.2e-5 는 ep5(0.727)까지 단조 상승 = 수렴 부족(early-stop 미발동).
#   → epochs 8 + patience 2(플래토 자동중단), lr 2e-5(수렴 가속; 발산 시 best 복원).
# ML 은 512 유지 — 프로젝트 측정상 640 은 전체 무이동(0.7802 vs 0.7801)인데 25% 느림.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=${GPU:-0}; TAG=qdebertaL2; ML=512; SER=base
echo "[dbl] $(date '+%H:%M') 시작 → tag=$TAG (ep8, lr2e-5, patience2)"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.train \
  --model microsoft/deberta-v3-large --out "runs/t_${TAG}" \
  --serialize $SER --max-length $ML --bf16 --grad-checkpoint --no-class-weight \
  --batch-size 64 --lr 2e-5 --warmup-ratio 0.1 \
  --epochs 8 --early-stop-patience 2
echo "[dbl] $(date '+%H:%M') 학습완료 → soft-label 캐시"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.gen_softlabels \
  --teacher-dir "runs/t_${TAG}/model" --tag "$TAG" --serialize $SER --max-length $ML
echo "[dbl] $(date '+%H:%M') 완료 ✅  OOF: $(grep -oE '\"macro_f1\": [0-9.]+' runs/t_${TAG}/metrics.json)"
