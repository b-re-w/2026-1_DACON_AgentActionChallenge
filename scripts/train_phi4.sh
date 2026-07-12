#!/usr/bin/env bash
# ── Phi-4 (14B) 이질 teacher 학습 + soft-label 캐시 ──────────────────────────
# Phi-4 는 합성데이터 사전학습 → Qwen 사다리와 오류 decorrelation 이 크다(아직 없는 계열).
#
# 기존 train_exotic.sh 대비 고친 것:
#   1) --early-stop-patience 1 → 2.  patience 1 은 ep2 에서 한 번만 흔들려도 즉사한다.
#      qmistral24b 가 정확히 이 함정에 빠져 2 epoch 만에 중단 → OOF 0.667(사실상 1 epoch).
#   2) --resume + supervisor 재시작 루프. 이 환경은 장시간 GPU job 을 조용히 SIGKILL 하는데,
#      31h 짜리가 중간에 죽으면 전부 날아간다. epoch 마다 체크포인트가 저장되므로
#      죽으면 마지막 epoch 부터 이어받아 재시작한다.
#
# 실측: 12.8 s/it × 8750 step(5ep, batch16×2) ≈ 31h. ML 은 640 유지(대회 특성상 미세이득도 중요).
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
GPU=${GPU:-0}
TAG=qphi4; MODEL=microsoft/phi-4; ML=640; SER=base
TARGET="qkv_proj,o_proj,gate_up_proj,down_proj"   # Phi3 arch: 융합 qkv/gate_up (검증확정)
# Phi-4 seqcls 는 transformers>=4.47 필요 → 오버레이.
# --extra dev 필수: QLoRA 의 peft/bitsandbytes 가 dev extra 에 있는데, --with 오버레이는
# 기본 extra 를 안 딸려오므로 빼면 ModuleNotFoundError: peft 로 즉사한다.
TF='--extra dev --with transformers==4.57.1'
MAX_RETRY=${MAX_RETRY:-20}

echo "[phi4] $(date '+%m-%d %H:%M') 시작  model=$MODEL  ML=$ML  patience=2  (예상 ~31h)"

for try in $(seq 1 $MAX_RETRY); do
  if [ -f "runs/t_${TAG}/model/config.json" ]; then
    echo "[phi4] 학습 완료 감지 → 다음 단계"; break
  fi
  echo "[phi4] --- 시도 $try/$MAX_RETRY ($(date '+%m-%d %H:%M')) ---"
  CUDA_VISIBLE_DEVICES=$GPU uv run $TF python -m ai_challenge.method.train \
    --model "$MODEL" --out "runs/t_${TAG}" --lora-target-modules "$TARGET" \
    --qlora --serialize $SER --max-length $ML --bf16 --grad-checkpoint --no-class-weight \
    --lr 1.5e-4 --batch-size 16 --grad-accum 2 \
    --epochs 5 --early-stop-patience 2 --resume
  rc=$?
  if [ -f "runs/t_${TAG}/model/config.json" ]; then
    echo "[phi4] 학습 정상 완료 (rc=$rc)"; break
  fi
  echo "[phi4] ⚠️ 죽음(rc=$rc) → 60초 후 마지막 체크포인트에서 재개"
  sleep 60
done

if [ ! -f "runs/t_${TAG}/model/config.json" ]; then
  echo "[phi4] ❌ $MAX_RETRY 회 시도했으나 완료 실패"; exit 1
fi

echo "[phi4] $(date '+%m-%d %H:%M') soft-label 캐시 생성"
CUDA_VISIBLE_DEVICES=$GPU uv run $TF python -m ai_challenge.method.gen_softlabels \
  --teacher-dir "runs/t_${TAG}/model" --tag "$TAG" --serialize $SER --max-length $ML --load-in-4bit
echo "[phi4] ✅ $(date '+%m-%d %H:%M') 완료  runs/_teacher_logits/${TAG}.npz"
echo "[phi4] OOF: $(grep -oE '\"macro_f1\": [0-9.]+' runs/t_${TAG}/metrics.json)"
