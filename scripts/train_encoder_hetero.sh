#!/usr/bin/env bash
# ── 이질 encoder teacher (DeBERTa-v3-large / ModernBERT-large) 학습 + soft-label 캐시 ──
# 기존 hetero teacher(Qwen 사다리·Mistral·Gemma…)는 전부 decoder LLM(QLoRA)이라, 양방향
# encoder + 다른 tokenizer 계열을 넣으면 아키텍처 다양성(오류 decorrelation)이 커진다.
# encoder 는 full fine-tune(QLoRA 불필요, 435M/395M → A100 여유). soft-label store 는
# 레코드 id 기준이라 tokenizer 가 달라도 앙상블 호환.
#
# 사용:  scripts/train_encoder_hetero.sh {deberta|modernbert}
#   결과:  runs/_teacher_logits/{qdebertaL|qmodernbertL}.npz  → 앙상블(distill)에 합류
set -euo pipefail
cd "$(dirname "$0")/.."

SER=base   # 기존 teacher store(qmistral24b 등)와 반드시 일치 → 앙상블 입력 신호 정합.
# encoder full-ft 안정 레시피. no-class-weight: soft-label 확률 왜곡 방지(teacher 는 calibrated 라벨).
# grad-checkpoint 필수: DeBERTa-v3 disentangled attention 은 activation 메모리가 커서
# 끄면 batch64 에 80GB OOM. grad-ckpt 로 눌러 batch 를 키우는 게 정답.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 단편화 완화
COMMON="--serialize $SER --bf16 --grad-checkpoint --no-class-weight \
        --epochs 5 --early-stop-patience 1 --warmup-ratio 0.1"

case "${1:-}" in
  deberta)    # 435M, disentangled attention → Qwen decoder 와 강한 이질. 현재 env(4.46.3)로 즉시.
    MODEL=microsoft/deberta-v3-large; TAG=qdebertaL
    ML=512                     # DeBERTa-v3 native 길이(512 학습). 안정.
    EXTRA="--batch-size 64 --lr 1.2e-5"   # grad-ckpt 로 memory 안전, batch64 로 step 절반
    TF="" ;;
  modernbert) # 395M, 최신 arch + RoPE + 8k context. transformers>=4.48 필요 → 오버레이.
    # lr5e-5: ModernBERT 는 higher-LR 설계 — lr2e-5(1차)는 0.598 undertrain 의 원인으로 확정.
    MODEL=answerdotai/ModernBERT-large; TAG=qmodernbertL2
    ML=640
    EXTRA="--batch-size 48 --lr 5e-5 --epochs 8 --early-stop-patience 2"
    TF="--extra dev --with transformers==4.57.1" ;;
  *) echo "사용법: $0 {deberta|modernbert}"; exit 1 ;;
esac

GPU=${GPU:-0}
echo "[enc-hetero] GPU$GPU  $MODEL  → tag=$TAG  max_length=$ML"
CUDA_VISIBLE_DEVICES=$GPU uv run $TF python -m ai_challenge.method.train $COMMON $EXTRA \
  --model "$MODEL" --max-length $ML --out "runs/t_${TAG}"
echo "[enc-hetero] 학습 완료 → soft-label 캐시(serialize=$SER)"
CUDA_VISIBLE_DEVICES=$GPU uv run $TF python -m ai_challenge.method.gen_softlabels \
  --teacher-dir "runs/t_${TAG}/model" --tag "$TAG" --serialize $SER --max-length $ML
echo "[enc-hetero] 완료 ✅  runs/_teacher_logits/${TAG}.npz"
