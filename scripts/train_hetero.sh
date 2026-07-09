#!/usr/bin/env bash
# ── 다른 A100(80GB) 서버 전용 ── 비-Qwen 이질 teacher 학습 + soft-label 캐시.
# 이질 앙상블 재료를 만든다. 결과 .npz 를 메인 서버로 scp 하면 앙상블에 합류.
#
# 사용:
#   export HF_HOME=/big_disk/hf                 # 큰 디스크 (70B 는 ~140GB)
#   export HF_TOKEN=hf_xxx                       # Llama/Gemma gated → 필요
#   scripts/train_hetero.sh mistral             # 또는 gemma / llama
#   # 끝나면:  scp runs/_teacher_logits/<tag>.{npz,json}  main:/data/agent_action/runs/_teacher_logits/
#
# 전제: uv sync --extra dev 완료, data/{train.jsonl,train_labels.csv,folds.csv} 를
#       메인 서버에서 복사(레코드 id·fold 일치가 앙상블 호환의 핵심).
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_HOME:?HF_HOME 를 큰 디스크로 export 하세요 (예: export HF_HOME=/big_disk/hf)}"

ML=640; SER=base
# 메인 Qwen 사다리와 동일 레시피(QLoRA·base·640·eff32·lr1.5e-4·early-stop) → 표현 일관.
COMMON="--qlora --serialize $SER --max-length $ML --bf16 --grad-checkpoint --no-class-weight \
        --lr 1.5e-4 --epochs 5 --early-stop-patience 1"

case "${1:-}" in
  mistral)  # open, 마찰 없음
    MODEL=mistralai/Mistral-Small-24B-Instruct-2501; TAG=qmistral24b
    EXTRA="--batch-size 16 --grad-accum 2" ;;
  gemma)    # gated. Gemma-2 는 soft-capping → eager attention 필수.
    MODEL=google/gemma-2-27b-it; TAG=qgemma27b
    EXTRA="--batch-size 16 --grad-accum 2 --attn-implementation eager" ;;
  llama)    # gated, 70B → batch 작게(우리 72B 와 동일)
    MODEL=meta-llama/Llama-3.3-70B-Instruct; TAG=qllama70b
    EXTRA="--batch-size 4 --grad-accum 8" ;;
  *) echo "사용법: $0 {mistral|gemma|llama}"; exit 1 ;;
esac
# 이 3개는 target_modules 기본값(q/k/v/o/gate/up/down_proj)이 그대로 맞음 → 지정 불필요.

GPU=${GPU:-0}
echo "[hetero] GPU$GPU  $MODEL  → tag=$TAG"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.train $COMMON $EXTRA \
  --model "$MODEL" --out "runs/t_${TAG}"
echo "[hetero] 학습 완료 → soft-label 캐시"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.gen_softlabels \
  --teacher-dir "runs/t_${TAG}/model" --tag "$TAG" --serialize $SER --max-length $ML --load-in-4bit
echo "[hetero] 완료 ✅  runs/_teacher_logits/${TAG}.npz"
echo "         메인 서버로:  scp runs/_teacher_logits/${TAG}.{npz,json} main:/data/agent_action/runs/_teacher_logits/"
