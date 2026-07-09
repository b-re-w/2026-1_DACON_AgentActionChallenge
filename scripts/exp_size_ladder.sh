#!/usr/bin/env bash
# =============================================================================
# 크기 사다리 실험 (Qwen 순수) — teacher 크기 vs 성능 / 증류 uplift / 앙상블
#
# 슬레이트:  GPU1=Qwen2.5-14B  GPU2=Qwen2.5-32B  GPU3=Qwen2.5-72B  (+ 기존 3B 재사용)
#            (GPU0 은 다른 작업 사용중 → 이 실험은 GPU1~3 만 사용)
# 방식:      QLoRA(4bit nf4 + LoRA, score head full-train) 단일 GPU 학습
# 산출:      teacher OOF(fold0) → soft-label 캐시 → 0.5B student 증류 → 부분집합 앙상블
#
# 주의: 통째로 실행하지 말고 Phase 단위로 돌릴 것(특히 72B 는 2ep ~10-20h).
#       각 Phase 는 앞 Phase 결과에 의존한다.
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# HF 캐시를 /data 로 고정 — 루트('/')는 60GB 뿐이라 32B(64GB)/72B(145GB) 다운로드가
# 터진다. /data(5.3TB)의 기존 프로젝트 캐시(14B 등 이미 존재)를 재사용.
export HF_HOME=/data/hf

RUN() { echo "+ $*"; uv run python -m "$@"; }
SER=base                 # teacher 입력: base 고정(rich 는 teacher 성능 하락, 실험기록 확인됨)
STU=Qwen/Qwen2.5-0.5B
ML=640                   # 입력 최대 길이. 512 초과 1.7%(p100=608)를 완전히 담아 잘림 0%.
                         # train/cache/distill 전부 동일 ML 이어야 정합(teacher↔student 길이 일치).
COMMON="--qlora --serialize ${SER} --bf16 --grad-checkpoint --no-class-weight --lr 1.5e-4 --epochs 5 --early-stop-patience 1 --max-length ${ML}"
# QLoRA lr 은 어댑터만 학습하므로 full-FT(1e-5)보다 높게. VRAM 여유가 커(14B=20GB,32B=27GB
# /80GB) eff batch 를 16→32 로 키우고 lr 을 1e-4→1.5e-4 로 함께 올림(배치↑ 시 lr 비례↑).
# epochs 5 는 상한 — 매 epoch val macro_f1 검증 후 1회 개선 없으면 조기중단 + best epoch 복원.
# (LoRA 는 epoch별 체크포인트가 어댑터+score head 뿐이라 조기중단 비용이 거의 없음)

# -----------------------------------------------------------------------------
# Phase 1 — teacher QLoRA 병렬 학습 (GPU0~3 동시). 각 metrics.json 에 OOF 기록.
#   크기 사다리 3B/14B/32B/72B 전부 동일 방식(QLoRA·base·640·eff batch32)로 학습해
#   "크기 vs 성능" 을 방법 일관되게 비교한다. (기존 t3b_base 는 512 full-FT 별도 baseline)
# -----------------------------------------------------------------------------
phase1_train() {
  # eff batch = 32 (batch×accum). VRAM 여유로 per-device 배치를 키움.
  CUDA_VISIBLE_DEVICES=0 uv run python -m ai_challenge.method.train $COMMON \
    --model Qwen/Qwen2.5-3B  --batch-size 32 --grad-accum 1 \
    --out runs/t3b_qlora   > runs/t3b_qlora.log 2>&1 &
  CUDA_VISIBLE_DEVICES=1 uv run python -m ai_challenge.method.train $COMMON \
    --model Qwen/Qwen2.5-14B --batch-size 16 --grad-accum 2 \
    --out runs/t14b_qlora  > runs/t14b_qlora.log 2>&1 &
  CUDA_VISIBLE_DEVICES=2 uv run python -m ai_challenge.method.train $COMMON \
    --model Qwen/Qwen2.5-32B --batch-size 16 --grad-accum 2 \
    --out runs/t32b_qlora  > runs/t32b_qlora.log 2>&1 &
  CUDA_VISIBLE_DEVICES=3 uv run python -m ai_challenge.method.train $COMMON \
    --model Qwen/Qwen2.5-72B --batch-size 4  --grad-accum 8 \
    --out runs/t72b_qlora  > runs/t72b_qlora.log 2>&1 &
  wait
  echo "[phase1] teacher OOF:"; grep -h macro_f1 runs/t{3b,14,32,72}b_qlora/metrics.json
}

# -----------------------------------------------------------------------------
# Phase 2 — soft-label 캐시 (teacher당 1회, 전체 70k). 이후 증류는 재추론 0회.
#   3B: 기존 fp16 모델(어댑터 아님) → --load-in-4bit 없이.  QLoRA: --load-in-4bit.
# -----------------------------------------------------------------------------
phase2_cache() {
  CUDA_VISIBLE_DEVICES=1 RUN ai_challenge.method.gen_softlabels --teacher-dir runs/t3b_qlora/model --tag q3b  --serialize $SER --max-length $ML --load-in-4bit
  CUDA_VISIBLE_DEVICES=1 RUN ai_challenge.method.gen_softlabels --teacher-dir runs/t14b_qlora/model --tag q14b --serialize $SER --max-length $ML --load-in-4bit
  CUDA_VISIBLE_DEVICES=2 RUN ai_challenge.method.gen_softlabels --teacher-dir runs/t32b_qlora/model --tag q32b --serialize $SER --max-length $ML --load-in-4bit
  CUDA_VISIBLE_DEVICES=3 RUN ai_challenge.method.gen_softlabels --teacher-dir runs/t72b_qlora/model --tag q72b --serialize $SER --max-length $ML --load-in-4bit
  ls -la runs/_teacher_logits/
}

# -----------------------------------------------------------------------------
# Phase 3a — 목표1·2: 크기별 단일 teacher 증류 uplift (student OOF). T3 α0.25(기존 최적).
# -----------------------------------------------------------------------------
phase3_single() {
  for tag in q3b q14b q32b q72b; do
    CUDA_VISIBLE_DEVICES=3 RUN ai_challenge.method.distill --teacher-logits $tag \
      --student $STU --out runs/kd_${tag} --bf16 --temperature 3 --alpha 0.25 --max-length $ML
  done
  echo "[phase3a] 크기별 student OOF:"; grep -h macro_f1 runs/kd_q*/metrics.json
}

# -----------------------------------------------------------------------------
# Phase 3b — 목표3: 이질(여기선 크기-이질) teacher 앙상블. 부분집합만 바꿔 재추론 0회.
# -----------------------------------------------------------------------------
phase3_ensemble() {
  CUDA_VISIBLE_DEVICES=3 RUN ai_challenge.method.distill --teacher-logits q3b q72b \
    --student $STU --out runs/kd_ens_3b72b --bf16 --temperature 3 --alpha 0.25 --max-length $ML
  CUDA_VISIBLE_DEVICES=3 RUN ai_challenge.method.distill --teacher-logits q3b q14b q32b q72b \
    --student $STU --out runs/kd_ens_all --bf16 --temperature 3 --alpha 0.25 --max-length $ML
  echo "[phase3b] 앙상블 student OOF:"; grep -h macro_f1 runs/kd_ens_*/metrics.json
}

case "${1:-help}" in
  train)     phase1_train ;;
  cache)     phase2_cache ;;
  single)    phase3_single ;;
  ensemble)  phase3_ensemble ;;
  *) echo "사용법: $0 {train|cache|single|ensemble}  (Phase 순서대로 실행)";
     echo "  train    — 14B/32B/72B QLoRA 병렬 학습 (GPU1/2/3, 72B는 장시간)";
     echo "  cache    — teacher soft-label 캐시 (3B 포함)";
     echo "  single   — 크기별 단일 teacher 증류 uplift";
     echo "  ensemble — 크기-이질 teacher 앙상블 측정" ;;
esac
