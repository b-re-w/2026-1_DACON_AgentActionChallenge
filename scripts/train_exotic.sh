#!/usr/bin/env bash
# ── 다른 A100(80GB) 서버 전용 ── Phi-4 / Jamba 이질 teacher 학습 + soft-label 캐시.
# 아키텍처가 다른(합성데이터 Phi, Mamba-하이브리드 Jamba) 이질 재료로 앙상블 다양성 확보.
# 결과 .npz 를 메인 서버 runs/_teacher_logits/ 로 scp → 앙상블 합류.
#
# 사전 준비:
#   export HF_HOME=/big_disk/hf
#   export HF_TOKEN=hf_xxx                 # Phi-4는 open(MIT), Jamba는 open(Jamba license)
#   uv pip install -U "transformers>=4.55"  # Phi-4/Jamba seqcls 지원 (메인 4.46.3과 분리)
#   # Jamba 만: 반드시 커널 설치(없으면 naive scan = 3-5배 느림)
#   uv pip install mamba-ssm causal-conv1d
#   scp main:/data/agent_action/data/{train.jsonl,train_labels.csv,folds.csv} data/
#
# 사용:  scripts/train_exotic.sh {phi4|phi4mini|jamba}
set -euo pipefail
cd "$(dirname "$0")/.."
: "${HF_HOME:?HF_HOME 를 큰 디스크로 export 하세요}"

ML=640; SER=base
COMMON="--qlora --serialize $SER --max-length $ML --bf16 --grad-checkpoint --no-class-weight \
        --lr 1.5e-4 --epochs 5 --early-stop-patience 1"

case "${1:-}" in
  phi4)      # 14B(Phi3 arch), 합성데이터 분포 = Qwen과 강한 decorrelation. ~10-15h.
    MODEL=microsoft/phi-4; TAG=qphi4
    TARGET="qkv_proj,o_proj,gate_up_proj,down_proj"   # Phi3: 융합 qkv/gate_up (검증확정)
    EXTRA="--batch-size 16 --grad-accum 2" ;;
  phi4mini)  # 3.8B(Phi3 arch), 빠름 ~4-7h
    MODEL=microsoft/Phi-4-mini-instruct; TAG=qphi4mini
    TARGET="qkv_proj,o_proj,gate_up_proj,down_proj"   # Phi3: 융합 qkv/gate_up (검증확정)
    EXTRA="--batch-size 32 --grad-accum 1" ;;
  jamba)     # 52B MoE + Mamba 하이브리드 = 아키텍처 최대 이질. 커널 필수. ~15-25h(커널) / 40h+(없이)
    MODEL=ai21labs/AI21-Jamba-1.5-Mini; TAG=qjamba
    # Mamba proj 포함, router(MoE 라우팅)·score 제외
    TARGET="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj,out_proj,x_proj,dt_proj"
    EXTRA="--batch-size 4 --grad-accum 8" ;;
  *) echo "사용법: $0 {phi4|phi4mini|jamba}"; exit 1 ;;
esac

GPU=${GPU:-0}
# Phi-4(≥4.47)·Jamba(seqcls 최신) → transformers 오버레이. 메인 4.46.3과 분리(이 서버는 npz만 산출).
TF='--with transformers==4.57.1'
echo "[exotic] GPU$GPU  $MODEL  → tag=$TAG  target=$TARGET"
CUDA_VISIBLE_DEVICES=$GPU uv run $TF python -m ai_challenge.method.train $COMMON $EXTRA \
  --model "$MODEL" --lora-target-modules "$TARGET" --out "runs/t_${TAG}"
echo "[exotic] 학습 완료 → soft-label 캐시(640, student 일관)"
CUDA_VISIBLE_DEVICES=$GPU uv run $TF python -m ai_challenge.method.gen_softlabels \
  --teacher-dir "runs/t_${TAG}/model" --tag "$TAG" --serialize $SER --max-length $ML --load-in-4bit
echo "[exotic] 완료 ✅  runs/_teacher_logits/${TAG}.npz"
echo "         메인으로:  scp runs/_teacher_logits/${TAG}.{npz,json} main:/data/agent_action/runs/_teacher_logits/"
