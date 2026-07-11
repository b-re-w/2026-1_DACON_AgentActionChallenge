#!/usr/bin/env bash
# 저녁 체인 (사용자 지정 순서):
#  0) 5-fold 완료 대기 → bias70k held-out 결과 출력
#  1) 프로브 A: Coder-0.5B student fold0
#  2) 프로브 B: Gemma3-270M student fold0 (transformers 4.55 오버레이, 새 teacher 블렌드)
#  3) 7B_s42 +1에폭 연장 (lr 3e-6) → export → 반분검증
#  4) gpt-oss-20b LoRA teacher (bf16+LoRA, 4.55 오버레이) → export
# GLM/72B 도착 시 이 체인과 별개로 워처가 알림 (수동 인터럽트 판단).
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

# 0) 5-fold 대기
while pgrep -f "kd_cycle5_folds" >/dev/null; do sleep 120; done
log "5-fold 완료 — bias70k 결과:"
cat runs/bias70k.json 2>/dev/null | python -c "import json,sys;d=json.load(sys.stdin);print(f\"  held-out fold0: {d['macro_f1_before']:.5f}→{d['macro_f1_after']:.5f} (Δ{d['heldout_gain']:+.5f})\")" 2>/dev/null || log "bias70k.json 없음"

# 1) 프로브 A: Coder-0.5B
log "프로브 A: Coder-0.5B student fold0"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-Coder-0.5B --preset base --fold 0 --max-length 640 \
  --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
  --name kd_coder05b_f0 > logs/kd_coder05b_f0.log 2>&1 \
  && grep -E "ckpt|result" logs/kd_coder05b_f0.log | tail -2 \
  || { log "Coder-0.5B 실패"; tail -3 logs/kd_coder05b_f0.log | tr '\r' '\n' | tail -3; }

# 2) 프로브 B: Gemma3-270M (오버레이 env)
log "프로브 B: Gemma3-270M student fold0"
uv run --with "transformers==4.55.0" python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student google/gemma-3-270m --preset base --fold 0 --max-length 640 \
  --epochs 3 --bs 32 --lr 2e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
  --name kd_gemma270m_f0 > logs/kd_gemma270m_f0.log 2>&1 \
  && grep -E "ckpt|result" logs/kd_gemma270m_f0.log | tail -2 \
  || { log "Gemma-270M 실패"; tail -3 logs/kd_gemma270m_f0.log | tr '\r' '\n' | tail -3; }

# 3) 7B +1ep 연장
log "7B_s42 +1에폭 연장 (lr 3e-6)"
uv run python -m ai_challenge.method.train \
  --model runs/qwen7b_base_f0/model --preset base --fold 0 --bs 4 --grad-accum 8 \
  --lr 3e-6 --epochs 1 --max-length 512 --grad-checkpoint --optim paged_adamw_8bit \
  --name qwen7b_ext3ep_f0 > logs/qwen7b_ext3ep_f0.log 2>&1 \
  && grep -E "result" logs/qwen7b_ext3ep_f0.log | tail -1 \
  || { log "7B 연장 실패"; tail -3 logs/qwen7b_ext3ep_f0.log | tr '\r' '\n' | tail -3; }
if [ -f runs/qwen7b_ext3ep_f0/metrics.json ]; then
  uv run python -m ai_challenge.method.export_teacher --runs runs/qwen7b_ext3ep_f0 \
    --note "7B s42 3ep 연장판" > logs/export_7b_ext.log 2>&1 || log "export 실패"
fi

# 4) gpt-oss-20b LoRA teacher (bf16 LoRA, 병합 저장)
log "gpt-oss-20b LoRA teacher 시작"
uv run --with "transformers==4.55.0" python -m ai_challenge.method.train \
  --model openai/gpt-oss-20b --preset base --fold 0 --bs 4 --grad-accum 8 \
  --lr 1e-4 --epochs 2 --max-length 512 --grad-checkpoint --lora --lora-r 16 \
  --name gptoss20b_f0 > logs/gptoss20b_f0.log 2>&1 \
  && grep -E "result" logs/gptoss20b_f0.log | tail -1 \
  || { log "gpt-oss 실패"; tail -5 logs/gptoss20b_f0.log | tr '\r' '\n' | tail -5; }
if [ -f runs/gptoss20b_f0/metrics.json ]; then
  uv run --with "transformers==4.55.0" python -m ai_challenge.method.export_teacher \
    --runs runs/gptoss20b_f0 --note "gpt-oss-20b LoRA 이질 teacher" > logs/export_gptoss.log 2>&1 || log "export 실패"
fi
log "저녁 체인 종료"
