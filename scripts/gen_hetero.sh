#!/usr/bin/env bash
# 비-Qwen 이질 teacher soft-label store 생성 (gemma-9b, GLM-32B) @640, 4bit adapter.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
LOG=runs/gen_hetero.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
# GLM-32B (overlay 4.55, GPU0)
[ -f runs/_teacher_logits/qglm32b.npz ] || { say "qglm32b 생성 (overlay 4.55, 4bit)"; CUDA_VISIBLE_DEVICES=0 \
  uv run --with "transformers==4.55.0" python -m ai_challenge.method.gen_softlabels \
  --teacher-dir runs/tglm32b/model --tag qglm32b --serialize base --max-length 640 --load-in-4bit >>"$LOG" 2>&1 && say "qglm32b 완료" || say "qglm32b 실패"; } &
# gemma-2-9b (main env, GPU2)
[ -f runs/_teacher_logits/qgemma9b.npz ] || { say "qgemma9b 생성 (main env, 4bit)"; CUDA_VISIBLE_DEVICES=2 \
  uv run python -m ai_challenge.method.gen_softlabels \
  --teacher-dir runs/tgemma9b/model --tag qgemma9b --serialize base --max-length 640 --load-in-4bit >>"$LOG" 2>&1 && say "qgemma9b 완료" || say "qgemma9b 실패"; } &
wait
say "=== gen_hetero 완료 ==="
