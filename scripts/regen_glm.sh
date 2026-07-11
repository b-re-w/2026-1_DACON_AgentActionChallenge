#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/regen_glm.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
say "GLM-32B store 재생성 (batch4, expandable, GPU0)"
CUDA_VISIBLE_DEVICES=0 uv run --with "transformers==4.55.0" python -m ai_challenge.method.gen_softlabels \
  --teacher-dir runs/tglm32b/model --tag qglm32b --serialize base --max-length 640 --load-in-4bit --batch-size 4 >>"$LOG" 2>&1 \
  && say "qglm32b 완료" || say "qglm32b 재실패"
