#!/usr/bin/env bash
# w6r_oss2g1 (4×3B + gptoss:2 + gpt5:1) all-data + pack. 최고 OOF 0.7878 → 제출 후보.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
G=1; LOG=runs/prep_oss2g1.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
T="q3bb q3b43 q3b44 q3b45 qgptoss:2 qgpt5:1"
if [ ! -f build/oss2g1_all.zip ]; then
  say "w6r_oss2g1 all-data 학습 :: $T"
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits $T \
    --student Qwen/Qwen2.5-0.5B --out runs/kd_oss2g1_all --all-data --bf16 --temperature 3 --alpha 0.25 \
    --max-length 640 --serialize base >runs/kd_oss2g1_all.log 2>&1 && \
  { say "pack(fp16)"; uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_oss2g1_all/model \
    --name oss2g1_all --serialize base --quant fp16 >>"$LOG" 2>&1; } || say "실패"
fi
say "=== oss2g1 준비완료: build/oss2g1_all.zip (제출 대기) ==="
