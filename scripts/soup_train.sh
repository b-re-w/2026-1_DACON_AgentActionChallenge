#!/usr/bin/env bash
# $1=GPU $2=seed. qw6 all-data 학습(model soup 재료).
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
CUDA_VISIBLE_DEVICES=$1 uv run python -m ai_challenge.method.distill \
  --teacher-logits qw6 --student Qwen/Qwen2.5-0.5B --all-data --bf16 \
  --temperature 3 --alpha 0.25 --max-length 640 --serialize base \
  --out runs/kd_qw6soup_s$2 --seed $2 > runs/kd_qw6soup_s$2.log 2>&1
