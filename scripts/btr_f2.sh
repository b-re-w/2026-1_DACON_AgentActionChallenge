#!/usr/bin/env bash
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
until grep -q "weights_lab 완료" runs/weights_lab.log 2>/dev/null; do sleep 180; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)" -lt 2000 ]; do sleep 60; done
CUDA_VISIBLE_DEVICES=3 uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 \
  --temperature 3 --alpha 0.25 --max-length 640 --serialize btrailr --teacher-logits qgptoss qw6 q7bteam \
  --val-fold 2 --out runs/kd_btr_f2 > runs/kd_btr_f2.log 2>&1
echo "[btr_f2] OOF=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_btr_f2/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+') (기준 f2=0.79367)" >> runs/weights_lab.log
