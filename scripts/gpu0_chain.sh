#!/usr/bin/env bash
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640"
[ -f runs/kd_3bt_f4/metrics.json ] || CUDA_VISIBLE_DEVICES=0 $D --serialize btrail --teacher-logits qgptoss qw6 q7bteam --val-fold 4 --out runs/kd_3bt_f4 > runs/kd_3bt_f4.log 2>&1
[ -f runs/kd_btm_f2/metrics.json ] || CUDA_VISIBLE_DEVICES=0 $D --serialize btrim --teacher-logits qgptoss qw6 q7bteam --val-fold 2 --out runs/kd_btm_f2 > runs/kd_btm_f2.log 2>&1
[ -f runs/kd_lw_all15/metrics.json ] || CUDA_VISIBLE_DEVICES=0 $D --serialize btrail --teacher-logits qgptoss qw6 q7bteam qgemma9b:0.15 qglm32b:0.15 q72b:0.15 --val-fold 2 --out runs/kd_lw_all15 > runs/kd_lw_all15.log 2>&1
echo "gpu0 chain done" >> runs/oss125_ext.log
