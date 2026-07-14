#!/usr/bin/env bash
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/seedrop.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
until [ -f runs/kd_ossw_f0/metrics.json ] && [ -f runs/kd_ossw_f2/metrics.json ]; do sleep 120; done
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss q3bb q3b44 qgpt5:2 q7bteam"
r(){ local g=$1 f=$2; local o=runs/kd_sd_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g seedrop fold$f"; CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out $o >$o.log 2>&1
  say "sd_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"; }
( r 1 0 ) & ( r 2 2 ) & wait
say "=== seedrop 완료 (약체 s43/s45 제거판, 기준 f0=0.78669 f2=0.79367) ==="
