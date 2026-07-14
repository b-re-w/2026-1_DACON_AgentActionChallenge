#!/usr/bin/env bash
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/lowW.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
# 직렬화 게이트 3종 완료 후
until [ -f runs/kd_btf_f2/metrics.json ] && [ -f runs/kd_btn_f2/metrics.json ] && [ -f runs/kd_btr_f2/metrics.json ]; do sleep 180; done
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --val-fold 2"
r(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --out $o >$o.log 2>&1
  say "$n f2=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+') (기준 0.79367)"; }
( r 1 lw_gem25 qgptoss qw6 q7bteam qgemma9b:0.25 ) &
( r 2 lw_glm25 qgptoss qw6 q7bteam qglm32b:0.25 ) &
( r 3 lw_all15 qgptoss qw6 q7bteam qgemma9b:0.15 qglm32b:0.15 q72b:0.15 ) &
wait
say "=== lowW 게이트 완료 ==="
