#!/usr/bin/env bash
# gpt-oss 가중 상향 테스트 (사용자 제안): 트리오에서 qgptoss:1.5, btrail (f0/f2 페어)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/ossw.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss:1.5 qw6 q7bteam"
r(){ local g=$1 f=$2; local o=runs/kd_ossw_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g ossw fold$f"; CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out $o >$o.log 2>&1
  say "ossw_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"; }
( r 3 0 ) &   # btf 끝나는 GPU3
( r 1 2 ) &   # btc_f0 끝나는 GPU1
wait
say "=== ossw 완료 (기준 btrail 1:1:1 f0=0.78669 f2=0.79367) ==="
