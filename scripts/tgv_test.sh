#!/usr/bin/env bash
# 사용자 가설: gemma(+)가 약한 Qwen성분(−)에 상쇄됐을 수 있다 → qw6 감량/제거 변형 (fold0).
# tgd_test 완료 후 GPU2 이어받아 순차 3종.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/tgv_test.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
until grep -q "tgd_test 완료" runs/tgd_test.log 2>/dev/null; do sleep 120; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 2000 ]; do sleep 60; done
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local n=$1; shift; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  say "$n :: $*"; CUDA_VISIBLE_DEVICES=2 $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
d tgv_half  qgptoss qw6:0.5 q7bteam qgemma9b:0.5
d tgv_noqw6 qgptoss q7bteam qgemma9b:0.5
d tgv_gpt5  qgptoss qgpt5 q7bteam qgemma9b:0.5
say "=== tgv_test 완료 (기준: tg5 f0=0.78388[epoch]/0.78337[step], trio f0=0.78205) ==="
