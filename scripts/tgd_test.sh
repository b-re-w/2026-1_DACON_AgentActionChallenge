#!/usr/bin/env bash
# 트리오+gemma:0.5 에 DeBERTa 추가 검증 (사용자 지시): :0.25 / :0.5 (fold0).
# GPU2 는 mts_f3 완료 후 안전하게 빔 (mtfix 는 GPU0/1 사용) → 거기서 순차 실행.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/tgd_test.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2 2>/dev/null)" -lt 2000 ]; }
say "GPU2 확보 대기 (mts_f3 완료 후)..."
until [ -f runs/kd_mts_f3/metrics.json ]; do sleep 120; done
until gpu_free; do sleep 60; done
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local n=$1; shift; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  say "$n :: $*"; CUDA_VISIBLE_DEVICES=2 $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
d tgd025 qgptoss qw6 q7bteam qgemma9b:0.5 qdebertaL2:0.25
d tgd05  qgptoss qw6 q7bteam qgemma9b:0.5 qdebertaL2:0.5
say "=== tgd_test 완료 (기준: trio_gem05 f0=0.78388, trio f0=0.78205) ==="
