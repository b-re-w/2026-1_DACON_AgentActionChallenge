#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
LOG=runs/glm_blend.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
say "qglm32b store 대기..."
until [ -f runs/_teacher_logits/qglm32b.npz ] || grep -qiE "qglm32b 재실패" runs/regen_glm.log 2>/dev/null; do sleep 120; done
[ -f runs/_teacher_logits/qglm32b.npz ] || { say "GLM store 재실패 — 중단"; exit 1; }
# gemma 블렌드 끝나 GPU1/2 비면 실행
until ! screen -ls 2>/dev/null|grep -q "\.gemblend"; do sleep 60; done; sleep 10
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  say "GPU$g $n :: $*"; CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.distill --teacher-logits "$@" \
    --student Qwen/Qwen2.5-0.5B --out $o --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base >$o.log 2>&1
  say "$n OOF=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')"; }
Q="q3bb q3b43 q3b44 q3b45"
( d 1 glm_qw6_05 qw6 qglm32b:0.5
  d 1 glm_qw6_1  qw6 qglm32b:1
  d 1 glm_gem_1  qw6 qglm32b:0.5 qgemma9b:0.5 ) &
( d 2 glm_2g1_05 $Q qgptoss:2 qgpt5:1 qglm32b:0.5
  d 2 glm_2g1gem $Q qgptoss:2 qgpt5:1 qglm32b:0.5 qgemma9b:0.5 ) &
wait
say "=== glm_blend 완료 ==="
for n in glm_qw6_05 glm_qw6_1 glm_gem_1 glm_2g1_05 glm_2g1gem; do echo "  $n: $(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"|tee -a "$LOG"; done
