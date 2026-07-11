#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
LOG=runs/gemma_blend.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && { say "$n=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')"; return; }
  say "GPU$g $n :: $*"; CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.distill --teacher-logits "$@" \
    --student Qwen/Qwen2.5-0.5B --out $o --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base >$o.log 2>&1
  say "$n OOF=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')"; }
# GPU1: qw6 + gemma (우리 최고 블렌드에 이질 추가)
( d 1 gem_qw6_05  qw6 qgemma9b:0.5
  d 1 gem_qw6_1   qw6 qgemma9b:1
  d 1 gem_qw6_2   qw6 qgemma9b:2 ) &
# GPU2: oss2g1 레시피 + gemma (개별 재조합)
Q="q3bb q3b43 q3b44 q3b45"
( d 2 gem_2g1_05  $Q qgptoss:2 qgpt5:1 qgemma9b:0.5
  d 2 gem_2g1_1   $Q qgptoss:2 qgpt5:1 qgemma9b:1
  d 2 gem_w6_1    $Q qgpt5:2 qgemma9b:1 ) &
wait
say "=== gemma_blend 완료 ==="
for n in gem_qw6_05 gem_qw6_1 gem_qw6_2 gem_2g1_05 gem_2g1_1 gem_w6_1; do echo "  $n: $(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"|tee -a "$LOG"; done
