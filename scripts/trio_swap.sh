#!/usr/bin/env bash
# 트리오(qw6+oss+7B)의 3번째 자리 교체 실험: 7B → 32B / GLM / gemma (사용자 제안).
# mega_blend 완료 대기 → 3종 병렬 (fold0).
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/trio_swap.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "mega_blend 완료 대기..."
until grep -q "mega_blend 완료" runs/mega_blend.log 2>/dev/null; do sleep 120; done; sleep 10

D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  say "GPU$g $n :: $*"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }

( d 0 trio_32b qgptoss qw6 q32b ) &                    # 7B → 32B (가장 큰 Qwen)
( d 1 trio_gem qgptoss qw6 qgemma9b ) &                # 7B → gemma (이질)
( if [ -f runs/_teacher_logits/qglm32b.npz ]; then d 2 trio_glm qgptoss qw6 qglm32b; else say "GLM store 없음 → trio_glm 스킵"; fi ) &
wait
say "=== trio_swap 완료 (기준: 트리오 원본 kd_trio_f0 OOF 참조) ==="
for n in trio_32b trio_gem trio_glm; do
  echo "  $n: $(grep -oE '"macro_f1": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+' || echo NA)"|tee -a "$LOG"; done
