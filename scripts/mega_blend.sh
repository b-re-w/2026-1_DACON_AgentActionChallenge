#!/usr/bin/env bash
# 메가 블렌드 (사용자 제안: qwen+gptoss+gpt5+gemma+glm 다 넣기).
# glm_retry 완료 대기 → 3종 병렬 (fold0): 동등 / 코어가중 / 트리오+이질경량.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/mega_blend.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "glm_retry 완료 대기..."
until grep -q "glm_retry 완료" runs/glm_retry.log 2>/dev/null; do sleep 120; done; sleep 10
[ -f runs/_teacher_logits/qglm32b.npz ] || { say "GLM store 없음 → GLM 빼고 진행"; GLM=""; }
GLM=${GLM-"qglm32b"}

D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  say "GPU$g $n :: $*"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }

if [ -n "$GLM" ]; then
  # ① 동등가중 전부 (사용자 제안 그대로): 4×3B + oss + gpt5 + gemma + glm
  ( d 0 mega_eq q3bb q3b43 q3b44 q3b45 qgptoss qgpt5 qgemma9b qglm32b ) &
  # ② 코어 무겁게 + 이질 가볍게
  ( d 1 mega_w qw6:4 qgptoss:1.5 qgpt5:1 q7bteam:1 qgemma9b:0.5 qglm32b:0.5 ) &
  # ③ 팀원 트리오 + 이질 경량 (그들 최고 base 위에 gemma/glm)
  ( d 2 mega_trio qgptoss qw6 q7bteam qgemma9b:0.5 qglm32b:0.5 ) &
else
  ( d 0 mega_eq q3bb q3b43 q3b44 q3b45 qgptoss qgpt5 qgemma9b ) &
  ( d 1 mega_w qw6:4 qgptoss:1.5 qgpt5:1 q7bteam:1 qgemma9b:0.5 ) &
  ( d 2 mega_trio qgptoss qw6 q7bteam qgemma9b:0.5 ) &
fi
wait
# Wave2 (사용자 제안: 큰 Qwen도 포함) — 32B(최대)/14B 추가 변형
if [ -n "$GLM" ]; then
  ( d 0 mega_all q3bb q3b43 q3b44 q3b45 qgptoss qgpt5 qgemma9b qglm32b q14b q32b q7bteam ) &   # 말그대로 전부(11종)
  ( d 1 mega_w32 qw6:4 qgptoss:1.5 qgpt5:1 q7bteam:1 q32b:1 qgemma9b:0.5 qglm32b:0.5 ) &        # 코어가중 + 32B
  ( d 2 mega_trio32 qgptoss qw6 q7bteam q32b:0.5 qgemma9b:0.5 qglm32b:0.5 ) &                   # 트리오 + 32B/이질 경량
else
  ( d 0 mega_all q3bb q3b43 q3b44 q3b45 qgptoss qgpt5 qgemma9b q14b q32b q7bteam ) &
  ( d 1 mega_w32 qw6:4 qgptoss:1.5 qgpt5:1 q7bteam:1 q32b:1 qgemma9b:0.5 ) &
  ( d 2 mega_trio32 qgptoss qw6 q7bteam q32b:0.5 qgemma9b:0.5 ) &
fi
wait
say "=== mega_blend 완료 (기준: qw6=0.7847, 트리오fold0은 kd_trio_f0 참조) ==="
for n in mega_eq mega_w mega_trio mega_all mega_w32 mega_trio32; do
  echo "  $n: $(grep -oE '"macro_f1": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+' || echo NA)"|tee -a "$LOG"; done
