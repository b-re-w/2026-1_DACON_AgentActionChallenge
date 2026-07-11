#!/usr/bin/env bash
# GLM 재시도(사용자 지시: 다른 계열이니 넣어봐야 안다) + 미시도 이질 조합.
# bias70k 캠페인 완료 대기 → GPU0: GLM store(batch16) / GPU1·2: 미시도 블렌드 병렬.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/glm_retry.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "bias70k 캠페인 완료 대기..."
until grep -q "bias70k_ours 완료" runs/bias70k_ours.log 2>/dev/null; do sleep 120; done; sleep 10

D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }

# GPU0: GLM-32B store 재생성 (batch16 — 지난 OOM은 batch128, 저속중단은 batch4)
( if [ ! -f runs/_teacher_logits/qglm32b.npz ]; then
    say "GLM store 생성 (batch16)"
    CUDA_VISIBLE_DEVICES=0 uv run --with "transformers==4.55.0" python -m ai_challenge.method.gen_softlabels \
      --teacher-dir runs/tglm32b/model --tag qglm32b --serialize base --max-length 640 --load-in-4bit --batch-size 16 >>"$LOG" 2>&1 \
      && say "qglm32b 완료" || say "qglm32b 실패"
  fi ) &
# GPU1: 트리오+gemma 저가중 (트리오 문맥에서 gemma 재시험)
( d 1 trio_gem05 qgptoss qw6 q7bteam qgemma9b:0.5 ) &
# GPU2: soup 추가시드 51 (편차 ±0.0013 확인됐으니 시드 수 늘릴 가치)
( bash scripts/soup_train.sh 2 51 ) &
wait

# GLM store 성공 시 GLM 블렌드 3종 (GPU 0/1/2)
if [ -f runs/_teacher_logits/qglm32b.npz ]; then
  say "GLM 블렌드 착수"
  ( d 0 glm_qw6_05 qw6 qglm32b:0.5 ) &
  ( d 1 glm_trio05 qgptoss qw6 q7bteam qglm32b:0.5 ) &
  ( d 2 glm_gem_dual qw6 qglm32b:0.5 qgemma9b:0.5 ) &
  wait
fi
say "=== glm_retry 완료 ==="
for n in trio_gem05 glm_qw6_05 glm_trio05 glm_gem_dual; do
  echo "  $n: $(grep -oE '"macro_f1": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+' || echo NA)"|tee -a "$LOG"; done
