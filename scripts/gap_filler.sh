#!/usr/bin/env bash
# glm_retry의 GPU1/2 유휴구간(GLM store 생성 대기) 필러:
# GPU1/2 작업(trio_gem05, soup51) 끝나면 GLM 불필요한 trio_32b + soup52 즉시 실행.
# trio_swap의 d() 가드가 중복 실행 방지(완료된 건 스킵).
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/gap_filler.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "GPU1/2 선행작업(trio_gem05·soup s51) 완료 대기..."
until [ -f runs/kd_trio_gem05/metrics.json ] && [ -f runs/kd_qw6soup_s51/model/config.json ]; do sleep 90; done
# GLM store가 이미 끝났으면(=GLM 블렌드가 곧 GPU 점유) 충돌 방지 위해 필러 스킵
if [ -f runs/_teacher_logits/qglm32b.npz ]; then say "GLM store 이미 완료 → 충돌방지 스킵"; exit 0; fi
say "갭 필러 시작: GPU1 trio_32b, GPU2 soup s52"
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
( [ -f runs/kd_trio_32b/metrics.json ] || CUDA_VISIBLE_DEVICES=1 $D --teacher-logits qgptoss qw6 q32b --val-fold 0 --out runs/kd_trio_32b >runs/kd_trio_32b.log 2>&1 ) &
( bash scripts/soup_train.sh 2 52 ) &
wait
say "갭 필러 완료: trio_32b OOF=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_trio_32b/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"
