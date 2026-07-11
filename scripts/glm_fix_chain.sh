#!/usr/bin/env bash
# GLM bf16 재생성(GPU0) → 검증 → GLM 블렌드 3종 재실행 (NaN 무효분 재시도)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf PYTHONPATH=/data/agent_action
LOG=runs/glm_fix.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "GLM bf16 재생성 시작"
CUDA_VISIBLE_DEVICES=0 uv run --with "transformers==4.55.0" python scripts/glm_bf16_gen.py >>"$LOG" 2>&1 || { say "bf16 재생성 실패"; exit 1; }
say "store 검증(argmax)"
CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42); f0=fold==0
y=np.array([CLASS_TO_ID[r.action] for r in recs])
lg=gather_teacher_logits(["qglm32b"],recs)
print(f"qglm32b argmax(fold0)={f1_score(y[f0],lg[f0].argmax(1),average='macro'):.5f} (기대 ~0.76)")
PY
# 무효였던 GLM 블렌드 재실행 (기존 0.0167 결과 삭제 후)
for n in glm_qw6_05 glm_trio05 glm_gem_dual trio_glm; do rm -rf runs/kd_$n runs/kd_$n.log; done
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
( d 0 glm_trio05 qgptoss qw6 q7bteam qglm32b:0.5 ) &
( until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do sleep 60; done; d 1 trio_glm qgptoss qw6 qglm32b ) &
( until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 2000 ]; do sleep 60; done; d 2 glm_qw6_05 qw6 qglm32b:0.5 ) &
wait
say "=== glm_fix 완료 ==="
