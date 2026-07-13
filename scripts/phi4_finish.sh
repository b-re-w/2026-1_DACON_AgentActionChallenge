#!/usr/bin/env bash
# Phi-4 종결 워처: 학습 완료 → best eval 판정 → 0.768+ 이면 store 생성(GPU0) + 트리오+Phi4 f0/f2 페어.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/phi4_finish.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
say "Phi-4 학습 완료 대기..."
until [ -f runs/t_qphi4/model/config.json ] || [ -f runs/t_qphi4/model/adapter_config.json ]; do sleep 300; done
best=$(uv run python -c "
import json,glob,re
cks=sorted(glob.glob('runs/t_qphi4/hf/checkpoint-*/trainer_state.json'), key=lambda p:int(re.search(r'checkpoint-(\\d+)',p).group(1)))
st=json.load(open(cks[-1])); print(st.get('best_metric') or 0)" 2>/dev/null)
say "Phi-4 best eval = $best"
if python3 -c "exit(0 if float('$best') >= 0.768 else 1)"; then
  say "0.768+ → store 생성 (GPU0)"
  until gpu_free 0; do sleep 120; done
  CUDA_VISIBLE_DEVICES=0 uv run --extra dev --with "transformers==4.57.1" python -m ai_challenge.method.gen_softlabels \
    --teacher-dir runs/t_qphi4/model --tag qphi4 --serialize base --max-length 640 --load-in-4bit --batch-size 16 >>"$LOG" 2>&1 \
    && say "qphi4 store 완료" || { say "store 실패"; exit 1; }
  # 검증 + 트리오+Phi4 페어
  CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42); f0=fold==0
y=np.array([CLASS_TO_ID[r.action] for r in recs])
lg=gather_teacher_logits(["qphi4"],recs)
assert not np.isnan(lg).any(), "NaN"
print(f"qphi4 argmax fold0 = {f1_score(y[f0],lg[f0].argmax(1),average='macro'):.5f}")
PY
  D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss qw6 q7bteam qphi4:1"
  until gpu_free 0; do sleep 60; done
  say "트리오+Phi4 (btrail) f0"
  CUDA_VISIBLE_DEVICES=0 $D --val-fold 0 --out runs/kd_btp4_f0 >runs/kd_btp4_f0.log 2>&1
  say "btp4_f0 OOF=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_btp4_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+') (기준 btrail f0=0.78669)"
  CUDA_VISIBLE_DEVICES=0 $D --val-fold 2 --out runs/kd_btp4_f2 >runs/kd_btp4_f2.log 2>&1
  say "btp4_f2 OOF=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_btp4_f2/metrics.json 2>/dev/null|grep -oE '[0-9.]+') (기준 0.79367)"
else
  say "0.768 미달 → Phi-4 라인 종료 (gemma/GLM 전례)"
fi
say "=== phi4_finish 완료 ==="
