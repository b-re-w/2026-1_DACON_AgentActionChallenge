#!/usr/bin/env bash
# 72B 완료(~01:00) → store 생성(GPU3, 4bit batch8) → argmax 검증 → 트리오×72B fold0 스크리닝.
# 아침에 결과 보고 표준구조(all-data+56k bias) 조립 여부 결정.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/chain_72b.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
say "72B 학습 완료 대기..."
until [ -f runs/t72b_qlora/model/adapter_config.json ]; do sleep 300; done
say "72B 완료 → 최종 eval: $(grep -oE \"'eval_macro_f1': [0-9.]+\" runs/t72b_qlora.log | grep -oE '[0-9.]+' | tr '\n' ' ')"
until gpu_free 3; do sleep 60; done
# store 생성 (72B 4bit — Qwen계열은 4bit 추론 정상 이력, batch8 안전)
[ -f runs/_teacher_logits/q72b.npz ] || { say "q72b store 생성 (GPU3, 4bit b8)"; CUDA_VISIBLE_DEVICES=3 \
  uv run python -m ai_challenge.method.gen_softlabels --teacher-dir runs/t72b_qlora/model \
  --tag q72b --serialize base --max-length 640 --load-in-4bit --batch-size 8 >>"$LOG" 2>&1 \
  && say "q72b 완료" || { say "q72b 실패"; exit 1; }; }
# 검증 (NaN/argmax)
CUDA_VISIBLE_DEVICES=3 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42); f0=fold==0
y=np.array([CLASS_TO_ID[r.action] for r in recs])
lg=gather_teacher_logits(["q72b"],recs)
assert not np.isnan(lg).any(), "NaN!"
print(f"q72b argmax fold0 = {f1_score(y[f0],lg[f0].argmax(1),average='macro'):.5f} (train eval 0.770+)")
PY
# fold0 스크리닝: 스왑(7B→72B) / 추가(+72B)
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
( d 1 tri72_swap qgptoss qw6 q72b ) &            # 7B → 72B 교체
( d 2 tri72_add  qgptoss qw6 q7bteam q72b ) &    # 72B 추가 (7B 유지)
( d 3 tri72_hv   qgptoss qw6 q72b:2 ) &          # 72B 무겁게
wait
say "=== chain_72b 완료 (기준 trio f0=0.78205) ==="
for n in tri72_swap tri72_add tri72_hv; do
  echo "  $n: $(grep -oE '"macro_f1": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+' || echo NA)"|tee -a "$LOG"; done
