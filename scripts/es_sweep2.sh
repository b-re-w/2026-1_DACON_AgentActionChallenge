#!/usr/bin/env bash
# es300 5-fold 평균이 es500+0.001 이상이면 → es100/200/400 5-fold 스윕 (밤샘, GPU당 1값)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/es_sweep2.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
until [ -f runs/oof70k_es3.npy ]; do sleep 300; done
MEAN=$(uv run python -c "
import numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
recs=load_records('data/train.jsonl','data/train_labels.csv')
fold=assign_folds(recs,n_splits=5,seed=42)
import numpy as np
y=np.array([CLASS_TO_ID[r.action] for r in recs])
L=np.load('runs/oof70k_es3.npy')
print(np.mean([f1_score(y[fold==f],L[fold==f].argmax(1),average='macro') for f in range(5)]))" 2>/dev/null)
say "es300 평균 = $MEAN (트리거: >0.78830)"
if ! python3 -c "exit(0 if float('$MEAN') > 0.78830 else 1)"; then
  say "트리거 미달 → es 스윕 생략"; exit 0
fi
say "트리거 통과 → es100/200/400 밤샘 스윕"
run_es(){ local g=$1 es=$2
  local D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss qw6 q7bteam --eval-steps $es"
  for f in 0 1 2 3 4; do
    [ -f runs/kd_es${es}_f$f/metrics.json ] && continue
    until gpu_free $g; do sleep 90; done
    say "GPU$g es$es fold$f"
    CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out runs/kd_es${es}_f$f > runs/kd_es${es}_f$f.log 2>&1
  done
  say "es$es 완주"
}
( run_es 1 200 ) & ( run_es 2 400 ) & ( run_es 3 100 ) & wait
say "=== es 스윕 전체 완료 → fold별 평균 비교 ==="
uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np, json
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42)
y=np.array([CLASS_TO_ID[r.action] for r in recs])
for es in [100,200,400]:
    vals=[]
    for f in range(5):
        try: vals.append(json.load(open(f"runs/kd_es{es}_f{f}/metrics.json"))["macro_f1"])
        except: vals.append(None)
    ok=[v for v in vals if v]
    print(f"es{es}: {[round(v,5) if v else None for v in vals]} 평균={np.mean(ok):.5f}" if ok else f"es{es}: 없음")
print("(es500 평균 0.78730 / es300은 oof70k_es3 참조)")
PY
