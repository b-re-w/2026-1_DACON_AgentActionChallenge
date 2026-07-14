#!/usr/bin/env bash
# 인코더 함대 (1등 방법 축소재현): 다양한 인코더 teacher 6~9개 학습(btrail/base 혼합, 20ep p2, no-cw)
# → 70k store 생성 → fold0 앙상블 게이트(≥0.785) → GO면 증류는 별도 스크립트.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/enc_fleet.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
T="uv run python -m ai_challenge.method.train --bf16 --no-class-weight --epochs 20 --early-stop-patience 2 --warmup-ratio 0.1 --val-fold 0"
tr(){ local g=$1 n=$2 model=$3 ser=$4 ml=$5; shift 5; local o=runs/tE_$n
  if [ ! -f $o/model/config.json ]; then
    until gpu_free $g; do sleep 90; done
    say "GPU$g teacher $n ($model, $ser, ml$ml)"
    CUDA_VISIBLE_DEVICES=$g $T --model $model --serialize $ser --max-length $ml --batch-size 64 "$@" --out $o >$o.log 2>&1 || { say "$n 실패"; return; }
    say "$n eval=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"
  fi
  # 70k store
  [ -f runs/_teacher_logits/qe_$n.npz ] || { until gpu_free $g; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.gen_softlabels --teacher-dir $o/model \
      --tag qe_$n --serialize $ser --max-length $ml >>"$LOG" 2>&1 && say "qe_$n store ✅"; }
}
# 함대: 아키텍처 × 직렬화 × 시드 다양성
( tr 3 klueb1 klue/roberta-base btrail 512 --lr 2e-5
  tr 3 mdebb2 microsoft/mdeberta-v3-base base 640 --lr 2e-5
  tr 3 klueb3 klue/roberta-base btrail 512 --lr 2e-5 --seed 43 ) &
( tr 1 mdebb1 microsoft/mdeberta-v3-base btrail 640 --lr 2e-5
  tr 1 xlmrb1 xlm-roberta-base btrail 512 --lr 2e-5
  tr 1 klueb4 klue/roberta-base bthist 512 --lr 2e-5 ) &
( tr 2 xlmrb2 xlm-roberta-base base 512 --lr 3e-5
  tr 2 kluel1 klue/roberta-large btrail 512 --lr 1e-5 --batch-size 32 --grad-accum 2
  tr 2 mdebb3 microsoft/mdeberta-v3-base btrail 640 --lr 3e-5 --seed 44 ) &
wait
say "함대 학습/store 완료 → fold0 앙상블 게이트"
uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np, glob, os
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42); f0=fold==0
y=np.array([CLASS_TO_ID[r.action] for r in recs])
tags=[os.path.basename(p)[:-4] for p in glob.glob("runs/_teacher_logits/qe_*.npz")]
print("함대 store:", tags)
Ps=[]
for t in tags:
    lg=gather_teacher_logits([t],recs)
    if np.isnan(lg).any(): print(f"  {t}: NaN 제외"); continue
    e=np.exp(lg-lg.max(1,keepdims=True)); p=e/e.sum(1,keepdims=True)
    s=f1_score(y[f0],lg[f0].argmax(1),average="macro")
    print(f"  {t}: fold0 argmax={s:.5f}")
    if s>0.60: Ps.append(p)
if Ps:
    ens=np.mean(Ps,axis=0)
    g=f1_score(y[f0],ens[f0].argmax(1),average="macro")
    print(f"★ 인코더 {len(Ps)}개 앙상블 fold0 argmax = {g:.5f} (게이트 0.785 / 참고 trio argmax~0.786)")
    # 트리오와 혼합도
    tri=gather_teacher_logits(["qgptoss","qw6","q7bteam"],recs)
    e=np.exp(tri-tri.max(1,keepdims=True)); ptri=e/e.sum(1,keepdims=True)
    for w in [0.3,0.5,0.7]:
        m=f1_score(y[f0],((1-w)*ptri+w*ens)[f0].argmax(1),average="macro")
        print(f"  trio+(enc x{w}): {m:.5f}")
PY
say "=== enc_fleet 완료 (게이트 결과 위) ==="
