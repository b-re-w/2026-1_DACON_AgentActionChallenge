#!/usr/bin/env bash
# 인코더 부활 최종 시도 (사용자 아이디어: 임베딩 프루닝 → 디코더+인코더 동시탑재 앙상블)
# base급 인코더 2종 CE 훈련 (btrail, val-fold2) → 70k store → fold2 정직 블렌드 게이트
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/enc_ce_f2.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
T="uv run python -m ai_challenge.method.train --serialize btrail --val-fold 2 --epochs 5 --lr 2e-5 --batch-size 32 --early-stop-patience 2 --no-class-weight"
tr(){ local g=$1 n=$2 model=$3 ml=$4; local o=runs/ce_$n
  if [ ! -f $o/metrics.json ]; then
    until gpu_free $g; do sleep 120; done
    say "GPU$g CE-enc $n 시작"
    CUDA_VISIBLE_DEVICES=$g $T --model $model --max-length $ml --out $o >$o.log 2>&1 || { say "$n 실패"; return 1; }
    say "$n f2 OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json|grep -oE '0\.[0-9]+')"
  fi
  [ -f runs/_teacher_logits/qce_$n.npz ] || { until gpu_free $g; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.gen_softlabels --teacher-dir $o/model \
      --tag qce_$n --serialize btrail --max-length $ml >>"$LOG" 2>&1 && say "qce_$n store 완료"; }
}
( tr 0 mdebb microsoft/mdeberta-v3-base 640 ) &
( tr 2 klueb klue/roberta-base 512 ) &
wait
say "=== 훈련 완료 → fold2 정직 블렌드 게이트 ==="
uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np, glob, os
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42); y=np.array([CLASS_TO_ID[r.action] for r in recs])
m2=fold==2
oof=np.load("runs/oof70k_bts.npy")
def sm(x): e=np.exp(x-x.max(1,keepdims=True)); return e/e.sum(1,keepdims=True)
P_s=sm(oof)
base=f1_score(y[m2],oof[m2].argmax(1),average="macro")
print(f"student f2 기준선: {base:.5f}")
for t in sorted(os.path.basename(p)[:-4] for p in glob.glob("runs/_teacher_logits/qce_*.npz")):
    lg=gather_teacher_logits([t],recs)
    if np.isnan(lg).any(): print(f"{t}: NaN 제외"); continue
    P_e=sm(lg)
    solo=f1_score(y[m2],lg[m2].argmax(1),average="macro")
    print(f"{t}: 단독 f2={solo:.5f} (0.774+ 룰 기준)")
    for w in [0.1,0.15,0.2,0.3]:
        b=(1-w)*P_s+w*P_e
        s=f1_score(y[m2],b[m2].argmax(1),average="macro")
        print(f"  블렌드 w={w}: f2={s:.5f} (Δ{s-base:+.5f})")
PY
say "=== enc_ce_f2 완료 ==="
