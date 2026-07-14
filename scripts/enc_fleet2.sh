#!/usr/bin/env bash
# KD 인코더 함대 (레버 총동원): 인코더를 distill.py로 학습 (트리오 soft-label KD)
#   + btrail/btrailf 직렬화 + eval-steps 500 step-best + 12ep.
# 완료 후: 70k store → fold0 앙상블 게이트 → 결과 보고 (증류 GO는 수동/후속).
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/enc_fleet2.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --bf16 --temperature 3 --alpha 0.25 --eval-steps 500 --epochs 12 --val-fold 0 --teacher-logits qgptoss qw6 q7bteam"
tr(){ local g=$1 n=$2 model=$3 ser=$4 ml=$5; shift 5; local o=runs/kE_$n
  if [ ! -f $o/metrics.json ]; then
    until gpu_free $g; do sleep 90; done
    say "GPU$g KD-enc $n ($model/$ser/ml$ml)"
    CUDA_VISIBLE_DEVICES=$g $D --student $model --serialize $ser --max-length $ml "$@" --out $o >$o.log 2>&1 || { say "$n 실패"; return; }
    say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"
  fi
  [ -f runs/_teacher_logits/qk_$n.npz ] || { until gpu_free $g; do sleep 60; done
    CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.gen_softlabels --teacher-dir $o/model \
      --tag qk_$n --serialize $ser --max-length $ml >>"$LOG" 2>&1 && say "qk_$n store"; }
}
# 함대 6+α: 아키텍처(3) × 직렬화(btrail/btrailf) × 시드
( tr 1 klueb_t  klue/roberta-base btrail 512
  tr 1 klueb_tf klue/roberta-base btrailf 512 --seed 43 ) &
( tr 2 mdebb_t  microsoft/mdeberta-v3-base btrail 640
  tr 2 mdebb_tf microsoft/mdeberta-v3-base btrailf 640 --seed 43 ) &
( tr 3 xlmrb_t  xlm-roberta-base btrail 512
  tr 3 xlmrb_tf xlm-roberta-base btrailf 512 --seed 43 ) &
wait
say "함대 완료 → fold0 앙상블 게이트"
until gpu_free 1; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np, glob, os
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42); f0=fold==0
y=np.array([CLASS_TO_ID[r.action] for r in recs])
tags=sorted(os.path.basename(p)[:-4] for p in glob.glob("runs/_teacher_logits/qk_*.npz"))
Ps=[]
for t in tags:
    lg=gather_teacher_logits([t],recs)
    if np.isnan(lg).any(): print(f"  {t}: NaN 제외"); continue
    e=np.exp(lg-lg.max(1,keepdims=True)); p=e/e.sum(1,keepdims=True)
    s=f1_score(y[f0],lg[f0].argmax(1),average="macro")
    print(f"  {t}: fold0={s:.5f}")
    Ps.append(p)
if Ps:
    ens=np.mean(Ps,axis=0)
    print(f"★ KD인코더 {len(Ps)}개 앙상블 fold0 = {f1_score(y[f0],ens[f0].argmax(1),average='macro'):.5f}")
    tri=gather_teacher_logits(["qgptoss","qw6","q7bteam"],recs)
    e=np.exp(tri-tri.max(1,keepdims=True)); ptri=e/e.sum(1,keepdims=True)
    print(f"  (참고 trio argmax fold0 = {f1_score(y[f0],ptri[f0].argmax(1),average='macro'):.5f})")
    for w in [0.2,0.35,0.5]:
        m=f1_score(y[f0],((1-w)*ptri+w*ens)[f0].argmax(1),average="macro")
        print(f"  trio+(encKD x{w}): {m:.5f}")
PY
say "=== enc_fleet2 완료 ==="
