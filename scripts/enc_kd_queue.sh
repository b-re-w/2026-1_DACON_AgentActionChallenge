#!/usr/bin/env bash
# 인코더 teacher 학습 종료 후 자동 실행:
#  A(GPU1): 디코더 student + btrail 직렬화 f0/f2 페어 (메인 라인에 TRAIL 신호 주입 테스트)
#  B(GPU2/3): 소형 인코더 KD student 3종 (btrail, ML512, 트리오 store) → 앙상블 재료
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/enc_kd.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
TRIO="qgptoss qw6 q7bteam"
D="uv run python -m ai_challenge.method.distill --bf16 --serialize btrail --temperature 3 --alpha 0.25 --teacher-logits $TRIO"
r(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $D "$@" --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
# A: 디코더 btrail 페어 (기준: trio base f0=0.78205 f2=0.79053)
( r 1 3bt_f0 --student Qwen/Qwen2.5-0.5B --max-length 640 --val-fold 0
  r 1 3bt_f2 --student Qwen/Qwen2.5-0.5B --max-length 640 --val-fold 2 ) &
# B: 소형 인코더 KD students (fold0) — 앙상블 게이트용
( r 2 e_klueb_f0 --student klue/roberta-base --max-length 512 --val-fold 0
  r 2 e_klues_f0 --student klue/roberta-small --max-length 512 --val-fold 0 ) &
( r 3 e_mdeb_f0 --student microsoft/mdeberta-v3-base --max-length 512 --val-fold 0 ) &
wait
say "=== enc_kd 완료 ==="
say "--- A 디코더+TRAIL: f0=$(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_3bt_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+') f2=$(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_3bt_f2/metrics.json 2>/dev/null|grep -oE '[0-9.]+') (기준 0.78205/0.79053)"
say "--- B 인코더 students: klueb=$(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_e_klueb_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+') klues=$(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_e_klues_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+') mdeb=$(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_e_mdeb_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"
# B 앙상블 로컬 게이트 (fold0 logit 평균)
until gpu_free 2; do sleep 60; done
CUDA_VISIBLE_DEVICES=2 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID, SERIALIZE_PRESETS
from ai_challenge.models.common import predict_logits
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42)
f0=[r for r,f in zip(recs,fold) if f==0]; y=np.array([CLASS_TO_ID[r.action] for r in f0])
sk=SERIALIZE_PRESETS["btrail"]; Ls=[]
for md in ["runs/kd_e_klueb_f0/model","runs/kd_e_klues_f0/model","runs/kd_e_mdeb_f0/model"]:
    try:
        L=predict_logits(md,f0,max_length=512,serialize_kwargs=sk); Ls.append(L)
        print(f"  {md.split('/')[1]}: {f1_score(y,L.argmax(1),average='macro'):.5f}")
    except Exception as e: print(md,"실패",str(e)[:50])
if len(Ls)>=2:
    ens=np.mean([l-l.mean(1,keepdims=True) for l in Ls],axis=0)
    print(f"★ 인코더 {len(Ls)}종 logit 앙상블 fold0 = {f1_score(y,ens.argmax(1),average='macro'):.5f} (게이트 0.78)")
PY
say "=== 전체 완료 ==="
