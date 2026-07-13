#!/usr/bin/env bash
# 밤 2차 캠페인 (std_bt 완료 후 자동):
#  WaveA: TRAIL 변형 페어 — btrail8(트레일8개) f0/f2 + bthist(TRAIL+히스토리확장) f0/f2
#  WaveB: TRAIL 레시피 시드 43/44 all-data (신분포 복권/백업 재료)
#  CPU:   bts OOF 나오면 TF-IDF 블렌드 재검증 (TRAIL이 n-gram 흡수했는지)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/night2.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
say "std_bt 완료 대기..."
until grep -q "std_bt 완료" runs/std_bt.log 2>/dev/null; do sleep 180; done
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --teacher-logits qgptoss qw6 q7bteam"
r(){ local g=$1 n=$2 ser=$3; shift 3; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g $n ($ser)"; CUDA_VISIBLE_DEVICES=$g $D --serialize $ser "$@" --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
# WaveA + WaveB
( r 1 bt8_f0 btrail8 --val-fold 0 ; r 1 bt8_f2 btrail8 --val-fold 2 ) &
( r 2 bth_f0 bthist  --val-fold 0 ; r 2 bth_f2 bthist  --val-fold 2 ) &
( r 3 bt_s43 btrail --all-data --seed 43 ; r 3 bt_s44 btrail --all-data --seed 44 ) &
# CPU: TF-IDF × bts(TRAIL students) 재검증
( until [ -f runs/oof70k_bts.npy ] && [ -f runs/tfidf_oof70k.npy ]; do sleep 120; done
  uv run python - <<'PY' >> runs/night2.log 2>&1
import numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
recs=load_records("data/train.jsonl","data/train_labels.csv")
fold=assign_folds(recs,n_splits=5,seed=42)
y=np.array([CLASS_TO_ID[r.action] for r in recs])
L=np.load("runs/oof70k_bts.npy"); P=np.exp(L-L.max(1,keepdims=True)); P/=P.sum(1,keepdims=True)
T=np.load("runs/tfidf_oof70k.npy")
gains=[]
for f in range(5):
    te=fold==f
    b=f1_score(y[te],P[te].argmax(1),average="macro")
    m=f1_score(y[te],(0.9*P[te]+0.1*T[te]).argmax(1),average="macro")
    gains.append(m-b); print(f"[tfidf x bts] fold{f}: {b:.5f} -> {m:.5f} (D{m-b:+.5f})", flush=True)
print(f"[tfidf x bts] 평균 D={np.mean(gains):+.5f} 양수 {sum(g>0 for g in gains)}/5", flush=True)
PY
  say "TF-IDF x bts 재검증 완료" ) &
wait
say "=== night2 완료 ==="
say "--- 결과 (기준: btrail f0=0.78669 f2=0.79367) ---"
for n in bt8_f0 bt8_f2 bth_f0 bth_f2; do
  echo "  $n: $(grep -oE '"macro_f1": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+' || echo NA)"|tee -a "$LOG"; done
