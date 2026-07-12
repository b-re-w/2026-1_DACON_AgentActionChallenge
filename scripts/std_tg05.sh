#!/usr/bin/env bash
# 표준 파이프라인 (56k best-ckpt + bias70k) × trio_gem05 (우리 고유 변형):
#  ① fold0~4 를 --eval-steps 500 로 학습 (fold0 = 제출 베이스, step피크 선택)
#  ② 5-fold 70k OOF → bias 피팅 (+정직검증 보고용)
#  ③ fold0-best 모델 + bias.json → pack (tg5b70.zip)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/std_tg05.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
T="qgptoss qw6 q7bteam qgemma9b:0.5"
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base --eval-steps 500 --teacher-logits $T"
tf(){ local g=$1 f=$2 o=runs/kd_tg5s_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g tg5s fold$f (eval-steps 500)"; CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out $o >$o.log 2>&1
  say "tg5s_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
( tf 0 0; tf 0 3 ) & ( tf 1 1; tf 1 4 ) & ( tf 2 2 ) & wait
say "5-fold 완료 → 70k OOF + bias"
until gpu_free 0; do sleep 60; done
CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import json, numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import predict_logits
from ai_challenge.method.tune_threshold import coordinate_ascent
recs = load_records("data/train.jsonl","data/train_labels.csv")
fold = assign_folds(recs, n_splits=5, seed=42); y = np.array([CLASS_TO_ID[r.action] for r in recs])
L = np.zeros((len(recs),14), np.float32)
for f in range(5):
    idx = np.where(fold==f)[0]
    L[idx] = predict_logits(f"runs/kd_tg5s_f{f}/model", [recs[i] for i in idx], max_length=640, serialize_kwargs={})
np.save("runs/oof70k_tg5s.npy", L)
per = [f1_score(y[fold==f], L[fold==f].argmax(1), average="macro") for f in range(5)]
print("fold별:", [round(v,5) for v in per], "| 평균:", round(float(np.mean(per)),5))
tr = fold!=0; te = fold==0
bh,_ = coordinate_ascent(L[tr], y[tr])
print(f"[정직검증] f0: {f1_score(y[te],L[te].argmax(1),average='macro'):.5f} → {f1_score(y[te],(L[te]+bh).argmax(1),average='macro'):.5f}")
base = f1_score(y, L.argmax(1), average="macro")
bias, tuned = coordinate_ascent(L, y)
print(f"[70k전체] {base:.5f} → {tuned:.5f}")
json.dump({"bias": bias.tolist(), "before": float(base), "after": float(tuned)}, open("runs/bias70k_tg5s.json","w"), indent=1)
PY
# fold0-best 모델 + bias → pack
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b = json.load(open("runs/bias70k_tg5s.json"))
dst="runs/kd_tg5b70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_tg5s_f0/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("fold0-best + bias 부착")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_tg5b70/model --name tg5b70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/tg5b70.zip ✅ (표준 파이프라인 × trio_gem05)"
say "=== std_tg05 완료 ==="
