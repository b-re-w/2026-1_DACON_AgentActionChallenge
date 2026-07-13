#!/usr/bin/env bash
# 표준 파이프라인 × btrail(TRAIL): step-students f0~4 + all-data 모델 + 56k bias → btb70.zip
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/std_bt.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss qw6 q7bteam"
tf(){ local g=$1 f=$2 o=runs/kd_bts_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g bts fold$f"; CUDA_VISIBLE_DEVICES=$g $D --eval-steps 500 --val-fold $f --out $o >$o.log 2>&1
  say "bts_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
( tf 1 1; tf 1 3 ) &
( tf 2 0; tf 2 4 ) &
( until gpu_free 3; do sleep 60; done
  if [ ! -f runs/kd_bt_all/model/config.json ]; then
    say "GPU3 btrail all-data"
    CUDA_VISIBLE_DEVICES=3 $D --all-data --out runs/kd_bt_all >runs/kd_bt_all.log 2>&1
    say "all-data 완료"
  fi
  tf 3 2 ) &
wait
say "5-fold+all-data 완료 → 70k OOF + bias"
until gpu_free 2; do sleep 60; done
CUDA_VISIBLE_DEVICES=2 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
import json, numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID, SERIALIZE_PRESETS
from ai_challenge.models.common import predict_logits
from ai_challenge.method.tune_threshold import coordinate_ascent
recs = load_records("data/train.jsonl","data/train_labels.csv")
fold = assign_folds(recs, n_splits=5, seed=42); y = np.array([CLASS_TO_ID[r.action] for r in recs])
sk = SERIALIZE_PRESETS["btrail"]
L = np.zeros((len(recs),14), np.float32)
for f in range(5):
    idx = np.where(fold==f)[0]
    L[idx] = predict_logits(f"runs/kd_bts_f{f}/model", [recs[i] for i in idx], max_length=640, serialize_kwargs=sk)
np.save("runs/oof70k_bts.npy", L)
per=[f1_score(y[fold==f], L[fold==f].argmax(1), average="macro") for f in range(5)]
print("bts fold별:", [round(v,5) for v in per], "| 평균:", round(float(np.mean(per)),5), "(tg5s 평균 0.78463)")
tr=fold!=0; te=fold==0
bias,_=coordinate_ascent(L[tr], y[tr])
hb=f1_score(y[te],L[te].argmax(1),average="macro"); ha=f1_score(y[te],(L[te]+bias).argmax(1),average="macro")
print(f"[정직검증] f0: {hb:.5f} -> {ha:.5f} (D{ha-hb:+.5f})")
json.dump({"bias": bias.tolist(), "honest_delta": float(ha-hb)}, open("runs/bias56k_bts.json","w"), indent=1)
PY
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b=json.load(open("runs/bias56k_bts.json"))
dst="runs/kd_btb70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_bt_all/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("btrail all-data + 56k bias 부착")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_btb70/model --name btb70 --serialize btrail --quant fp16 >>"$LOG" 2>&1 && say "build/btb70.zip OK (제출은 아침 승인)"
say "=== std_bt 완료 ==="
