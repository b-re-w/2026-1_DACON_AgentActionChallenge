#!/usr/bin/env bash
# 0.79371 정확재현: 트리오 student fold1~4 학습 → 트리오 70k OOF → bias 정직검증 → trio_all에 부착.
# GPU1/2 선행작업(trio_gem05, soup51) 끝나는 유휴구간 활용. GPU 점유 가드 포함.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/trio_folds.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base --teacher-logits qgptoss qw6 q7bteam"
tf(){ local g=$1 f=$2 o=runs/kd_trio_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g trio fold$f"; CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out $o >$o.log 2>&1
  say "trio_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
say "GPU1/2 선행작업 완료 대기..."
until [ -f runs/kd_trio_gem05/metrics.json ] && [ -f runs/kd_qw6soup_s51/model/config.json ]; do sleep 90; done
( tf 1 1; tf 1 3 ) &
( tf 2 2; tf 2 4 ) &
wait
# 트리오 70k OOF → bias (정직검증 포함)
say "트리오 70k OOF 수집 + bias"
until gpu_free 1; do sleep 60; done
CUDA_VISIBLE_DEVICES=1 uv run python - <<'PY' 2>>"$LOG"
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
    L[idx] = predict_logits(f"runs/kd_trio_f{f}/model", [recs[i] for i in idx], max_length=640, serialize_kwargs={})
np.save("runs/oof70k_trio.npy", L)
tr = fold!=0; te = fold==0
b_h,_ = coordinate_ascent(L[tr], y[tr])
h_b = f1_score(y[te], L[te].argmax(1), average="macro"); h_a = f1_score(y[te], (L[te]+b_h).argmax(1), average="macro")
print(f"[정직검증] fold1-4피팅→fold0: {h_b:.5f} → {h_a:.5f} (Δ{h_a-h_b:+.5f})", flush=True)
base = f1_score(y, L.argmax(1), average="macro")
bias, tuned = coordinate_ascent(L, y)
print(f"[70k전체] {base:.5f} → {tuned:.5f} (Δ{tuned-base:+.5f})", flush=True)
json.dump({"bias": bias.tolist(), "before": float(base), "after": float(tuned),
           "honest_f0_delta": float(h_a-h_b)}, open("runs/bias70k_trio.json","w"), indent=1)
PY
# trio_all + bias 부착 pack
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b = json.load(open("runs/bias70k_trio.json"))
dst="runs/kd_trio_b70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_trio_all/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("trio_all + trio-bias 부착")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_trio_b70/model --name trio_b70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/trio_b70.zip ✅ (0.79371 정확재현판)"
say "=== trio_folds 완료 ==="
