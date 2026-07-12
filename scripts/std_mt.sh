#!/usr/bin/env bash
# 표준 파이프라인 (56k step-best + bias70k) × mega_trio (트리오+gemma:0.5+GLM:0.5 이질듀얼).
# 사용자 승인: gem+glm 은 다른 계열이라 fold0 중첩판정이 부정확할 수 있음 → LB 실측.
# 완료 시 제출까지 (사용자 사전 승인분).
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/std_mt.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
T="qgptoss qw6 q7bteam qgemma9b:0.5 qglm32b:0.5"
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base --eval-steps 500 --teacher-logits $T"
tf(){ local g=$1 f=$2 o=runs/kd_mts_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g mts fold$f"; CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out $o >$o.log 2>&1
  say "mts_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
# GPU2 즉시 시작, GPU0/1은 tg5 시드 끝나면 가드 통과
( tf 2 0; tf 2 3 ) & ( tf 0 1; tf 0 4 ) & ( tf 1 2 ) & wait
say "5-fold 완료 → 70k OOF + bias"
until gpu_free 2; do sleep 60; done
CUDA_VISIBLE_DEVICES=2 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
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
    L[idx] = predict_logits(f"runs/kd_mts_f{f}/model", [recs[i] for i in idx], max_length=640, serialize_kwargs={})
np.save("runs/oof70k_mts.npy", L)
per = [f1_score(y[fold==f], L[fold==f].argmax(1), average="macro") for f in range(5)]
print("fold별:", [round(v,5) for v in per], "| 평균:", round(float(np.mean(per)),5))
tr = fold!=0; te = fold==0
bh,_ = coordinate_ascent(L[tr], y[tr])
print(f"[정직검증] f0: {f1_score(y[te],L[te].argmax(1),average='macro'):.5f} → {f1_score(y[te],(L[te]+bh).argmax(1),average='macro'):.5f}")
base = f1_score(y, L.argmax(1), average="macro")
bias, tuned = coordinate_ascent(L, y)
print(f"[70k전체] {base:.5f} → {tuned:.5f}")
json.dump({"bias": bias.tolist(), "before": float(base), "after": float(tuned)}, open("runs/bias70k_mts.json","w"), indent=1)
PY
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b = json.load(open("runs/bias70k_mts.json"))
dst="runs/kd_mtb70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_mts_f0/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("fold0-best + bias 부착")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_mtb70/model --name mtb70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/mtb70.zip ✅"
# 검증 후 제출 (사용자 사전 승인)
uv run python -m ai_challenge.utils.pack eval --zip build/mtb70.zip >>"$LOG" 2>&1
uv run python -c "import zipfile;z=zipfile.ZipFile('build/mtb70.zip');assert not z.testzip() and 'model/bias.json' in z.namelist();print('zip+bias 검증 OK')" >>"$LOG" 2>&1 || { say "zip 검증 실패 — 제출 중단"; exit 1; }
say "제출 (사용자 승인분)"
uv run python -m ai_challenge.utils.submission submit build/mtb70.zip \
  --memo "(동연) mtb70 = trio+gemma:0.5+GLM:0.5 이질듀얼, 56k step-best + 70k bias" --yes 2>&1 | grep -oiE "isSubmitted[^,}]*|Success" | tee -a "$LOG"
say "=== std_mt 완료 (제출됨) ==="
