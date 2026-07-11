#!/usr/bin/env bash
# 팀원 0.79371 방법 재현 (우리 재료):
#  A) qw6 레시피 student를 fold1~4 학습(fold0=kd_qw6_t3a25 기존) → 70k OOF → bias 피팅
#     → qw6_all640/soup9 에 bias.json 부착 → pack (제출후보 2종)
#  B) 트리오(qgptoss+qw6+q7bteam 1:1:1) all-data KD → pack (그들 kd3 base 재현)
# 제출은 사용자 판단.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/bias70k_ours.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"

# Wave1: qw6 fold 1/2/3 (GPU 0/1/2)
say "Wave1: qw6 fold1/2/3"
for f in 1 2 3; do
  g=$((f-1))
  [ -f runs/kd_qw6_f$f/metrics.json ] || CUDA_VISIBLE_DEVICES=$g $D --teacher-logits qw6 --val-fold $f --out runs/kd_qw6_f$f > runs/kd_qw6_f$f.log 2>&1 &
done
wait; say "Wave1 완료"

# Wave2: fold4(GPU0) + 트리오 all-data(GPU1) + 트리오 fold0(GPU2, 트리오 bias용 OOF 시작점)
say "Wave2: fold4 + trio"
( [ -f runs/kd_qw6_f4/metrics.json ] || CUDA_VISIBLE_DEVICES=0 $D --teacher-logits qw6 --val-fold 4 --out runs/kd_qw6_f4 > runs/kd_qw6_f4.log 2>&1 ) &
( [ -f runs/kd_trio_all/model/config.json ] || CUDA_VISIBLE_DEVICES=1 $D --teacher-logits qgptoss qw6 q7bteam --all-data --out runs/kd_trio_all > runs/kd_trio_all.log 2>&1 ) &
( [ -f runs/kd_trio_f0/metrics.json ] || CUDA_VISIBLE_DEVICES=2 $D --teacher-logits qgptoss qw6 q7bteam --val-fold 0 --out runs/kd_trio_f0 > runs/kd_trio_f0.log 2>&1 ) &
wait; say "Wave2 완료"
for n in kd_qw6_f1 kd_qw6_f2 kd_qw6_f3 kd_qw6_f4 kd_trio_f0; do
  say "  $n OOF=$(grep -oE '"macro_f1": [0-9.]+' runs/$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; done

# 3) 70k OOF logits 수집 → bias 피팅 (GPU0)
say "70k OOF 수집 + bias 피팅"
CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY' 2>>"$LOG"
import json, numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import predict_logits
from ai_challenge.method.tune_threshold import coordinate_ascent
recs = load_records("data/train.jsonl","data/train_labels.csv")
fold = assign_folds(recs, n_splits=5, seed=42)
y_all = np.array([CLASS_TO_ID[r.action] for r in recs])
L = np.zeros((len(recs),14), np.float32)
runs = {0:"runs/kd_qw6_t3a25", 1:"runs/kd_qw6_f1", 2:"runs/kd_qw6_f2", 3:"runs/kd_qw6_f3", 4:"runs/kd_qw6_f4"}
for f, rd in runs.items():
    idx = np.where(fold==f)[0]
    smp = [recs[i] for i in idx]
    L[idx] = predict_logits(f"{rd}/model", smp, max_length=640, serialize_kwargs={})
    print(f"fold{f} OOF F1 = {f1_score(y_all[idx], L[idx].argmax(1), average='macro'):.5f}", flush=True)
base = f1_score(y_all, L.argmax(1), average="macro")
bias, tuned = coordinate_ascent(L, y_all)
print(f"★ 70k 전체 OOF: {base:.5f} → +bias {tuned:.5f} (Δ{tuned-base:+.5f})")
json.dump({"bias": bias.tolist(), "before": float(base), "after": float(tuned)}, open("runs/bias70k_qw6.json","w"), indent=1)
np.save("runs/oof70k_qw6.npy", L)
PY

# 4) bias 부착 pack 2종 (qw6_all640+bias, soup9+bias)
say "pack: qw6_all640+bias / soup9+bias"
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b = json.load(open("runs/bias70k_qw6.json"))
for src, dst in [("runs/kd_qw6_all640/model","runs/kd_qw6b70/model"),("runs/kd_qw6soup9/model","runs/kd_soup9b70/model")]:
    if os.path.exists(dst): shutil.rmtree(dst)
    shutil.copytree(src, dst)
    json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("bias.json 부착 완료")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_qw6b70/model  --name qw6b70  --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/qw6b70.zip ✅"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_soup9b70/model --name soup9b70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/soup9b70.zip ✅"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_trio_all/model --name trio_all --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/trio_all.zip ✅"
say "=== bias70k_ours 완료 — 제출후보: qw6b70 / soup9b70 / trio_all (사용자 판단) ==="
