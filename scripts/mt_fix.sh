#!/usr/bin/env bash
# std_mt 교정: 5-fold 완료 감지 → stdmt 중단(잘못된 구조로 제출 방지) →
# 올바른 구조(= 팀원 신기록 구조)로 재구성: mega_trio ALL-DATA 모델 + 56k-피팅 bias → 제출.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/mt_fix.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "mts 5-fold 완료 대기..."
until [ -f runs/kd_mts_f0/metrics.json ] && [ -f runs/kd_mts_f1/metrics.json ] && [ -f runs/kd_mts_f2/metrics.json ] && [ -f runs/kd_mts_f3/metrics.json ] && [ -f runs/kd_mts_f4/metrics.json ]; do sleep 60; done
say "5-fold 완료 → stdmt 중단 (fold0-모델 구조 제출 방지)"
screen -S stdmt -X quit 2>/dev/null; sleep 5; pkill -f "kd_mtb70" 2>/dev/null; sleep 5
# GPU0: mega_trio ALL-DATA 학습 (올바른 베이스) — GPU1: OOF 수집 병렬
( until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ]; do sleep 60; done
  say "mega_trio all-data 학습 (GPU0)"
  CUDA_VISIBLE_DEVICES=0 uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 \
    --temperature 3 --alpha 0.25 --max-length 640 --serialize base \
    --teacher-logits qgptoss qw6 q7bteam qgemma9b:0.5 qglm32b:0.5 --all-data \
    --out runs/kd_mt_all > runs/kd_mt_all.log 2>&1; say "all-data 완료" ) &
( until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do sleep 60; done
  say "mts 70k OOF 수집 (GPU1)"
  CUDA_VISIBLE_DEVICES=1 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
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
print("mts fold별:", [round(v,5) for v in per], "| 평균:", round(float(np.mean(per)),5))
tr = fold!=0; te = fold==0
bias, _ = coordinate_ascent(L[tr], y[tr])   # 56k 피팅 (팀원 신기록 방식)
h_b = f1_score(y[te], L[te].argmax(1), average="macro"); h_a = f1_score(y[te], (L[te]+bias).argmax(1), average="macro")
print(f"[56k피팅 held-out] f0: {h_b:.5f} → {h_a:.5f} (Δ{h_a-h_b:+.5f})")
json.dump({"bias": bias.tolist(), "honest_delta": float(h_a-h_b)}, open("runs/bias56k_mts.json","w"), indent=1)
PY
  say "OOF+bias 완료" ) &
wait
# 올바른 구조 조립: all-data 모델 + 56k bias → pack → 제출 (사용자 승인분: gem+glm LB 실측)
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b = json.load(open("runs/bias56k_mts.json"))
dst="runs/kd_mtallb70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_mt_all/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("mega_trio all-data + 56k bias 부착")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_mtallb70/model --name mtallb70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/mtallb70.zip ✅"
uv run python -c "import zipfile;z=zipfile.ZipFile('build/mtallb70.zip');assert not z.testzip() and 'model/bias.json' in z.namelist()" >>"$LOG" 2>&1 || { say "zip 검증 실패 — 제출 중단"; exit 1; }
uv run python -m ai_challenge.utils.pack eval --zip build/mtallb70.zip >>"$LOG" 2>&1
say "제출 (사용자 승인분: gem+glm 이질듀얼)"
uv run python -m ai_challenge.utils.submission submit build/mtallb70.zip \
  --memo "(동연) mtallb70 = trio+gem:0.5+GLM:0.5 all-data + 56k피팅 bias (팀원 신기록 구조)" --yes 2>&1 | grep -oiE "isSubmitted[^,}]*|Success" | tee -a "$LOG"
say "=== mt_fix 완료 (제출됨) ==="
