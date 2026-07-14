#!/usr/bin/env bash
# es300 5-fold 완주 → OOF 수집 → bagged(x1.0) bias 재피팅 → 정직검증 → v5 조립 (제출은 사용자 판단)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/es300_finish.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
until [ -f runs/kd_es3_f4/metrics.json ]; do sleep 180; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ]; do sleep 60; done
say "es300 완주 → OOF 수집"
CUDA_VISIBLE_DEVICES=0 uv run python - <<'PY' 2>>"$LOG" | tee -a "$LOG"
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
    L[idx] = predict_logits(f"runs/kd_es3_f{f}/model", [recs[i] for i in idx], max_length=640, serialize_kwargs=sk)
np.save("runs/oof70k_es3.npy", L)
per=[f1_score(y[fold==f], L[fold==f].argmax(1), average="macro") for f in range(5)]
print("es300 fold별:", [round(v,5) for v in per], "| 평균:", round(float(np.mean(per)),5), "(es500 평균 0.78730)")
# bagged(x1.0) bias — es300 OOF 재료
tr=fold!=0; te=fold==0
lofo=[coordinate_ascent(L[np.isin(fold,[x for x in range(5) if x not in (0,fo)])],y[np.isin(fold,[x for x in range(5) if x not in (0,fo)])])[0] for fo in [1,2,3,4]]
bh=np.mean(lofo,axis=0)
hb=f1_score(y[te],L[te].argmax(1),average="macro"); ha=f1_score(y[te],(L[te]+bh).argmax(1),average="macro")
print(f"[정직검증-es300재료] f0: {hb:.5f} -> {ha:.5f} (D{ha-hb:+.5f}) [es500재료땐 +0.00206]")
bd=[coordinate_ascent(L[fold!=f],y[fold!=f])[0] for f in range(5)]
json.dump({"bias": np.mean(bd,axis=0).tolist()}, open("runs/bias_es3_deploy.json","w"), indent=1)
PY
say "v5 조립 (bt_all + es300-OOF bagged bias)"
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b=json.load(open("runs/bias_es3_deploy.json"))
dst="runs/kd_btb70v5/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_bt_all/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("v5 조립")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_btb70v5/model --name btb70v5 --serialize btrail --quant fp16 >>"$LOG" 2>&1 && say "build/btb70v5.zip 준비 (제출은 사용자 판단)"
say "=== es300_finish 완료 ==="
