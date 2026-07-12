#!/usr/bin/env bash
# 밤샘: tg5(트리오+gemma) all-data 시드 43/44 학습 → 42(기존)와 3시드 soup + 56k bias → pack (제출은 아침 승인).
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/tg5soup_night.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
# mtfix(교정 제출) 완료 후 시작 (GPU0/1 충돌 방지)
say "mtfix 완료 대기..."
until grep -q "mt_fix 완료" runs/mt_fix.log 2>/dev/null; do sleep 180; done; sleep 30
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base --teacher-logits qgptoss qw6 q7bteam qgemma9b:0.5 --all-data"
tr(){ local g=$1 s=$2 o=runs/kd_tg5all_s$2; [ -f $o/model/config.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g tg5 all-data seed$s"; CUDA_VISIBLE_DEVICES=$g $D --out $o --seed $s >$o.log 2>&1; say "seed$s 완료"; }
( tr 0 43 ) & ( tr 1 44 ) & wait
say "3시드 soup 병합 + 56k bias 부착"
uv run python - <<'PY' 2>>"$LOG"
import os, glob, shutil, json
from safetensors.torch import load_file, save_file
paths=[f"{d}/model/model.safetensors" for d in ["runs/kd_trio_gem05_all","runs/kd_tg5all_s43","runs/kd_tg5all_s44"] if os.path.exists(f"{d}/model/model.safetensors")]
acc=None
for p in paths:
    sd=load_file(p)
    if acc is None: acc={k:v.float().clone() for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()
for k in acc: acc[k]/=len(paths)
os.makedirs("runs/kd_tg5soup/model",exist_ok=True)
for f in glob.glob("runs/kd_trio_gem05_all/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_tg5soup/model/")
save_file({k:v.half().contiguous() for k,v in acc.items()},"runs/kd_tg5soup/model/model.safetensors", metadata={"format":"pt"})
b=json.load(open("runs/bias56k_tg5s.json")) if os.path.exists("runs/bias56k_tg5s.json") else None
if b is None:
    import numpy as np
    from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
    from ai_challenge.method.tune_threshold import coordinate_ascent
    recs=load_records("data/train.jsonl","data/train_labels.csv"); fold=assign_folds(recs,n_splits=5,seed=42)
    y=np.array([CLASS_TO_ID[r.action] for r in recs]); L=np.load("runs/oof70k_tg5s.npy")
    bias,_=coordinate_ascent(L[fold!=0],y[fold!=0]); b={"bias":bias.tolist()}
    json.dump(b,open("runs/bias56k_tg5s.json","w"),indent=1)
json.dump({"bias": b["bias"]}, open("runs/kd_tg5soup/model/bias.json","w"))
print(f"[tg5soup] {len(paths)}시드 평균 + 56k bias")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_tg5soup/model --name tg5soupb70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/tg5soupb70.zip ✅ (아침 승인 대기)"
say "=== tg5soup_night 완료 ==="
