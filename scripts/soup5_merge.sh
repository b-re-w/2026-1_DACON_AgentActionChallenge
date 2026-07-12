#!/usr/bin/env bash
# 시드 45/46 완료 대기 → 5시드(42~46) soup + 56k bias → tg5soup5b70.zip (제출은 아침 승인)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/soup5.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
until [ -f runs/kd_tg5all_s45/model/config.json ] && [ -f runs/kd_tg5all_s46/model/config.json ]; do sleep 120; done
uv run python - <<'PY' 2>>"$LOG"
import os, glob, shutil, json
from safetensors.torch import load_file, save_file
seeds=["runs/kd_trio_gem05_all"]+[f"runs/kd_tg5all_s{s}" for s in (43,44,45,46)]
paths=[f"{d}/model/model.safetensors" for d in seeds if os.path.exists(f"{d}/model/model.safetensors")]
acc=None
for p in paths:
    sd=load_file(p)
    if acc is None: acc={k:v.float().clone() for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()
for k in acc: acc[k]/=len(paths)
os.makedirs("runs/kd_tg5soup5/model",exist_ok=True)
for f in glob.glob("runs/kd_trio_gem05_all/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_tg5soup5/model/")
save_file({k:v.half().contiguous() for k,v in acc.items()},"runs/kd_tg5soup5/model/model.safetensors", metadata={"format":"pt"})
b=json.load(open("runs/bias56k_tg5s.json"))
json.dump({"bias": b["bias"]}, open("runs/kd_tg5soup5/model/bias.json","w"))
print(f"[soup5] {len(paths)}시드 평균 + bias")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_tg5soup5/model --name tg5soup5b70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/tg5soup5b70.zip ✅ (아침 승인 대기)"
say "=== soup5 완료 ==="
