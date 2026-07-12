#!/usr/bin/env bash
# 42-가중 soup (사용자 관찰: seed42가 LB 유리 가능성): 42×2 + 43~46 각1 가중평균 + bias.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/soup5w42.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
until [ -f build/tg5soup5b70.zip ]; do sleep 120; done   # 동등판 완료 후
uv run python - <<'PY' 2>>"$LOG"
import os, glob, shutil, json
from safetensors.torch import load_file, save_file
w = {"runs/kd_trio_gem05_all": 2.0}   # seed42 두 배
for s in (43,44,45,46): w[f"runs/kd_tg5all_s{s}"] = 1.0
acc=None; tot=0.0
for d, wt in w.items():
    p=f"{d}/model/model.safetensors"
    if not os.path.exists(p): continue
    sd=load_file(p); tot+=wt
    if acc is None: acc={k:v.float()*wt for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()*wt
for k in acc: acc[k]/=tot
os.makedirs("runs/kd_tg5soup5w42/model",exist_ok=True)
for f in glob.glob("runs/kd_trio_gem05_all/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_tg5soup5w42/model/")
save_file({k:v.half().contiguous() for k,v in acc.items()},"runs/kd_tg5soup5w42/model/model.safetensors", metadata={"format":"pt"})
b=json.load(open("runs/bias56k_tg5s.json"))
json.dump({"bias": b["bias"]}, open("runs/kd_tg5soup5w42/model/bias.json","w"))
print(f"[soup5w42] 42x2 가중 soup (총가중 {tot})")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_tg5soup5w42/model --name tg5soup5w42 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/tg5soup5w42.zip ✅"
say "=== soup5w42 완료 ==="
