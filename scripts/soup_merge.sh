#!/usr/bin/env bash
# seed42(기존)+43+44 완료 대기 → 가중평균(metadata 포함) → pack. 제출은 사용자 판단.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/soup.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
say "soup seed 43/44 완료 대기..."
until [ -f runs/kd_qw6soup_s43/model/config.json ] && [ -f runs/kd_qw6soup_s44/model/config.json ]; do sleep 120; done
say "merge (seed 42/43/44)"
uv run python - <<'PY' 2>>"$LOG"
import os, glob, shutil
from safetensors.torch import load_file, save_file
paths=["runs/kd_qw6_all640/model/model.safetensors","runs/kd_qw6soup_s43/model/model.safetensors","runs/kd_qw6soup_s44/model/model.safetensors"]
paths=[p for p in paths if os.path.exists(p)]
acc=None
for p in paths:
    sd=load_file(p)
    if acc is None: acc={k:v.float().clone() for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()
for k in acc: acc[k]/=len(paths)
os.makedirs("runs/kd_qw6soup/model",exist_ok=True)
for f in glob.glob("runs/kd_qw6_all640/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_qw6soup/model/")
save_file({k:v.half().contiguous() for k,v in acc.items()},"runs/kd_qw6soup/model/model.safetensors", metadata={"format":"pt"})
print(f"[soup] saved {len(paths)} models 평균")
PY
say "pack soup"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_qw6soup/model --name qw6soup --serialize base --quant fp16 >>"$LOG" 2>&1 \
  && say "=== soup zip 완료: build/qw6soup.zip ===" || say "pack 실패"
