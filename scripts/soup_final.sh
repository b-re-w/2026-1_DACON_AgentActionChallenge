#!/usr/bin/env bash
# seed 45/46/47 완료 대기 → seed 42~47(6개) 가중평균 → pack. 제출은 사용자 판단.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/soup_final.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
say "seed 45/46/47 완료 대기..."
until [ -f runs/kd_qw6soup_s45/model/config.json ] && [ -f runs/kd_qw6soup_s46/model/config.json ] && [ -f runs/kd_qw6soup_s47/model/config.json ]; do sleep 120; done
say "6시드 병합 (42~47)"
uv run python - <<'PY' 2>>"$LOG"
import os, glob, shutil
from safetensors.torch import load_file, save_file
seeds=["runs/kd_qw6_all640","runs/kd_qw6soup_s43","runs/kd_qw6soup_s44","runs/kd_qw6soup_s45","runs/kd_qw6soup_s46","runs/kd_qw6soup_s47"]
paths=[f"{d}/model/model.safetensors" for d in seeds if os.path.exists(f"{d}/model/model.safetensors")]
acc=None
for p in paths:
    sd=load_file(p)
    if acc is None: acc={k:v.float().clone() for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()
for k in acc: acc[k]/=len(paths)
os.makedirs("runs/kd_qw6soup6/model",exist_ok=True)
for f in glob.glob("runs/kd_qw6_all640/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_qw6soup6/model/")
save_file({k:v.half().contiguous() for k,v in acc.items()},"runs/kd_qw6soup6/model/model.safetensors", metadata={"format":"pt"})
print(f"[soup6] {len(paths)}시드 평균 저장")
PY
say "pack soup6"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_qw6soup6/model --name qw6soup6 --serialize base --quant fp16 >>"$LOG" 2>&1 \
  && say "=== soup6 zip 완료: build/qw6soup6.zip ===" || say "pack 실패"
