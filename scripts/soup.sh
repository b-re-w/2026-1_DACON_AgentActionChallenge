#!/usr/bin/env bash
# model soup: qw6_all640(seed42, 기존) + seed43/44 학습 → 가중평균 → LB 후보(variance 감소).
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
LOG=runs/soup.log; say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
T="qw6"  # 승리 레시피 (qw6_all640과 동일 타깃)
tr(){ local g=$1 s=$2 o=runs/kd_qw6soup_s$s; [ -f $o/model/config.json ] && { say "s$s 있음"; return; }
  say "GPU$g qw6 all-data seed$s"; CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.distill \
    --teacher-logits $T --student Qwen/Qwen2.5-0.5B --out $o --all-data --bf16 --temperature 3 --alpha 0.25 \
    --max-length 640 --serialize base --seed $s >$o.log 2>&1; say "s$s 완료"; }
( tr 1 43 ) & ( tr 2 44 ) & wait
# soup: seed42(기존 qw6_all640) + 43 + 44 가중평균
say "soup 병합 (seed 42/43/44)"
uv run python - <<'PY' 2>>"$LOG"
import torch, glob, os
from safetensors.torch import load_file, save_file
paths=["runs/kd_qw6_all640/model/model.safetensors",
       "runs/kd_qw6soup_s43/model/model.safetensors",
       "runs/kd_qw6soup_s44/model/model.safetensors"]
paths=[p for p in paths if os.path.exists(p)]
print("soup inputs:",len(paths))
acc=None
for p in paths:
    sd=load_file(p)
    if acc is None: acc={k:v.float().clone() for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()
for k in acc: acc[k]/=len(paths)
import shutil
os.makedirs("runs/kd_qw6soup/model",exist_ok=True)
for f in glob.glob("runs/kd_qw6_all640/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_qw6soup/model/")
save_file({k:v.half() for k,v in acc.items()},"runs/kd_qw6soup/model/model.safetensors")
print("soup saved: runs/kd_qw6soup/model")
PY
say "pack soup"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_qw6soup/model --name qw6soup --serialize base --quant fp16 >>"$LOG" 2>&1
say "=== soup 완료: build/qw6soup.zip (제출 대기) ==="
