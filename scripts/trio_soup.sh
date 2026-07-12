#!/usr/bin/env bash
# trio_soup: 트리오 all-data 시드 43/44 학습 → 42(기존 trio_all)와 3시드 평균 → pack.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/trio_soup.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base --teacher-logits qgptoss qw6 q7bteam --all-data"
tr(){ local g=$1 s=$2 o=runs/kd_trio_s$2; [ -f $o/model/config.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g trio seed$s"; CUDA_VISIBLE_DEVICES=$g $D --out $o --seed $s >$o.log 2>&1; say "seed$s 완료"; }
( tr 1 43 ) & ( tr 2 44 ) & wait
say "3시드 병합 (42=trio_all, 43, 44)"
uv run python - <<'PY' 2>>"$LOG"
import os, glob, shutil
from safetensors.torch import load_file, save_file
paths=[f"{d}/model/model.safetensors" for d in ["runs/kd_trio_all","runs/kd_trio_s43","runs/kd_trio_s44"] if os.path.exists(f"{d}/model/model.safetensors")]
acc=None
for p in paths:
    sd=load_file(p)
    if acc is None: acc={k:v.float().clone() for k,v in sd.items()}
    else:
        for k in acc: acc[k]+=sd[k].float()
for k in acc: acc[k]/=len(paths)
os.makedirs("runs/kd_trio_soup/model",exist_ok=True)
for f in glob.glob("runs/kd_trio_all/model/*"):
    if not f.endswith(".safetensors"): shutil.copy(f,"runs/kd_trio_soup/model/")
save_file({k:v.half().contiguous() for k,v in acc.items()},"runs/kd_trio_soup/model/model.safetensors", metadata={"format":"pt"})
print(f"[trio_soup] {len(paths)}시드 평균")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_trio_soup/model --name trio_soup --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/trio_soup.zip ✅"
say "=== trio_soup 완료 ==="
