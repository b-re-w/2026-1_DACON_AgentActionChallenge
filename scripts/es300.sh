#!/usr/bin/env bash
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/es300.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss qw6 q7bteam --eval-steps 300"
for f in 0 1 2 3 4; do
  [ -f runs/kd_es3_f$f/metrics.json ] && continue
  say "es300 fold$f"
  CUDA_VISIBLE_DEVICES=0 $D --val-fold $f --out runs/kd_es3_f$f > runs/kd_es3_f$f.log 2>&1
done
say "=== es300 5-fold 완료 ==="
