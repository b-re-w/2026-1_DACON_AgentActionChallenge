#!/usr/bin/env bash
# oss:1.25 확장심사: f1 + f4 추가 → 유효 3-fold(f1,f2,f4) 평균으로 최종 판정
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/oss125_ext.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss:1.25 qw6 q7bteam"
r(){ local g=$1 f=$2; local o=runs/kd_w_oss125_f$2; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g oss125 f$f"; CUDA_VISIBLE_DEVICES=$g $D --val-fold $f --out $o >$o.log 2>&1
  say "oss125_f$f=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"; }
( r 1 1 ) & ( r 2 4 ) & wait
say "=== oss125 확장 완료 — 유효3fold 평균 계산 ==="
uv run python - <<'PY' | tee -a "$LOG"
import json
base = (0.78717 + 0.79367 + 0.78788) / 3   # 기준 트리오 f1/f2/f4
try:
    v = [json.load(open(f"runs/kd_w_oss125_f{f}/metrics.json"))["macro_f1"] for f in (1,2,4)]
    m = sum(v)/3
    print(f"oss125 유효3fold: f1={v[0]:.5f} f2={v[1]:.5f} f4={v[2]:.5f} 평균={m:.5f} (기준 {base:.5f}, D{m-base:+.5f})")
except Exception as e: print("미완:", e)
PY
