#!/usr/bin/env bash
# teacher 가중 변형 6종 (btrail) — 전부 f0+f2 페어 (사용자 지시: 스크리닝 없이)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/weights_lab.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail"
declare -A W=(
  [w_oss125]="qgptoss:1.25 qw6 q7bteam"
  [w_oss20]="qgptoss:2 qw6 q7bteam"
  [w_7b15]="qgptoss qw6 q7bteam:1.5"
  [w_7b05]="qgptoss qw6 q7bteam:0.5"
  [w_core15]="qgptoss qw6:1.5 q7bteam"
  [w_gpt53]="qgptoss q3bb q3b43 q3b44 q3b45 qgpt5:3 q7bteam"
)
r(){ local g=$1 n=$2 f=$3; local o=runs/kd_${n}_f$3; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g $n f$f"
  CUDA_VISIBLE_DEVICES=$g $D --teacher-logits ${W[$n]} --val-fold $f --out $o >$o.log 2>&1
  say "${n}_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"; }
# seedrop 완료 후 시작
until [ -f runs/kd_sd_f0/metrics.json ] && [ -f runs/kd_sd_f2/metrics.json ]; do sleep 180; done
say "=== 전 변형 f0+f2 페어 시작 ==="
( r 1 w_oss125 0; r 1 w_oss125 2; r 1 w_7b05 0; r 1 w_7b05 2 ) &
( r 2 w_oss20 0;  r 2 w_oss20 2;  r 2 w_core15 0; r 2 w_core15 2 ) &
( r 3 w_7b15 0;   r 3 w_7b15 2;   r 3 w_gpt53 0;  r 3 w_gpt53 2 ) &
wait
say "=== weights_lab 완료 (기준 f0=0.78669 f2=0.79367 mean=0.79018) ==="
for n in "${!W[@]}"; do
  a=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f0/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')
  b=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f2/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')
  echo "  $n: f0=$a f2=$b"|tee -a "$LOG"
done
