#!/usr/bin/env bash
# teacher 가중 변형 6종 (btrail): f0 스크리닝 → f0가 기준-0.001 이상인 것만 f2 페어 확정.
# 변형: oss 1.25/2.0, 7B 1.5/0.5, 코어(qw6) 1.5, GPT5 내부×3(풀어쓴 qw6)
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
  say "GPU$g $n f$f :: ${W[$n]}"
  CUDA_VISIBLE_DEVICES=$g $D --teacher-logits ${W[$n]} --val-fold $f --out $o >$o.log 2>&1
  say "${n}_f$f OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')"; }
# seedrop 완료 후 시작 (GPU 충돌 방지)
until [ -f runs/kd_sd_f0/metrics.json ] && [ -f runs/kd_sd_f2/metrics.json ]; do sleep 180; done
say "=== f0 스크리닝 6종 ==="
( r 1 w_oss125 0; r 1 w_7b05 0 ) &
( r 2 w_oss20 0;  r 2 w_core15 0 ) &
( r 3 w_7b15 0;   r 3 w_gpt53 0 ) &
wait
say "--- f0 스크리닝 결과 (기준 btrail f0=0.78669, 컷 0.78569) ---"
PASS=""
for n in "${!W[@]}"; do
  v=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f0/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+' || echo 0)
  echo "  $n: $v"|tee -a "$LOG"
  ok=$(python3 -c "print(1 if $v >= 0.78569 else 0)"); [ "$ok" = "1" ] && PASS="$PASS $n"
done
say "f2 확정 대상:$PASS"
i=0
for n in $PASS; do
  g=$((i % 3 + 1)); ( r $g $n 2 ) & i=$((i+1))
done
wait
say "=== weights_lab 완료 ==="
for n in $PASS; do
  a=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f0/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')
  b=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f2/metrics.json 2>/dev/null|grep -oE '0\.[0-9]+')
  echo "  $n: f0=$a f2=$b (기준 0.78669/0.79367)"|tee -a "$LOG"
done
