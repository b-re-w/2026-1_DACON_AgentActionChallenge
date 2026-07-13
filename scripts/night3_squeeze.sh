#!/usr/bin/env bash
# 밤 3차 (조건부 짜내기): night2 결과를 읽고 이긴 가지만 심화.
#  분기1: btrail8 승 → k=10 페어 + bt8 all-data 준비
#  분기2: bthist 승 → bthist+paths 페어
#  분기3: 둘 다 패 → btrail(원본) 시드45 all-data (티켓 보강)
#  (TF-IDF×bts, Phi-4는 별도 워처)
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/night3.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
say "night2 완료 대기..."
until grep -q "night2 완료" runs/night2.log 2>/dev/null; do sleep 180; done
m(){ grep -oE '"macro_f1": [0-9.]+' runs/kd_$1/metrics.json 2>/dev/null | grep -oE '[0-9.]+' || echo 0; }
BT_F0=0.78669; BT_F2=0.79367   # btrail 기준
bt8_mean=$(python3 -c "print((($(m bt8_f0))+($(m bt8_f2)))/2)")
bth_mean=$(python3 -c "print((($(m bth_f0))+($(m bth_f2)))/2)")
base_mean=$(python3 -c "print(($BT_F0+$BT_F2)/2)")
say "2fold평균: btrail=$base_mean bt8=$bt8_mean bthist=$bth_mean"
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --teacher-logits qgptoss qw6 q7bteam"
r(){ local g=$1 n=$2 ser=$3; shift 3; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 60; done
  say "GPU$g $n ($ser)"; CUDA_VISIBLE_DEVICES=$g $D --serialize $ser "$@" --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
win8=$(python3 -c "print(1 if $bt8_mean > $base_mean else 0)")
winh=$(python3 -c "print(1 if $bth_mean > $base_mean else 0)")
if [ "$win8" = "1" ]; then
  say "분기1: btrail8 승 → k10 페어 + bt8 all-data"
  # btrail10 프리셋 즉석 추가 (있으면 스킵)
  uv run python - <<'PY'
p="ai_challenge/datasets/serialize.py"; s=open(p).read()
if '"btrail10"' not in s:
    s=s.replace('    "btrail8": {"include_trail": True, "trail_k": 8},',
                '    "btrail8": {"include_trail": True, "trail_k": 8},\n    "btrail10": {"include_trail": True, "trail_k": 10},')
    open(p,"w").write(s); print("btrail10 추가")
PY
  ( r 1 bt10_f0 btrail10 --val-fold 0 ; r 1 bt10_f2 btrail10 --val-fold 2 ) &
  ( r 2 bt8_all btrail8 --all-data ) &
fi
if [ "$winh" = "1" ]; then
  say "분기2: bthist 승 → bthist all-data 준비 + f4 3fold 확정"
  ( r 3 bth_f4 bthist --val-fold 4 ) &
fi
if [ "$win8" = "0" ] && [ "$winh" = "0" ]; then
  say "분기3: 변형 전패 → btrail 원본 유지, 시드45 티켓 보강"
  ( r 1 bt_s45 btrail --all-data --seed 45 ) &
fi
wait
say "=== night3 완료 ==="
