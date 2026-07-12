#!/usr/bin/env bash
# 9시 전 GPU 유휴 제로 캠페인:
#  A(01:50~): 트리오 KD 하이퍼 재튜닝 (T4/T2/α0.15/Instruct-student) — 전부 f0+f2 페어
#  B(72B 스크리닝 후): 승자 fold2 페어 + all-data + bias 조립 → 제출후보 zip
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/morning.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
TRIO="qgptoss qw6 q7bteam"
base="uv run python -m ai_challenge.method.distill --bf16 --max-length 640 --serialize base --teacher-logits $TRIO"
r(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $base "$@" --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }

say "=== Phase A: 트리오 하이퍼 재튜닝 (시드45/46 후 GPU1/2) ==="
( r 1 3T4_f0 --student Qwen/Qwen2.5-0.5B --temperature 4 --alpha 0.25 --val-fold 0
  r 1 3T4_f2 --student Qwen/Qwen2.5-0.5B --temperature 4 --alpha 0.25 --val-fold 2
  r 1 3T2_f0 --student Qwen/Qwen2.5-0.5B --temperature 2 --alpha 0.25 --val-fold 0
  r 1 3T2_f2 --student Qwen/Qwen2.5-0.5B --temperature 2 --alpha 0.25 --val-fold 2 ) &
( r 2 3a15_f0 --student Qwen/Qwen2.5-0.5B --temperature 3 --alpha 0.15 --val-fold 0
  r 2 3a15_f2 --student Qwen/Qwen2.5-0.5B --temperature 3 --alpha 0.15 --val-fold 2
  r 2 3sI_f0  --student Qwen/Qwen2.5-0.5B-Instruct --temperature 3 --alpha 0.25 --val-fold 0
  r 2 3sI_f2  --student Qwen/Qwen2.5-0.5B-Instruct --temperature 3 --alpha 0.25 --val-fold 2 ) &
wait
say "--- Phase A 결과 (기준 trio T3a25: f0=0.78205 f2=0.79053 mean=0.78629) ---"
for n in 3T4 3T2 3a15 3sI; do
  a=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+')
  b=$(grep -oE '"macro_f1": [0-9.]+' runs/kd_${n}_f2/metrics.json 2>/dev/null|grep -oE '[0-9.]+')
  echo "  $n: f0=$a f2=$b"|tee -a "$LOG"; done

say "=== Phase B: 72B 스크리닝 완료 대기 ==="
until grep -q "chain_72b 완료" runs/chain_72b.log 2>/dev/null; do sleep 180; done
# 승자 판정 (fold0 기준 상위 2)
uv run python - <<'PY' > runs/b_winners.txt 2>>"$LOG"
import json
cands={}
for n in ["tri72_swap","tri72_add","tri72_hv"]:
    try: cands[n]=json.load(open(f"runs/kd_{n}/metrics.json"))["macro_f1"]
    except: pass
top=sorted(cands.items(), key=lambda kv:-kv[1])
print("\n".join(f"{k} {v:.5f}" for k,v in top))
PY
say "72B 스크리닝 순위: $(tr '\n' ' ' < runs/b_winners.txt)"
W1=$(head -1 runs/b_winners.txt | cut -d' ' -f1); W2=$(sed -n 2p runs/b_winners.txt | cut -d' ' -f1)
tset(){ case $1 in tri72_swap) echo "qgptoss qw6 q72b";; tri72_add) echo "qgptoss qw6 q7bteam q72b";; tri72_hv) echo "qgptoss qw6 q72b:2";; esac; }
# 승자 f2 페어 (GPU1/2) + 1위 all-data (GPU3)
( [ -n "$W1" ] && r 1 ${W1}_f2 --student Qwen/Qwen2.5-0.5B --temperature 3 --alpha 0.25 --val-fold 2 --teacher-logits $(tset $W1) ) &
( [ -n "$W2" ] && r 2 ${W2}_f2 --student Qwen/Qwen2.5-0.5B --temperature 3 --alpha 0.25 --val-fold 2 --teacher-logits $(tset $W2) ) &
( [ -n "$W1" ] && { until gpu_free 3; do sleep 90; done
    say "GPU3 ${W1} all-data"
    CUDA_VISIBLE_DEVICES=3 uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 \
      --temperature 3 --alpha 0.25 --max-length 640 --serialize base --teacher-logits $(tset $W1) --all-data \
      --out runs/kd_${W1}_all >runs/kd_${W1}_all.log 2>&1; } ) &
wait
# 1위 all-data + bias(tg5s 56k 근사) → 제출후보 zip (제출 안 함)
if [ -n "$W1" ] && [ -f runs/kd_${W1}_all/model/config.json ]; then
  uv run python - <<PY 2>>"$LOG"
import json, shutil, os
b=json.load(open("runs/bias56k_tg5s.json"))
dst="runs/kd_${W1}_b70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_${W1}_all/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("${W1} all-data + bias 부착")
PY
  uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_${W1}_b70/model --name ${W1}b70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/${W1}b70.zip ✅ (제출후보)"
fi
say "=== morning_campaign 완료 ==="
