#!/usr/bin/env bash
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
G=1; ML=640; LOG=runs/w6recon.log
say(){ echo "[$(date '+%H:%M')] $*"|tee -a "$LOG"; }
# 1) 개별 3B teacher 640 store (full 모델이라 4bit 불필요). GPT5=qgpt5, oss=qgptoss 이미 있음.
declare -A M=( [q3bb]=t3b_base [q3b43]=t3b_s43 [q3b44]=t3b_s44 [q3b45]=t3b_s45 )
for tag in "${!M[@]}"; do
  [ -f runs/_teacher_logits/$tag.npz ] && { say "$tag store 있음"; continue; }
  say "$tag store 생성 (${M[$tag]})"
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.gen_softlabels \
    --teacher-dir runs/${M[$tag]}/model --tag $tag --serialize base --max-length $ML >>"$LOG" 2>&1
done
# 2) 재구성 + 다양성 스윕 (fold0). 기준 control=0.7847
Q="q3bb q3b43 q3b44 q3b45"
run(){ local n=$1 t=$2; local o=runs/kd_$n; [ -f $o/metrics.json ] && { say "$n 있음"; return; }
  say "$n :: $t"; CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill \
    --teacher-logits $t --student Qwen/Qwen2.5-0.5B --out $o --bf16 --temperature 3 --alpha 0.25 \
    --max-length $ML --serialize base >"$o.log" 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9]+\.[0-9]+' $o/metrics.json|grep -oE '[0-9]+\.[0-9]+')"; }
run w6r_g20     "$Q qgpt5:2"            # w6 재현(GPT5x2) — 검증 ~0.7847 기대
run w6r_oss20   "$Q qgptoss:2"          # ★ GPT5 대신 gpt-oss (님 질문)
run w6r_both    "$Q qgpt5:1 qgptoss:1"  # ★ GPT5+gpt-oss 둘 다
run w6r_g20oss1 "$Q qgpt5:2 qgptoss:1"  # ★ w6 + gpt-oss 추가
run w6r_g25     "$Q qgpt5:2.5"          # GPT5 가중 스윕
run w6r_g15     "$Q qgpt5:1.5"
say "=== w6recon 완료 ==="
grep -oE "kd_w6r_[a-z0-9]+ OOF=[0-9.]+" "$LOG"|tail -8
