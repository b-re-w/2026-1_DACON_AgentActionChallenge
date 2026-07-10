#!/usr/bin/env bash
# gpt-oss@512 완료 후: oss+gpt5 조합 스윕 + qw6+q14b 가중 스윕 @640 (stores 준비됨).
# 전부 제출 없음 — OOF만.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
G=1; LOG=runs/post512.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
# 0) gpt-oss@512 fold0 완료 대기
say "gpt-oss@512 완료 대기..."
until [ -f runs/kd_oss512_f0/metrics.json ] || ! screen -ls 2>/dev/null|grep -q "\.oss512"; do sleep 90; done
sleep 20
say "gpt-oss@512 OOF = $(grep -oE '"macro_f1": [0-9.]+' runs/kd_oss512_f0/metrics.json 2>/dev/null|grep -oE '[0-9.]+' || echo NA)"

# 1) oss+gpt5 조합 스윕 @640 (개별 store 재조합, gen 불필요)
dist(){ local n=$1 t=$2 o=runs/kd_$1; [ -f $o/metrics.json ] && { say "$n 있음=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')"; return; }
  say "$n :: $t"; CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits $t \
    --student Qwen/Qwen2.5-0.5B --out $o --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base >$o.log 2>&1
  say "$n OOF=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')"; }
Q="q3bb q3b43 q3b44 q3b45"
dist w6r_both    "$Q qgpt5:1 qgptoss:1"    # 4×3B + GPT5 + oss (동등)
dist w6r_g20oss1 "$Q qgpt5:2 qgptoss:1"    # 4×3B + GPT5×2 + oss
dist w6r_oss2g1  "$Q qgptoss:2 qgpt5:1"    # 4×3B + oss×2 + GPT5 (oss 우위 반영)
# 14B 가중 스윕 (승리레시피 qw6 + q14b, 14B 저가중부터). 뭉친 qw6(512) + q14b(640) id-평균.
dist w6_14b_51   "qw6:5 q14b:1"            # 14B 매우 저가중 (순수 qw6에 근접)
dist w6_14b_31   "qw6:3 q14b:1"            # 14B 저가중
dist w6_14b_11   "qw6:1 q14b:1"            # 동등 (희석 예상 확인)
say "--- oss+gpt5·14B 조합 요약 (기준: qw6=0.7847, w6r_oss20=0.7847, w6r_g20=0.7816) ---"
for n in w6r_both w6r_g20oss1 w6r_oss2g1 w6_14b_51 w6_14b_31 w6_14b_11; do echo "  $n: $(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"|tee -a "$LOG"; done
say "=== post512 완료 (제출 없음) ==="
