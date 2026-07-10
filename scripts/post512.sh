#!/usr/bin/env bash
# gpt-oss@512 완료 후 순차 실행: (1) oss+gpt5 조합 스윕 @640 (stores 준비됨) → (2) t3b_640 학습.
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
say "--- oss+gpt5 조합 요약 (기준: qw6=0.7847, w6r_oss20=0.7847, w6r_g20=0.7816) ---"
for n in w6r_both w6r_g20oss1 w6r_oss2g1; do echo "  $n: $(grep -oE '\"macro_f1\": [0-9.]+' runs/kd_$n/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"|tee -a "$LOG"; done

# 2) t3b_640 학습 (님 가설: teacher 640 학습이 512보다 나은가)
if [ ! -f runs/t3b_640/model/config.json ]; then
  say "t3b_640 학습 (Qwen2.5-3B full-FT 640 seed42)"
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.train --model Qwen/Qwen2.5-3B \
    --serialize base --max-length 640 --lr 2e-5 --epochs 3 --batch-size 16 --grad-accum 2 \
    --grad-checkpoint --seed 42 --bf16 --out runs/t3b_640 >runs/t3b_640.log 2>&1 || { say "t3b_640 실패"; exit 1; }
fi
[ -f runs/_teacher_logits/q3b640.npz ] || CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.gen_softlabels \
  --teacher-dir runs/t3b_640/model --tag q3b640 --serialize base --max-length 640 >>"$LOG" 2>&1
st(){ local o=runs/kd_st_$1; [ -f $o/metrics.json ]||CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits $1 --student Qwen/Qwen2.5-0.5B --out $o --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base >$o.log 2>&1; grep -oE '"macro_f1": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+'; }
say "★ teacher 640학습 → single student = $(st q3b640)  |  512학습(t3b_base) → single student = $(st q3bb5)"
say "=== post512 완료 (제출 없음) ==="
