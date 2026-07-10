#!/usr/bin/env bash
# 사용자 가설 검증: teacher를 "처음부터 640으로 학습"하면 (재추론 아님) student가 오르나?
# 3B teacher를 640 full-FT → single-teacher student 비교 (기존 512 t3b_base vs 신규 640).
# 제출 없음 — OOF만.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
G=1; LOG=runs/t3b640_test.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
# 0) GPU1 free 대기 (gpt-oss@512 완료)
say "GPU1 free 대기 (gpt-oss@512 완료)..."
until [ -f runs/kd_oss512_f0/metrics.json ] || ! screen -ls 2>/dev/null|grep -q "\.oss512"; do sleep 90; done
sleep 20
# 1) 3B full-FT @640 (seed42, base, lr2e-5 ep3, eff-batch32=16×2, grad-ckpt)
if [ ! -f runs/t3b_640/model/config.json ]; then
  say "t3b_640 학습 시작 (Qwen2.5-3B full-FT max_length=640 seed42)"
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.train --model Qwen/Qwen2.5-3B \
    --serialize base --max-length 640 --lr 2e-5 --epochs 3 --batch-size 16 --grad-accum 2 \
    --grad-checkpoint --seed 42 --bf16 --out runs/t3b_640 >runs/t3b_640.log 2>&1 || { say "학습 실패"; exit 1; }
fi
say "t3b_640 학습 완료"
# 2) soft-label @640
[ -f runs/_teacher_logits/q3b640.npz ] || { say "q3b640 store(640)"; CUDA_VISIBLE_DEVICES=$G \
  uv run python -m ai_challenge.method.gen_softlabels --teacher-dir runs/t3b_640/model \
  --tag q3b640 --serialize base --max-length 640 >>"$LOG" 2>&1; }
# 3) single-teacher student 비교 (둘 다 student 640)
st(){ local tag=$1 out=runs/kd_st_$1; [ -f $out/metrics.json ] && { echo $(grep -oE '"macro_f1": [0-9.]+' $out/metrics.json|grep -oE '[0-9.]+'); return; }
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits $tag \
    --student Qwen/Qwen2.5-0.5B --out $out --bf16 --temperature 3 --alpha 0.25 --max-length 640 \
    --serialize base >$out.log 2>&1; echo $(grep -oE '"macro_f1": [0-9.]+' $out/metrics.json|grep -oE '[0-9.]+'); }
say "baseline: t3b_base(512학습,512추론) single student"
B=$(st q3bb5)          # q3bb5 = t3b_base @512, gptoss512.sh가 생성
say "test: t3b_640(640학습,640추론) single student"
T=$(st q3b640)
say "★★★ 512학습 teacher → student = $B  |  640학습 teacher → student = $T"
say "→ 640 > 512 이면 '처음부터 640 학습' 유효 (가설 지지). 아니면 512가 맞음."
say "=== t3b640_test 완료 (제출 없음) ==="
