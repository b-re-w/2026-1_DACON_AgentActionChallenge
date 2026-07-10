#!/usr/bin/env bash
# gpt-oss@512-teacher 테스트: 4×3B(512) + gpt-oss(512)×2 → student 640. fold0 OOF만 산출(제출 X).
# 사용자가 OOF 보고 판단 → 승인 시 별도로 all-data+제출.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
G=1; LOG=runs/gptoss512.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
# 1) 512 store: 3B=메인env, gpt-oss=Qwen3 오버레이
for m in "q3bb5:t3b_base" "q3b43_5:t3b_s43" "q3b44_5:t3b_s44" "q3b45_5:t3b_s45"; do
  t=${m%%:*}; d=${m##*:}; [ -f runs/_teacher_logits/$t.npz ] && { say "$t 있음"; continue; }
  say "$t (512)"; CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.gen_softlabels \
    --teacher-dir runs/$d/model --tag $t --serialize base --max-length 512 >>"$LOG" 2>&1
done
[ -f runs/_teacher_logits/qgptoss5.npz ] || { say "qgptoss5 (512,overlay)"; CUDA_VISIBLE_DEVICES=$G \
  uv run --with "transformers==4.55.0" python -m ai_challenge.method.gen_softlabels \
  --teacher-dir runs/t_gptoss4b/model --tag qgptoss5 --serialize base --max-length 512 >>"$LOG" 2>&1; }
# 2) fold0 (teacher 512, student 640 — qw6_all640과 동일 setup). 제출 없음.
T="q3bb5 q3b43_5 q3b44_5 q3b45_5 qgptoss5:2"
say "gpt-oss@512 fold0 distill :: $T"
CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits $T \
  --student Qwen/Qwen2.5-0.5B --out runs/kd_oss512_f0 --bf16 --temperature 3 --alpha 0.25 \
  --max-length 640 --serialize base >runs/kd_oss512_f0.log 2>&1
oof=$(grep -oE '"macro_f1": [0-9]+\.[0-9]+' runs/kd_oss512_f0/metrics.json|grep -oE '[0-9]+\.[0-9]+')
say "★★ gpt-oss@512 fold0 OOF = $oof (기준: qw6 control=0.7847 / w6r_oss20@640=0.7847)"
say "=== 완료. 제출 없음 — 사용자 판단 대기 ==="
