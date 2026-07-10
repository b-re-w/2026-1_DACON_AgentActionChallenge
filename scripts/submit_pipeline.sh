#!/usr/bin/env bash
# all-data 재학습 → pack(fp16) → local eval → 제출. 미세차 후보를 LB로 실측(val 세트 차이 검증).
# GPU1 이 비면(w6oss_up 완료) 순차 실행. 슬롯: 후보당 1개.
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf
GPU=${GPU:-1}; ML=640; STU=Qwen/Qwen2.5-0.5B; LOG=runs/submit_pipe.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

run_one(){  # name  "teacher-logits"  alpha  memo
  local name="$1" teachers="$2" alpha="$3" memo="$4" out="runs/kd_$1"
  if [ ! -f "$out/model/config.json" ]; then
    say "$name: all-data 학습 (teachers=$teachers a=$alpha)"
    CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.distill \
      --teacher-logits $teachers --student $STU --out "$out" --all-data --bf16 \
      --temperature 3 --alpha $alpha --max-length $ML --serialize base > "$out.log" 2>&1 \
      || { say "$name 학습 실패"; return 1; }
  else say "$name: 학습물 있음 — 스킵"; fi
  say "$name: pack"
  uv run python -m ai_challenge.utils.pack pack --model-dir "$out/model" --name "$name" \
    --serialize base --quant fp16 >> "$LOG" 2>&1 || { say "$name pack 실패"; return 1; }
  uv run python -m ai_challenge.utils.pack eval --zip "build/${name}.zip" 2>&1 | grep -E "rows|invalid" | tee -a "$LOG"
  say "$name: 제출"
  uv run python -m ai_challenge.utils.submission submit "build/${name}.zip" --memo "$memo" --yes 2>&1 \
    | grep -oiE "isSubmitted[^,}]*|Success|detail[^,}]*" | tee -a "$LOG"
  say "$name: 완료"
}

until [ -f runs/kd_w6oss_up/metrics.json ]; do sleep 60; done
say "=== 제출 파이프라인 시작 (①②③) ==="
run_one "qw6_all640"   "qw6"               0.25 "(동연) qw6 all-data 640 T3a25 — w6 640 remake"
run_one "w6oss5_all"   "qw6 qgptoss qgpt5" 0.25 "(동연) w6oss5 all-data 640 — qw6+gptoss+gpt5 다양성 LB test"
run_one "qw6_a15_all"  "qw6"               0.15 "(동연) qw6 all-data 640 T3a15 — alpha0.15 변형"
say "=== 파이프라인 완료 ==="
