#!/usr/bin/env bash
# 640 fold0 student 들의 per-class bias 튜닝 + held-out 이득(base vs tuned) 비교. 재학습 0.
# 이득이 유의미(>+0.001)할 때만 bias.json 유지, 아니면 제거(무익한 bias 방지).
# 사용: GPU=2 scripts/bias_students.sh   (여유 GPU 지정; 각 ~1분)
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf
GPU=${GPU:-2}; ML=640
LOG=runs/bias_students.log
for tag in q3b q14b q32b q72b ens_all ens_3b72b; do
  m=runs/kd_${tag}/model
  [ -f "$m/config.json" ] || { echo "[$tag] student 없음 — 스킵"; continue; }
  [ -f "$m/.bias_done" ]  && { echo "[$tag] 이미 처리 — 스킵"; continue; }
  OUT=$(CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.tune_threshold \
        --model-dir "$m" --val-fold 0 --max-length $ML 2>&1)
  b=$(echo "$OUT" | grep -oE "base=[0-9.]+"  | tail -1 | cut -d= -f2)
  t=$(echo "$OUT" | grep -oE "tuned=[0-9.]+" | tail -1 | cut -d= -f2)
  if [ -n "${b:-}" ] && [ -n "${t:-}" ] && awk "BEGIN{exit !($t > $b + 0.001)}" 2>/dev/null; then
    msg="[$tag] (640) held-out $b→$t  ✅ 이득 → bias 유지"
  else
    rm -f "$m/bias.json"; msg="[$tag] (640) held-out ${b:-?}→${t:-?}  ✗ 무익 → bias 제거"
  fi
  touch "$m/.bias_done"; echo "$msg" | tee -a "$LOG"
done
echo "[bias_students] 처리 완료(미완 student 는 재실행 시 이어서)."