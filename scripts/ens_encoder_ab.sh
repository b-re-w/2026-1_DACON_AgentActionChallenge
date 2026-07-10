#!/usr/bin/env bash
# ── 이질 encoder teacher 앙상블 A/B ──
# "decoder teacher 단독 vs +encoder teacher" 로 이질 encoder(DeBERTa/ModernBERT)가
# 앙상블 KD 에 실제로 기여하는지 student(Qwen0.5B) OOF 로 직접 측정.
# 이 머신엔 teacher store 가 qmistral24b/qdebertaL/qmodernbertL 3개만 존재.
# ens_sweep.sh 와 달리 HF_HOME 을 건드리지 않는다(기본 ~/.cache/huggingface 사용).
set -uo pipefail
cd "$(dirname "$0")/.."
GPU=${GPU:-0}; STU=Qwen/Qwen2.5-0.5B; ML=640; LOG=runs/ens_encoder_ab.log
# 조합: name=comma-tags  (distill 은 temperature3/alpha0.25 — ens_sweep 관례와 동일)
SPECS=(
  "m=qmistral24b"                               # decoder 단독 (baseline)
  "md=qmistral24b,qdebertaL"                    # +DeBERTa
  "mmb=qmistral24b,qmodernbertL"                # +ModernBERT
  "mdmb=qmistral24b,qdebertaL,qmodernbertL"     # +둘 다
  "dmb=qdebertaL,qmodernbertL"                  # encoder 둘만 (참고)
)
echo "[ab] start $(date '+%m-%d %H:%M')  student=$STU ML=$ML GPU=$GPU" | tee -a "$LOG"
for spec in "${SPECS[@]}"; do
  name=${spec%%=*}; tags=${spec#*=}
  out=runs/kd_ab_$name
  [ -f "$out/metrics.json" ] && { echo "[$name] skip(완료)" | tee -a "$LOG"; continue; }
  echo "[$(date '+%H:%M')] [$name] teachers=$tags" | tee -a "$LOG"
  CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.distill \
    --teacher-logits ${tags//,/ } --student "$STU" --out "$out" --bf16 \
    --temperature 3 --alpha 0.25 --max-length $ML --serialize base \
    > "$out.log" 2>&1 || { echo "[$name] 실패 → $out.log" | tee -a "$LOG"; continue; }
  f1=$(grep -oE '"macro_f1": [0-9.]+' "$out/metrics.json" | grep -oE '[0-9.]+')
  echo "[$name] OOF=$f1  teachers=$tags" | tee -a "$LOG"
done
echo "[ab] done $(date '+%m-%d %H:%M')" | tee -a "$LOG"
echo "=== 결과 요약 ===" | tee -a "$LOG"
grep -E "^\[.*\] OOF=" "$LOG" | tail -8