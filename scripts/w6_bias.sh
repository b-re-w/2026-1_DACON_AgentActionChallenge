#!/usr/bin/env bash
# w6(현재 최고 LB 0.787396)에 per-class bias 를 붙이기 위한 fold0 프록시 튜닝.
# w6 는 all-data 라 자기 OOF 가 오염됨 → 같은 타깃으로 fold0 student 를 새로 학습해
# 그 held-out OOF 로 bias 를 튜닝하고, bias.json 을 제출 모델에 복사한다.
# (repack + 재제출은 슬롯이 걸린 outward 액션이라 사람이 결정 — 여기선 bias.json 까지만)
#
# 실행: GPU=1 scripts/w6_bias.sh    (기본 GPU1)
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf
GPU=${GPU:-1}; ML=512; SER=base    # ⚠️ w6 원본과 동일: 512 / base / T3 / a0.25
LOG=runs/w6_bias.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

say "1) w6 teacher 타깃(all-data teacher_cache) → id-키 store qw6 로 변환 (CPU, 재추론 0)"
uv run python - <<'PY'
import numpy as np
from ai_challenge.datasets import load_records
from ai_challenge.models.common import TRAIN_JSONL, TRAIN_LABELS, save_teacher_logits
recs = load_records(TRAIN_JSONL, TRAIN_LABELS)
tl = np.load("runs/kd_w6_all/teacher_cache.npz")["train"]
assert len(tl) == len(recs), (len(tl), len(recs))
p = save_teacher_logits("qw6", [s.id for s in recs], tl,
                        meta={"from": "kd_w6_all/teacher_cache", "max_length": 512, "serialize": "base"})
print("[qw6] store 생성", tl.shape, p)
PY

say "2) fold0 w6 student 학습 (GPU$GPU, ~40min) — w6 원본과 동일 하이퍼(512·T3·a0.25)"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.distill \
  --teacher-logits qw6 --student Qwen/Qwen2.5-0.5B --out runs/kd_w6_f0 \
  --val-fold 0 --bf16 --temperature 3 --alpha 0.25 --max-length $ML --serialize $SER \
  >> "$LOG" 2>&1

say "3) bias 튜닝 (fold0 student OOF → coordinate ascent → bias.json)"
CUDA_VISIBLE_DEVICES=$GPU uv run python -m ai_challenge.method.tune_threshold \
  --model-dir runs/kd_w6_f0/model --val-fold 0 --max-length $ML >> "$LOG" 2>&1

say "4) bias.json 을 제출 모델(kd_w6_all/model)에 복사"
cp runs/kd_w6_f0/model/bias.json runs/kd_w6_all/model/bias.json
say "완료 ✅  runs/kd_w6_all/model/bias.json"
say "다음(수동): pack --model-dir runs/kd_w6_all/model --name submit_w6_bias --serialize base"
say "           → local eval → submission submit (슬롯 1개)"
grep -oE "base.*macro_f1.*|tuned.*|\[saved\].*" "$LOG" | tail -4