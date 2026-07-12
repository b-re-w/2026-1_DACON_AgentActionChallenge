#!/usr/bin/env bash
# gpt-oss fold0 품질 게이트: kd7_v2 종료 후 fold0 14k만 추론(bs8) → OOF 측정.
# ≥0.78 이면 70k 전체 export(bs8) + 반분검증까지, 미만이면 즉시 폐기 보고.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
TF="transformers==4.57.1"

while pgrep -f "kd7_v2.sh" >/dev/null; do sleep 120; done
log "gpt-oss fold0 프로브 시작 (bs8)"
uv run --with "$TF" python - <<'PY'
import json, numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_records, load_folds, SERIALIZE_PRESETS
from ai_challenge.models.common import predict_logits
recs = load_records("data/train.jsonl","data/train_labels.csv")
fold = load_folds("data/folds.csv", recs)
val = [r for r, f in zip(recs, fold) if f == 0]
lg = predict_logits("runs/gptoss20b_f0/model", val, max_length=512, batch_size=8,
                    serialize_kwargs=SERIALIZE_PRESETS["base"])
y = np.array([CLASS_TO_ID[r.action] for r in val])
m = f1_score(y, lg.argmax(1), average="macro")
np.save("runs/gptoss20b_f0/oof_logits.npy", lg.astype(np.float32))
json.dump([r.id for r in val], open("runs/gptoss20b_f0/oof_ids.json","w"))
print(f"GPTOSS_FOLD0 {m:.5f}")
PY
M=$(grep -oE "GPTOSS_FOLD0 [0-9.]+" logs/gptoss_probe.log | tail -1 | awk '{print $2}')
log "gpt-oss fold0 OOF = ${M:-측정실패}"
GO=$(python -c "print(1 if float('${M:-0}' or 0) >= 0.78 else 0)")
if [ "$GO" = 1 ]; then
  log "게이트 통과 → 70k 전체 export (bs8, ~2.5h)"
  uv run --with "$TF" python -m ai_challenge.method.export_teacher \
    --runs runs/gptoss20b_f0 --max-length 512 --batch-size 8 \
    --note "gpt-oss-20b LoRA 1ep (fold0 $M)" \
    >> logs/export_gptoss.log 2>&1 && log "export 완료" || log "export 실패"
else
  log "게이트 미달(<0.78) → gpt-oss 폐기"
fi
log "프로브 종료"
