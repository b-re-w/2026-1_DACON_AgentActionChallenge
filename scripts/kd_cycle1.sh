#!/usr/bin/env bash
# KD 사이클 1: 3B base 완료 대기 → teacher 로짓 확보 → 블렌드 탐색 →
# KD fold0(proxy·bias) → KD all-data(제출용) → pack → 로컬 T4 모사 검증.
# 제출(submit)은 하지 않는다 — 일 10회 팀 공유 한도라 사람 확인 후 진행.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date +%H:%M:%S)] $*"; }

# 0) 3B base 학습 종료 대기
until [ -f runs/qwen3b_base_f0/metrics.json ]; do
  if ! pgrep -f "ai_challenge.method.train" >/dev/null && [ ! -f runs/qwen3b_base_f0/metrics.json ]; then
    log "3B base 가 비정상 종료된 듯 — 계속 대기하지 않고 7B/기존 teacher 만으로 진행"
    break
  fi
  sleep 60
done
[ -f runs/qwen3b_base_f0/metrics.json ] && log "3B base 완료" || log "3B base 없음(스킵)"

# 1) teacher 로짓 확보 (GPU 단독)
log "7B 로짓 추출"
uv run python -m ai_challenge.method.export_teacher --runs runs/qwen7b_base_f0 \
  --note "7B base 2ep (OOF 0.77605, +bias 0.78081)" >> logs/export_7b.log 2>&1 \
  || { log "7B export 실패"; tail -3 logs/export_7b.log; }
if [ -f runs/qwen3b_base_f0/metrics.json ]; then
  log "3B base 로짓 추출"
  uv run python -m ai_challenge.method.export_teacher --runs runs/qwen3b_base_f0 \
    --note "3B base 3ep 레시피 수정판" >> logs/export_3b_base.log 2>&1 \
    || log "3B base export 실패"
fi

# 2) 블렌드 가중 탐색(OOF) → kd1_blend.json
log "블렌드 가중 탐색"
uv run python - <<'PY'
import json, itertools, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_labels
from ai_challenge.method.tune_threshold import load_oof, softmax, coordinate_ascent

cands = {  # run_dir: 존재하는 것만 사용
    "runs/qwen7b_base_f0": [1, 2, 3, 4],
    "runs/qwen3b_base_f0": [0, 1, 2],
    "runs/qwen3b_cueshist_f0": [0, 1, 2],
    "runs/qwen05b_cueshist_f0": [0, 0.5, 1],
}
cands = {k: v for k, v in cands.items() if Path(k, "oof_logits.npy").exists()}
id2lab = load_labels("data/train_labels.csv")
probs, base_ids, order = {}, None, None
for rd in cands:
    ids, lg = load_oof(Path(rd)); p = softmax(lg)
    if base_ids is None:
        base_ids = ids; order = {i: n for n, i in enumerate(ids)}
    remap = np.array([order[i] for i in ids]); buf = np.zeros_like(p); buf[remap] = p
    probs[rd] = buf
y = np.array([CLASS_TO_ID[id2lab[i]] for i in base_ids])
best = None
keys = list(cands)
for ws in itertools.product(*[cands[k] for k in keys]):
    if sum(ws) == 0: continue
    blend = sum(w * probs[k] for k, w in zip(keys, ws)) / sum(ws)
    m = f1_score(y, blend.argmax(1), average="macro")
    if best is None or m > best[0]: best = (m, ws)
m, ws = best
blend = sum(w * probs[k] for k, w in zip(keys, ws)) / sum(ws)
bias, tuned = coordinate_ascent(y, blend)
sel = [(k, w) for k, w in zip(keys, ws) if w > 0]
out = {"teachers": [k + "/model" for k, _ in sel], "weights": [w for _, w in sel],
       "blend_macro": float(m), "blend_macro_bias": float(tuned),
       "bias_vector": [float(b) for b in bias]}
json.dump(out, open("runs/kd1_blend.json", "w"), indent=2)
print(f"[blend] {dict(zip(keys, ws))} macro={m:.5f} +bias={tuned:.5f}")
PY

TEACHERS=$(python -c "import json;print(' '.join(json.load(open('runs/kd1_blend.json'))['teachers']))")
WEIGHTS=$(python -c "import json;print(' '.join(str(w) for w in json.load(open('runs/kd1_blend.json'))['weights']))")
log "teachers=$TEACHERS weights=$WEIGHTS"

# 3) KD fold0 (proxy 점수 + student bias 확보)
log "KD fold0 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $TEACHERS --weights $WEIGHTS \
  --student Qwen/Qwen2.5-0.5B --preset base --fold 0 \
  --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
  --name kd1_f0 > logs/kd1_f0.log 2>&1 || { log "KD fold0 실패"; tail -5 logs/kd1_f0.log; exit 1; }
grep -E "teacher-blend|result" logs/kd1_f0.log | tail -2
uv run python -m ai_challenge.method.tune_threshold --runs runs/kd1_f0 2>&1 | tail -2

# 4) KD all-data (제출용)
log "KD all-data 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $TEACHERS --weights $WEIGHTS \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data \
  --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
  --name kd1_all > logs/kd1_all.log 2>&1 || { log "KD all-data 실패"; tail -5 logs/kd1_all.log; exit 1; }

# 5) pack (+ fold0 에서 얻은 student bias 주입) → 로컬 T4 모사 검증
log "pack + local eval"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd1_all/model \
  --serialize base --bias runs/kd1_f0/class_bias.json --name submit_kd1 2>&1 | tail -3
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd1.zip 2>&1 | tail -4
log "KD 사이클 1 완료 — 제출은 사람 확인 후"
