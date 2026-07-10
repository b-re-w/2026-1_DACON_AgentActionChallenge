#!/usr/bin/env bash
# KD 사이클 2: Coder-7B 완료 대기 → export → 팀+우리 teacher 블렌드 재탐색 →
# KD fold0(bias) → KD all-data → pack(bias/무bias 2종) → 검증 → 제출(자율권 부여됨)
# → 이후 7B 3ep 야간 학습 시작.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

# 0) Coder-7B 종료 대기
until [ -f runs/coder7b_base_f0/metrics.json ]; do
  pgrep -f "method.train" >/dev/null || { log "coder7b 프로세스 소멸(실패?) — 기존 teacher 로 진행"; break; }
  sleep 120
done
CODER_OK=0; [ -f runs/coder7b_base_f0/metrics.json ] && CODER_OK=1
log "coder7b 상태: $CODER_OK"

# 1) coder7b 로짓 export
if [ "$CODER_OK" = 1 ]; then
  log "coder7b 로짓 추출"
  uv run python -m ai_challenge.method.export_teacher --runs runs/coder7b_base_f0 \
    --note "Coder-7B base 2ep (이종 백본)" > logs/export_coder7b.log 2>&1 || log "coder export 실패"
fi

# 2) 블렌드 재탐색 (팀 정직 teacher + 우리 전체) → runs/kd2_blend.json
log "블렌드 재탐색"
uv run python - <<'PY'
import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_records, load_folds
from ai_challenge.models.teacher_store import load_teacher_logits, logits_path
from ai_challenge.method.tune_threshold import softmax, coordinate_ascent

recs = load_records("data/train.jsonl", "data/train_labels.csv")
fold = load_folds("data/folds.csv", recs)
y = np.array([CLASS_TO_ID[r.action] for r in recs])[fold == 0]
f0_idx = np.where(fold == 0)[0]
gpos = {recs[i].id: k for k, i in enumerate(f0_idx)}

P = {}
for tag in ["team_qgpt5","team_qgptoss","team_q14b","team_q32b","team_q3b","team_qw6"]:
    P[tag] = softmax(load_teacher_logits(tag)[f0_idx].astype(np.float64))
OURS = {  # run_dir → store key
  "qwen7b_base_f0": "qwen7b_base_f0__base",
  "qwen3b_base_f0": "qwen3b_base_f0__base",
  "qwen3b_cueshist_f0": "qwen3b_cueshist_f0__cues_hist",
  "coder7b_base_f0": "coder7b_base_f0__base",
}
for run, key in list(OURS.items()):
    if not Path(f"runs/{run}/oof_logits.npy").exists() or not logits_path(key).exists():
        OURS.pop(run); continue
    ids = json.loads(Path(f"runs/{run}/oof_ids.json").read_text())
    lg = np.load(f"runs/{run}/oof_logits.npy").astype(np.float64)
    buf = np.zeros((len(f0_idx), 14)); buf[[gpos[i] for i in ids]] = softmax(lg)
    P[key] = buf

keys = list(P)
w = {k: 0.0 for k in keys}; w["team_qw6"] = 1.0
def score(w):
    tot = sum(w.values())
    blend = sum(wt * P[k] for k, wt in w.items() if wt > 0) / tot
    return f1_score(y, blend.argmax(1), average="macro"), blend
best, _ = score(w)
for _ in range(4):
    improved = False
    for k in keys:
        cur = w[k]
        for v in [0, 0.5, 1, 1.5, 2, 3]:
            w[k] = v
            if sum(w.values()) == 0: continue
            s, _ = score(w)
            if s > best + 1e-6: best = s; cur = v; improved = True
        w[k] = cur
    if not improved: break
_, blend = score(w)
bias, tuned = coordinate_ascent(y, blend)
sel = {k: v for k, v in w.items() if v > 0}
print(f"[blend] {sel} macro={best:.5f} +bias={tuned:.5f}")
json.dump({"teachers": list(sel), "weights": list(sel.values()),
           "blend_macro": best, "blend_macro_bias": tuned,
           "bias_vector": bias.tolist()}, open("runs/kd2_blend.json","w"), indent=2)
PY

TEACHERS=$(python -c "import json;print(' '.join(json.load(open('runs/kd2_blend.json'))['teachers']))")
WEIGHTS=$(python -c "import json;print(' '.join(str(w) for w in json.load(open('runs/kd2_blend.json'))['weights']))")
log "teachers=$TEACHERS weights=$WEIGHTS"

# 3) KD fold0 → student bias  (max_length 640: 서연 qw6_all640 LB 0.7905 검증 레시피)
log "KD2 fold0"
uv run python -m ai_challenge.method.distill \
  --teachers $TEACHERS --weights $WEIGHTS \
  --student Qwen/Qwen2.5-0.5B --preset base --fold 0 --max-length 640 \
  --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
  --name kd2_f0 > logs/kd2_f0.log 2>&1 || { log "KD2 fold0 실패"; tail -4 logs/kd2_f0.log; exit 1; }
grep -E "result" logs/kd2_f0.log | tail -1
uv run python -m ai_challenge.method.tune_threshold --runs runs/kd2_f0 2>&1 | tail -2

# 4) KD all-data
log "KD2 all-data"
uv run python -m ai_challenge.method.distill \
  --teachers $TEACHERS --weights $WEIGHTS \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
  --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
  --name kd2_all > logs/kd2_all.log 2>&1 || { log "KD2 all 실패"; tail -4 logs/kd2_all.log; exit 1; }

# 5) pack 2종 + 검증
log "pack (bias/무bias)"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd2_all/model \
  --serialize base --bias runs/kd2_f0/class_bias.json --name submit_kd2b 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd2_all/model \
  --serialize base --name submit_kd2n 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd2b.zip 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd2n.zip 2>&1 | tail -2

# 6) 제출 (사용자 자율권 — bias 효과 A/B 측정)
PROXY=$(python -c "import json;print(round(json.load(open('runs/kd2_f0/metrics.json'))['macro_f1'],4))")
log "제출 (fold0 proxy=$PROXY)"
uv run python -m ai_challenge.utils.submission submit build/submit_kd2b.zip \
  --memo "(claude) kd2+bias: qw6+gptoss+our7B(+coder) blend KD->0.5B all70k (proxy $PROXY+bias)" --yes 2>&1 | tail -1
sleep 20
uv run python -m ai_challenge.utils.submission submit build/submit_kd2n.zip \
  --memo "(claude) kd2 no-bias: 동일모델 bias 미적용 A/B" --yes 2>&1 | tail -1

# 7) 야간: 7B 3ep
log "야간 7B 3ep 시작"
uv run python -m ai_challenge.method.train \
  --model Qwen/Qwen2.5-7B --preset base --fold 0 --bs 4 --grad-accum 8 --lr 7e-6 \
  --epochs 3 --max-length 512 --grad-checkpoint --optim paged_adamw_8bit \
  --name qwen7b_3ep_f0 > logs/qwen7b_3ep_f0.log 2>&1 || log "7B 3ep 실패"
log "사이클 2 체인 종료"
