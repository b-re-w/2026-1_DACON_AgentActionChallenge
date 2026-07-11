#!/usr/bin/env bash
# 사이클 5 준비: student 5-fold OOF 완성 (fold0=kd3_f0 기존, fold1~4 신규).
# 완성 후: folds1~4 OOF(56k)로 bias 피팅 → fold0(14k)로 정직 검증.
# +0.002 이상 이득일 때만 kd3n+bias 변형 제출 후보로 보고.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"
for f in 1 2 3 4; do
  log "kd5_f$f 시작"
  uv run python -m ai_challenge.method.distill \
    --teachers $T --weights $W \
    --student Qwen/Qwen2.5-0.5B --preset base --fold $f --max-length 640 \
    --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
    --name kd5_f$f > logs/kd5_f$f.log 2>&1 || { log "kd5_f$f 실패"; tail -3 logs/kd5_f$f.log; continue; }
  grep -E "ckpt|result" logs/kd5_f$f.log | tail -2
done

log "5-fold OOF 집계 + bias 정직검증 (train: folds1-4, test: fold0)"
uv run python - <<'PY'
import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_records, load_folds
from ai_challenge.method.tune_threshold import softmax, coordinate_ascent

recs = load_records("data/train.jsonl","data/train_labels.csv")
fold = load_folds("data/folds.csv", recs)
y_all = np.array([CLASS_TO_ID[r.action] for r in recs])
pos = {r.id: i for i, r in enumerate(recs)}

# 각 fold 런의 OOF 를 글로벌 인덱스로 배치
probs = np.zeros((len(recs), 14)); have = np.zeros(len(recs), bool)
for f, run in [(0,"kd3_f0"),(1,"kd5_f1"),(2,"kd5_f2"),(3,"kd5_f3"),(4,"kd5_f4")]:
    p = Path(f"runs/{run}/oof_logits.npy")
    if not p.exists(): print(f"fold{f} 없음"); continue
    ids = json.loads(Path(f"runs/{run}/oof_ids.json").read_text())
    idx = np.array([pos[i] for i in ids])
    probs[idx] = softmax(np.load(p).astype(np.float64)); have[idx] = True
print(f"OOF 커버: {have.sum()}/70000")

tr = have & (fold != 0); te = have & (fold == 0)
base_tr = f1_score(y_all[tr], probs[tr].argmax(1), average="macro")
base_te = f1_score(y_all[te], probs[te].argmax(1), average="macro")
bias, tuned_tr = coordinate_ascent(y_all[tr], probs[tr])
tuned_te = f1_score(y_all[te], (probs[te]+bias).argmax(1), average="macro")
print(f"[bias 56k 피팅] train(1-4): {base_tr:.5f}→{tuned_tr:.5f} | held-out fold0: {base_te:.5f}→{tuned_te:.5f} (Δ{tuned_te-base_te:+.5f})")
json.dump({"bias_vector": bias.tolist(), "heldout_gain": float(tuned_te-base_te),
           "macro_f1_before": base_te, "macro_f1_after": tuned_te},
          open("runs/bias70k.json","w"), indent=2)
PY
log "사이클 5 준비 완료"
