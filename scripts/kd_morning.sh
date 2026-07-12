#!/usr/bin/env bash
# 아침 체인 (7/12):
#  1) 5ep 언더트레이닝 프로브 (kd7_5ep_f0) — 곡선상 3ep 이후에도 상승 여부 확정
#     ≥ 0.7855 (kd3_f0 +0.003) 면 all-data 승격 + bias70k pack + 자동 제출
#  2) 1.5B student fold0 (코랩 T4 벤치 통과 대비 품질 측정)
#  3) gpt-oss-20b 재도전: 1ep, in-loop eval 끔(OOM 회피) → OOF(bs16) → export → 반분검증
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

# 1) 5ep 프로브
log "kd7_5ep_f0 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-0.5B --preset base --fold 0 --max-length 640 \
  --epochs 5 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
  --name kd7_5ep_f0 > logs/kd7_5ep_f0.log 2>&1 || { log "5ep 프로브 실패"; tail -3 logs/kd7_5ep_f0.log; }
grep -E "ckpt|result" logs/kd7_5ep_f0.log | tail -2
M5=$(python -c "import json;print(json.load(open('runs/kd7_5ep_f0/metrics.json'))['macro_f1'])" 2>/dev/null || echo 0)
GO5=$(python -c "print(1 if float('$M5' or 0) >= 0.7855 else 0)")
if [ "$GO5" = 1 ]; then
  BS=$(python -c "import json;d=json.load(open('runs/kd7_5ep_f0/metrics.json'));print(d['best_step'])")
  EP=$(python -c "print(round(max($BS/1750, 0.5),2))")
  log "5ep GO (fold0 $M5, best_step $BS → all-data ${EP}ep)"
  uv run python -m ai_challenge.method.distill \
    --teachers $T --weights $W \
    --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
    --epochs $EP --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
    --name kd7_all > logs/kd7_all.log 2>&1 || { log "kd7_all 실패"; exit 1; }
  uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd7_all/model \
    --serialize base --bias runs/bias70k.json --name submit_kd7b 2>&1 | tail -1
  uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd7b.zip 2>&1 | tail -2 || exit 1
  uv run python -m ai_challenge.utils.submission submit build/submit_kd7b.zip \
    --memo "(claude) kd7: 5ep 언더트레이닝 해소 (fold0 $M5) + bias70k" --yes 2>&1 | tail -1
else
  log "5ep 프로브 기준 미달 ($M5 < 0.7855) — 승격 스킵"
fi

# 2) 1.5B student fold0 (코랩 벤치 통과 대비)
log "kd_15b_f0 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-1.5B --preset base --fold 0 --max-length 640 \
  --epochs 3 --bs 16 --grad-accum 2 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
  --name kd_15b_f0 > logs/kd_15b_f0.log 2>&1 || { log "1.5B 실패"; tail -3 logs/kd_15b_f0.log; }
grep -E "ckpt|result" logs/kd_15b_f0.log | tail -2

# 3) gpt-oss 재도전 (1ep, eval 끔)
log "gpt-oss 1ep 재도전"
uv run --with "transformers==4.57.1" python -m ai_challenge.method.train \
  --model openai/gpt-oss-20b --preset base --fold 0 --bs 8 --grad-accum 4 \
  --lr 1e-4 --epochs 1 --max-length 512 --grad-checkpoint --lora --lora-r 16 \
  --no-inloop-eval --name gptoss20b_f0 > logs/gptoss20b_f0.log 2>&1 \
  || { log "gpt-oss 재실패"; tr '\r' '\n' < logs/gptoss20b_f0.log | grep -E "Error" | tail -2; }
if [ -d runs/gptoss20b_f0/model ]; then
  log "gpt-oss OOF+export (bs16)"
  uv run --with "transformers==4.57.1" python -m ai_challenge.method.export_teacher \
    --runs runs/gptoss20b_f0 --max-length 512 --note "gpt-oss-20b LoRA 1ep" \
    > logs/export_gptoss.log 2>&1 || log "export 실패"
  # fold0 OOF 측정 + 반분검증
  uv run python - <<'PY'
import json, numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_records, load_folds
from ai_challenge.models.teacher_store import load_teacher_logits
from ai_challenge.method.tune_threshold import softmax
recs = load_records("data/train.jsonl","data/train_labels.csv")
fold = load_folds("data/folds.csv", recs); f0_idx = np.where(fold==0)[0]
y = np.array([CLASS_TO_ID[r.action] for r in recs])[f0_idx]
gpos = {recs[i].id: k for k,i in enumerate(f0_idx)}
g = load_teacher_logits("gptoss20b_f0__base")
print(f"gptoss fold0 OOF: {f1_score(y, g[f0_idx].argmax(1), average='macro'):.4f}")
P = {t: softmax(load_teacher_logits(t)[f0_idx].astype(np.float64)) for t in ["team_qw6","team_qgptoss"]}
P["goss"] = softmax(g[f0_idx].astype(np.float64))
ids = json.loads(Path("runs/qwen7b_base_f0/oof_ids.json").read_text())
buf = np.zeros((len(f0_idx),14)); buf[[gpos[i] for i in ids]] = softmax(np.load("runs/qwen7b_base_f0/oof_logits.npy").astype(np.float64))
P["7b"] = buf
rng = np.random.RandomState(7); h = rng.permutation(len(y)); A,B = h[:len(y)//2], h[len(y)//2:]
def sc(w, idx):
    tot=sum(w.values()); return f1_score(y[idx],(sum(v*P[k][idx] for k,v in w.items() if v>0)/tot).argmax(1),average="macro")
for name,w in {"cycle3":{"team_qw6":1,"team_qgptoss":1,"7b":1},
               "+goss1":{"team_qw6":1,"team_qgptoss":1,"7b":1,"goss":1}}.items():
    print(f"{name:8s} A={sc(w,A):.4f} B={sc(w,B):.4f} full={sc(w,np.arange(len(y))):.4f}")
PY
fi
log "아침 체인 종료"
