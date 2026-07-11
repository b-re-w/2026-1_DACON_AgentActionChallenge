#!/usr/bin/env bash
# 밤 체인: gpt-oss-20b LoRA teacher (tf 4.57.1) → export → 반분검증 →
# 블렌드 개선(≥+0.002) 시 KD 사이클6 (fold0 + all-data) → bias70k pack → 자동 제출.
# 마지막에 gemma3-270m student 프로브(4.57.1)도 시도.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
TF="transformers==4.57.1"

# 1) gpt-oss-20b LoRA teacher
log "gpt-oss-20b LoRA 시작 (tf 4.57.1)"
uv run --with "$TF" python -m ai_challenge.method.train \
  --model openai/gpt-oss-20b --preset base --fold 0 --bs 8 --grad-accum 4 \
  --lr 1e-4 --epochs 2 --max-length 512 --grad-checkpoint --lora --lora-r 16 \
  --name gptoss20b_f0 > logs/gptoss20b_f0.log 2>&1 \
  || { log "gpt-oss 실패"; tr '\r' '\n' < logs/gptoss20b_f0.log | grep -E "Error|error" | tail -3; }
grep -E "result" logs/gptoss20b_f0.log | tail -1

GO=0
if [ -f runs/gptoss20b_f0/metrics.json ]; then
  log "gpt-oss 로짓 export"
  uv run --with "$TF" python -m ai_challenge.method.export_teacher \
    --runs runs/gptoss20b_f0 --note "gpt-oss-20b LoRA 이질 teacher (자체)" \
    > logs/export_gptoss.log 2>&1 || log "export 실패"
  # 반분검증: cycle3 트리오 + gptoss20b
  uv run python - > runs/gptoss_blend_test.txt 2>&1 <<'PY'
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
P = {t: softmax(load_teacher_logits(t)[f0_idx].astype(np.float64))
     for t in ["team_qw6","team_qgptoss"]}
for run,key in [("qwen7b_base_f0","7b"), ("gptoss20b_f0","goss")]:
    ids = json.loads(Path(f"runs/{run}/oof_ids.json").read_text())
    buf = np.zeros((len(f0_idx),14)); buf[[gpos[i] for i in ids]] = softmax(np.load(f"runs/{run}/oof_logits.npy").astype(np.float64))
    P[key] = buf
rng = np.random.RandomState(7); h = rng.permutation(len(y)); A,B = h[:len(y)//2], h[len(y)//2:]
def sc(w, idx):
    tot=sum(w.values()); return f1_score(y[idx],(sum(v*P[k][idx] for k,v in w.items() if v>0)/tot).argmax(1),average="macro")
res = {}
for name,w in {"cycle3":{"team_qw6":1,"team_qgptoss":1,"7b":1},
               "+goss05":{"team_qw6":1,"team_qgptoss":1,"7b":1,"goss":0.5},
               "+goss1":{"team_qw6":1,"team_qgptoss":1,"7b":1,"goss":1},
               "+goss2":{"team_qw6":1,"team_qgptoss":1,"7b":1,"goss":2}}.items():
    full = sc(w, np.arange(len(y)))
    print(f"{name:9s} A={sc(w,A):.4f} B={sc(w,B):.4f} full={full:.4f}")
    res[name] = full
best = max(res, key=res.get)
gain = res[best] - res["cycle3"]
print(f"BEST {best} gain {gain:+.4f}")
print(f"DECISION {'GO' if gain >= 0.002 and best!='cycle3' else 'NOGO'} {best}")
PY
  cat runs/gptoss_blend_test.txt
  grep -q "DECISION GO" runs/gptoss_blend_test.txt && GO=1
fi

# 2) 개선 시 KD 사이클 6 자동 (표준: 640, best-ckpt, bias70k 56k벡터, 무/유 bias 2종 제출)
if [ "$GO" = 1 ]; then
  BESTW=$(grep "DECISION GO" runs/gptoss_blend_test.txt | awk '{print $3}')
  case "$BESTW" in
    +goss05) GW=0.5;; +goss1) GW=1;; +goss2) GW=2;; *) GW=1;;
  esac
  T="team_qgptoss team_qw6 qwen7b_base_f0__base gptoss20b_f0__base"; W="1 1 1 $GW"
  log "사이클6 GO (goss 가중 $GW) — fold0"
  uv run python -m ai_challenge.method.distill \
    --teachers $T --weights $W \
    --student Qwen/Qwen2.5-0.5B --preset base --fold 0 --max-length 640 \
    --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
    --name kd6_f0 > logs/kd6_f0.log 2>&1 || { log "kd6_f0 실패"; exit 1; }
  grep -E "ckpt|result" logs/kd6_f0.log | tail -2
  log "사이클6 all-data"
  uv run python -m ai_challenge.method.distill \
    --teachers $T --weights $W \
    --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
    --epochs 2.86 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
    --name kd6_all > logs/kd6_all.log 2>&1 || { log "kd6_all 실패"; exit 1; }
  uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd6_all/model \
    --serialize base --bias runs/bias70k.json --name submit_kd6b 2>&1 | tail -1
  uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd6b.zip 2>&1 | tail -2 || exit 1
  P0=$(python -c "import json;print(round(json.load(open('runs/kd6_f0/metrics.json'))['macro_f1'],4))")
  uv run python -m ai_challenge.utils.submission submit build/submit_kd6b.zip \
    --memo "(claude) kd6+bias70k: cycle3+gptoss20b(w$GW) blend (fold0 $P0)" --yes 2>&1 | tail -1
else
  log "gpt-oss 블렌드 개선 없음 → 사이클6 스킵"
fi

# 3) gemma3-270m student 프로브 (tf 4.57.1)
log "gemma3-270m student 프로브"
uv run --with "$TF" python -m ai_challenge.method.distill \
  --teachers team_qgptoss team_qw6 qwen7b_base_f0__base --weights 1 1 1 \
  --student google/gemma-3-270m --preset base --fold 0 --max-length 640 \
  --epochs 3 --bs 32 --lr 2e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
  --name kd_gemma270m_f0 > logs/kd_gemma270m_f0.log 2>&1 \
  && grep -E "ckpt|result" logs/kd_gemma270m_f0.log | tail -2 \
  || { log "gemma 프로브 실패"; tr '\r' '\n' < logs/kd_gemma270m_f0.log | grep -E "Error" | tail -2; }
log "밤 체인 종료"
