#!/usr/bin/env bash
# KD 사이클 3 (체크포인트 선택 + A/B 검증):
#  1) kd3_f0: fold0 KD, 스텝단위 eval → macro-F1 피크 체크포인트 자동 선택(load_best)
#  2) bias 튜닝 (베스트 체크포인트 OOF)
#  3) 제출 ①: fold0-베스트 모델 그대로 (+bias)   ← "fold0 모델 직접 제출" 검증
#  4) kd3_all: all-data, fold0 베스트 에폭 이식(fractional epochs)
#  5) 제출 ②: all-data (+bias) / 제출 ③: all-data (bias 없음, bias A/B)
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

TEACHERS=$(python -c "import json;print(' '.join(json.load(open('runs/kd2_blend.json'))['teachers']))")
WEIGHTS=$(python -c "import json;print(' '.join(str(w) for w in json.load(open('runs/kd2_blend.json'))['weights']))")
log "blend: $TEACHERS ($WEIGHTS)"

# 1) fold0 KD + 스텝단위 베스트 선택
log "kd3_f0 시작 (eval-steps 500)"
uv run python -m ai_challenge.method.distill \
  --teachers $TEACHERS --weights $WEIGHTS \
  --student Qwen/Qwen2.5-0.5B --preset base --fold 0 --max-length 640 \
  --epochs 3 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --eval-steps 500 \
  --name kd3_f0 > logs/kd3_f0.log 2>&1 || { log "kd3_f0 실패"; tail -4 logs/kd3_f0.log | tr '\r' '\n' | tail -4; exit 1; }
grep -E "ckpt|result" logs/kd3_f0.log | tail -2

# 2) bias
uv run python -m ai_challenge.method.tune_threshold --runs runs/kd3_f0 2>&1 | tail -2

# 3) 제출 ①: fold0-베스트 직접
log "pack+제출 ① fold0-best"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd3_f0/model \
  --serialize base --bias runs/kd3_f0/class_bias.json --name submit_kd3f 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd3f.zip 2>&1 | tail -2 || exit 1
P=$(python -c "import json;d=json.load(open('runs/kd3_f0/metrics.json'));print(round(d['macro_f1'],4))")
BS=$(python -c "import json;d=json.load(open('runs/kd3_f0/metrics.json'));print(d.get('best_step'))")
uv run python -m ai_challenge.utils.submission submit build/submit_kd3f.zip \
  --memo "(claude) kd3-fold0best: qw6+gptoss+7B blend, ckpt step$BS 선택, 56k만 학습 +bias (oof $P)" --yes 2>&1 | tail -1

# 4) all-data (베스트 에폭 이식)
EPOCHS=$(python -c "
import json; d=json.load(open('runs/kd3_f0/metrics.json'))
spe = 56000//32
e = (d.get('best_step') or spe*3)/spe
print(round(max(e, 0.5), 2))")
log "kd3_all 시작 (epochs=$EPOCHS 이식)"
uv run python -m ai_challenge.method.distill \
  --teachers $TEACHERS --weights $WEIGHTS \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
  --epochs $EPOCHS --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
  --name kd3_all > logs/kd3_all.log 2>&1 || { log "kd3_all 실패"; tail -4 logs/kd3_all.log | tr '\r' '\n' | tail -4; exit 1; }

# 5) 제출 ②③: all-data bias / no-bias
log "pack+제출 ②③ all-data"
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd3_all/model \
  --serialize base --bias runs/kd3_f0/class_bias.json --name submit_kd3a 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd3_all/model \
  --serialize base --name submit_kd3n 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd3a.zip 2>&1 | tail -2 || exit 1
uv run python -m ai_challenge.utils.submission submit build/submit_kd3a.zip \
  --memo "(claude) kd3-alldata+bias: 동일 blend, epochs $EPOCHS 이식 (fold0쌍둥이 oof $P)" --yes 2>&1 | tail -1
sleep 20
uv run python -m ai_challenge.utils.submission submit build/submit_kd3n.zip \
  --memo "(claude) kd3-alldata-nobias: bias A/B 대조군" --yes 2>&1 | tail -1
log "사이클 3 제출 완료"
