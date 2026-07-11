#!/usr/bin/env bash
# 사이클 4: 검증된 cycle3 블렌드 고정, student 직교 레버 2개.
#  A) kd4_s43: kd3_all 동일 레시피 + seed 43 (seed 재추첨)
#  B) kd4_warm: kd_qw6_all640(LB 0.7905) 웜스타트 + cycle3 블렌드 1ep 저LR
# 제출은 무bias (A/B 로 확정된 표준). 72B/GLM teacher 나오면 사이클 5 에서 사용.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

# A) seed 43 재추첨 (kd3_all: epochs 2.86, ml640, T3 a0.25)
log "kd4_s43 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
  --epochs 2.86 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --seed 43 \
  --name kd4_s43 > logs/kd4_s43.log 2>&1 || { log "s43 실패"; tail -3 logs/kd4_s43.log; exit 1; }
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd4_s43/model \
  --serialize base --name submit_kd4s43 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd4s43.zip 2>&1 | tail -2 || exit 1
uv run python -m ai_challenge.utils.submission submit build/submit_kd4s43.zip \
  --memo "(claude) kd4-s43: kd3 동일레시피 seed43 재추첨 (kd3n LB 0.79195)" --yes 2>&1 | tail -1

# B) 웜스타트 (0.7905 ckpt → cycle3 블렌드, 1ep lr 3e-6)
log "kd4_warm 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student runs/kd_qw6_all640/model --preset base --all-data --max-length 640 \
  --epochs 1 --bs 32 --lr 3e-6 --kd-alpha 0.25 --kd-T 3 \
  --name kd4_warm > logs/kd4_warm.log 2>&1 || { log "warm 실패"; tail -3 logs/kd4_warm.log; exit 1; }
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd4_warm/model \
  --serialize base --name submit_kd4wm 2>&1 | tail -2
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd4wm.zip 2>&1 | tail -2 || exit 1
uv run python -m ai_challenge.utils.submission submit build/submit_kd4wm.zip \
  --memo "(claude) kd4-warm: qw6_all640(0.7905) 웜스타트 + qw6/oss/7B blend 1ep lr3e-6" --yes 2>&1 | tail -1
log "사이클 4 완료"
