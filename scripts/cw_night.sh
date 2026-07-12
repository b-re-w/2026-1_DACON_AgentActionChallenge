#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"
log "kd10_cw (sqrt 가중 KD) 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
  --epochs 2.86 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --class-weight sqrt \
  --name kd10_cw > logs/kd10_cw.log 2>&1 || { log "실패"; tail -3 logs/kd10_cw.log; exit 1; }
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd10_cw/model \
  --serialize base --bias runs/bias70k.json --name submit_kd10cw 2>&1 | tail -1 \
&& uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd10cw.zip 2>&1 | tail -1 \
&& uv run python -m ai_challenge.utils.submission submit build/submit_kd10cw.zip \
     --memo "(claude) kd10-cw: sqrt 클래스가중 KD(약클래스 손실 가중) + bias70k" --yes 2>&1 | tail -1
log "완료"
