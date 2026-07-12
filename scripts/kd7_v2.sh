#!/usr/bin/env bash
# kd7 v2: 5ep fold0 피크를 '스텝 수'로 이식 (8500스텝 = all-data 3.89ep).
# 아침 체인(gpt-oss) 종료 후 실행. bias70k 판단은 kd7n(무bias) 채점 보고 수동.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

while pgrep -f "kd_morning.sh" >/dev/null; do sleep 300; done
log "kd7_all_v2 시작 (3.89ep = 8500스텝 이식)"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
  --epochs 3.89 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 \
  --name kd7_all_v2 > logs/kd7_all_v2.log 2>&1 || { log "실패"; tail -3 logs/kd7_all_v2.log; exit 1; }
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd7_all_v2/model \
  --serialize base --name submit_kd7v2n 2>&1 | tail -1
uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd7v2n.zip 2>&1 | tail -2 || exit 1
uv run python -m ai_challenge.utils.submission submit build/submit_kd7v2n.zip \
  --memo "(claude) kd7v2 무bias: 5ep피크 스텝이식(8500step=3.89ep) — 에폭이식 과적합 수정" --yes 2>&1 | tail -1
log "kd7 v2 완료"
