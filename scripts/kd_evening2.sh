#!/usr/bin/env bash
# 저녁 복권 2장 (all-data 직행, LB 로 직접 측정):
#  A) kd9_tb : 2.86ep + target-bias 증류 + 추론 bias70k
#  B) kd9_s45: 2.86ep seed45 + 추론 bias70k
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

run_one () {  # $1 name, $2 extra distill args, $3 memo
  local name=$1 extra=$2 memo=$3
  log "$name 시작"
  uv run python -m ai_challenge.method.distill \
    --teachers $T --weights $W \
    --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
    --epochs 2.86 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 $extra \
    --name $name > logs/$name.log 2>&1 || { log "$name 실패"; tail -3 logs/$name.log; return 1; }
  uv run python -m ai_challenge.utils.pack pack --model-dir runs/$name/model \
    --serialize base --bias runs/bias70k.json --name submit_$name 2>&1 | tail -1 \
  && uv run python -m ai_challenge.utils.pack eval --zip build/submit_$name.zip 2>&1 | tail -1 \
  && uv run python -m ai_challenge.utils.submission submit build/submit_$name.zip \
       --memo "$memo" --yes 2>&1 | tail -1
}

run_one kd9_tb "--target-bias runs/bias70k.json" \
  "(claude) kd9-tb: target-bias 증류(경계를 타겟에 학습) + 추론bias70k, 2.86ep"
sleep 30
run_one kd9_s45 "--seed 45" \
  "(claude) kd9-s45: kd3n 레시피 seed45 재추첨 + bias70k (s42=0.79371)"
log "저녁 복권 완료"
