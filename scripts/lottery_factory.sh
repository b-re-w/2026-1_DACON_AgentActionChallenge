#!/usr/bin/env bash
# 복권 공장 (비축 모드): kd3n 레시피 지터 변형을 연속 학습 + pack 만 하고 제출은 안 함.
# 자정 이후 build/*.zip 을 수동/스크립트로 제출.
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }
T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

# 진행 중 학습(kd10_cw) 종료 대기
while pgrep -f "method.distill" >/dev/null; do sleep 180; done

# (변형이름, 추가인자) — seed·lr·epoch 지터
declare -a NAMES=(kd11_s46 kd11_s47 kd11_lr09 kd11_lr11 kd11_ep27 kd11_ep30 kd11_s48)
declare -a ARGS=(
  "--seed 46"
  "--seed 47"
  "--seed 46 --lr 0.9e-5"
  "--seed 47 --lr 1.1e-5"
  "--seed 48 --epochs 2.7"
  "--seed 49 --epochs 3.0"
  "--seed 48"
)
for i in "${!NAMES[@]}"; do
  n=${NAMES[$i]}; a=${ARGS[$i]}
  [ -f "build/submit_$n.zip" ] && { log "$n 이미 있음, 스킵"; continue; }
  log "$n 학습 시작 ($a)"
  # epochs 지터 변형은 --epochs 가 ARGS 에 있으면 그걸 쓰고, 없으면 2.86
  if echo "$a" | grep -q epochs; then EP=""; else EP="--epochs 2.86"; fi
  uv run python -m ai_challenge.method.distill \
    --teachers $T --weights $W \
    --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
    $EP --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 $a \
    --name $n > logs/$n.log 2>&1 || { log "$n 실패"; continue; }
  uv run python -m ai_challenge.utils.pack pack --model-dir runs/$n/model \
    --serialize base --bias runs/bias70k.json --name submit_$n 2>&1 | tail -1
  uv run python -m ai_challenge.utils.pack eval --zip build/submit_$n.zip 2>&1 | tail -1 \
    && log "$n 비축 완료" || log "$n 검증 실패"
  rm -rf runs/$n/hf  # 디스크 관리
done
log "복권 공장 종료 — build/ 비축분 확인"
ls -la build/submit_kd11*.zip 2>/dev/null
