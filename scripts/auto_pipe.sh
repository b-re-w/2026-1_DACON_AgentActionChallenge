#!/usr/bin/env bash
# 각 teacher(14/32/72B) 학습 완료를 감지해 자동으로 soft-label 캐시 + 0.5B 증류를
# GPU0 에서 순차 실행. 셋 다 끝나면 이질(크기) 앙상블까지 돌린다. 완료 순서대로 처리.
#   실행:  screen -dmS autopipe bash scripts/auto_pipe.sh
#   진행:  tail -f runs/auto_pipe.log  (이 스크립트 자체 로그) / 각 runs/kd_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=/data/hf
ML=640; SER=base; STU=Qwen/Qwen2.5-0.5B
LOG=runs/auto_pipe.log
say() { echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

# tag → teacher out 디렉터리 / 그 teacher 가 쓰던 GPU (완료 시 그 GPU 가 비므로 거기서 처리).
# GPU0 은 GLM 학습이 점유하므로 auto_pipe 는 절대 GPU0 을 쓰지 않는다.
declare -A DIR=( [q14b]=runs/t14b_qlora [q32b]=runs/t32b_qlora [q72b]=runs/t72b_qlora )
declare -A GPUOF=( [q14b]=1 [q32b]=2 [q72b]=3 )
pending="q14b q32b q72b"

# teacher metrics.json 은 train.py 가 가장 마지막에 쓴다 → 존재하면 model/ 저장 완료 보장.
process() {  # $1=tag
  local tag=$1 dir=${DIR[$1]} g=${GPUOF[$1]}
  if [ -f "runs/kd_${tag}/metrics.json" ]; then say "$tag 이미 완료 — 스킵"; return; fi
  say "$tag teacher 완료 감지 → 캐시 시작 (GPU$g — 방금 빈 그 teacher 의 GPU)"
  CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.gen_softlabels \
    --teacher-dir "${dir}/model" --tag "$tag" --serialize $SER --max-length $ML --load-in-4bit \
    > "runs/gen_${tag}.log" 2>&1 || { say "$tag 캐시 실패 — 로그 확인"; return 1; }
  say "$tag 캐시 완료 → 증류 시작 (GPU$g)"
  CUDA_VISIBLE_DEVICES=$g uv run python -m ai_challenge.method.distill \
    --teacher-logits "$tag" --student $STU --out "runs/kd_${tag}" --bf16 \
    --temperature 3 --alpha 0.25 --max-length $ML > "runs/kd_${tag}.log" 2>&1 \
    || { say "$tag 증류 실패 — 로그 확인"; return 1; }
  say "$tag 증류 완료: $(grep -oE '"macro_f1": [0-9.]+' runs/kd_${tag}/metrics.json)"
}

say "auto_pipe 시작 — 대기: $pending"
while [ -n "${pending// }" ]; do
  did=""
  for tag in $pending; do
    if [ -f "${DIR[$tag]}/metrics.json" ]; then
      process "$tag"
      pending=$(echo "$pending" | tr ' ' '\n' | grep -vx "$tag" | tr '\n' ' ')
      did=1; break   # 목록 갱신 후 재순회
    fi
  done
  [ -z "$did" ] && sleep 120
done

# 전체 teacher 캐시 완료 → 이질(크기) 앙상블 증류. q3b 는 이미 캐시돼 있음.
# 이 시점엔 Qwen GPU(1/2/3) 전부 비어 있으므로 GPU1 사용(GPU0 은 GLM).
say "단일 증류 전부 완료 → 앙상블 시작 (GPU1)"
CUDA_VISIBLE_DEVICES=1 uv run python -m ai_challenge.method.distill \
  --teacher-logits q3b q72b --student $STU --out runs/kd_ens_3b72b --bf16 \
  --temperature 3 --alpha 0.25 --max-length $ML > runs/kd_ens_3b72b.log 2>&1
CUDA_VISIBLE_DEVICES=1 uv run python -m ai_challenge.method.distill \
  --teacher-logits q3b q14b q32b q72b --student $STU --out runs/kd_ens_all --bf16 \
  --temperature 3 --alpha 0.25 --max-length $ML > runs/kd_ens_all.log 2>&1
say "앙상블 완료. 전체 student OOF:"
grep -h '"macro_f1"' runs/kd_q3b/metrics.json runs/kd_q{14,32,72}b/metrics.json runs/kd_ens_*/metrics.json 2>/dev/null | tee -a "$LOG"
say "=== auto_pipe 종료 ==="
