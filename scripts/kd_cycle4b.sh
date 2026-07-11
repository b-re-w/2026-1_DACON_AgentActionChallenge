#!/usr/bin/env bash
# 사이클 4b (72B 대기 중 GPU 활용):
#  A) kd4_s44: student seed 추첨 1장 (50분) → 제출
#  B) qwen7b_s43: 7B teacher seed 43 풀FT (7h) → 로짓 export (72B 도착 대비 블렌드 풀 강화)
set -uo pipefail
cd "$(dirname "$0")/.."
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date +%H:%M:%S)] $*"; }

T="team_qgptoss team_qw6 qwen7b_base_f0__base"; W="1 1 1"

# A) student seed 44
log "kd4_s44 시작"
uv run python -m ai_challenge.method.distill \
  --teachers $T --weights $W \
  --student Qwen/Qwen2.5-0.5B --preset base --all-data --max-length 640 \
  --epochs 2.86 --bs 32 --lr 1e-5 --kd-alpha 0.25 --kd-T 3 --seed 44 \
  --name kd4_s44 > logs/kd4_s44.log 2>&1 || { log "s44 실패"; tail -3 logs/kd4_s44.log; }
if [ -d runs/kd4_s44/model ]; then
  uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd4_s44/model \
    --serialize base --name submit_kd4s44 2>&1 | tail -1
  uv run python -m ai_challenge.utils.pack eval --zip build/submit_kd4s44.zip 2>&1 | tail -2 \
    && uv run python -m ai_challenge.utils.submission submit build/submit_kd4s44.zip \
       --memo "(claude) kd4-s44: seed 재추첨 3장째 (s42=0.79195 s43=0.79067)" --yes 2>&1 | tail -1
fi

# B) 7B teacher seed 43 (풀FT 2ep)
log "qwen7b_s43 teacher 시작"
uv run python -m ai_challenge.method.train \
  --model Qwen/Qwen2.5-7B --preset base --fold 0 --bs 4 --grad-accum 8 --lr 7e-6 \
  --epochs 2 --max-length 512 --grad-checkpoint --optim paged_adamw_8bit --seed 43 \
  --name qwen7b_s43_f0 > logs/qwen7b_s43_f0.log 2>&1 || { log "7B s43 실패"; tail -3 logs/qwen7b_s43_f0.log; exit 1; }
grep -E "result" logs/qwen7b_s43_f0.log | tail -1
log "7B s43 로짓 export"
uv run python -m ai_challenge.method.export_teacher --runs runs/qwen7b_s43_f0 \
  --note "7B base 2ep seed43 (블렌드 다양성)" > logs/export_7b_s43.log 2>&1 || log "export 실패"
log "사이클 4b 완료"
