#!/usr/bin/env bash
# 인코더 teacher (DeBERTa-v3-large, OOF 0.7418) 를 트리오에 저가중 결합 (fold0).
# 다른서버는 qw6 기반(w6d1 등) 진행중 → 우리는 트리오 기반으로 상보 테스트.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/trio_enc.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize base"
d(){ local g=$1 n=$2; shift 2; local o=runs/kd_$n; [ -f $o/metrics.json ] && return
  until gpu_free $g; do sleep 90; done
  say "GPU$g $n"; CUDA_VISIBLE_DEVICES=$g $D --teacher-logits "$@" --val-fold 0 --out $o >$o.log 2>&1
  say "$n OOF=$(grep -oE '"macro_f1": [0-9.]+' $o/metrics.json 2>/dev/null|grep -oE '[0-9.]+')"; }
# stdtg05의 f2가 GPU2를 먼저 놓아줌 → 거기서 순차 2종
until [ -f runs/kd_tg5s_f2/metrics.json ]; do sleep 120; done
d 2 trio_deb05 qgptoss qw6 q7bteam qdebertaL2:0.5
d 2 trio_deb025 qgptoss qw6 q7bteam qdebertaL2:0.25
say "=== trio_enc 완료 (기준 trio f0=0.78205, +gemma=0.78388) ==="
