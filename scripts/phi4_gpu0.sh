#!/usr/bin/env bash
set -uo pipefail; cd /data/agent_action
export HF_HOME=/data/hf GPU=0
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ]; do sleep 60; done
echo "[phi4-gpu0] GPU0 확보 → train_phi4.sh 시작 ($(date '+%m-%d %H:%M'))"
exec bash scripts/train_phi4.sh
