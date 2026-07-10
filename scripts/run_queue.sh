#!/usr/bin/env bash
# 단일 GPU 순차 실험 큐. 현재 실행 중인 학습이 끝나길 기다린 뒤,
# 큐 파일의 각 줄(= train.py 인자)을 하나씩 순차 실행한다(경합 방지).
#
# 큐 파일 형식: 한 줄에 train.py 인자. '#' 시작 줄/빈 줄 무시. 반드시 --name 포함.
# 사용: nohup scripts/run_queue.sh queue.txt > logs/queue.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
QUEUE="${1:?queue file}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 진행 중인 학습이 있으면 끝날 때까지 대기(GPU 독점 보장)
while pgrep -f "ai_challenge.method.train" >/dev/null; do sleep 30; done

while IFS= read -r line || [ -n "$line" ]; do
  case "$line" in ''|\#*) continue;; esac
  name=$(echo "$line" | grep -oE -- '--name [^ ]+' | awk '{print $2}')
  name=${name:-unnamed_$(echo "$line" | md5sum | cut -c1-6)}
  echo "===== [$(date +%H:%M:%S)] START $name ====="
  # shellcheck disable=SC2086
  uv run python -m ai_challenge.method.train $line > "logs/${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "!!!!! [$(date +%H:%M:%S)] FAIL $name (rc=$rc) — tail:"
    tail -c 600 "logs/${name}.log" | tr '\r' '\n' | tail -4
  else
    grep -E "result" "logs/${name}.log" | tail -1 | sed "s/^/  $name /"
  fi
done < "$QUEUE"
echo "===== [$(date +%H:%M:%S)] QUEUE DONE ====="
