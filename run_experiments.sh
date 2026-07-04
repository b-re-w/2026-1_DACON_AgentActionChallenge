#!/bin/bash
# B(Qwen) 개선 실험을 순차 실행 (A100 단일 job 포화 → 순차). 각 run 은 fold0 OOF
# macro_f1 을 metrics.json 에 남긴다. 요약은 runs/exp_queue.log.
cd /home/tta/agent_action
export TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0

wait_gpu_free() { while pgrep -f "ai_challenge.train" >/dev/null; do sleep 30; done; }

run() {
  local name=$1; shift
  mkdir -p "runs/$name"
  wait_gpu_free
  echo "[$(date +%H:%M)] START $name :: $*" >> runs/exp_queue.log
  uv run python -m ai_challenge.train "$@" --out "runs/$name" --bf16 --num-workers 4 \
    > "runs/$name/train.log" 2>&1
  local f1
  f1=$(python3 -c "import json;print(json.load(open('runs/$name/metrics.json'))['macro_f1'])" 2>/dev/null)
  echo "[$(date +%H:%M)] DONE  $name -> OOF macro_f1=$f1" >> runs/exp_queue.log
}

# 큐 (B5=5에폭은 별도 실행중). 기대가치 순.
run binst --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3 --batch-size 32 --max-length 512 --lr 2e-5
run bnw   --model Qwen/Qwen2.5-0.5B          --epochs 3 --batch-size 32 --max-length 512 --lr 2e-5 --no-class-weight
run blr3  --model Qwen/Qwen2.5-0.5B          --epochs 5 --batch-size 32 --max-length 512 --lr 3e-5
echo "[$(date +%H:%M)] ALL DONE" >> runs/exp_queue.log
