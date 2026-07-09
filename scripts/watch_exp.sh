#!/usr/bin/env bash
# 크기 사다리 실험 진행 대시보드. 사용: scripts/watch_exp.sh   (Ctrl-C 로 종료)
cd "$(dirname "$0")/.."
watch -n 10 '
for m in 14 32 72; do
  log=runs/t${m}b_qlora.log
  met=runs/t${m}b_qlora/metrics.json
  printf "── %sB ──\n" "$m"
  if [ -f "$met" ]; then
    printf "  DONE  OOF macro_f1=%s\n" "$(grep -oE "\"macro_f1\": [0-9.]+" "$met" | grep -oE "[0-9.]+")"
  elif [ -f "$log" ]; then
    grep -oE "Downloading shards: +[0-9]+%|Loading checkpoint shards: +[0-9]+%|[0-9]+/[0-9]+ \[[0-9:]+<[^]]+\]" "$log" | tail -1 | sed "s/^/  진행: /"
    grep -oE "eval_macro_f1: [0-9.]+|.loss.: [0-9.]+" "$log" | tail -1 | sed "s/^/  최근: /"
  else
    echo "  (기동 대기)"
  fi
done
echo
echo "── GPU ──"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader | sed "s/^/  GPU/"
'
