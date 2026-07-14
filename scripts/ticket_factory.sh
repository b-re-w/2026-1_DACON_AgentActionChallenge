#!/usr/bin/env bash
# 시드 티켓 공장: btt43 LB 0.79521 신기록 → 시드 45/46/47(/48) 추가 발행
# 레시피 = btt43과 동일: 트리오 KD + btrail + all-data(70k) + bagged bias → pack
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/ticket_factory.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
D="uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 --temperature 3 --alpha 0.25 --max-length 640 --serialize btrail --teacher-logits qgptoss qw6 q7bteam --all-data"
mk(){ local g=$1 s=$2; local o=runs/kd_bt_s$s
  if [ ! -f $o/model/config.json ]; then
    until gpu_free $g; do sleep 120; done
    say "GPU$g 시드$s 훈련 시작"
    CUDA_VISIBLE_DEVICES=$g $D --seed $s --out $o >$o.log 2>&1 || { say "시드$s 실패"; return 1; }
  fi
  cp runs/bias_bagged_deploy.json $o/model/bias.json
  [ -f build/btt$s.zip ] || uv run python -m ai_challenge.utils.pack pack --model-dir $o/model --name btt$s --serialize btrail --quant fp16 >>"$LOG" 2>&1
  say "build/btt$s.zip ✅ ($(ls -la build/btt$s.zip | awk '{printf "%.0fMB", $5/1048576}'))"
}
( mk 1 45 ) &
( mk 2 46 ) &
( mk 0 47 ) &
wait
( mk 3 48 ) &   # es100 종료 후 보너스 티켓
wait
say "=== ticket_factory 완료 ==="
