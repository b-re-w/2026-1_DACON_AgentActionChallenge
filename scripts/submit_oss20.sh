#!/usr/bin/env bash
# w6r_oss20 자동제출 — 사용자 명시 승인("내 승인 안받고 자동제출해줘", 2026-07-11).
# zip 준비(all-data+pack 완료)되면 즉시 제출.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
LOG=runs/submit_oss20.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
say "w6r_oss20 zip 대기 (all-data+pack 완료)..."
until [ -f build/w6r_oss20_all.zip ] || grep -qiE "w6r_oss20 준비 실패" runs/post512.log 2>/dev/null; do sleep 60; done
[ -f build/w6r_oss20_all.zip ] || { say "zip 없음(준비 실패) — 제출 중단"; exit 1; }
sleep 5
say "w6r_oss20 자동제출 (사용자 명시 승인)"
uv run python -m ai_challenge.utils.submission submit build/w6r_oss20_all.zip \
  --memo "(동연) w6r_oss20 = 4x3B + gpt-oss x2 @640 (foldOOF 0.784701, qw6 동률 다른모델 LB실측)" --yes 2>&1 \
  | grep -oiE "isSubmitted[^,}]*|Success|detail[^,}]*|error[^,}]*|leaderboard[^,}]*" | tee -a "$LOG"
say "=== w6r_oss20 제출 완료 ==="
