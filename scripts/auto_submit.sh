#!/usr/bin/env bash
# w6recon fold0 결과가 임계 이상이면 all-data 재학습→pack→제출 (시간·슬롯 제약 대응).
# 핵심 결과(g20/oss20) 나오면 저순위 sweep 중단하고 winner 우선 제출.
set -uo pipefail; cd "$(dirname "$0")/.."; export HF_HOME=/data/hf
G=1; ML=640; THR=0.7845; CAP=4; LOG=runs/auto_submit.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
# name → all-data teacher 조합
declare -A CFG=(
  [w6r_g20]="q3bb q3b43 q3b44 q3b45 qgpt5:2"
  [w6r_oss20]="q3bb q3b43 q3b44 q3b45 qgptoss:2"
  [w6r_both]="q3bb q3b43 q3b44 q3b45 qgpt5:1 qgptoss:1"
  [w6r_g20oss1]="q3bb q3b43 q3b44 q3b45 qgpt5:2 qgptoss:1"
)
oof_of(){ grep -oE '"macro_f1": [0-9]+\.[0-9]+' runs/kd_$1/metrics.json 2>/dev/null|grep -oE '[0-9]+\.[0-9]+'; }

# 1) 핵심 2개(g20/oss20) 완료 대기
say "핵심 결과(g20/oss20) 대기..."
until [ -f runs/kd_w6r_g20/metrics.json ] && [ -f runs/kd_w6r_oss20/metrics.json ]; do sleep 90; done
# 2) 저순위 sweep 중단(GPU1 확보) — 이미 끝난 both/g20oss1 은 유지
say "핵심 결과 확보 → w6recon sweep 중단(GPU1 확보)"
screen -S w6recon -X quit 2>/dev/null; pkill -9 -f "kd_w6r_g25\|kd_w6r_g15" 2>/dev/null; sleep 5

# 3) 사용 가능한 후보를 OOF 내림차순으로 제출 (≥THR, CAP개)
subs=0
for name in $(for n in w6r_g20 w6r_oss20 w6r_both w6r_g20oss1; do v=$(oof_of $n); [ -n "$v" ] && echo "$v $n"; done | sort -rn | awk '{print $2}'); do
  oof=$(oof_of $name)
  awk "BEGIN{exit !($oof >= $THR)}" || { say "$name OOF=$oof < $THR → 스킵"; continue; }
  [ "$subs" -ge "$CAP" ] && { say "슬롯 캡($CAP) 도달 → 중단"; break; }
  out=runs/kd_${name}_all
  say "$name OOF=$oof ≥ $THR → all-data 학습"
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits ${CFG[$name]} \
    --student Qwen/Qwen2.5-0.5B --out $out --all-data --bf16 --temperature 3 --alpha 0.25 \
    --max-length $ML --serialize base > $out.log 2>&1 || { say "$name all-data 실패"; continue; }
  uv run python -m ai_challenge.utils.pack pack --model-dir $out/model --name ${name}_all --serialize base --quant fp16 >>"$LOG" 2>&1 || { say "$name pack 실패"; continue; }
  r=$(uv run python -m ai_challenge.utils.submission submit build/${name}_all.zip --memo "(동연) ${name} all640 (foldOOF $oof)" --yes 2>&1 | grep -oiE "isSubmitted[^,}]*|Success")
  subs=$((subs+1)); say "$name 제출 ($r) — 누적 $subs/$CAP"
done
say "=== auto_submit 완료: $subs개 제출 ==="
