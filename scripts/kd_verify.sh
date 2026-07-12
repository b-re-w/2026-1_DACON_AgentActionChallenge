#!/usr/bin/env bash
# 스크리닝 상위 후보 실제 Qwen0.5B KD 검증 (fold0 OOF). 정식 레시피: T3/alpha0.25/ML640/base.
# HF_HOME 은 건드리지 않음(이 머신 기본 캐시). control=w6r_g20 재현 목표 ~0.7847.
set -uo pipefail; cd "$(dirname "$0")/.."
G=${GPU:-0}; ML=640; STU=Qwen/Qwen2.5-0.5B; LOG=logs/kd_verify.log
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
Q="q3bb q3b43 q3b44 q3b45"   # w6 뼈대(4×3B full-FT 시드)
run(){ local n=$1 t=$2 o=runs/kd_$n
  [ -f "$o/metrics.json" ] && { say "$n 있음=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')"; return; }
  say "$n :: $t"
  CUDA_VISIBLE_DEVICES=$G uv run python -m ai_challenge.method.distill --teacher-logits $t \
    --student "$STU" --out "$o" --bf16 --temperature 3 --alpha 0.25 --max-length $ML --serialize base >"$o.log" 2>&1 \
    && say "$n OOF=$(grep -oE '\"macro_f1\": [0-9.]+' $o/metrics.json|grep -oE '[0-9.]+')" \
    || say "$n 실패 → $o.log"; }

say "=== KD 검증 스윕 시작 ==="
# 우선순위: control → gpt-oss 레버 → encoder(DeBERTa) 효과
run w6r_g20      "$Q qgpt5:2"                  # control (w6 재현) ~0.7847
run w6r_oss20    "$Q qgptoss:2"                # ★ gpt-oss 스왑 (프록시 최상 0.7863)
run w6oss_deb    "$Q qgptoss:2 qdebertaL:1"    # ★ 최상base + DeBERTa(encoder 이질)
run qw6_deb      "qw6:6 qdebertaL:1"           # DeBERTa 단독 효과(qw6 위)
run qw6_gptoss_deb "qw6:6 qgptoss:1 qdebertaL:1"  # gptoss+deberta (프록시 0.7851)
run w6r_oss2g1   "$Q qgptoss:2 qgpt5:1"        # oss×2 + gpt5 (둘 다)
say "=== KD 검증 완료 ==="
grep -oE "kd_[a-z0-9_]+ OOF=[0-9.]+" "$LOG" | tail -8
