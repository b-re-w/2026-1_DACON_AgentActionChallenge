#!/usr/bin/env bash
# ── 이질 앙상블 teacher "순차 전체 실행" (Gemma → Llama) ──
# 이 스크립트 하나만 실행하면 Gemma-2-27B 와 Llama-3.3-70B 를 차례로
# QLoRA 학습하고 soft-label(.npz) 을 만든다. 잘 몰라도 되게 만들어 뒀다:
#   - 시작 전에 자동으로 사전 점검(preflight)을 돌린다.
#   - 이미 만들어진 .npz 가 있으면 그 모델은 건너뛴다(중복 학습 방지).
#   - 각 모델 로그는 logs/ 에 따로 남는다.
#   - 한 모델이 실패해도 다음 모델은 계속 시도하고, 끝에 요약을 보여준다.
#
# 사용:
#   export HF_HOME=/큰디스크/hf         # 예: export HF_HOME=$HOME/hf
#   export HF_TOKEN=hf_xxx              # 메인 담당자에게 받은 토큰
#   scripts/run_hetero_all.sh          # (오래 걸리니 nohup 사용 권장 — 가이드 참고)
#
# 특정 모델만 돌리고 싶으면:  scripts/run_hetero_all.sh gemma   (또는 llama)
set -uo pipefail
cd "$(dirname "$0")/.."

# 인자 없으면 gemma llama 둘 다. 인자 있으면 그것만.
if [ "$#" -gt 0 ]; then MODELS=("$@"); else MODELS=(gemma llama); fi

mkdir -p logs runs/_teacher_logits

# tag 매핑 (train_hetero.sh 와 동일해야 함)
tag_of() { case "$1" in gemma) echo qgemma27b;; llama) echo qllama70b;; mistral) echo qmistral24b;; *) echo "";; esac; }

echo "########################################################"
echo "#  이질 앙상블 teacher 순차 실행:  ${MODELS[*]}"
echo "#  시작 전 사전 점검을 먼저 돌립니다..."
echo "########################################################"
if ! bash scripts/preflight_hetero.sh; then
  echo ""
  echo "‼️  사전 점검에서 문제가 발견됐습니다. 위의 ❌ 를 해결한 뒤 다시 실행하세요."
  exit 1
fi

declare -a RESULTS
for m in "${MODELS[@]}"; do
  tag=$(tag_of "$m")
  if [ -z "$tag" ]; then echo "알 수 없는 모델: $m (gemma|llama 만 가능)"; RESULTS+=("$m: 잘못된 이름"); continue; fi
  npz="runs/_teacher_logits/${tag}.npz"
  log="logs/${tag}.log"

  echo ""
  echo "======================================================="
  echo "▶ [$m] 시작  (tag=$tag)  로그: $log"
  echo "======================================================="

  # 이미 완성된 .npz 가 있으면 건너뛴다
  if [ -s "$npz" ]; then
    echo "  이미 $npz 가 있습니다 → 건너뜀 (다시 하려면 이 파일을 지우세요)."
    RESULTS+=("$m: 건너뜀(이미 완료)")
    continue
  fi

  # 학습 + soft-label 생성 (train_hetero.sh 재사용). 로그는 파일+화면 동시.
  if bash scripts/train_hetero.sh "$m" 2>&1 | tee "$log"; then
    if [ -s "$npz" ]; then
      echo "  ✅ [$m] 완료 → $npz"
      RESULTS+=("$m: ✅ 완료 → $npz")
    else
      echo "  ⚠️  [$m] 학습은 끝났지만 $npz 가 안 생겼습니다. 로그($log) 확인 필요."
      RESULTS+=("$m: ⚠️ npz 미생성 (로그 확인)")
    fi
  else
    echo "  ❌ [$m] 실행 중 오류. 로그($log) 마지막 부분을 메인 담당자에게 보내세요."
    RESULTS+=("$m: ❌ 실패 (로그 확인)")
  fi
done

echo ""
echo "########################################################"
echo "#  전체 요약"
for r in "${RESULTS[@]}"; do echo "#   - $r"; done
echo "#"
echo "#  생성된 .npz 목록:"
ls -la runs/_teacher_logits/*.npz 2>/dev/null | sed 's/^/#   /' || echo "#   (아직 없음)"
echo "#"
echo "#  다음 할 일: 아래 .npz(+.json) 를 메인 담당자에게 전달하세요."
echo "#   예)  scp runs/_teacher_logits/qgemma27b.{npz,json} main:/data/agent_action/runs/_teacher_logits/"
echo "#   (scp 가 어려우면 파일을 그대로 압축해 전달해도 됩니다 — 개당 ~7MB)"
echo "########################################################"
