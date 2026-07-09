#!/usr/bin/env bash
# ── 이질 앙상블 teacher 학습 전 "사전 점검" ──
# PRO 6000(또는 다른 GPU) 서버에서 학습을 시작하기 전에, 필요한 게 다 준비됐는지
# 하나씩 확인하고 ✅/❌ 로 알려준다. ❌ 가 하나라도 있으면 그걸 먼저 고쳐야 한다.
#
# 사용:  scripts/preflight_hetero.sh
set -uo pipefail
cd "$(dirname "$0")/.."

ok=0; fail=0; warn=0
pass() { echo "  ✅ $1"; ok=$((ok+1)); }
bad()  { echo "  ❌ $1"; fail=$((fail+1)); }
note() { echo "  ⚠️  $1"; warn=$((warn+1)); }

echo "==================== 사전 점검 시작 ===================="

# 1) uv
echo "[1/8] uv 설치 확인"
if command -v uv >/dev/null 2>&1; then pass "uv 있음 ($(uv --version 2>/dev/null))"
else bad "uv 없음 → https://docs.astral.sh/uv/ 참고해 설치"; fi

# 2) 가상환경(uv sync 완료 여부)
echo "[2/8] 파이썬 환경(.venv) 확인"
if [ -d .venv ]; then pass ".venv 있음 (uv sync 완료)"
else bad ".venv 없음 → 'uv sync --extra dev' 실행 필요"; fi

# 3) GPU 드라이버
echo "[3/8] GPU 확인"
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
  pass "GPU: $gpu"
else bad "nvidia-smi 없음 → GPU 드라이버 미설치/미인식"; fi

# 4) torch + CUDA + bitsandbytes (Blackwell 호환성의 핵심)
echo "[4/8] torch·CUDA·bitsandbytes 동작 확인 (PRO 6000=Blackwell 이라 중요)"
py_out=$(uv run python - <<'PY' 2>&1
import sys
try:
    import torch
    print("TORCH", torch.__version__, "CUDA", torch.version.cuda, "avail", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPUNAME", torch.cuda.get_device_name(0))
        cc = torch.cuda.get_device_capability(0)
        print("CC", f"{cc[0]}.{cc[1]}")  # Blackwell = 12.0
except Exception as e:
    print("TORCHERR", repr(e))
try:
    import bitsandbytes as bnb
    print("BNB", bnb.__version__)
    # 실제 4bit 텐서를 GPU 로 올려 CUDA 커널이 로드되는지 확인
    import torch
    from bitsandbytes.nn import Linear4bit
    lin = Linear4bit(16, 16).cuda()
    x = torch.randn(2, 16, device="cuda", dtype=torch.float16)
    _ = lin(x)
    print("BNB4BIT_OK")
except Exception as e:
    print("BNBERR", repr(e))
PY
)
echo "$py_out" | sed 's/^/      /'
if echo "$py_out" | grep -q "avail True"; then pass "torch 에서 CUDA 인식됨"
else bad "torch 가 CUDA 를 못 씀 → GPU용 torch 재설치 필요"; fi
if echo "$py_out" | grep -q "BNB4BIT_OK"; then pass "bitsandbytes 4bit 커널 정상 (QLoRA 가능)"
else bad "bitsandbytes 4bit 실패 → Blackwell 지원 버전 필요: 'uv pip install -U bitsandbytes' (필요시 torch 도 최신 CUDA 빌드로)"; fi

# 5) HF 환경변수
echo "[5/8] HuggingFace 설정 확인"
if [ -n "${HF_HOME:-}" ]; then
  if mkdir -p "$HF_HOME" 2>/dev/null && [ -w "$HF_HOME" ]; then pass "HF_HOME=$HF_HOME (쓰기 가능)"
  else bad "HF_HOME=$HF_HOME 에 쓸 수 없음 (권한/경로 확인)"; fi
else bad "HF_HOME 미설정 → 큰 디스크로 export (예: export HF_HOME=\$HOME/hf)"; fi
if [ -n "${HF_TOKEN:-}" ]; then pass "HF_TOKEN 설정됨 (${HF_TOKEN:0:3}...${HF_TOKEN: -3})"
else bad "HF_TOKEN 미설정 → export HF_TOKEN=hf_xxx"; fi

# 6) HF 인증
echo "[6/8] HuggingFace 로그인 확인"
who=$(uv run hf auth whoami 2>/dev/null | head -1)
if [ -n "$who" ] && ! echo "$who" | grep -qi "not logged"; then pass "HF 로그인: $who"
else bad "HF 인증 실패 → HF_TOKEN 값이 유효한지 확인"; fi

# 7) 데이터
echo "[7/8] 학습 데이터 확인"
for f in data/train.jsonl data/train_labels.csv; do
  if [ -s "$f" ]; then pass "$f ($(wc -l < "$f") 줄)"
  else bad "$f 없음 → 메인 담당자에게 받아 data/ 에 넣기"; fi
done

# 8) gated 모델 라이선스 접근 (Gemma/Llama)
echo "[8/8] gated 모델 라이선스 접근 확인 (HF 웹에서 미리 수락 필요)"
check_gate() {
  local repo="$1" name="$2"
  if uv run hf download "$repo" --include "config.json" >/dev/null 2>&1; then pass "$name 접근 가능 (라이선스 수락됨)"
  else note "$name 접근 불가 → https://huggingface.co/$repo 에서 라이선스 수락(또는 승인 대기중)"; fi
}
check_gate "google/gemma-2-27b-it" "Gemma-2-27B"
check_gate "meta-llama/Llama-3.3-70B-Instruct" "Llama-3.3-70B"

echo "======================================================="
echo "결과:  ✅ $ok   ⚠️ $warn   ❌ $fail"
if [ "$fail" -gt 0 ]; then
  echo "→ ❌ 항목을 먼저 해결한 뒤 다시 이 스크립트를 실행하세요."
  exit 1
else
  echo "→ 준비 완료! 이제 'scripts/run_hetero_all.sh' 를 실행하면 됩니다."
  [ "$warn" -gt 0 ] && echo "  (⚠️ 항목은 해당 모델 학습 시작 전까지만 해결하면 됩니다.)"
  exit 0
fi
