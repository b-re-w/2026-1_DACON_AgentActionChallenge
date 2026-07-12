#!/usr/bin/env bash
# trio+GLM:0.5 단독판 (사용자 지적: GLM-only는 LB 미검증 — gemma와의 교락 분리).
# 신기록 구조: all-data 모델 + 56k피팅 bias(mts 것 근사 사용) → 자정 후 제출.
set -uo pipefail; cd /data/agent_action; export HF_HOME=/data/hf
LOG=runs/glm_only.log; say(){ echo "[$(date '+%m-%d %H:%M')] $*"|tee -a "$LOG"; }
gpu_free(){ [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 2>/dev/null)" -lt 2000 ]; }
# tg5night 시드(GPU0/1) 끝나면 GPU1에서 all-data 학습
say "GPU1 확보 대기 (tg5night 시드 후)..."
until [ -f runs/kd_tg5all_s44/model/config.json ]; do sleep 120; done
until gpu_free 1; do sleep 60; done
[ -f runs/kd_tglm_all/model/config.json ] || { say "trio+GLM:0.5 all-data 학습 (GPU1)"
  CUDA_VISIBLE_DEVICES=1 uv run python -m ai_challenge.method.distill --student Qwen/Qwen2.5-0.5B --bf16 \
    --temperature 3 --alpha 0.25 --max-length 640 --serialize base \
    --teacher-logits qgptoss qw6 q7bteam qglm32b:0.5 --all-data --out runs/kd_tglm_all >runs/kd_tglm_all.log 2>&1; }
say "bias 부착 (mts 56k bias 근사 — 같은 트리오 계열)"
uv run python - <<'PY' 2>>"$LOG"
import json, shutil, os
b = json.load(open("runs/bias56k_mts.json"))
dst="runs/kd_tglmb70/model"
if os.path.exists(dst): shutil.rmtree(dst)
shutil.copytree("runs/kd_tglm_all/model", dst)
json.dump({"bias": b["bias"]}, open(f"{dst}/bias.json","w"))
print("all-data + 56k bias(mts근사) 부착")
PY
uv run python -m ai_challenge.utils.pack pack --model-dir runs/kd_tglmb70/model --name tglmb70 --serialize base --quant fp16 >>"$LOG" 2>&1 && say "build/tglmb70.zip ✅"
uv run python -c "import zipfile;z=zipfile.ZipFile('build/tglmb70.zip');assert not z.testzip() and 'model/bias.json' in z.namelist()" 2>>"$LOG" || { say "zip 검증 실패"; exit 1; }
# 자정 이후에만 제출 (오늘 슬롯 보존) — 사용자 요청 실험
while [ "$(date '+%H')" -ge 12 ]; do sleep 300; done   # 00~11시 사이가 되면 통과
say "자정 지남 → 제출"
uv run python -m ai_challenge.utils.submission submit build/tglmb70.zip \
  --memo "(동연) tglmb70 = trio+GLM:0.5 단독 all-data + 56k bias (gemma 교락 분리)" --yes 2>&1 | grep -oiE "isSubmitted[^,}]*|Success" | tee -a "$LOG"
say "=== glm_only 완료 (제출됨) ==="
