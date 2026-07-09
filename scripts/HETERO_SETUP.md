# 이질 앙상블 — 다른 A100(80GB) 서버 셋업 가이드

메인 서버(Qwen 사다리)와 별개로, 이 서버에서 **비-Qwen teacher**(Llama/Gemma/Mistral)를
QLoRA로 학습해 soft-label `.npz`만 만든다. 그 `.npz`를 메인 서버로 옮겨 앙상블에 합류.
거대한 모델을 옮길 필요 없음 — `.npz`는 개당 ~7MB(14-class).

## 왜 이 방식인가
- 승리 레버 = **이질(계열 다양성) 앙상블**. Qwen만으론 오류가 비슷해 앙상블 이득이 작다.
- soft-label store는 **레코드 id 기준**이라, 같은 데이터 + `serialize=base` + `max_length=640`
  이면 어느 서버에서 만들었든 앙상블 호환.

## 1. 코드
```bash
git clone https://github.com/b-re-w/agent_action.git
cd agent_action
git checkout big_model_lim
uv sync --extra dev            # peft·bitsandbytes 포함
```

## 2. 데이터 (⚠️ 앙상블 호환의 핵심)
메인 서버의 데이터를 그대로 복사 — 레코드 id·fold가 일치해야 함:
```bash
scp main:/data/agent_action/data/{train.jsonl,train_labels.csv,folds.csv} data/
```

## 3. HF 캐시 · 토큰
```bash
export HF_HOME=/big_disk/hf     # 70B는 ~140GB 받으니 큰 디스크로
export HF_TOKEN=hf_xxxxx        # Llama·Gemma는 gated → 토큰 필요 (Mistral은 open)
```
Llama-3.3-70B / Gemma-2-27B는 HF 모델 페이지에서 **라이선스 수락** 먼저(계정으로 1회).

## 4. 학습 + soft-label 생성 (1 GPU라 순차)
```bash
scripts/train_hetero.sh mistral    # open, 마찰 없음 (추천 1순위)
scripts/train_hetero.sh gemma      # gated, eager attention 자동 적용
scripts/train_hetero.sh llama      # gated, 70B (오래 걸림)
```
각각 `runs/_teacher_logits/{qmistral24b|qgemma27b|qllama70b}.npz` 생성.

## 5. `.npz`를 메인 서버로
```bash
scp runs/_teacher_logits/qmistral24b.{npz,json} main:/data/agent_action/runs/_teacher_logits/
```

## 6. 메인 서버에서 앙상블 (거기서 실행)
```bash
uv run python -m ai_challenge.method.distill \
  --teacher-logits q3b q14b q32b q72b qglm32b qmistral24b qgemma27b qllama70b \
  --student Qwen/Qwen2.5-0.5B --out runs/kd_hetero_full --bf16 \
  --temperature 3 --alpha 0.25 --max-length 640
```

## 참고 (80GB QLoRA 사양)
| 모델 | 계열 | gated | batch | VRAM(추정) |
|---|---|---|---|---|
| Mistral-Small-24B-Instruct-2501 | Mistral | open | 16×2 | ~40GB |
| gemma-2-27b-it | Google | gated | 16×2 (eager) | ~45GB |
| Llama-3.3-70B-Instruct | Meta | gated | 4×8 | ~52GB |

- 이 3개는 `target_modules` 기본값(q/k/v/o/gate/up/down_proj)이 그대로 맞음.
- transformers 4.46.3(uv sync 기본)로 3개 다 로드됨. 혹시 "unknown architecture" 뜨면
  `uv pip install -U transformers` (이 서버는 메인과 분리라 버전 올려도 무방).
- 학습 시간(early-stop 기준): 24~27B ~20-30h, 70B ~40h+.
