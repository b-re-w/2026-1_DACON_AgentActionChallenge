# ai_challenge — 패키지 라우팅

DACON 236694 (AI Agent Action 예측, Macro-F1) 코드 패키지. 전체 접근·전략은 루트
[PROJECT.md](../PROJECT.md), 실험 기록 규칙은 [LOGGING.md](../LOGGING.md) 참고.

## 서브패키지 (각 폴더 AGENTS.md 참고)

| 폴더 | 역할 | 상세 |
|------|------|------|
| `datasets/` | 원천 로딩·직렬화·fold·torch Dataset·자동 다운로드 | [datasets/AGENTS.md](datasets/AGENTS.md) |
| `models/` | 공통 학습 로직(build/trainer/args/predict) · 커스텀 모델 코드 | [models/AGENTS.md](models/AGENTS.md) |
| `method/` | 실행 방법: train · distill · tune_threshold (CLI 진입점) | [method/AGENTS.md](method/AGENTS.md) |
| `utils/` | 제출 패키징/검증(pack) · 리더보드 · 제출 API · 실험 로그 | [utils/AGENTS.md](utils/AGENTS.md) |

## 의존 방향 (단방향)

```
datasets  ←  models  ←  method
                    ↖  utils (pack 은 datasets 만 사용)
```

## 주요 진입점

```bash
# 학습 (인코더/디코더 공용)
uv run python -m ai_challenge.method.train --model Qwen/Qwen2.5-0.5B --out runs/B --bf16 --no-class-weight
# distillation (teacher → student)
uv run python -m ai_challenge.method.distill --teacher-dir runs/teacher/model --out runs/kd --bf16
# 제출 패키징 → build/<name>.zip, 로컬 검증
uv run python -m ai_challenge.utils.pack pack --model-dir runs/B/model --name submit_B
uv run python -m ai_challenge.utils.pack eval --zip build/submit_B.zip
# 리더보드 / 제출
uv run python -m ai_challenge.utils.leaderboard --team 클피티
uv run python -m ai_challenge.utils.submission list --mine
```

## 규칙

- 실험이 끝나면 `ai_challenge.utils.experiment_log.log_experiment(...)` 로 기록(LOGGING.md).
- 산출물(`runs/`, `build/`)은 커밋하지 않는다.
