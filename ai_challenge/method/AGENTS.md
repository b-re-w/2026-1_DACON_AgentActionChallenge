# method — 실행 방법 (CLI 진입점)

각 파일은 하나의 "방법"이며, 공통 로직은 `ai_challenge.models.common` 에서 가져다 쓰고
여기서는 **CLI 인자 + 오케스트레이션만** 담당한다(중복 최소화).

## 파일

| 파일 | 방법 | 진입점 |
|------|------|--------|
| `train.py` | 지도학습 (인코더/디코더 공용 seq-cls) | `python -m ai_challenge.method.train` |
| `distill.py` | Knowledge Distillation (teacher soft-label → student) | `python -m ai_challenge.method.distill` |
| `tune_threshold.py` | OOF per-class bias 튜닝 (Macro-F1 직접 최적화, 재학습 X) | `python -m ai_challenge.method.tune_threshold` |

## 공용 인자 (train/distill)

`--model`/`--student` · `--out` · `--epochs` · `--batch-size` · `--grad-accum` ·
`--max-length` · `--lr` · `--bf16` · `--num-workers` · `--grad-checkpoint` ·
`--all-data`(전체 70k, 최종 제출용). train 은 `--no-class-weight`, distill 은
`--teacher-dir`·`--temperature`·`--alpha` 추가.

## 규칙

- 실험이 끝나면 결과를 `log_experiment(...)` 로 기록(LOGGING.md).
- 산출물은 `--out runs/<name>` 아래. 제출 zip 은 `utils/pack.py` 로 `build/` 에 생성.
