# models — 공통 학습 로직 · 모델 코드

`method/` 의 train·distill·tune_threshold 가 공유하는 빌딩 블록을 모은다.
**모델 수정(커스텀 분류 헤드, 아키텍처 변경, 새 모델 래퍼 등)이 가해질 때에는 그
모델 코드도 이 `models/` 패키지에 새 모듈로 넣는다.**

## `common.py` 공개 함수

| 함수/클래스 | 역할 |
|------|------|
| `build_tokenizer(model)` | 토크나이저 + pad 토큰(디코더) + 섹션 특수 토큰 |
| `build_model(model, tok, grad_checkpoint=)` | 14-way seq-cls (id2label 부여, 임베딩 resize) |
| `load_train_records()` / `get_folds()` | train 로드 / folds.csv 캐시 공유 |
| `build_datasets(...)` | (train_ds, val_ds) — all_data 면 전체 학습 |
| `build_training_args(...)` | train·distill 공용 TrainingArguments (bf16/grad-accum/grad-ckpt/all-data) |
| `WeightedTrainer` | class-weighted CE (weights=None → 일반 CE) |
| `compute_metrics` / `compute_class_weights` | macro-F1·acc / balanced weight |
| `save_submission_model` / `write_oof` / `write_metrics` | 제출용 저장 · OOF · 지표 |
| `predict_logits(model_dir, samples)` | 저장 모델 → 14-class logits (길이정렬 배칭). tune·distill 공용 |

## 규칙

- 새 실행 방법을 추가할 때 반복 로직은 여기로 추출해 `method/` 는 CLI+오케스트레이션만 유지.
- 커스텀 모델은 `common` 이 아니라 별도 모듈(예: `heads.py`)로 두고 `build_model` 에서 분기.
