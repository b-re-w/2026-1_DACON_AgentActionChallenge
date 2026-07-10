# Teacher soft-label 저장소

KD(지식증류)용 teacher 로짓 모음. 각 `.npy` 는 shape **(70000, 14)** float32,
행 순서는 **train.jsonl 줄 순서**(`load_records` 반환 순서), 열 순서는
`ai_challenge.datasets.schema.ACTION_CLASSES` (14클래스 canonical 순서).

블렌드/증류에 쓸 때는 `ai_challenge.method.distill` 의 `--teachers` 에
run 의 `model/` 경로 대신 이 저장소 key 를 그대로 쓸 수 있다(로짓 캐시 히트).

| key | 백본 | teacher 프리셋 | 학습 fold | fold0 OOF macro-F1 | 저장일 | 비고 |
|-----|------|---------------|----------|--------------------|--------|------|
| coder7b_base_f0__base | Qwen/Qwen2.5-Coder-7B | base | 0 | 0.76072 | 2026-07-10 | Coder-7B base 2ep (이종 백본) |
| qwen3b_base_f0__base | Qwen/Qwen2.5-3B | base | 0 | 0.76625 | 2026-07-10 | 3B base 3ep 레시피 수정판 |
| qwen3b_cueshist_f0__cues_hist | Qwen/Qwen2.5-3B | cues_hist | 0 | 0.76474 | 2026-07-10 | teacher 1호기: 팀 3B 레시피 재현(fold0) |
| qwen7b_base_f0__base | Qwen/Qwen2.5-7B | base | 0 | 0.77605 | 2026-07-10 | 7B base 2ep (OOF 0.77605, +bias 0.78081) |
| team_q14b | Qwen2.5-14B QLoRA ml640 (팀) | base | all-train | - | 2026-07-10 | val OOF 0.7651(LOG). train argmax 0.8347 | 70k argmax=0.8347 |
| team_q32b | Qwen2.5-32B QLoRA ml640 (팀) | base | all-train | - | 2026-07-10 | train argmax 0.7928 | 70k argmax=0.7928 |
| team_q3b | Qwen2.5-3B QLoRA ml640 (팀) | base | all-train | - | 2026-07-10 | val OOF 0.7639(LOG) | 70k argmax=0.8216 |
| team_q3b_bl3 | q3b 라벨보정 blend λ0.3 (팀) | base | all-train | - | 2026-07-10 | correct_store | 70k argmax=0.9393 |
| team_q3b_ka | q3b 라벨보정 ka λ0.3 (팀) | base | all-train | - | 2026-07-10 | correct_store | 70k argmax=0.9997 |
| team_qgpt5 | Qwen3-4B GPT5증류 teacher (팀) | base | all-train | - | 2026-07-10 | val OOF 0.7741(LOG). w6 핵심 teacher | 70k argmax=0.8166 |
| team_qgptoss | gpt-oss 증류 4B teacher (팀) | base | all-train | - | 2026-07-10 | train argmax 0.8227 | 70k argmax=0.8227 |
| team_qw6 | kd_w6_all teacher blend 캐시 (팀) | base | all-train | - | 2026-07-10 | LB 0.7874를 만든 최종 soft-target | 70k argmax=0.8213 |
