# 실험 로그 인덱스 (LOG.md)

각 실험 결과를 **한 줄**로 요약한다(타이틀 + 핵심 수치). 상세는 컨셉 폴더의 날짜별 파일 참고.
자동 기록: `ai_challenge.utils.experiment_log.log_experiment(...)`.

| 날짜 | 컨셉 | 타이틀 | 핵심 수치 | 상세 |
|------|------|--------|-----------|------|
| 2026-07-02 | encoder_team | 접근 A — mDeBERTa-v3-base baseline | oof_macro_f1=0.6552, lb=0.6431 | [로그](encoder_team/2026-07-02.md) |
| 2026-07-02 | decoder_team | 접근 B — Qwen2.5-0.5B (class-weighted) | oof_macro_f1=0.7575, lb=0.7601 | [로그](decoder_team/2026-07-02.md) |
| 2026-07-03 | decoder_team | bnw — Qwen0.5B no-class-weight | oof_macro_f1=0.7669, lb=0.7732 | [로그](decoder_team/2026-07-03.md) |
| 2026-07-03 | decoder_team | Bfull — bnw + 전체 70k (제출 최고) | lb=0.7761, rank=4위(당시) | [로그](decoder_team/2026-07-03.md) |
| 2026-07-03 | decoder_team | ablation — 5에폭 / Instruct / 4에폭 | B5_5ep=0.7461, binst=0.7575, ncw4_4ep=0.7584 | [로그](decoder_team/2026-07-03.md) |
| 2026-07-03 | others | 효과 없던 레버 — threshold / 앙상블 | threshold_heldout=+0.0002, ensemble_heldout=+0.0013 | [로그](others/2026-07-03.md) |
| 2026-07-05 | others | 데이터 누수 점검 — 테스트는 안전 | train_next_step_leak=100%/83.3%, test_leak=none | [로그](others/2026-07-05.md) |
| 2026-07-05 | speculative_team | teacher 벤치 — Qwen2.5-1.5B | oof_macro_f1=0.7565 | [로그](speculative_team/2026-07-05.md) |
