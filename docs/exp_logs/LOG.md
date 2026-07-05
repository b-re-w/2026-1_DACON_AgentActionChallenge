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
| 2026-07-05 | speculative_team | teacher 벤치 — Qwen2.5-3B | oof_macro_f1=0.7737, epochs=2 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | speculative_team | KD 3B→0.5B student (T3 α0.5) | oof_macro_f1=0.776, teacher=3B(0.774), student_solo=0.767 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | others | 입력 피처 실험 (직렬화 프리셋) | base=0.76, paths=0.7646, hist=0.7652, rich=0.7658 | [로그](others/2026-07-05.md) |
| 2026-07-05 | speculative_team | KD α 스윕 (3B→0.5B, T3) | alpha0.5=0.776, alpha0.3=0.7788 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | speculative_team | KD α 스윕 완료 (α0.2) | alpha0.5=0.776, alpha0.3=0.7788, alpha0.2=0.7789 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | speculative_team | KD 하이퍼 튜닝 완료 (T·α) | best=T3_a0.25, oof=0.7789, T4a0.25=0.7775 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | speculative_team | KD 최종 비교 (base vs rich 입력) | base+KD=0.7801, baseT_richS=0.778, richT_richS=0.7747, 입력rich_직접=+0.006 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | speculative_team | 제출: base+KD 전체70k → LB 0.7839 | lb=0.7839, oof=0.7801, prev_best_lb=0.7761 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | speculative_team | Teacher 앙상블 (4-teacher KD) | ensemble4_oof=0.781, single_oof=0.7801, delta=+0.0009 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | others | 다른 접근 총정리 — 전부 Qwen0.5B-KD 미달 | lgbm_structural=0.42, xlmr_student=0.64, phi_teacher_ep1=0.57, ensemble_delta=+0.0009 | [로그](others/2026-07-05.md) |
| 2026-07-05 | speculative_team | Teacher 앙상블 seed별 3B OOF | base_s42=0.7745, s43=0.7717, s44=0.7721, s45=0.7713, s47=0.7762, s46_diverged=0.7334, rich_s42=0.7676 | [로그](speculative_team/2026-07-05.md) |
| 2026-07-05 | others | 에러분석 — 천장 진단(탐색도구 애매성) | edit/write/patch/respond_F1=0.95~0.996, read/grep/list/glob_F1=0.53~0.67 | [로그](others/2026-07-05.md) |
| 2026-07-05 | decoder_team | 다른 아키텍처 student (인코더 KD) | xlm_roberta_base=0.6398, mdeberta_v3_base=0.5968, qwen0.5B_ref=0.779 | [로그](decoder_team/2026-07-05.md) |
