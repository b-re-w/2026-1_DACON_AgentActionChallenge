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
| 2026-07-06 | speculative_team | Qwen3 라운드 — teacher·student 전부 미달/동률 | q3_4b_teacher=0.7727, q3_1.7b_teacher=0.7702, qwen2.5_3b_teacher=0.7745, q3_0.6b_student=0.7794, gemma3_270m_student=0.7763, qwen2.5_0.5b_student=0.779 | [로그](speculative_team/2026-07-06.md) |
| 2026-07-06 | speculative_team | 다양성 앙상블 제출 → LB 0.78621 (최고 갱신) | lb=0.78621, fold0_oof=0.7816, prev_best_lb=0.78519, delta=+0.001 | [로그](speculative_team/2026-07-06.md) |
| 2026-07-06 | others | 대형 teacher·5.x·FSDP 환경 제약 정리 | qwen7b=진행중, 14b_fsdp=PCIe 146h 비현실적, claude4b=0.7719, gpt5_4b=0.7741, r1_7b=0.7655 | [로그](others/2026-07-06.md) |
| 2026-07-08 | speculative_team | KD 그리드 완성 — rich teacher → base student (비대칭) | richT_baseS_oof=0.7757, latency_ms_per_sample=2.789, proj_30k_s=83.7, base_KD_ref=0.7801, baseT_richS=0.778, richT_richS=0.7747, teacher_val_argmax=0.7681 | [로그](speculative_team/2026-07-08.md) |
| 2026-07-08 | others | max_length 512→640 (잘림 제거) — 타깃 +24%p지만 전체 무이동 | ml640_oof=0.7802, ml512_oof_ref=0.7801, trunc111_acc_512=0.5495, trunc111_acc_640=0.7928, trunc_delta=0.2432, trunc_rate=0.0079, proj_30k_s_640=80.1, proj_30k_s_512=90.1, teacher_argmax_512=0.7745, teacher_argmax_640=0.7766 | [로그](others/2026-07-08.md) |
| 2026-07-09 | others | QLoRA 대형 teacher 사다리 + 재사용 store — 크기↑ 무익 | t3b_qlora=0.7639, t14b_qlora=0.7651, student_3b=0.7747, student_14b=0.7732, store_MB=6.5 | [로그](others/2026-07-09.md) |
| 2026-07-09 | others | 단일-student 레버 총점검 — 전부 실패(앙상블만 생존) | threshold_q3b_student=-0.0019, threshold_w6_student=-0.0030, dkd=0.7633(vanilla 0.7747), cues=하락 | [로그](others/2026-07-09.md) |
| 2026-07-09 | others | 이질 teacher(GLM 격리env) + 서버 tooling + 제출 | glm4_32b=격리4.55, submit_q3b_all_lb=0.781638, best_w6=0.787396, hetero_tooling=train_hetero.sh | [로그](others/2026-07-09.md) |
| 2026-07-09 | others | feature engineering(CUES) 재검토 — OOF·LB 둘 다 하락 | cues_teacher_oof=0.7693(base 0.7745), cues_student_lb=0.782716(w6 0.787396) | [로그](feature_engineering.md) |
