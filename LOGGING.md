# 실험 로깅 규칙 (실험 에이전트 필독)

모든 실험 에이전트는 **실험이 끝나면 반드시** 결과를 기록한다. 기록 위치는 두 곳:

1. **상세 기록** — `docs/exp_logs/<컨셉>/<YYYY-MM-DD>.md`
   컨셉별 폴더 · **날짜별 파일**(형식 `YYYY-MM-DD`로 통일). 같은 날 여러 실험은
   시각 섹션(`## [HH:MM] 타이틀`)으로 한 파일에 이어붙인다.
2. **요약 인덱스** — `docs/exp_logs/LOG.md`
   실험마다 **한 줄**: 날짜 · 컨셉 · 타이틀 · 핵심 수치 · 상세 링크.

## 컨셉 폴더 (컨셉별로 로그 폴더가 따로 있음)

| 폴더 | 컨셉 |
|------|------|
| `docs/exp_logs/encoder_team/` | 접근 A · Encoder-based (mDeBERTa/XLM-R) |
| `docs/exp_logs/decoder_team/` | 접근 B · Decoder-based SLM (Qwen3 등) |
| `docs/exp_logs/speculative_team/` | 접근 C · Speculative Decoding/MatFormer (Gemma 3n) |
| `docs/exp_logs/others/` | 기타 (데이터·검증·앙상블·전처리 등) |

## 사용법 — 손으로 파일 쓰지 말고 헬퍼를 호출한다

```python
from ai_challenge.utils.experiment_log import log_experiment

log_experiment(
    concept="encoder_team",
    title="mDeBERTa-v3 baseline (GroupKFold OOF)",
    metrics={"oof_macro_f1": 0.712, "folds": 5},     # 핵심 수치 → LOG.md 요약에 그대로
    notes="class-weighted CE, max_len=512, seed=42",  # 선택: 상세 서술(마크다운)
)
```

- 날짜 형식(`YYYY-MM-DD`)과 파일 배치는 헬퍼가 자동 처리한다.
- `metrics` 는 **간결하게** — 이 값이 LOG.md 요약 줄에 들어간다.
- `notes` 에는 하이퍼파라미터·관찰·다음 액션을 자유롭게(상세 파일에만 기록).

## 원칙

- **실패·무효 실험도 기록한다** — 왜 안 됐는지가 자산이다.
- **LB 제출 점수**는 어떤 컨셉/설정인지와 함께 반드시 기록한다.
- LOG.md 를 손으로 편집하기보다 `log_experiment` 를 사용해 포맷을 통일한다.
- 스켈레톤이 없으면 `python -m ai_challenge.utils.experiment_log` 로 생성된다.
