# utils — 제출·조회·로깅 도구

학습과 무관한 주변 도구. 대부분 stdlib 로 작성(제출 zip 에는 안 들어감).

## 파일

| 파일 | 역할 | CLI |
|------|------|-----|
| `pack.py` | 제출 zip 패키징(fp16, 1GB 검증) + 로컬 검증(평가서버 모사) + `SCRIPT_PY`(제출 script.py 원본) | `pack pack …` / `pack eval …` |
| `submission.py` | DACON 제출 + 제출기록 조회(memo/점수). 공유 팀이라 `--mine`(user_id) 필터 | `submission submit\|list` |
| `leaderboard.py` | 공개 리더보드 순위 조회(Node vm 로 Nuxt 파싱) | `leaderboard --team …` |
| `experiment_log.py` | 실험 로그 헬퍼 `log_experiment()` → `docs/exp_logs/` | — |

## 규칙

- 제출 zip 은 항상 **`build/<name>.zip`** 로 생성(루트 오염 금지). 제출 전 `pack eval` 로 검증.
- `pack.py` 의 `SCRIPT_PY` 안 `serialize()` 는 `datasets/serialize.py` 와 **동일 로직 유지**.
- 제출은 일 10회 한도 소모(비가역) — `--yes` 필수. 조회는 [[dacon-submission-api]] 참고.
