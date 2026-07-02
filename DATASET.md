## DATASET 구조
1. train.jsonl - 학습 입력
2. train_labels.csv - 학습 정답

### 입력 (train.jsonl / test.jsonl)
한 줄 = 한 샘플(JSON 객체), 4개 최상위 키:

| 키 | 내용 |
|---|---|
| `id` | 샘플 식별자 (`sess_..._step_02`) |
| `session_meta` | 세션·작업공간 메타정보 |
| `history` | 이전 대화·행동 이력 (0~12개, user↔action 교대) |
| `current_prompt` | 현재 사용자 발화 → 이 시점의 다음 행동이 타깃 |

- session_meta 세부
    - 스칼라: user_tier(enterprise/pro/free), language_pref(ko/en/mixed), budget_tokens_remaining(int), turn_index(int), elapsed_session_sec(int)
    - workspace: language_mix(언어비율 dict), loc(int), git_dirty(bool), open_files(list), last_ci_status(passed/failed/none)

- history 항목 두 종류 (시간순 교대)
    - user 턴: role, content
    - action 턴: role, name(14클래스 중 하나), args, result_summary

### 타깃 (train_labels.csv)
id + action, 14개 클래스 (대소문자 정확히 일치 필수): id로 train.jsonl과 매칭되는 형태.

| 그룹 | 클래스 |
|---|---|
| 탐색(읽기) | `read_file`, `grep_search`, `list_directory`, `glob_pattern` |
| 수정(쓰기) | `edit_file`, `write_file`, `apply_patch` |
| 실행 | `run_bash`, `run_tests`, `lint_or_typecheck` |
| 대화/기타 | `ask_user`, `plan_task`, `web_search`, `respond_only` |


## DATA 분석 (간략)
1. history 마지막 행동 → 다음 행동 (강한 전이 패턴)
에이전트 행동엔 뚜렷한 흐름이 있습니다.
- write_file 다음엔 edit_file 40% (새로 만들고 바로 수정)
- read_file 다음엔 edit_file 29% (읽고 고침)
- edit_file 다음엔 run_tests 23% (고치고 테스트)
- lint_or_typecheck 다음엔 apply_patch 25% (검사하고 패치)
- list_directory 다음엔 read_file 25% (목록 보고 읽기)
- history 없음(세션 첫 스텝)이면 list_directory 20% / read_file 17% / plan_task 12% — 탐색·계획으로 시작

2. workspace 상태 → 다음 행동 (매우 강한 신호, lift 높음)
- open_files=[](열린 파일 없음)일 때: write_file lift 2.83, list_directory 2.34, plan_task 1.92 — 파일이 안 열려 있으면 새로 만들거나 탐색부터 함. 이게 가장 강한 단일 신호.
- last_ci_status=failed: apply_patch(1.40)·run_tests(1.23)↑ — CI 깨지면 고치고 다시 테스트
- last_ci_status=passed: respond_only lift 1.48 — 통과했으니 그냥 응답하고 끝
- git_dirty=True: run_tests·lint·apply_patch↑ — 변경 있으니 검증

3. current_prompt 키워드 → 다음 행동 (직접적, 텍스트 신호)
- "write"/"create" → write_file lift 10배 (희소 클래스인데 키워드가 확실)
- "plan" → plan_task 4.1배, "patch" → edit_file 3.5배, "grep" → grep_search 2.7배, "run" → run_bash 2.7배
- 이건 TF-IDF가 이미 어느 정도 잡습니다. 단, "write/create → write_file"처럼 희소 클래스를 살리는 키워드가 Macro-F1에 결정적입니다.
