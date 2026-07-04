# datasets — 데이터 계층

원천 데이터 로딩 → 직렬화 → 검증분할 → torch Dataset. `ai_challenge` 하위에 두어
HuggingFace `datasets` 와의 이름 충돌(shadowing)을 피한다.

## 파일

| 파일 | 역할 |
|------|------|
| `schema.py` | 14 action 클래스 상수·매핑(`CLASS_TO_ID`/`ID_TO_CLASS`)·세션id 추출. **단일 출처** |
| `io.py` | `ActionSample` dataclass + jsonl/labels 로딩(`load_records`) |
| `serialize.py` | ActionSample → 텍스트 직렬화 `[META][LAST_ACTION][HISTORY][PROMPT]` |
| `splits.py` | StratifiedGroupKFold(group=세션) + 누수 검증 + fold 저장/로드 |
| `action_dataset.py` | `ActionDataset`(토크나이즈) + `AgentActionDataset`(torchvision 규격 자동 다운로드) |

## 규칙

- 14클래스 문자열·순서는 **`schema.py` 한 곳**에서만 정의(대소문자 정확히 일치 필수).
- 직렬화 로직을 바꾸면 제출 추론 코드(`utils/pack.py` 의 `SCRIPT_PY`)의 `serialize()` 도
  **동일하게** 맞춰야 한다(학습/추론 표현 일치가 정확도에 직결).
- `AgentActionDataset.download()` 는 압축 해제 후 아카이브(open.zip)를 삭제한다.
