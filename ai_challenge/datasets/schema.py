"""데이터 스키마 상수 및 라벨 매핑.

14개 action 클래스는 대회 규정상 대소문자까지 정확히 일치해야 하므로,
여기 정의한 canonical 순서/문자열을 프로젝트 전역에서 단일 출처로 사용한다.
"""

from __future__ import annotations

# 14개 예측 대상 클래스 (대소문자 고정, canonical 순서)
ACTION_CLASSES: list[str] = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

NUM_CLASSES: int = len(ACTION_CLASSES)

CLASS_TO_ID: dict[str, int] = {c: i for i, c in enumerate(ACTION_CLASSES)}
ID_TO_CLASS: dict[int, str] = {i: c for i, c in enumerate(ACTION_CLASSES)}

# id 형식: "sess_sim_20260522_006284-step_01" → 세션 = "sess_sim_20260522_006284"
_STEP_SEP = "-step_"


def session_id_from_id(sample_id: str) -> str:
    """샘플 id에서 세션 id를 추출한다 (그룹 분할의 group 키).

    같은 세션의 여러 step은 history/워크스페이스 맥락을 공유하므로
    반드시 같은 fold에 묶여야 한다 (누수 방지).
    """
    return sample_id.split(_STEP_SEP, 1)[0]
