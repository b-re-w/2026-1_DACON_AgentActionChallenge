"""ActionSample → 텍스트 직렬화.

인코더/디코더 어느 모델에 넣든 동일한 입력 표현을 쓰도록 featurization 을
이 한 곳에 모은다. DATASET.md 의 강한 신호(직전 action, open_files, ci_status,
git_dirty, current_prompt 키워드)를 명시적 토큰으로 드러내는 것을 기본값으로 한다.

입력 신호 실험을 위해 프리셋(SERIALIZE_PRESETS)으로 표현을 바꿀 수 있다. 학습·추론이
**같은 프리셋**을 써야 하므로, 제출 시 pack.py 의 SCRIPT_PY 도 동일 프리셋에 맞춘다.
"""

from __future__ import annotations

import os

from .io import ActionSample

# 섹션 구분 특수 토큰 (모델 토크나이저에 special token 으로 추가하면 더 좋다)
TOK_META = "[META]"
TOK_LAST = "[LAST_ACTION]"
TOK_HISTORY = "[HISTORY]"
TOK_PROMPT = "[PROMPT]"


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())  # 공백/개행 정규화
    return text if len(text) <= limit else text[:limit] + "…"


def _format_meta(sample: ActionSample, open_files_names: int = 0) -> str:
    sm = sample.session_meta
    ws = sample.workspace
    open_files = ws.get("open_files") or []
    parts = [
        f"tier={sm.get('user_tier', 'na')}",
        f"lang={sm.get('language_pref', 'na')}",
        f"turn={sm.get('turn_index', 0)}",
        f"budget={sm.get('budget_tokens_remaining', 0)}",
        f"ci={ws.get('last_ci_status', 'none')}",
        f"git={'dirty' if ws.get('git_dirty') else 'clean'}",
        f"open_files={len(open_files)}",  # open_files=0 여부가 최강 신호
        f"loc={ws.get('loc', 0)}",
    ]
    # 열린 파일의 basename(확장자 포함) — 어떤 파일이 열려있는지가 다음 action 을 강하게 시사
    if open_files_names and open_files:
        names = [os.path.basename(str(p)) for p in open_files[:open_files_names]]
        parts.append("openf=" + ",".join(names))
    lang_mix = ws.get("language_mix") or {}
    if lang_mix:
        top = sorted(lang_mix.items(), key=lambda kv: -kv[1])[:3]
        parts.append("codemix=" + ",".join(f"{k}:{v:.2f}" for k, v in top))
    return " ".join(parts)


def _format_history_turn(turn: dict, text_limit: int, arg_value_limit: int = 40, n_args: int = 4) -> str:
    role = turn.get("role")
    if role == "user":
        return "U: " + _truncate(turn.get("content", ""), text_limit)
    # assistant_action
    name = turn.get("name", "?")
    args = turn.get("args") or {}
    arg_str = ",".join(f"{k}={_truncate(v, arg_value_limit)}" for k, v in list(args.items())[:n_args])
    result = _truncate(turn.get("result_summary", ""), text_limit)
    return f"A: {name}({arg_str}) -> {result}"


def serialize_sample(
    sample: ActionSample,
    *,
    max_history_turns: int = 12,
    prompt_limit: int = 512,
    history_text_limit: int = 160,
    open_files_names: int = 0,
    arg_value_limit: int = 40,
    n_args: int = 4,
    include_meta: bool = True,
    include_history: bool = True,
    include_last_action: bool = True,
) -> str:
    """샘플을 단일 문자열로 직렬화.

    구조: [META] ... [LAST_ACTION] x [HISTORY] U:.. A:.. [PROMPT] ...
    history 는 최근 max_history_turns 개만 사용한다(시간순 유지).
    """
    chunks: list[str] = []

    if include_meta:
        chunks.append(f"{TOK_META} {_format_meta(sample, open_files_names=open_files_names)}")

    if include_last_action:
        chunks.append(f"{TOK_LAST} {sample.last_action or 'none'}")

    if include_history and sample.history:
        recent = sample.history[-max_history_turns:]
        lines = [
            _format_history_turn(t, history_text_limit, arg_value_limit, n_args)
            for t in recent
        ]
        chunks.append(TOK_HISTORY + " " + " ".join(lines))

    chunks.append(f"{TOK_PROMPT} {_truncate(sample.current_prompt, prompt_limit)}")
    return " ".join(chunks)


# 입력 신호 실험용 프리셋 (serialize_sample 의 kwargs 묶음).
#   base : 현재 기본값(대조군)
#   paths: 열린 파일 basename 추가(최강 신호 후보)
#   hist : history/args/result 를 더 길고 풍부하게
#   rich : paths + hist 결합
SERIALIZE_PRESETS: dict[str, dict] = {
    "base": {},
    "paths": {"open_files_names": 8},
    "hist": {"max_history_turns": 16, "history_text_limit": 280, "arg_value_limit": 80, "n_args": 8},
    "rich": {"open_files_names": 8, "max_history_turns": 16, "history_text_limit": 280,
             "arg_value_limit": 80, "n_args": 8},
}


# 토크나이저에 추가하면 좋은 special token 목록 (선택)
SPECIAL_TOKENS: list[str] = [TOK_META, TOK_LAST, TOK_HISTORY, TOK_PROMPT]
