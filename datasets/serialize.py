"""ActionSample → 텍스트 직렬화.

인코더/디코더 어느 모델에 넣든 동일한 입력 표현을 쓰도록 featurization 을
이 한 곳에 모은다. DATASET.md 의 강한 신호(직전 action, open_files, ci_status,
git_dirty, current_prompt 키워드)를 명시적 토큰으로 드러내는 것을 기본값으로 한다.
"""

from __future__ import annotations

from .io import ActionSample

# 섹션 구분 특수 토큰 (모델 토크나이저에 special token 으로 추가하면 더 좋다)
TOK_META = "[META]"
TOK_LAST = "[LAST_ACTION]"
TOK_HISTORY = "[HISTORY]"
TOK_PROMPT = "[PROMPT]"


def _truncate(text: str, limit: int) -> str:
    text = " ".join(str(text).split())  # 공백/개행 정규화
    return text if len(text) <= limit else text[:limit] + "…"


def _format_meta(sample: ActionSample) -> str:
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
    lang_mix = ws.get("language_mix") or {}
    if lang_mix:
        top = sorted(lang_mix.items(), key=lambda kv: -kv[1])[:3]
        parts.append("codemix=" + ",".join(f"{k}:{v:.2f}" for k, v in top))
    return " ".join(parts)


def _format_history_turn(turn: dict, text_limit: int) -> str:
    role = turn.get("role")
    if role == "user":
        return "U: " + _truncate(turn.get("content", ""), text_limit)
    # assistant_action
    name = turn.get("name", "?")
    args = turn.get("args") or {}
    arg_str = ",".join(f"{k}={_truncate(v, 40)}" for k, v in list(args.items())[:4])
    result = _truncate(turn.get("result_summary", ""), text_limit)
    return f"A: {name}({arg_str}) -> {result}"


def serialize_sample(
    sample: ActionSample,
    *,
    max_history_turns: int = 12,
    prompt_limit: int = 512,
    history_text_limit: int = 160,
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
        chunks.append(f"{TOK_META} {_format_meta(sample)}")

    if include_last_action:
        chunks.append(f"{TOK_LAST} {sample.last_action or 'none'}")

    if include_history and sample.history:
        recent = sample.history[-max_history_turns:]
        lines = [_format_history_turn(t, history_text_limit) for t in recent]
        chunks.append(TOK_HISTORY + " " + " ".join(lines))

    chunks.append(f"{TOK_PROMPT} {_truncate(sample.current_prompt, prompt_limit)}")
    return " ".join(chunks)


# 토크나이저에 추가하면 좋은 special token 목록 (선택)
SPECIAL_TOKENS: list[str] = [TOK_META, TOK_LAST, TOK_HISTORY, TOK_PROMPT]
