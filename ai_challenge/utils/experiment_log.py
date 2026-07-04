"""실험 로그 유틸 — 실험이 끝나면 결과를 표준 포맷으로 기록한다.

규칙(자세한 지침은 프로젝트 루트 LOGGING.md 참고):
- **상세 기록**: `docs/logs/<concept>/<YYYY-MM-DD>.md` (컨셉별 폴더 · 날짜별 파일,
  같은 날 여러 실험은 시각 섹션으로 이어붙인다)
- **요약 인덱스**: `docs/logs/LOG.md` 에 한 줄(날짜·컨셉·타이틀·핵심 수치·링크)

에이전트는 실험이 끝나면 반드시 `log_experiment(...)` 를 호출한다.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = _ROOT / "docs" / "logs"
INDEX = LOGS_DIR / "LOG.md"

# 컨셉 슬러그 → 설명 (IDEA.md 의 접근과 대응). 각 컨셉은 별도 로그 폴더를 가진다.
CONCEPTS: dict[str, str] = {
    "encoder_team": "접근 A · Encoder-based (mDeBERTa/XLM-R)",
    "decoder_team": "접근 B · Decoder-based SLM (Qwen3 등)",
    "speculative_team": "접근 C · Speculative Decoding/MatFormer (Gemma 3n)",
    "others": "기타 (데이터·검증·앙상블·전처리 등)",
}

_INDEX_HEADER = """# 실험 로그 인덱스 (LOG.md)

각 실험 결과를 **한 줄**로 요약한다(타이틀 + 핵심 수치). 상세는 컨셉 폴더의 날짜별 파일 참고.
자동 기록: `ai_challenge.utils.experiment_log.log_experiment(...)`.

| 날짜 | 컨셉 | 타이틀 | 핵심 수치 | 상세 |
|------|------|--------|-----------|------|
"""


def init_log_dirs() -> None:
    """docs/logs 스켈레톤(컨셉 폴더 + LOG.md)을 생성(멱등)."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for slug in CONCEPTS:
        d = LOGS_DIR / slug
        d.mkdir(parents=True, exist_ok=True)
        gk = d / ".gitkeep"
        if not gk.exists():
            gk.write_text("", encoding="utf-8")
    # 없거나 비어 있으면 헤더 작성 (사용자가 만든 빈 LOG.md 포함)
    if not INDEX.exists() or not INDEX.read_text(encoding="utf-8").strip():
        INDEX.write_text(_INDEX_HEADER, encoding="utf-8")


def _fmt_metrics(metrics: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in metrics.items()) if metrics else "-"


def log_experiment(
    concept: str,
    title: str,
    metrics: dict | None = None,
    notes: str = "",
    date: str | None = None,
) -> Path:
    """실험 결과 1건을 기록한다.

    Parameters
    ----------
    concept : CONCEPTS 의 키 중 하나 (예: "encoder").
    title   : 실험 제목 (예: "mDeBERTa-v3 baseline (GroupKFold OOF)").
    metrics : 핵심 수치 dict (예: {"oof_macro_f1": 0.712, "folds": 5}).
    notes   : 상세 파일에 덧붙일 자유 서술(마크다운 가능).
    date    : YYYY-MM-DD. 생략 시 오늘.

    Returns 상세 로그 파일 경로.
    """
    if concept not in CONCEPTS:
        raise ValueError(f"알 수 없는 컨셉 '{concept}'. 가능: {list(CONCEPTS)}")
    metrics = metrics or {}
    now = datetime.now()
    date = date or now.strftime("%Y-%m-%d")
    time_s = now.strftime("%H:%M")
    init_log_dirs()

    # 1) 컨셉별·날짜별 상세 파일 (같은 날이면 이어붙임)
    detail = LOGS_DIR / concept / f"{date}.md"
    if not detail.exists():
        detail.write_text(f"# {concept} — {date}\n\n> {CONCEPTS[concept]}\n", encoding="utf-8")
    block = [f"\n## [{time_s}] {title}\n"]
    for k, v in metrics.items():
        block.append(f"- **{k}**: {v}")
    if notes:
        block.append(f"\n{notes}")
    with detail.open("a", encoding="utf-8") as f:
        f.write("\n".join(block) + "\n")

    # 2) 요약 인덱스에 한 줄
    if not INDEX.exists():
        INDEX.write_text(_INDEX_HEADER, encoding="utf-8")
    row = f"| {date} | {concept} | {title} | {_fmt_metrics(metrics)} | [로그]({concept}/{date}.md) |\n"
    with INDEX.open("a", encoding="utf-8") as f:
        f.write(row)

    return detail


if __name__ == "__main__":
    init_log_dirs()
    print("[OK] docs/logs 스켈레톤 생성:", LOGS_DIR)
