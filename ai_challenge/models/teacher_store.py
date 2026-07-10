"""teacher soft-label(로짓) 중앙 저장소 — `teacher_logits/`.

KD 의 핵심 자산인 teacher 별 전체 train(70k) 로짓을 한 폴더에 모아 보관한다.
나중에 어떤 조합으로든 다시 블렌드/증류할 수 있도록:

  teacher_logits/
    ├── TEACHERS.md      사람용 목록(모델·프리셋·OOF 점수·비고) — registry 에서 자동 생성
    ├── registry.json    기계용 레지스트리(항목별 메타)
    └── <key>.npy        로짓 (N=70000, C=14) float32 — train.jsonl 줄 순서와 동일

key = "<run이름>__<teacher프리셋>" (예: qwen3b_cueshist_f0__cues_hist).
로짓 순서는 **train.jsonl 파일 순서**(load_train_records 반환 순서)와 동일하다.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

STORE_DIR = Path("teacher_logits")
REGISTRY = STORE_DIR / "registry.json"
INDEX_MD = STORE_DIR / "TEACHERS.md"

_MD_HEADER = """# Teacher soft-label 저장소

KD(지식증류)용 teacher 로짓 모음. 각 `.npy` 는 shape **(70000, 14)** float32,
행 순서는 **train.jsonl 줄 순서**(`load_records` 반환 순서), 열 순서는
`ai_challenge.datasets.schema.ACTION_CLASSES` (14클래스 canonical 순서).

블렌드/증류에 쓸 때는 `ai_challenge.method.distill` 의 `--teachers` 에
run 의 `model/` 경로 대신 이 저장소 key 를 그대로 쓸 수 있다(로짓 캐시 히트).

| key | 백본 | teacher 프리셋 | 학습 fold | fold0 OOF macro-F1 | 저장일 | 비고 |
|-----|------|---------------|----------|--------------------|--------|------|
"""


def _load_registry() -> dict:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf-8"))
    return {}


def _write_index_md(reg: dict) -> None:
    lines = [_MD_HEADER.rstrip("\n")]
    for key in sorted(reg):
        e = reg[key]
        oof = e.get("oof_macro_f1")
        oof_s = f"{oof:.5f}" if isinstance(oof, (int, float)) else "-"
        lines.append(
            f"| {key} | {e.get('backbone','?')} | {e.get('preset','?')} "
            f"| {e.get('fold','?')} | {oof_s} | {e.get('saved_at','?')[:10]} "
            f"| {e.get('note','')} |"
        )
    INDEX_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def logits_path(key: str) -> Path:
    return STORE_DIR / f"{key}.npy"


def make_key(run_name: str, preset: str) -> str:
    return f"{run_name}__{preset}"


def save_teacher_logits(
    key: str,
    logits: np.ndarray,
    *,
    backbone: str,
    preset: str,
    fold: str | int,
    model_dir: str | None = None,
    oof_macro_f1: float | None = None,
    note: str = "",
) -> Path:
    """로짓 저장 + registry/TEACHERS.md 갱신(멱등: 같은 key 는 덮어씀)."""
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    path = logits_path(key)
    np.save(path, logits.astype(np.float32))

    reg = _load_registry()
    reg[key] = {
        "backbone": backbone,
        "preset": preset,
        "fold": str(fold),
        "model_dir": model_dir,
        "oof_macro_f1": oof_macro_f1,
        "n_rows": int(len(logits)),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
    }
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_index_md(reg)
    print(f"[teacher-store] saved {path}  ({len(logits)}x{logits.shape[1]})")
    return path


def load_teacher_logits(key: str) -> np.ndarray | None:
    """key 로 로짓 로드(없으면 None)."""
    path = logits_path(key)
    return np.load(path) if path.exists() else None


def update_entry(key: str, **fields) -> None:
    """registry 항목 메타 갱신(예: OOF 점수 추가 기입)."""
    reg = _load_registry()
    if key not in reg:
        raise KeyError(f"teacher-store 에 없는 key: {key}")
    reg[key].update(fields)
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_index_md(reg)


def list_teachers() -> dict:
    return _load_registry()
