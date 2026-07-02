"""원천 데이터 로딩: jsonl + labels.csv → ActionSample 리스트."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ActionSample:
    """한 세션의 특정 시점 상태(1개 샘플).

    action 은 train 에서만 존재하고 test 에서는 None 이다.
    """

    id: str
    session_meta: dict[str, Any]
    history: list[dict[str, Any]]
    current_prompt: str
    action: str | None = None

    @property
    def workspace(self) -> dict[str, Any]:
        return self.session_meta.get("workspace", {}) or {}

    @property
    def last_action(self) -> str | None:
        """history 내 가장 최근 assistant_action 의 name (없으면 None).

        DATASET.md 기준 가장 강한 단일 전이 신호.
        """
        for turn in reversed(self.history):
            if turn.get("role") == "assistant_action" and turn.get("name"):
                return turn["name"]
        return None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """JSON Lines 파일을 dict 리스트로 로드 (빈 줄 무시)."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_labels(path: str | Path) -> dict[str, str]:
    """train_labels.csv → {id: action} 매핑."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def load_records(
    jsonl_path: str | Path,
    labels_path: str | Path | None = None,
) -> list[ActionSample]:
    """jsonl(+선택적 labels)을 ActionSample 리스트로 결합.

    labels_path 를 주면 train, 생략하면 test(라벨 None)로 취급한다.
    labels 가 있는데 특정 id 라벨이 없으면 KeyError 로 조기 실패시킨다.
    """
    rows = load_jsonl(jsonl_path)
    labels = load_labels(labels_path) if labels_path is not None else None

    samples: list[ActionSample] = []
    for r in rows:
        action = labels[r["id"]] if labels is not None else None
        samples.append(
            ActionSample(
                id=r["id"],
                session_meta=r.get("session_meta", {}) or {},
                history=r.get("history", []) or [],
                current_prompt=r.get("current_prompt", "") or "",
                action=action,
            )
        )
    return samples
