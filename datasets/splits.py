"""StratifiedGroupKFold 기반 교차검증 분할.

group = 세션 id (같은 세션 step 은 한 fold 에 몰아 누수 차단),
stratify = action 라벨 (fold 간 클래스 비율 유지 → Macro-F1 추정 안정화).
분할 결과는 (id, fold) CSV 로 저장해 여러 실험/도구에서 재사용한다.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .io import ActionSample
from .schema import CLASS_TO_ID, session_id_from_id


def assign_folds(
    samples: list[ActionSample],
    n_splits: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """각 샘플에 검증 fold 번호(0..n_splits-1)를 부여한 배열을 반환.

    fold[i] = i 번 샘플이 '검증셋'으로 쓰이는 fold. 라벨이 없으면 실패한다.
    """
    if any(s.action is None for s in samples):
        raise ValueError("assign_folds 는 라벨이 있는 train 샘플에만 사용할 수 있다.")

    y = np.array([CLASS_TO_ID[s.action] for s in samples])
    groups = np.array([session_id_from_id(s.id) for s in samples])
    X = np.zeros(len(samples))  # 분할에 X 값은 쓰이지 않음

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = np.full(len(samples), -1, dtype=int)
    for f, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
        fold[val_idx] = f

    assert (fold >= 0).all(), "일부 샘플이 어떤 fold 에도 배정되지 않았다."
    verify_no_leakage(samples, fold)
    return fold


def verify_no_leakage(samples: list[ActionSample], fold: np.ndarray) -> None:
    """한 세션이 두 개 이상 fold 에 걸치지 않는지 검증(누수 방지 확인)."""
    sess_to_fold: dict[str, int] = {}
    for s, f in zip(samples, fold):
        sess = session_id_from_id(s.id)
        prev = sess_to_fold.setdefault(sess, int(f))
        if prev != int(f):
            raise AssertionError(f"세션 {sess} 이 fold {prev}, {f} 에 걸쳐 누수 발생.")


def train_val_indices(fold: np.ndarray, val_fold: int) -> tuple[np.ndarray, np.ndarray]:
    """지정 val_fold 에 대한 (train_idx, val_idx)."""
    val_idx = np.where(fold == val_fold)[0]
    train_idx = np.where(fold != val_fold)[0]
    return train_idx, val_idx


def save_folds(samples: list[ActionSample], fold: np.ndarray, path: str | Path) -> None:
    """(id, fold) 를 CSV 로 저장."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "fold"])
        for s, fo in zip(samples, fold):
            w.writerow([s.id, int(fo)])


def load_folds(path: str | Path, samples: list[ActionSample]) -> np.ndarray:
    """저장된 fold CSV 를 로드해 samples 순서에 맞춘 배열로 반환."""
    path = Path(path)
    with path.open(encoding="utf-8", newline="") as f:
        id_to_fold = {row["id"]: int(row["fold"]) for row in csv.DictReader(f)}
    return np.array([id_to_fold[s.id] for s in samples], dtype=int)
