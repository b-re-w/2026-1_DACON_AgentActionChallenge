"""OOF 예측(oof.csv) → per-class 지표 + 혼동행렬 상위 오분류쌍 리포트.

병목 클래스(어떤 클래스가 어떤 클래스로 새는지)를 빠르게 진단해 다음 실험
(직렬화 신호·threshold·클래스가중)을 겨냥한다.

사용:
    uv run python -m ai_challenge.method.analyze_oof --runs runs/mdeberta_base_f0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ai_challenge.datasets import ACTION_CLASSES, CLASS_TO_ID


def load_oof_csv(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (run_dir / "oof.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    y = np.array([CLASS_TO_ID[r["true"]] for r in rows])
    p = np.array([CLASS_TO_ID[r["pred"]] for r in rows])
    return y, p


def main() -> None:
    ap = argparse.ArgumentParser(description="OOF 혼동행렬/병목 리포트")
    ap.add_argument("--runs", nargs="+", required=True, help="oof.csv 를 가진 run 디렉터리(합쳐서 집계)")
    ap.add_argument("--top", type=int, default=15, help="상위 오분류쌍 개수")
    args = ap.parse_args()

    ys, ps = [], []
    for rd in args.runs:
        y, p = load_oof_csv(Path(rd))
        ys.append(y); ps.append(p)
    y = np.concatenate(ys); p = np.concatenate(ps)

    macro = f1_score(y, p, average="macro")
    print(f"=== OOF 집계: n={len(y)}  macro_f1={macro:.5f} ===\n")
    print(classification_report(y, p, target_names=ACTION_CLASSES, digits=4, zero_division=0))

    cm = confusion_matrix(y, p, labels=list(range(len(ACTION_CLASSES))))
    # 대각 제외 최다 오분류쌍
    pairs = []
    for i in range(len(ACTION_CLASSES)):
        row_tot = cm[i].sum()
        for j in range(len(ACTION_CLASSES)):
            if i != j and cm[i, j] > 0:
                pairs.append((cm[i, j], cm[i, j] / max(row_tot, 1), i, j))
    pairs.sort(reverse=True)
    print(f"\n=== 상위 {args.top} 오분류쌍 (true → pred : count, row%) ===")
    for cnt, frac, i, j in pairs[:args.top]:
        print(f"  {ACTION_CLASSES[i]:16s} → {ACTION_CLASSES[j]:16s} : {cnt:4d}  ({frac*100:4.1f}%)")


if __name__ == "__main__":
    main()
