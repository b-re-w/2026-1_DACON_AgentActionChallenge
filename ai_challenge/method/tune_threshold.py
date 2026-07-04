"""OOF logits 기반 per-class bias(threshold) 튜닝 — Macro-F1 직접 최적화.

argmax(logits + bias) 의 bias 를 coordinate ascent 로 최적화한다. 재학습 불필요.
과적합 점검: OOF(val fold)를 tune/check 로 나눠 일반화를 확인한다.
공통 로직은 ``ai_challenge.models.common`` (OOF logits = predict_logits).

산출물: <model-dir>/bias.json

사용:
    uv run python -m ai_challenge.method.tune_threshold --model-dir runs/B/model --val-fold 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ai_challenge.datasets import ACTION_CLASSES, CLASS_TO_ID, NUM_CLASSES
from ai_challenge.models.common import get_folds, load_train_records, predict_logits


def compute_oof_logits(model_dir, val_fold, max_length):
    """저장된 모델로 val fold 의 logits 와 정답을 계산한다."""
    records = load_train_records()
    fold = get_folds(records)
    val = [s for s, f in zip(records, fold) if f == val_fold]
    logits = predict_logits(model_dir, val, max_length=max_length)
    y = np.array([CLASS_TO_ID[s.action] for s in val])
    return logits, y


def macro_f1(logits, y, bias):
    return f1_score(y, (logits + bias).argmax(1), average="macro")


def coordinate_ascent(logits, y, rounds=8):
    """per-class bias 를 coordinate ascent 로 최적화(coarse→fine)."""
    bias = np.zeros(NUM_CLASSES, dtype=np.float32)
    best = macro_f1(logits, y, bias)
    for grid in (np.arange(-4, 4.01, 0.25), np.arange(-1, 1.01, 0.05)):
        for _ in range(rounds):
            improved = False
            for c in range(NUM_CLASSES):
                base = bias[c]
                cand, cand_f1 = base, best
                for d in grid:
                    bias[c] = base + d
                    f = macro_f1(logits, y, bias)
                    if f > cand_f1:
                        cand, cand_f1 = base + d, f
                bias[c] = cand
                if cand_f1 > best + 1e-9:
                    best, improved = cand_f1, True
            if not improved:
                break
    return bias, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logits, y = compute_oof_logits(args.model_dir, args.val_fold, args.max_length)
    base_f1 = macro_f1(logits, y, np.zeros(NUM_CLASSES))
    print(f"[base] OOF macro_f1 (bias=0) = {base_f1:.4f}")

    # 과적합 점검: tune/check 분할
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(y))
    cut = int(len(y) * 0.7)
    ti, ci = idx[:cut], idx[cut:]
    bias_t, _ = coordinate_ascent(logits[ti], y[ti])
    f1_check_base = macro_f1(logits[ci], y[ci], np.zeros(NUM_CLASSES))
    f1_check_tuned = macro_f1(logits[ci], y[ci], bias_t)
    print(f"[generalization] check-split: base={f1_check_base:.4f} → tuned={f1_check_tuned:.4f} "
          f"(Δ{f1_check_tuned - f1_check_base:+.4f})")

    # 최종 bias: 전체 OOF 로 적합
    bias_full, f1_full = coordinate_ascent(logits, y)
    print(f"[final] full-OOF macro_f1: {base_f1:.4f} → {f1_full:.4f} (Δ{f1_full - base_f1:+.4f})")

    out = Path(args.model_dir) / "bias.json"
    out.write_text(json.dumps({"bias": bias_full.tolist(), "classes": ACTION_CLASSES}), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
