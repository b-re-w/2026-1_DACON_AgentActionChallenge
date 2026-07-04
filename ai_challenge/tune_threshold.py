"""OOF logits 기반 per-class bias(threshold) 튜닝 — Macro-F1 직접 최적화.

argmax(logits + bias) 의 bias 를 coordinate ascent 로 최적화한다. Macro-F1 은
클래스 균등 가중이라, 희소 클래스의 결정 경계를 살짝 낮추면(bias↑) 크게 오른다.
재학습이 필요 없다.

과적합 점검: OOF(val fold)를 tune/check 로 나눠 tune 에서 최적화한 bias 를
check 에서 평가한다(일반화 확인). 최종 bias 는 전체 OOF 로 다시 적합해 저장한다.

산출물: <model-dir>/bias.json  (submit_script.py 가 있으면 로드해 logits 에 더함)

사용:
    uv run python -m ai_challenge.tune_threshold --model-dir runs/B/model --val-fold 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from ai_challenge.datasets import (
    ACTION_CLASSES,
    CLASS_TO_ID,
    NUM_CLASSES,
    load_folds,
    load_records,
)

DATA_DIR = Path("data")


def compute_oof_logits(model_dir: str, val_fold: int, max_length: int, batch_size: int = 128):
    """저장된 모델로 val fold 의 logits 와 정답을 계산한다(길이정렬 배칭)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from ai_challenge.datasets import serialize_sample

    records = load_records(DATA_DIR / "train.jsonl", DATA_DIR / "train_labels.csv")
    fold = load_folds(DATA_DIR / "folds.csv", records)
    val = [s for s, f in zip(records, fold) if f == val_fold]

    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).cuda().half().eval()

    texts = [serialize_sample(s) for s in val]
    y = np.array([CLASS_TO_ID[s.action] for s in val])
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    logits = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
    with torch.inference_mode():
        for st in range(0, len(order), batch_size):
            chunk = order[st:st + batch_size]
            enc = tok([texts[i] for i in chunk], truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to("cuda")
            out = model(**enc).logits.float().cpu().numpy()
            for pos, i in enumerate(chunk):
                logits[i] = out[pos]
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
    bias_t, f1_tune = coordinate_ascent(logits[ti], y[ti])
    f1_check_base = macro_f1(logits[ci], y[ci], np.zeros(NUM_CLASSES))
    f1_check_tuned = macro_f1(logits[ci], y[ci], bias_t)
    print(f"[generalization] check-split: base={f1_check_base:.4f} → tuned={f1_check_tuned:.4f} "
          f"(Δ{f1_check_tuned - f1_check_base:+.4f})")

    # 최종 bias: 전체 OOF 로 적합
    bias_full, f1_full = coordinate_ascent(logits, y)
    print(f"[final] full-OOF macro_f1: {base_f1:.4f} → {f1_full:.4f} (Δ{f1_full - base_f1:+.4f})")
    print("[bias]", dict(zip(ACTION_CLASSES, np.round(bias_full, 3).tolist())))

    out = Path(args.model_dir) / "bias.json"
    out.write_text(json.dumps({"bias": bias_full.tolist(), "classes": ACTION_CLASSES}), encoding="utf-8")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
