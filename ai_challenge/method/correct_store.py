"""GT-보정 soft-label store 생성 — teacher 오분류 샘플을 GT 쪽으로 '그럴듯하게' 보정.

원본 teacher store 의 logits 를 GT 로 보정한 새 store 를 만든다(재추론 0). 이후 그 store 로
증류해 vanilla 대비 이득을 본다. 병목이 라벨 애매성이라 이득이 없을(또는 하락) 수 있음 — 검증용.

모드:
  ka    : Knowledge Adjustment (Probability Shift, Ding et al.) — 오분류 샘플에서
          GT logit ↔ argmax logit swap. GT 를 top-1 로 만들되 나머지 상대구조(dark
          knowledge)는 보존. 온도 무관. → 보정후 train argmax 는 100% 정답.
  blend : (1-λ)·teacher_softmax + λ·onehot(GT) 를 다시 logit(log-prob)으로. 오분류만
          (--only-wrong) 또는 전체.

사용:
  uv run python -m ai_challenge.method.correct_store --src q3b --tag q3b_ka --mode ka
  uv run python -m ai_challenge.method.correct_store --src q3b --tag q3b_bl --mode blend --lam 0.3
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from ai_challenge.datasets import CLASS_TO_ID, NUM_CLASSES, load_records
from ai_challenge.models.common import (
    TRAIN_JSONL,
    TRAIN_LABELS,
    resolve_teacher_logits_path,
    save_teacher_logits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="원본 store 태그/경로")
    ap.add_argument("--tag", required=True, help="출력 store 태그")
    ap.add_argument("--mode", choices=["ka", "blend"], default="ka")
    ap.add_argument("--lam", type=float, default=0.3, help="blend 모드 λ (GT one-hot 비중)")
    ap.add_argument("--all-samples", action="store_true",
                    help="blend 을 전체 샘플에 적용(기본은 오분류만)")
    args = ap.parse_args()

    recs = load_records(TRAIN_JSONL, TRAIN_LABELS)
    y = np.array([CLASS_TO_ID[s.action] for s in recs])
    z = np.load(resolve_teacher_logits_path(args.src), allow_pickle=True)
    row = {rid: i for i, rid in enumerate(z["ids"].tolist())}
    L = z["logits"].astype(np.float32)[[row[s.id] for s in recs]].copy()  # recs 순서로 정렬

    pred = L.argmax(1)
    wrong = pred != y
    base_f1 = f1_score(y, pred, average="macro")
    print(f"[correct] N={len(y)} teacher 오분류={int(wrong.sum())} ({100*wrong.mean():.1f}%) "
          f"원본 argmax macro_f1={base_f1:.4f}", flush=True)

    if args.mode == "ka":
        w = np.where(wrong)[0]
        gt, am = y[w], pred[w]
        L[w, gt], L[w, am] = L[w, am].copy(), L[w, gt].copy()  # swap
    else:  # blend
        p = np.exp(L - L.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        oh = np.eye(NUM_CLASSES, dtype=np.float32)[y]
        mask = np.ones(len(y), bool) if args.all_samples else wrong
        p[mask] = (1.0 - args.lam) * p[mask] + args.lam * oh[mask]
        L = np.log(np.clip(p, 1e-8, None)).astype(np.float32)

    save_teacher_logits(args.tag, [s.id for s in recs], L,
                        meta={"corrected_from": args.src, "mode": args.mode,
                              "lam": args.lam, "all_samples": args.all_samples})
    new_f1 = f1_score(y, L.argmax(1), average="macro")
    print(f"[correct] 보정후 argmax macro_f1={new_f1:.4f} (원본 {base_f1:.4f}) → {args.tag}", flush=True)


if __name__ == "__main__":
    main()
