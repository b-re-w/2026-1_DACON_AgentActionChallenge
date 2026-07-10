"""OOF 로짓 → per-class 가산 bias 탐색으로 Macro-F1 최적화.

argmax(logits + bias) 의 bias 를 coordinate-ascent 로 조정해 macro-F1 을 올린다.
희소 클래스(web_search·write_file 등)의 결정 경계를 유리하게 밀어 macro-F1 을 끌어올리는
표준 기법. 여러 run 의 OOF 를 평균(앙상블)해 함께 튜닝할 수도 있다.

산출: <out>/class_bias.json  (bias 벡터 + before/after macro-F1)
이 bias 는 제출 추론(script.py)에서 logits 에 더해 argmax 하면 그대로 재현된다.

사용:
    uv run python -m ai_challenge.method.tune_threshold \
        --runs runs/mdeberta_base_f0 [runs/other_f0 ...] --out runs/blend_f0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from ai_challenge.datasets import CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES, load_labels


def load_oof(run_dir: Path) -> tuple[list[str], np.ndarray]:
    ids = json.loads((run_dir / "oof_ids.json").read_text())
    logits = np.load(run_dir / "oof_logits.npy").astype(np.float64)
    if len(ids) != len(logits):
        raise ValueError(f"{run_dir}: ids({len(ids)}) != logits({len(logits)})")
    return ids, logits


def softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def macro_f1(y: np.ndarray, logits: np.ndarray, bias: np.ndarray) -> float:
    preds = (logits + bias).argmax(1)
    return f1_score(y, preds, average="macro")


def coordinate_ascent(
    y: np.ndarray, logits: np.ndarray,
    rounds: int = 12, grid: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """클래스별 bias 를 coordinate ascent 로 탐색. logits 는 확률(softmax) 권장."""
    if grid is None:
        grid = np.concatenate([
            np.linspace(-3.0, 3.0, 61),  # 넓게
        ])
    bias = np.zeros(NUM_CLASSES, dtype=np.float64)
    best = macro_f1(y, logits, bias)
    for r in range(rounds):
        improved = False
        # 희소 클래스부터(빈도 오름차순) 조정하면 수렴이 빠르다
        order = np.argsort(np.bincount(y, minlength=NUM_CLASSES))
        for c in order:
            cur = bias[c]
            best_b, best_s = cur, best
            for delta in grid:
                bias[c] = cur + delta
                s = macro_f1(y, logits, bias)
                if s > best_s:
                    best_s, best_b = s, cur + delta
            bias[c] = best_b
            if best_s > best + 1e-9:
                improved = True
                best = best_s
        # 라운드마다 그리드를 촘촘히(미세조정)
        grid = grid * 0.6
        if not improved:
            break
    return bias, best


def main() -> None:
    ap = argparse.ArgumentParser(description="OOF per-class bias 튜닝(Macro-F1)")
    ap.add_argument("--runs", nargs="+", required=True, help="OOF 를 가진 run 디렉터리들(평균 앙상블)")
    ap.add_argument("--labels", default="data/train_labels.csv")
    ap.add_argument("--out", default=None, help="bias 저장 위치(기본: 첫 run)")
    ap.add_argument("--use-logits", action="store_true", help="softmax 대신 raw logits 로 튜닝")
    args = ap.parse_args()

    id2lab = load_labels(args.labels)

    # 공통 id 정렬로 여러 run 의 OOF 를 평균(각 run 은 같은 fold 분할이어야 정합)
    base_ids, base_logits = load_oof(Path(args.runs[0]))
    order = {i: k for k, i in enumerate(base_ids)}
    acc = softmax(base_logits) if not args.use_logits else base_logits.copy()
    n_used = 1
    for rd in args.runs[1:]:
        ids, logits = load_oof(Path(rd))
        probs = softmax(logits) if not args.use_logits else logits
        remap = np.array([order[i] for i in ids])
        buf = np.zeros_like(acc)
        buf[remap] = probs
        acc += buf
        n_used += 1
    probs = acc / n_used

    y = np.array([CLASS_TO_ID[id2lab[i]] for i in base_ids])
    base_macro = f1_score(y, probs.argmax(1), average="macro")
    bias, tuned_macro = coordinate_ascent(y, probs)

    out = Path(args.out) if args.out else Path(args.runs[0])
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "runs": args.runs, "n_runs": n_used, "use_logits": args.use_logits,
        "space": "logits" if args.use_logits else "probs",
        "macro_f1_before": float(base_macro),
        "macro_f1_after": float(tuned_macro),
        "gain": float(tuned_macro - base_macro),
        "class_bias": {ID_TO_CLASS[i]: float(bias[i]) for i in range(NUM_CLASSES)},
        "bias_vector": [float(b) for b in bias],
    }
    (out / "class_bias.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[tune] runs={n_used} space={payload['space']}")
    print(f"  macro_f1: {base_macro:.5f} -> {tuned_macro:.5f}  (+{payload['gain']:.5f})")
    print(f"  saved: {out/'class_bias.json'}")


if __name__ == "__main__":
    main()
