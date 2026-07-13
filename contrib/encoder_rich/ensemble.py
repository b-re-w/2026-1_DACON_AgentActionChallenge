# -*- coding: utf-8 -*-
"""OOF 앙상블/블렌딩 — 여러 실험의 val 예측(oof_logits.npy)을 결합해 macro-F1 을 비교.

사용:
  python experiments/ensemble.py exp07_xlmr_rich512_ce exp0X_mdeberta_rich512 ...
  python experiments/ensemble.py --all          # oof 가 있는 모든 run 자동 사용

전제: 결합할 실험들이 '같은 fold·seed'(같은 oof_idx)여야 정렬됨.
      (train.py 가 실험마다 runs/<name>/oof_logits.npy, oof_y.npy, oof_idx.npy 저장)
주의: exp01~06 은 oof 저장 도입 전 학습분이라 oof 파일이 없을 수 있음 → 재학습 필요.

앙상블이 단일보다 좋으면 → 그 조합을 'teacher' 로 삼아 student 하나로 증류(1GB 제출).
"""
import os, sys, glob
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
sys.path.insert(0, HERE)
import exp_lib as L


def softmax(x):
    x = x - x.max(1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(1, keepdims=True)


def macro(y, pred):
    return L._macro(y, pred)


def load_oof(name):
    d = os.path.join(RUNS, name)
    lg = np.load(os.path.join(d, "oof_logits.npy"))
    y = np.load(os.path.join(d, "oof_y.npy"))
    idx = np.load(os.path.join(d, "oof_idx.npy"))
    return lg, y, idx


def find_all_oof():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if os.path.exists(os.path.join(d, "oof_logits.npy")):
            out.append(os.path.basename(d))
    return out


def optimize_weights(probs, y, rounds=12):
    """coordinate ascent 로 블렌드 가중치 탐색 (macro-F1(no-bias) 최대화)."""
    n = len(probs)
    w = np.ones(n) / n
    best = macro(y, sum(wi * p for wi, p in zip(w, probs)).argmax(1))
    for _ in range(rounds):
        improved = False
        for i in range(n):
            for d in (-0.2, -0.1, -0.05, 0.05, 0.1, 0.2):
                t = w.copy(); t[i] += d
                t = np.clip(t, 0, None)
                if t.sum() == 0:
                    continue
                t /= t.sum()
                s = macro(y, sum(wi * p for wi, p in zip(t, probs)).argmax(1))
                if s > best + 1e-5:
                    best, w, improved = s, t, True
        if not improved:
            break
    return w, best


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    names = find_all_oof() if "--all" in sys.argv else args
    if not names:
        print("사용: python experiments/ensemble.py <run이름...>  또는  --all")
        print("현재 oof 보유 run:", find_all_oof() or "(없음 — 새 train.py 로 재학습 필요)")
        return

    data, base_idx, ys, probs = [], None, None, []
    for n in names:
        lg, y, idx = load_oof(n)
        if base_idx is None:
            base_idx, ys = idx, y
        elif not np.array_equal(idx, base_idx):
            print(f"⚠️ {n}: fold/idx 불일치 → 제외 (같은 fold·seed 여야 함)")
            continue
        probs.append(softmax(lg))
    used = [n for n in names if True][:len(probs)]

    print(f"=== 결합 대상 {len(probs)}개 (val n={len(ys)}) ===")
    print("--- 단일 모델 (none / bias_tune) ---")
    for n, p in zip(used, probs):
        _, fb = L.tune_bias(np.log(p + 1e-12), ys)
        print(f"  {n:34s} none={macro(ys, p.argmax(1)):.4f}  bias={fb:.4f}")

    if len(probs) < 2:
        print("\n(앙상블하려면 2개 이상 필요)")
        return

    # 1) 단순 평균
    avg = sum(probs) / len(probs)
    _, avg_b = L.tune_bias(np.log(avg + 1e-12), ys)
    print(f"\n--- 단순평균: none={macro(ys, avg.argmax(1)):.4f}  bias={avg_b:.4f}")

    # 2) 가중 최적화
    w, w_none = optimize_weights(probs, ys)
    blend = sum(wi * p for wi, p in zip(w, probs))
    _, w_bias = L.tune_bias(np.log(blend + 1e-12), ys)
    print(f"--- 가중최적: none={w_none:.4f}  bias={w_bias:.4f}")
    print("    weights: " + ", ".join(f"{n.split('_')[0]}={wi:.2f}" for n, wi in zip(used, w)))
    print(f"\n>>> 최고 앙상블 bias-tune macro-F1 = {max(avg_b, w_bias):.4f}")


if __name__ == "__main__":
    main()
