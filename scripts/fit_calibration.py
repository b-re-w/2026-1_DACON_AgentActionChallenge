"""혼동행렬 재보정: p' = softmax(A·log p + b) 를 5-fold OOF 로 피팅.

bias70k(b만, LB +0.0018 검증)의 상위호환 — A 가 쌍별 혼동(read↔grep 등)을 보정한다.
프로토콜: folds1-4(56k) 피팅 → fold0(14k) held-out 채점. torch LBFGS + macro-F1 은
비미분이라 CE 로 학습하되 λ 로 A 를 단위행렬에 정칙화, 최종 선택은 held-out macro.
산출: runs/calib_matrix.json {A, b, heldout}
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_records, load_folds
from ai_challenge.method.tune_threshold import softmax, coordinate_ascent

recs = load_records("data/train.jsonl", "data/train_labels.csv")
fold = load_folds("data/folds.csv", recs)
y_all = np.array([CLASS_TO_ID[r.action] for r in recs])
pos = {r.id: i for i, r in enumerate(recs)}
probs = np.zeros((len(recs), 14))
for f, run in [(0, "kd3_f0"), (1, "kd5_f1"), (2, "kd5_f2"), (3, "kd5_f3"), (4, "kd5_f4")]:
    ids = json.loads(Path(f"runs/{run}/oof_ids.json").read_text())
    probs[np.array([pos[i] for i in ids])] = softmax(np.load(f"runs/{run}/oof_logits.npy").astype(np.float64))

tr = fold != 0
te = fold == 0
logp = np.log(np.clip(probs, 1e-9, None))
Xtr = torch.tensor(logp[tr], dtype=torch.float32)
ytr = torch.tensor(y_all[tr], dtype=torch.long)
Xte, yte = logp[te], y_all[te]

base_te = f1_score(yte, probs[te].argmax(1), average="macro")
# 기준선: b-only (bias70k 방식, 56k 피팅)
bias_b, _ = coordinate_ascent(y_all[tr], probs[tr])
bonly_te = f1_score(yte, (probs[te] + bias_b).argmax(1), average="macro")
print(f"기준: raw {base_te:.5f} | b-only(56k) {bonly_te:.5f}")

best = {"heldout": bonly_te, "kind": "b-only", "A": np.eye(14).tolist(),
        "b_log": [0.0]*14, "b_prob": bias_b.tolist()}
cands = []
for lam in [10.0, 3.0, 1.0, 0.3, 0.1]:
    A = torch.eye(14, requires_grad=True)
    b = torch.zeros(14, requires_grad=True)
    opt = torch.optim.LBFGS([A, b], lr=0.5, max_iter=100)
    I = torch.eye(14)
    def closure():
        opt.zero_grad()
        z = Xtr @ A.T + b
        loss = F.cross_entropy(z, ytr) + lam * ((A - I) ** 2).sum() + 0.01 * (b ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        An, bn = A.numpy(), b.numpy()
        m = f1_score(yte, (Xte @ An.T + bn).argmax(1), average="macro")
    print(f"lam={lam:5.1f}  matrix held-out {m:.5f}", flush=True)
    cands.append((m, lam, An.copy(), bn.copy()))
    if m > best["heldout"] + 1e-6:
        best = {"heldout": float(m), "kind": f"matrix(lam={lam})",
                "A": An.tolist(), "b_log": bn.tolist(), "b_prob": [0.0]*14}

# 최고 행렬 위에 b'(확률공간) 좌표상승 1회만
m0, lam0, A0, b0 = max(cands, key=lambda c: c[0])
z_tr = logp[tr] @ A0.T + b0
bias2, _ = coordinate_ascent(y_all[tr], softmax(z_tr))
m2 = f1_score(yte, (softmax(Xte @ A0.T + b0) + bias2).argmax(1), average="macro")
print(f"matrix(lam={lam0})+b' held-out {m2:.5f}", flush=True)
if m2 > best["heldout"] + 1e-6:
    best = {"heldout": float(m2), "kind": f"matrix+b(lam={lam0})",
            "A": A0.tolist(), "b_log": b0.tolist(), "b_prob": bias2.tolist()}

print(f"\nBEST: {best['kind']}  held-out {best['heldout']:.5f} "
      f"(raw {base_te:.5f}, b-only {bonly_te:.5f})")
json.dump(best, open("runs/calib_matrix.json", "w"), indent=2)
