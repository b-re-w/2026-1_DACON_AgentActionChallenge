"""main식 coordinate-ascent 자동 블렌드탐색 (우리 재료). teacher-argmax F1 최대화.
주의: teacher가 all-70k 학습이라 fold0 argmax는 inflated(스크리닝 proxy). 최종은 student+LB로."""
import numpy as np, json, sys
from sklearn.metrics import f1_score
from ai_challenge.datasets import load_records, assign_folds, CLASS_TO_ID
from ai_challenge.models.common import gather_teacher_logits

recs = load_records("data/train.jsonl", "data/train_labels.csv")
fold = assign_folds(recs, n_splits=5, seed=42)
f0 = [r for r, f in zip(recs, fold) if f == 0]
y = np.array([CLASS_TO_ID[r.action] for r in f0])
print(f"fold0 N={len(f0)}", flush=True)

pool = ["q3bb", "q3b43", "q3b44", "q3b45", "qgptoss", "qgpt5", "q14b", "q32b", "qgemma9b"]
import os
pool = [t for t in pool if os.path.exists(f"runs/_teacher_logits/{t}.npz")]

def sm(x):
    x = x.astype(np.float64); e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

Ps = {}
print("=== 개별 teacher argmax F1 (검증) ===", flush=True)
for t in pool:
    lg = gather_teacher_logits([t], f0)
    Ps[t] = sm(lg)
    print(f"  {t:10s} argmax F1 = {f1_score(y, lg.argmax(1), average='macro'):.5f}", flush=True)

def score(w):
    tot = sum(w.values())
    if tot <= 0: return 0.0
    blend = sum(wt * Ps[t] for t, wt in w.items() if wt > 0) / tot
    return f1_score(y, blend.argmax(1), average="macro")

# 시작점: oss2g1 (4×3B + gptoss:2 + gpt5:1)
w = {t: 0.0 for t in pool}
for t in ["q3bb", "q3b43", "q3b44", "q3b45"]:
    if t in w: w[t] = 1.0
if "qgptoss" in w: w["qgptoss"] = 2.0
if "qgpt5" in w: w["qgpt5"] = 1.0
best = score(w)
print(f"\nstart(oss2g1) argmax={best:.5f}", flush=True)
grid = [0, 0.5, 1, 1.5, 2, 3]
for it in range(8):
    improved = False
    for t in pool:
        cur = w[t]; bestv = cur
        for v in grid:
            w[t] = v; s = score(w)
            if s > best + 1e-6: best = s; bestv = v; improved = True
        w[t] = bestv
    if not improved: break
sel = {t: v for t, v in w.items() if v > 0}
print(f"\n★ BEST blend argmax F1 = {best:.5f}", flush=True)
print(f"  weights: {sel}", flush=True)
json.dump({"weights": sel, "argmax_f1": best}, open("runs/blend_search.json", "w"), indent=1)
