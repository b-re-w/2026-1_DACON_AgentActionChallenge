"""bias 를 70k 전체 OOF 로 재피팅 (fold0 포함) → runs/bias70k_full.json."""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID, load_records, load_folds
from ai_challenge.method.tune_threshold import softmax, coordinate_ascent

recs = load_records("data/train.jsonl", "data/train_labels.csv")
fold = load_folds("data/folds.csv", recs)
y = np.array([CLASS_TO_ID[r.action] for r in recs])
pos = {r.id: i for i, r in enumerate(recs)}
probs = np.zeros((len(recs), 14))
for f, run in [(0, "kd3_f0"), (1, "kd5_f1"), (2, "kd5_f2"), (3, "kd5_f3"), (4, "kd5_f4")]:
    ids = json.loads(Path(f"runs/{run}/oof_ids.json").read_text())
    probs[np.array([pos[i] for i in ids])] = softmax(np.load(f"runs/{run}/oof_logits.npy").astype(np.float64))
base = f1_score(y, probs.argmax(1), average="macro")
bias, tuned = coordinate_ascent(y, probs)
print(f"70k 전체: {base:.5f} -> {tuned:.5f}")
json.dump({"bias_vector": bias.tolist(), "macro_f1_before": base, "macro_f1_after": tuned},
          open("runs/bias70k_full.json", "w"), indent=2)
print("saved")
