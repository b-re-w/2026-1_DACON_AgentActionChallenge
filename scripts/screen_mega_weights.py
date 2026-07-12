"""mega 계열 가중 그리드 CPU 스크리닝 — trio(1:1:1) 고정 + gem/glm/deberta 가중 스윕.
세션분할 A/B 각각에서 trio 기준선 대비 delta 를 재고, 양쪽 모두 + 인 조합만 통과(일관성 필터).
순위는 min(dA,dB) 비관적 기준."""
import itertools, numpy as np
from sklearn.metrics import f1_score
from ai_challenge.datasets import CLASS_TO_ID
from ai_challenge.models.common import load_train_records, get_folds, resolve_teacher_logits_path

records = load_train_records(); fold = get_folds(records)
val = [records[i] for i, f in enumerate(fold) if f == 0]
y = np.array([CLASS_TO_ID[s.action] for s in val])
sess = [s.id.rsplit("_step", 1)[0] for s in val]
uniq = sorted(set(sess)); half = set(uniq[::2])
A = np.array([i for i, g in enumerate(sess) if g in half])
B = np.array([i for i, g in enumerate(sess) if g not in half])

def load(tag):
    z = np.load(resolve_teacher_logits_path(tag), allow_pickle=True)
    row = {r: i for i, r in enumerate(z["ids"].tolist())}
    return z["logits"][[row[s.id] for s in val]].astype(np.float64)

L = {t: load(t) for t in ["qgptoss", "qw6", "q7bteam", "qgemma9b", "qglm32b", "qdebertaL2"]}
TRIO = L["qgptoss"] + L["qw6"] + L["q7bteam"]

def mf1(logits, idx): return f1_score(y[idx], logits[idx].argmax(1), average="macro")
bA, bB = mf1(TRIO/3, A), mf1(TRIO/3, B)
print(f"기준선 trio: A={bA:.5f}  B={bB:.5f}  full={f1_score(y, (TRIO/3).argmax(1), average='macro'):.5f}")

rows = []
for wg, wl, wd in itertools.product([0,0.25,0.5,0.75,1.0],[0,0.25,0.5,0.75,1.0],[0,0.25,0.5]):
    if wg==wl==wd==0: continue
    tot = 3+wg+wl+wd
    M = (TRIO + wg*L["qgemma9b"] + wl*L["qglm32b"] + wd*L["qdebertaL2"]) / tot
    dA, dB = mf1(M, A)-bA, mf1(M, B)-bB
    rows.append((wg, wl, wd, dA, dB, min(dA,dB), (dA+dB)/2))

rows.sort(key=lambda r: -r[5])
print(f"\n{'gem':>5} {'glm':>5} {'deb':>5} {'ΔA':>8} {'ΔB':>8} {'min':>8} {'mean':>8}  (일관성=양쪽+)")
n_pass = 0
for wg, wl, wd, dA, dB, mn, mean in rows[:14]:
    mark = "✅" if mn > 0 else "  "
    n_pass += mn > 0
    print(f"{wg:5.2f} {wl:5.2f} {wd:5.2f} {dA:+8.4f} {dB:+8.4f} {mn:+8.4f} {mean:+8.4f} {mark}")
print(f"\n일관 통과(min>0): {sum(1 for r in rows if r[5]>0)}/{len(rows)}개")
