"""gemma-2-27b soft-label 재생성 (NaN 수리판) — 팀원 서버용.

기존 qgemma27b.npz 가 전량 NaN(4bit/fp16 오버플로)이라 bf16 풀로드 + eager attention 으로 재추출.
gemma-2 는 soft-capping 때문에 eager 필수, 4bit 금지.

사용법 (t_qgemma27b 가 있는 서버에서):
    git pull
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/gemma27_bf16_gen.py
    # 완료 후: argmax fold0 ≈ 0.77 인지 출력 확인 → npz 를 git push
    #   git add -f runs/_teacher_logits/qgemma27b.npz runs/_teacher_logits/qgemma27b.json
"""
import numpy as np, torch, json, sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from pathlib import Path
from ai_challenge.datasets import load_records, serialize_sample, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES
from ai_challenge.models.common import save_teacher_logits

MD = "runs/t_qgemma27b/model"
is_adapter = Path(f"{MD}/adapter_config.json").exists()
base_name = json.load(open(f"{MD}/adapter_config.json"))["base_model_name_or_path"] if is_adapter else MD
print(f"[gemma27-bf16] base={base_name} adapter={is_adapter}", flush=True)
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForSequenceClassification.from_pretrained(
    base_name, num_labels=NUM_CLASSES,
    torch_dtype=torch.bfloat16,          # bf16 풀로드 (4bit 금지 — NaN 원인)
    attn_implementation="eager",         # gemma-2 soft-capping 필수
    device_map={"": 0},
    id2label={i: c for i, c in ID_TO_CLASS.items()}, label2id=dict(CLASS_TO_ID))
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
if is_adapter:
    model = PeftModel.from_pretrained(model, MD)
model = model.eval()

recs = load_records("data/train.jsonl", "data/train_labels.csv")
texts = [serialize_sample(r) for r in recs]

def fwd(batch_texts):
    enc = tok(batch_texts, truncation=True, max_length=640, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        return model(**enc).logits.float().cpu().numpy()

s = fwd(texts[:8])
print(f"[sanity8] mean={s.mean():.3f} std={s.std():.3f} NaN={np.isnan(s).sum()}", flush=True)
if np.isnan(s).any():
    print("[abort] bf16+eager 인데도 NaN — 중단"); sys.exit(1)

order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
out = np.zeros((len(texts), 14), np.float32)
B = 16
for k in range(0, len(order), B):
    idx = order[k:k+B]
    out[idx] = fwd([texts[i] for i in idx])
    if (k // B) % 200 == 0:
        print(f"  {k}/{len(order)} NaN={np.isnan(out).sum()}", flush=True)
assert not np.isnan(out).any(), "NaN 발생"
p = save_teacher_logits("qgemma27b", [r.id for r in recs], out,
                        meta={"teacher_dir": MD, "serialize": "base", "max_length": 640, "dtype": "bf16-eager"})
# 필수 sanity: fold0 argmax
from sklearn.metrics import f1_score
from ai_challenge.datasets import assign_folds
fold = assign_folds(recs, n_splits=5, seed=42)
y = np.array([CLASS_TO_ID[r.action] for r in recs]); f0 = fold == 0
print(f"[done] {p}", flush=True)
print(f"[sanity] argmax fold0 = {f1_score(y[f0], out[f0].argmax(1), average='macro'):.5f} (기대 ~0.77 — 0.5대면 push 금지!)", flush=True)
