"""GLM-32B soft-label을 bf16 풀로드로 재생성 (4bit 경로 NaN 회피). 먼저 8샘플 sanity 후 전체 70k."""
import numpy as np, torch, json, sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
from ai_challenge.datasets import load_records, serialize_sample, CLASS_TO_ID, ID_TO_CLASS, NUM_CLASSES
from ai_challenge.models.common import save_teacher_logits

MD = "runs/tglm32b/model"
base_name = json.load(open(f"{MD}/adapter_config.json"))["base_model_name_or_path"]
print(f"[bf16-gen] base={base_name}", flush=True)
tok = AutoTokenizer.from_pretrained(MD)
model = AutoModelForSequenceClassification.from_pretrained(
    base_name, num_labels=NUM_CLASSES, torch_dtype=torch.bfloat16, device_map={"": 0},
    id2label={i: c for i, c in ID_TO_CLASS.items()}, label2id=dict(CLASS_TO_ID))
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id or tok.eos_token_id
model = PeftModel.from_pretrained(model, MD).eval()

recs = load_records("data/train.jsonl", "data/train_labels.csv")
texts = [serialize_sample(r) for r in recs]

def fwd(batch_texts):
    enc = tok(batch_texts, truncation=True, max_length=640, padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        return model(**enc).logits.float().cpu().numpy()

# sanity 8개
s = fwd(texts[:8])
print(f"[sanity] mean={s.mean():.3f} std={s.std():.3f} NaN={np.isnan(s).sum()}", flush=True)
if np.isnan(s).any():
    print("[abort] bf16도 NaN — 중단"); sys.exit(1)

# 전체 (길이정렬 배칭)
order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
out = np.zeros((len(texts), 14), np.float32)
B = 24
for k in range(0, len(order), B):
    idx = order[k:k+B]
    out[idx] = fwd([texts[i] for i in idx])
    if (k // B) % 100 == 0:
        print(f"  {k}/{len(order)} NaN누적={np.isnan(out).sum()}", flush=True)
assert not np.isnan(out).any(), "NaN 발생"
p = save_teacher_logits("qglm32b", [r.id for r in recs], out,
                        meta={"teacher_dir": MD, "serialize": "base", "max_length": 640, "dtype": "bf16-full"})
print(f"[done] {p}", flush=True)
