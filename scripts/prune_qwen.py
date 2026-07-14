"""Qwen 0.5B vocab 프루닝 킷 — 임베딩에서 미사용 토큰 행 제거 + ID 리맵 테이블 생성.

원리: 남긴 행은 비트 동일 → keep-list 내 입력의 로짓 완전 동일.
OOV(keep-list 밖 토큰)는 바이트 토큰(0..255)으로 분해 리맵 (Qwen byte-BPE라 항상 가능).

사용:
    uv run python scripts/prune_qwen.py --model-dir runs/kd_bt_all/model --out runs/pruned_bt_all
    # → out/model (절삭본) + out/id_remap.npy (orig_id -> new_id, OOV는 -1)
    #   + out/byte_decomp.json (orig_id -> [new_byte_ids]) + 일치율 sanity
추론측 사용법 (script.py):
    ids = tokenizer(text)  # 원본 토크나이저 그대로
    new = []
    for t in ids:
        n = remap[t]
        new.extend(byte_decomp[t] if n < 0 else [n])
"""
import argparse, json
import numpy as np
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--pad-topk", type=int, default=0, help="상위빈도(저ID) 토큰 N개 추가 보존")
ap.add_argument("--sanity-n", type=int, default=300)
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.model_dir)
from ai_challenge.datasets import load_records, serialize_sample
recs = load_records("data/train.jsonl", "data/train_labels.csv")
texts = [serialize_sample(r, include_trail=True) for r in recs]
used = set()
for i in range(0, len(texts), 2000):
    for ids in tok(texts[i:i+2000], truncation=True, max_length=640)["input_ids"]:
        used.update(ids)
used |= set(range(256 + args.pad_topk))          # 바이트 256 (+옵션 고빈도 패딩)
used |= set(tok.all_special_ids)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir, torch_dtype=torch.float32)
V = model.get_input_embeddings().weight.shape[0]
keep = sorted(t for t in used if t < V)
print(f"vocab {V:,} -> keep {len(keep):,}")

remap = np.full(V, -1, dtype=np.int64)
for new, orig in enumerate(keep):
    remap[orig] = new

# 임베딩 절삭 (행 복사 — 비트 동일)
emb = model.get_input_embeddings().weight.data
new_emb = torch.nn.Embedding(len(keep), emb.shape[1])
new_emb.weight.data = emb[keep].clone()
model.set_input_embeddings(new_emb)
model.config.vocab_size = len(keep)
if model.config.pad_token_id is not None and model.config.pad_token_id >= 0:
    model.config.pad_token_id = int(remap[model.config.pad_token_id])

# OOV 바이트 분해 테이블: 모든 orig 토큰 -> 바이트토큰 시퀀스 (새 ID 공간)
# Qwen GPT-2식 byte-BPE: 토큰문자열의 각 유니코드문자 = 바이트 1개에 대응하는 단일토큰
b2t = {}  # 바이트값 -> orig 바이트토큰 id
vocab = tok.get_vocab()
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode
b2u = bytes_to_unicode()
for b, u in b2u.items():
    if u in vocab:
        b2t[b] = vocab[u]
u2b = {u: b for b, u in b2u.items()}  # 토큰문자열 -> 바이트열 역매핑
byte_decomp = {}
for t in range(V):
    if remap[t] >= 0:
        continue
    s = tok.convert_ids_to_tokens(t)
    if s is None:
        continue
    try:
        byte_decomp[t] = [int(remap[b2t[u2b[c]]]) for c in s]
    except KeyError:
        byte_decomp[t] = []  # 특수토큰류 — 입력에 나올 수 없음

out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
model.half().save_pretrained(out / "model")
tok.save_pretrained(out / "model")
np.save(out / "id_remap.npy", remap)
json.dump({str(k): v for k, v in byte_decomp.items()}, open(out / "byte_decomp.json", "w"))
sz = sum(f.stat().st_size for f in (out / "model").glob("*")) / 1048576
print(f"절삭본 저장: {out}/model ({sz:.0f}MB)")

# ── 일치율 sanity: 원본 vs 절삭 (fp32, CPU, 동일 샘플) ──
def remap_ids(ids):
    o = []
    for t in ids:
        n = remap[t]
        o.extend(byte_decomp.get(t, []) if n < 0 else [int(n)])
    return o

orig = AutoModelForSequenceClassification.from_pretrained(args.model_dir, torch_dtype=torch.float32).eval()
pruned = AutoModelForSequenceClassification.from_pretrained(out / "model", torch_dtype=torch.float32).eval()
idx = np.random.RandomState(0).choice(len(texts), args.sanity_n, replace=False)
agree, mx = 0, 0.0
with torch.inference_mode():
    for i in idx:
        ids = tok(texts[i], truncation=True, max_length=640)["input_ids"]
        l1 = orig(input_ids=torch.tensor([ids])).logits[0]
        l2 = pruned(input_ids=torch.tensor([remap_ids(ids)])).logits[0]
        agree += int(l1.argmax() == l2.argmax())
        mx = max(mx, (l1 - l2).abs().max().item())
print(f"[sanity] 일치 {agree}/{args.sanity_n} | max|Δlogit|={mx:.2e} (기대: 전원일치, Δ<1e-4)")
