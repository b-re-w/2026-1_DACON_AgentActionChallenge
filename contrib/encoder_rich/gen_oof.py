# -*- coding: utf-8 -*-
"""저장된 모델로 fold val 예측(OOF)을 재생성 → runs/<name>/oof_*.npy.

OOF 저장 기능 도입 전 학습된 모델(예: exp07)을 앙상블에 쓰기 위해 사용.
사용:  python experiments/gen_oof.py exp07_xlmr_rich512_ce
전제:  같은 fold·n_folds (GroupKFold 는 결정적이라 seed 무관, df 순서 동일 → idx 정렬됨)
"""
import os, sys, json
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(HERE), "models_cache"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, HERE)
import exp_lib as L
from sklearn.model_selection import GroupKFold
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RUNS = os.path.join(HERE, "runs")


def gen(name, fold=0, n_folds=5, batch=64):
    run = os.path.join(RUNS, name)
    model_dir = os.path.join(run, "model")
    assert os.path.exists(os.path.join(model_dir, "config.json")), f"저장모델 없음: {model_dir}"
    ic = json.load(open(os.path.join(model_dir, "infer_config.json")))
    feats, max_len = ic.get("features", "full"), ic.get("max_len", 192)
    print(f"{name}: features={feats} max_len={max_len}")

    df = L.load_data()
    y = df["action"].map(L.LABEL2ID).values
    _, va_idx = list(GroupKFold(n_folds).split(df, y, df["group"]))[fold]
    va_df = df.iloc[va_idx].reset_index(drop=True)
    va_y = va_df["action"].map(L.LABEL2ID).to_numpy()
    texts = [L.build_text(r, feats) for _, r in va_df.iterrows()]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    if dev == "cuda":
        model = model.half()

    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(dev)
            out.append(model(**enc).logits.float().cpu().numpy())
    logits = np.concatenate(out, 0)

    np.save(os.path.join(run, "oof_logits.npy"), logits.astype(np.float32))
    np.save(os.path.join(run, "oof_y.npy"), va_y)
    np.save(os.path.join(run, "oof_idx.npy"), np.asarray(va_idx))
    f1 = L._macro(va_y, logits.argmax(1))
    print(f"  OOF 저장 완료 (n={len(va_y)}, no-bias macro-F1={f1:.4f})")


if __name__ == "__main__":
    for n in sys.argv[1:]:
        gen(n)
