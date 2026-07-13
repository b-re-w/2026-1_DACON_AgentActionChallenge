# -*- coding: utf-8 -*-
"""증류용 teacher soft-label 생성 → experiments/teacher_soft.npy (전체 70k, 확률 [N,14]).

teacher(저장된 강한 모델/앙상블)로 전체 train 을 추론해 확률을 평균 → student 가 KD 로 흡수.
사용:
  python experiments/distill.py exp07_xlmr_rich512_ce exp08_mdeberta_rich512_ce
  (각 teacher 는 자기 features/max_len 로 추론. 기본 동일가중 평균 — ensemble.py에서 단순평균이 최고였음)

이후: config 에 "teacher_soft": "experiments/teacher_soft.npy", "kd_alpha": 0.5 를 주고 train.py 실행
      → student(제출가능 단일 인코더)가 앙상블 지식을 흡수. make_submit 로 패키징.
"""
import os, sys, json
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(HERE), "models_cache"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
sys.path.insert(0, HERE)
import exp_lib as L
from transformers import AutoTokenizer, AutoModelForSequenceClassification

RUNS = os.path.join(HERE, "runs")
OUT = os.path.join(HERE, "teacher_soft.npy")


def infer_probs(name, df, batch=64):
    model_dir = os.path.join(RUNS, name, "model")
    ic = json.load(open(os.path.join(model_dir, "infer_config.json")))
    feats, max_len = ic.get("features", "full"), ic.get("max_len", 192)
    texts = [L.build_text(r, feats) for _, r in df.iterrows()]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    if dev == "cuda":
        model = model.half()

    probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], truncation=True, max_length=max_len,
                      padding=True, return_tensors="pt").to(dev)
            p = torch.softmax(model(**enc).logits.float(), dim=1)
            probs.append(p.cpu().numpy())
    print(f"  {name}: features={feats} max_len={max_len} 추론완료")
    return np.concatenate(probs, 0)


def main(names, weights=None):
    df = L.load_data()  # 전체 70k, 행순서 = df index (train.py 와 동일)
    y = df["action"].map(L.LABEL2ID).to_numpy()
    w = np.array(weights, float) if weights else np.ones(len(names))
    w = w / w.sum()
    acc = np.zeros((len(df), len(L.ALL_CLASSES)), np.float32)
    for n, wi in zip(names, w):
        acc += wi * infer_probs(n, df)
    np.save(OUT, acc)
    f1 = L._macro(y, acc.argmax(1))
    print(f"\nteacher 앙상블({len(names)}개, 가중={w.round(2).tolist()}) → {OUT}")
    print(f"  teacher 전체데이터 macro-F1={f1:.4f} (train 포함 낙관치, 참고용)")
    print("  다음: config 에 teacher_soft/kd_alpha 넣고 student 학습 → make_submit")


if __name__ == "__main__":
    main(sys.argv[1:])
