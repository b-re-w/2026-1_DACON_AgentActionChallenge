# -*- coding: utf-8 -*-
"""QLoRA teacher 학습 — 큰 decoder(Coder-7B 등)를 4bit+LoRA로 파인튜닝해 강한 teacher 확보.

teacher 전용(제출 안 함). 학습 후:
  · fold0 val 점수 보고 (앙상블에 넣을 가치 판단)
  · runs/<name>/oof_logits.npy 저장 (앙상블용)
  · experiments/teacher_soft_<name>.npy 저장 (전체 70k soft prob, 증류용)

사용:  python experiments/train_qlora.py experiments/configs/exp15_coder7b_qlora.json
config: name, model_name, features(rich), max_len(512), batch_size(4), grad_accum(4),
        epochs(3), lr(1e-4), lora_r(16), lora_alpha(32), fold(0), n_folds(5)
"""
import os, sys, json, time, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(HERE), "models_cache"))
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, HERE)
import exp_lib as L
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          BitsAndBytesConfig, get_cosine_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

RUNS = os.path.join(HERE, "runs")
DEF = dict(features="rich", max_len=512, batch_size=4, grad_accum=4, epochs=3,
           lr=1e-4, lora_r=16, lora_alpha=32, lora_dropout=0.05,
           fold=0, n_folds=5, seed=42, n_limit=0, patience=2)


def fmt(s): m, s = divmod(int(s), 60); return f"{m}:{s:02d}"


def encode(texts, tok, ml):
    e = tok(list(texts), truncation=True, max_length=ml, padding="max_length", return_tensors="pt")
    return e["input_ids"], e["attention_mask"]


@torch.no_grad()
def predict(model, dl, dev):
    model.eval(); out = []
    for ids, mask in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
        out.append(lg.float().cpu().numpy())
    return np.concatenate(out, 0)


def main(cfg_path):
    cfg = DEF.copy(); cfg.update(json.load(open(cfg_path, encoding="utf-8")))
    name = cfg["name"]; out_dir = os.path.join(RUNS, name); os.makedirs(out_dir, exist_ok=True)
    dev = "cuda"
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    print(f"=== {name} (QLoRA) === model={cfg['model_name']} max_len={cfg['max_len']} "
          f"bs={cfg['batch_size']}x{cfg['grad_accum']} lr={cfg['lr']} r={cfg['lora_r']}", flush=True)

    df = L.load_data(n_limit=cfg["n_limit"])
    y = df["action"].map(L.LABEL2ID).values
    class_w = compute_class_weight("balanced", classes=np.arange(14), y=y)
    class_prior = np.bincount(y, minlength=14) / len(y)
    tr_idx, va_idx = list(GroupKFold(cfg["n_folds"]).split(df, y, df["group"]))[cfg["fold"]]
    tr_df, va_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    print(f"samples={len(df)} | fold{cfg['fold']}: train {len(tr_df)} / val {len(va_df)}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # QLoRA: 7B를 4bit(nf4)로 로드(~4.5GB) + LoRA. (UnicodeError 스레드경고는 무해)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=14, id2label=L.ID2LABEL, label2id=L.LABEL2ID,
        quantization_config=bnb, device_map={"": 0})
    model.config.pad_token_id = tok.pad_token_id
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(task_type="SEQ_CLS", r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"],
                      lora_dropout=cfg["lora_dropout"],
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    tr_txt = [L.build_text(r, cfg["features"]) for _, r in tr_df.iterrows()]
    va_txt = [L.build_text(r, cfg["features"]) for _, r in va_df.iterrows()]
    tr_y = tr_df["action"].map(L.LABEL2ID).to_numpy()
    va_y = va_df["action"].map(L.LABEL2ID).to_numpy()
    ii, mm = encode(tr_txt, tok, cfg["max_len"])
    tr_dl = DataLoader(TensorDataset(ii, mm, torch.tensor(tr_y)), batch_size=cfg["batch_size"], shuffle=True)
    ii, mm = encode(va_txt, tok, cfg["max_len"])
    va_dl = DataLoader(TensorDataset(ii, mm), batch_size=cfg["batch_size"] * 2)

    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float, device=dev))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])
    steps = len(tr_dl) // cfg["grad_accum"] * cfg["epochs"]
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)

    spe = len(tr_dl); log_every = max(1, spe // 15)
    best_f1, best_logits, no_imp = -1, None, 0
    t0 = time.time()
    for ep in range(cfg["epochs"]):
        model.train(); et = time.time(); run = 0.0; opt.zero_grad()
        for step, (ids, mask, lab) in enumerate(tr_dl, 1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lg = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
                loss = loss_fn(lg, lab.to(dev)) / cfg["grad_accum"]
            loss.backward(); run += loss.item() * cfg["grad_accum"]
            if step % cfg["grad_accum"] == 0:
                opt.step(); sched.step(); opt.zero_grad()
            if step % log_every == 0 or step == spe:
                el = time.time() - et; eta = el / step * (spe - step)
                print(f"    ep{ep} {step}/{spe} loss={run/step:.3f} el={fmt(el)} eta={fmt(eta)}", flush=True)
        lg = predict(model, va_dl, dev)
        f1 = L._macro(va_y, lg.argmax(1))
        print(f"  epoch {ep}: macro-F1(no-bias)={f1:.4f}  ({fmt(time.time()-et)})", flush=True)
        if f1 > best_f1:
            best_f1, best_logits, no_imp = f1, lg, 0
            model.save_pretrained(os.path.join(out_dir, "lora"))
        else:
            no_imp += 1
            if no_imp >= cfg["patience"]:
                print(f"  early stop (best={best_f1:.4f})", flush=True); break

    variants, best_v, params, per_class, best_pred = L.evaluate_all(best_logits, va_y, class_prior)
    print(f"\n[{name}] variants={variants} best={best_v} ({variants[best_v]})", flush=True)
    np.save(os.path.join(out_dir, "oof_logits.npy"), best_logits.astype(np.float32))
    np.save(os.path.join(out_dir, "oof_y.npy"), va_y)
    np.save(os.path.join(out_dir, "oof_idx.npy"), np.asarray(va_idx))
    json.dump(dict(name=name, model_name=cfg["model_name"], best_f1=variants[best_v],
                   variants=variants, per_class=per_class, config=cfg),
              open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    # 전체 70k soft-label 생성 (증류 teacher용)
    print("teacher_soft 생성(전체 70k)...", flush=True)
    all_txt = [L.build_text(r, cfg["features"]) for _, r in df.iterrows()]
    ii, mm = encode(all_txt, tok, cfg["max_len"])
    all_dl = DataLoader(TensorDataset(ii, mm), batch_size=cfg["batch_size"] * 2)
    all_lg = predict(model, all_dl, dev)
    soft = np.exp(all_lg - all_lg.max(1, keepdims=True)); soft /= soft.sum(1, keepdims=True)
    tpath = os.path.join(HERE, f"teacher_soft_{name}.npy")
    np.save(tpath, soft.astype(np.float32))
    print(f"저장: {tpath}  | fold0 best_f1={variants[best_v]:.4f}  총 {fmt(time.time()-t0)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
