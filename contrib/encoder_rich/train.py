# -*- coding: utf-8 -*-
"""단일 실험 실행기 — config(JSON) 하나를 받아 학습·평가하고 결과를 기록한다.

사용:  python experiments/train.py experiments/configs/exp02_mdeberta.json
결과:  experiments/runs/<name>/metrics.json  (+ 선택적 model/)
       experiments/leaderboard.csv 에 한 줄 append

config 키(기본값):
  name, model_name, features(full), loss(ce), max_len(192), batch_size(32),
  epochs(3), lr(2e-5), fold(0), n_folds(5), seed(42), n_limit(0),
  save_model(false), run_baseline(false)
"""
import os, sys, json, time, csv, random, platform
import numpy as np

# HF 캐시를 '프로젝트 로컬 models_cache'로 고정 → 자기완결(오프라인 재현).
# 폴더명이 ASCII(SW_Competition)라 sentencepiece 한글경로 문제 없음.
# (외부 hf_cache 우회는 폴더 리네임 후 불필요해져 제거)
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(_PROJ_ROOT, "models_cache"))
# SP → fast 토크나이저 변환 시 protobuf 신버전 호환 (Descriptors 에러 회피)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import exp_lib as L

import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_cosine_schedule_with_warmup

DEFAULTS = dict(features="full", loss="ce", max_len=192, batch_size=32, epochs=20,
                lr=2e-5, fold=0, n_folds=5, seed=42, n_limit=0, patience=3,
                lr_decay_epochs=6, use_wandb=False,
                save_model=False, run_baseline=False,
                teacher_soft=None, kd_alpha=0.5)  # 증류: teacher_soft(전체 70k prob npy) 주면 KD 활성


def fmt(sec):
    m, s = divmod(int(sec), 60)
    return f"{m:d}:{s:02d}"
HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
LEADERBOARD = os.path.join(HERE, "leaderboard.csv")


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def save_charts(out_dir, name, epoch_hist, best_pred, va_y, best_f1):
    """실험 폴더에 학습곡선(curve.png) + 혼동행렬(confusion.png) 저장."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
    ALL = L.ALL_CLASSES
    # 1) 학습곡선
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(len(epoch_hist)), epoch_hist, "-o", color="#2e6fb7")
    bi = int(np.argmax(epoch_hist))
    ax.scatter([bi], [epoch_hist[bi]], color="#c0392b", zorder=5,
               label=f"best ep{bi} = {epoch_hist[bi]:.4f}")
    ax.set_xlabel("epoch"); ax.set_ylabel("val macro-F1")
    ax.set_title(f"{name} — 학습곡선", fontweight="bold", color="#1f3a5f")
    ax.grid(alpha=.3); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "curve.png"), dpi=150); plt.close()
    # 2) 혼동행렬 (행 정규화 %)
    cm = confusion_matrix(va_y, best_pred, labels=range(len(ALL))).astype(float)
    cmn = cm / cm.sum(1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(9, 7.4))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(ALL))); ax.set_yticks(range(len(ALL)))
    ax.set_xticklabels(ALL, rotation=45, ha="right", fontsize=8); ax.set_yticklabels(ALL, fontsize=8)
    for i in range(len(ALL)):
        for j in range(len(ALL)):
            v = cmn[i, j]
            if v >= 8:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if v > 55 else "black")
    ax.set_ylabel("실제 정답 (True)", fontweight="bold"); ax.set_xlabel("모델 예측 (Predicted)", fontweight="bold")
    ax.set_title(f"{name} — 혼동행렬 (best_f1={best_f1:.4f})", fontweight="bold", color="#1f3a5f")
    plt.colorbar(im, fraction=0.046, pad=0.04, label="%")
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "confusion.png"), dpi=150); plt.close()


def encode(texts, tok, max_len):
    enc = tok(list(texts), truncation=True, max_length=max_len, padding="max_length", return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict_logits(model, dl, device):
    model.eval(); out = []
    for ids, mask, _ in dl:
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
        out.append(logits.cpu())
    return torch.cat(out).numpy()


def baseline_f1(tr_df, va_df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import f1_score
    pipe = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=80000,
                                               sublinear_tf=True, lowercase=True)),
                     ("clf", LogisticRegression(max_iter=500, class_weight="balanced", C=2.0))])
    pipe.fit(tr_df["current_prompt"].tolist(), tr_df["action"].tolist())
    pred = pipe.predict(va_df["current_prompt"].tolist())
    return round(f1_score(va_df["action"].tolist(), pred, labels=L.ALL_CLASSES,
                          average="macro", zero_division=0), 4)


def main(cfg_path):
    cfg = DEFAULTS.copy()
    cfg.update(json.load(open(cfg_path, encoding="utf-8")))
    name = cfg["name"]
    out_dir = os.path.join(RUNS, name); os.makedirs(out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(cfg["seed"])
    t_start = time.time()

    print(f"=== {name} ===")
    print(f"model={cfg['model_name']} features={cfg['features']} loss={cfg['loss']} "
          f"epochs={cfg['epochs']} max_len={cfg['max_len']} bs={cfg['batch_size']}")
    print(f"torch {torch.__version__} | transformers {transformers.__version__} | device {device}", flush=True)

    wb = None
    if cfg["use_wandb"]:
        import wandb
        wb = wandb.init(project="sw-action-pred", name=name, config=cfg, reinit=True)

    df = L.load_data(n_limit=cfg["n_limit"])
    y = df["action"].map(L.LABEL2ID).values
    class_w = compute_class_weight("balanced", classes=np.arange(len(L.ALL_CLASSES)), y=y)
    class_prior = np.bincount(y, minlength=len(L.ALL_CLASSES)) / len(y)

    splits = list(GroupKFold(cfg["n_folds"]).split(df, y, df["group"]))
    tr_idx, va_idx = splits[cfg["fold"]]
    tr_df, va_df = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
    print(f"samples={len(df)} sessions={df['group'].nunique()} | fold{cfg['fold']}: "
          f"train {len(tr_df)} / val {len(va_df)}", flush=True)

    base = baseline_f1(tr_df, va_df) if cfg["run_baseline"] else None
    if base is not None:
        print(f"[baseline] macro-F1 = {base}", flush=True)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"], num_labels=len(L.ALL_CLASSES), id2label=L.ID2LABEL, label2id=L.LABEL2ID).to(device)
    # decoder LLM(Qwen2.5 등) 호환: pad_token 이 없으면 eos 로 대체하고 config 에도 반영해야
    # SequenceClassification 이 '마지막 비-pad 토큰'을 찾을 수 있다. (encoder 는 이미 pad 있어 no-op)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    tr_texts = [L.build_text(r, cfg["features"]) for _, r in tr_df.iterrows()]
    va_texts = [L.build_text(r, cfg["features"]) for _, r in va_df.iterrows()]
    tr_y = tr_df["action"].map(L.LABEL2ID).to_numpy()
    va_y = va_df["action"].map(L.LABEL2ID).to_numpy()

    # 증류(KD): teacher_soft(전체 70k prob) 주면 해당 fold train 부분을 teacher 로 사용
    use_kd = bool(cfg.get("teacher_soft"))
    if use_kd:
        ts_all = np.load(cfg["teacher_soft"])
        # teacher_soft 는 원본 df 순서 기준. n_limit 은 앞에서 자르므로 ts_all[:n_limit] 과 정렬됨.
        assert ts_all.shape[0] >= len(df), f"teacher_soft 행수({ts_all.shape[0]}) < 데이터({len(df)})"
        tr_teacher = torch.tensor(ts_all[tr_idx], dtype=torch.float)
        print(f"  KD 활성: teacher_soft={cfg['teacher_soft']}  alpha={cfg['kd_alpha']}", flush=True)
    else:
        tr_teacher = torch.zeros(len(tr_y), len(L.ALL_CLASSES))

    ii, mm = encode(tr_texts, tok, cfg["max_len"])
    tr_dl = DataLoader(TensorDataset(ii, mm, torch.tensor(tr_y), tr_teacher),
                       batch_size=cfg["batch_size"], shuffle=True)
    ii, mm = encode(va_texts, tok, cfg["max_len"])
    va_dl = DataLoader(TensorDataset(ii, mm, torch.tensor(va_y)), batch_size=cfg["batch_size"] * 2)

    loss_fn = L.make_loss(cfg["loss"], class_w, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    # LR 스케줄 지평선을 early-stop 상한(epochs)과 분리한다.
    # → epochs 를 20 처럼 크게 잡아도 LR 은 lr_decay_epochs(기본 6)에 맞춰 정상 감쇠.
    #   (스케줄을 epochs=20 에 묶으면 LR 이 너무 천천히 줄어 수렴이 나빠짐)
    decay_epochs = min(cfg["epochs"], cfg["lr_decay_epochs"])
    total = len(tr_dl) * decay_epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.1 * total), total)
    scaler = torch.amp.GradScaler("cuda")

    steps_per_epoch = len(tr_dl)
    log_every = max(1, steps_per_epoch // 20)      # 에포크당 ~20회 진행 로그
    best_f1, best_logits, no_improve = -1.0, None, 0
    epoch_hist = []
    train_start = time.time()
    for ep in range(cfg["epochs"]):
        model.train(); t0 = time.time(); run_loss = 0.0
        for step, (ids, mask, labels, tsoft) in enumerate(tr_dl, 1):
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(input_ids=ids, attention_mask=mask).logits
                loss = loss_fn(logits, labels)
                if use_kd:  # soft-CE 증류: alpha*CE(정답) + (1-alpha)*KL(student‖teacher)
                    tsoft = tsoft.to(device)
                    kd = -(tsoft * torch.log_softmax(logits, dim=1)).sum(1).mean()
                    loss = cfg["kd_alpha"] * loss + (1 - cfg["kd_alpha"]) * kd
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            run_loss += loss.item()
            if step % log_every == 0 or step == steps_per_epoch:
                el = time.time() - t0
                eta = el / step * (steps_per_epoch - step)          # 이번 에포크 남은 시간
                pct = step / steps_per_epoch * 100
                print(f"    ep{ep} {step:>4}/{steps_per_epoch} ({pct:4.0f}%) "
                      f"loss={run_loss/step:.3f} el={fmt(el)} eta={fmt(eta)}", flush=True)
                if wb:
                    wb.log({"train/loss": run_loss / step, "epoch": ep})
        logits = predict_logits(model, va_dl, device)
        f1 = L._macro(va_y, logits.argmax(1))
        epoch_hist.append(f1)
        improved = f1 > best_f1
        tag = "  ← best" if improved else ""
        print(f"  epoch {ep}: macro-F1(no-bias)={f1:.4f}  ({fmt(time.time()-t0)})  "
              f"[total {fmt(time.time()-train_start)}]{tag}", flush=True)
        if wb:
            wb.log({"val/macro_f1_nobias": f1, "epoch": ep})
        if improved:
            best_f1, best_logits, no_improve = f1, logits, 0
            if cfg["save_model"]:
                model.half().save_pretrained(os.path.join(out_dir, "model"))
                model.float()  # 계속 학습 위해 복원
                tok.save_pretrained(os.path.join(out_dir, "model"))
        else:
            no_improve += 1
            if no_improve >= cfg["patience"]:
                print(f"  early stop: {cfg['patience']} 에포크 개선 없음 (best={best_f1:.4f})", flush=True)
                break

    variants, best_variant, params, per_class, best_pred = L.evaluate_all(best_logits, va_y, class_prior)
    elapsed = round(time.time() - t_start, 1)
    print(f"\n[{name}] variants={variants}  best={best_variant} ({variants[best_variant]})", flush=True)

    save_charts(out_dir, name, epoch_hist, best_pred, va_y, variants[best_variant])
    print(f"  charts saved → {out_dir}/curve.png, confusion.png", flush=True)

    if wb:
        import wandb
        wb.log({"val/best_f1": variants[best_variant], **{f"val/{k}": v for k, v in variants.items()},
                "curve": wandb.Image(os.path.join(out_dir, "curve.png")),
                "confusion": wandb.Image(os.path.join(out_dir, "confusion.png"))})
        wb.summary["best_f1"] = variants[best_variant]
        wb.finish()

    metrics = dict(name=name, model_name=cfg["model_name"], features=cfg["features"], loss=cfg["loss"],
                   epochs=cfg["epochs"], max_len=cfg["max_len"], fold=cfg["fold"],
                   baseline=base, variants=variants, best_variant=best_variant,
                   best_f1=variants[best_variant], per_class=per_class,
                   params=params, elapsed_sec=elapsed, config=cfg)
    json.dump(metrics, open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # OOF(out-of-fold) val 예측 저장 → 앙상블·증류·스태킹의 토대.
    # 모든 실험이 같은 fold·seed 를 쓰면 oof_idx 가 동일 → 모델 간 정렬되어 블렌딩 가능.
    np.save(os.path.join(out_dir, "oof_logits.npy"), np.asarray(best_logits, np.float32))
    np.save(os.path.join(out_dir, "oof_y.npy"), np.asarray(va_y))
    np.save(os.path.join(out_dir, "oof_idx.npy"), np.asarray(va_idx))

    if cfg["save_model"]:
        np.save(os.path.join(out_dir, "model", "class_bias.npy"), np.array(params["bias"], np.float32))
        # features 를 기록 → 제출 script.py 가 학습과 '동일한' 입력 구성을 재현.
        json.dump(dict(max_len=cfg["max_len"], model_name=cfg["model_name"], features=cfg["features"]),
                  open(os.path.join(out_dir, "model", "infer_config.json"), "w"))

    # 리더보드 append
    head = not os.path.exists(LEADERBOARD)
    with open(LEADERBOARD, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if head:
            w.writerow(["name", "model_name", "features", "loss", "epochs", "baseline",
                        "f1_none", "f1_bias_tune", "f1_logit_adjust", "best_variant", "best_f1", "elapsed_sec"])
        w.writerow([name, cfg["model_name"], cfg["features"], cfg["loss"], cfg["epochs"], base,
                    variants["none"], variants["bias_tune"], variants["logit_adjust"],
                    best_variant, variants[best_variant], elapsed])
    print(f"saved → {out_dir}/metrics.json  &  leaderboard.csv (best_f1={variants[best_variant]})", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
