"""Knowledge Distillation: 큰 teacher(예: Qwen2.5-1.5B) → 작은 student(Qwen2.5-0.5B).

이 태스크는 train==test 가 같은 시뮬레이터 분포라, 큰 모델이 시뮬레이터 결정함수를
더 잘 근사한다. 그 soft-label(dark knowledge)을 1GB·10분 제약을 지키는 0.5B student
에 전달해, 크기·속도는 그대로 두고 정확도를 올린다.

KD loss = alpha * CE(hard) + (1-alpha) * T^2 * KL(student/T || teacher/T)

teacher soft-label 은 student 의 train/val 샘플에 대해 미리 계산(길이정렬 배칭)한다.
teacher 는 val_fold 을 holdout 해 학습됐으므로 fold0 소프트라벨은 OOF(정직), 나머지는
in-sample(표준 KD 관행).

사용:
    uv run python -m ai_challenge.distill --teacher-dir runs/teacher15/model \
        --student Qwen/Qwen2.5-0.5B --out runs/student_kd --val-fold 0 \
        --temperature 3 --alpha 0.5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from ai_challenge.datasets import (
    ACTION_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    SPECIAL_TOKENS,
    load_folds,
    load_records,
    serialize_sample,
)

DATA_DIR = Path("data")


@torch.inference_mode()
def teacher_logits(teacher_dir: str, samples, max_length: int, batch_size: int = 128):
    """teacher 로 샘플들의 14-class logits 를 계산(길이정렬 배칭)."""
    tok = AutoTokenizer.from_pretrained(teacher_dir)
    model = AutoModelForSequenceClassification.from_pretrained(teacher_dir).cuda().half().eval()
    texts = [serialize_sample(s) for s in samples]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
    for st in range(0, len(order), batch_size):
        chunk = order[st:st + batch_size]
        enc = tok([texts[i] for i in chunk], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to("cuda")
        logit = model(**enc).logits.float().cpu().numpy()
        for pos, i in enumerate(chunk):
            out[i] = logit[pos]
    del model
    torch.cuda.empty_cache()
    return out


class KDDataset(Dataset):
    def __init__(self, samples, t_logits, tokenizer, max_length):
        self.samples = samples
        self.t_logits = t_logits
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tok(serialize_sample(s), truncation=True, max_length=self.max_length)
        item = {k: enc[k] for k in enc}
        item["labels"] = CLASS_TO_ID[s.action]
        item["teacher_logits"] = self.t_logits[idx].tolist()
        return item


class KDCollator:
    """teacher_logits 를 분리해 스택하고, 나머지는 표준 동적 패딩."""

    def __init__(self, tokenizer):
        self.base = DataCollatorWithPadding(tokenizer)

    def __call__(self, features):
        tl = torch.tensor([f.pop("teacher_logits") for f in features], dtype=torch.float32)
        batch = self.base(features)
        batch["teacher_logits"] = tl
        return batch


class KDTrainer(Trainer):
    def __init__(self, *args, temperature=3.0, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temperature
        self.alpha = alpha

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher = inputs.pop("teacher_logits")
        labels = inputs.pop("labels")
        out = model(**inputs)
        s = out.logits
        ce = F.cross_entropy(s, labels)
        T = self.T
        kd = F.kl_div(
            F.log_softmax(s / T, dim=-1),
            F.softmax(teacher.to(s.device) / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        loss = self.alpha * ce + (1.0 - self.alpha) * kd
        return (loss, out) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {"macro_f1": f1_score(labels, preds, average="macro"),
            "acc": accuracy_score(labels, preds)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="CE 가중(나머지는 KD)")
    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)

    records = load_records(DATA_DIR / "train.jsonl", DATA_DIR / "train_labels.csv")
    fold = load_folds(DATA_DIR / "folds.csv", records)
    if args.all_data:
        train_samples = records
        val_samples = []
    else:
        train_samples = [s for s, f in zip(records, fold) if f != args.val_fold]
        val_samples = [s for s, f in zip(records, fold) if f == args.val_fold]
    print(f"[data] train={len(train_samples)} val={len(val_samples)}", flush=True)

    # teacher soft-label 캐시 (train/val 각각)
    cache = out / "teacher_cache.npz"
    if cache.exists():
        z = np.load(cache)
        tl_train, tl_val = z["train"], z["val"]
        print("[teacher] cache 사용", flush=True)
    else:
        print("[teacher] soft-label 생성...", flush=True)
        tl_train = teacher_logits(args.teacher_dir, train_samples, args.max_length)
        tl_val = (teacher_logits(args.teacher_dir, val_samples, args.max_length)
                  if val_samples else np.zeros((0, NUM_CLASSES), np.float32))
        np.savez(cache, train=tl_train, val=tl_val)
    # teacher 자기평가(참고): OOF fold 소프트라벨 argmax vs 정답
    if val_samples:
        yv = np.array([CLASS_TO_ID[s.action] for s in val_samples])
        print(f"[teacher] val argmax macro_f1 = {f1_score(yv, tl_val.argmax(1), average='macro'):.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    train_ds = KDDataset(train_samples, tl_train, tok, args.max_length)
    val_ds = KDDataset(val_samples, tl_val, tok, args.max_length) if val_samples else None

    model = AutoModelForSequenceClassification.from_pretrained(
        args.student, num_labels=NUM_CLASSES,
        id2label={i: c for i, c in ID_TO_CLASS.items()}, label2id=dict(CLASS_TO_ID),
    )
    model.resize_token_embeddings(len(tok))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    targs = TrainingArguments(
        output_dir=str(out / "hf"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 64),
        learning_rate=args.lr, warmup_ratio=0.06, weight_decay=0.01,
        bf16=True,
        eval_strategy="no" if args.all_data else "epoch",
        save_strategy="no" if args.all_data else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not args.all_data,
        metric_for_best_model="macro_f1", greater_is_better=True,
        dataloader_num_workers=args.num_workers, logging_steps=50, report_to=[],
    )
    trainer = KDTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=KDCollator(tok), compute_metrics=compute_metrics,
        temperature=args.temperature, alpha=args.alpha,
    )
    trainer.train()

    model_dir = out / "model"
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    (model_dir / "infer_config.json").write_text(json.dumps({"max_length": args.max_length}), encoding="utf-8")

    if args.all_data:
        print(f"[done] all-data KD 학습 완료 -> {model_dir}", flush=True)
        return

    metrics = trainer.evaluate()
    print(f"[eval] {metrics}", flush=True)
    (out / "metrics.json").write_text(json.dumps({
        "student": args.student, "teacher": args.teacher_dir, "val_fold": args.val_fold,
        "temperature": args.temperature, "alpha": args.alpha,
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "acc": float(metrics.get("eval_acc", 0.0)),
    }, indent=2), encoding="utf-8")
    print(f"[done] student KD macro_f1={metrics.get('eval_macro_f1'):.4f} -> {model_dir}", flush=True)


if __name__ == "__main__":
    main()
