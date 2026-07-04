"""Knowledge Distillation — 큰 teacher → 작은 student (제출용 0.5B).

train==test 가 같은 시뮬레이터 분포라, 큰 모델이 결정함수를 더 잘 근사한다. 그
soft-label(dark knowledge)을 1GB·10분 제약을 지키는 작은 student 에 전달한다.
공통 로직은 ``ai_challenge.models.common`` (teacher soft-label = predict_logits).

KD loss = alpha * CE(hard) + (1-alpha) * T^2 * KL(student/T || teacher/T)

사용:
    uv run python -m ai_challenge.method.distill --teacher-dir runs/teacher15/model \
        --student Qwen/Qwen2.5-0.5B --out runs/student_kd --bf16 --temperature 3 --alpha 0.5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, Trainer

from ai_challenge.datasets import (
    CLASS_TO_ID,
    NUM_CLASSES,
    load_folds,
    load_records,
    serialize_sample,
)
from ai_challenge.models.common import (
    DATA_DIR,
    build_model,
    build_tokenizer,
    build_training_args,
    compute_metrics,
    predict_logits,
    save_submission_model,
    write_metrics,
)


class KDDataset(Dataset):
    """student 입력 + teacher soft-label(14-dim) 을 함께 반환."""

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--temperature", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.5, help="CE 가중(나머지는 KD)")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--all-data", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)

    records = load_records(DATA_DIR / "train.jsonl", DATA_DIR / "train_labels.csv")
    fold = load_folds(DATA_DIR / "folds.csv", records)
    if args.all_data:
        train_samples, val_samples = records, []
    else:
        train_samples = [s for s, f in zip(records, fold) if f != args.val_fold]
        val_samples = [s for s, f in zip(records, fold) if f == args.val_fold]
    print(f"[data] train={len(train_samples)} val={len(val_samples)}", flush=True)

    # teacher soft-label 캐시 (train/val 각각) — 공통 predict_logits 사용
    cache = out / "teacher_cache.npz"
    if cache.exists():
        z = np.load(cache)
        tl_train, tl_val = z["train"], z["val"]
        print("[teacher] cache 사용", flush=True)
    else:
        print("[teacher] soft-label 생성...", flush=True)
        tl_train = predict_logits(args.teacher_dir, train_samples, max_length=args.max_length)
        tl_val = (predict_logits(args.teacher_dir, val_samples, max_length=args.max_length)
                  if val_samples else np.zeros((0, NUM_CLASSES), np.float32))
        np.savez(cache, train=tl_train, val=tl_val)
    if val_samples:
        yv = np.array([CLASS_TO_ID[s.action] for s in val_samples])
        print(f"[teacher] val argmax macro_f1 = {f1_score(yv, tl_val.argmax(1), average='macro'):.4f}", flush=True)

    tok = build_tokenizer(args.student)
    train_ds = KDDataset(train_samples, tl_train, tok, args.max_length)
    val_ds = KDDataset(val_samples, tl_val, tok, args.max_length) if val_samples else None

    model = build_model(args.student, tok, grad_checkpoint=args.grad_checkpoint)
    targs = build_training_args(
        out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        grad_accum=args.grad_accum, bf16=args.bf16, num_workers=args.num_workers,
        grad_checkpoint=args.grad_checkpoint, all_data=args.all_data,
    )
    trainer = KDTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=KDCollator(tok), compute_metrics=compute_metrics,
        temperature=args.temperature, alpha=args.alpha,
    )
    trainer.train()
    save_submission_model(trainer, tok, out, args.max_length)

    if args.all_data:
        print(f"[done] all-data KD 학습 완료 -> {out}/model", flush=True)
        return

    metrics = trainer.evaluate()
    print(f"[eval] {metrics}", flush=True)
    write_metrics(out, {
        "student": args.student, "teacher": args.teacher_dir, "val_fold": args.val_fold,
        "temperature": args.temperature, "alpha": args.alpha,
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "acc": float(metrics.get("eval_acc", 0.0)),
    })
    print(f"[done] student KD macro_f1={metrics.get('eval_macro_f1'):.4f} -> {out}/model", flush=True)


if __name__ == "__main__":
    main()
