"""AI Agent action 분류 학습 (접근 A 인코더 / 접근 B 디코더 SLM 공용).

두 접근 모두 `AutoModelForSequenceClassification` (14-way) 로 통일한다:
  - A: microsoft/mdeberta-v3-base  (양방향 인코더)
  - B: Qwen/Qwen2.5-0.5B           (디코더 SLM, 마지막 토큰 분류 헤드)

핵심 설계
  - 검증: StratifiedGroupKFold (group=세션) — data/folds.csv 캐시로 A/B 동일 분할 공유.
  - 불균형: class-weighted CrossEntropy (Macro-F1 이 지표라 희소 클래스 가중).
  - 산출물: out/model (save_pretrained, id2label 포함) + out/metrics.json + out/oof.csv.

사용:
    uv run python -m ai_challenge.train --model microsoft/mdeberta-v3-base --out runs/A
    uv run python -m ai_challenge.train --model Qwen/Qwen2.5-0.5B --out runs/B --bf16
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from datasets import (
    ACTION_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    SPECIAL_TOKENS,
    assign_folds,
    build_fold_datasets,
    load_folds,
    load_records,
    save_folds,
)

DATA_DIR = Path("data")
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
FOLDS_CSV = DATA_DIR / "folds.csv"


class WeightedTrainer(Trainer):
    """class-weighted CrossEntropy 를 쓰는 Trainer (Macro-F1 대응)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = (
            self.class_weights.to(outputs.logits.device)
            if self.class_weights is not None
            else None
        )
        loss = F.cross_entropy(outputs.logits, labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "acc": accuracy_score(labels, preds),
    }


def get_folds(records):
    """A/B 가 동일 분할을 쓰도록 folds.csv 를 캐시/재사용한다."""
    if FOLDS_CSV.exists():
        return load_folds(FOLDS_CSV, records)
    fold = assign_folds(records, n_splits=5, seed=42)
    save_folds(records, fold, FOLDS_CSV)
    return fold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--no-class-weight", action="store_true")
    ap.add_argument("--all-data", action="store_true",
                    help="검증셋 없이 전체 70k로 학습(최종 제출용). OOF 미산출.")
    ap.add_argument("--grad-checkpoint", action="store_true",
                    help="gradient checkpointing (큰 teacher 모델 메모리 절약)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)

    # ---- 데이터 ----
    records = load_records(TRAIN_JSONL, TRAIN_LABELS)
    fold = get_folds(records)
    print(f"[data] records={len(records)} folds cached at {FOLDS_CSV}", flush=True)

    # ---- 토크나이저 (섹션 특수 토큰 추가) ----
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:  # Qwen 등 디코더는 pad 토큰이 없음
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    if args.all_data:
        from datasets import ActionDataset

        train_ds = ActionDataset(records, tok, max_length=args.max_length)
        val_ds = None
        print(f"[data] all-data train={len(train_ds)} (val 없음)", flush=True)
    else:
        train_ds, val_ds = build_fold_datasets(
            records, fold, args.val_fold, tokenizer=tok, max_length=args.max_length
        )
        print(f"[data] train={len(train_ds)} val={len(val_ds)}", flush=True)

    # ---- class weights (train fold 기준) ----
    class_weights = None
    if not args.no_class_weight:
        y_train = np.array([CLASS_TO_ID[s.action] for s in train_ds.samples])
        counts = np.bincount(y_train, minlength=NUM_CLASSES).astype(np.float64)
        w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))  # balanced
        class_weights = torch.tensor(w, dtype=torch.float32)
        print(f"[cw] {dict(zip(ACTION_CLASSES, np.round(w, 3)))}", flush=True)

    # ---- 모델 (id2label 부여 → 저장 모델이 클래스명을 self-describe) ----
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=NUM_CLASSES,
        id2label={i: c for i, c in ID_TO_CLASS.items()},
        label2id=dict(CLASS_TO_ID),
    )
    model.resize_token_embeddings(len(tok))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    targs = TrainingArguments(
        output_dir=str(out / "hf"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(args.batch_size, 64),
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        bf16=args.bf16,
        fp16=not args.bf16,
        eval_strategy="no" if args.all_data else "epoch",
        save_strategy="no" if args.all_data else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not args.all_data,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=args.num_workers,
        gradient_checkpointing=args.grad_checkpoint,
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
    )
    if args.grad_checkpoint:
        model.config.use_cache = False

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()

    # ---- 저장: 제출용 model/ 디렉터리 ----
    model_dir = out / "model"
    trainer.save_model(str(model_dir))
    tok.save_pretrained(str(model_dir))
    (model_dir / "infer_config.json").write_text(
        json.dumps({"max_length": args.max_length}), encoding="utf-8"
    )

    if args.all_data:  # OOF 없음 → 저장만 하고 종료
        print(f"[done] all-data 학습 완료 -> {model_dir}", flush=True)
        return

    metrics = trainer.evaluate()
    print(f"[eval] {metrics}", flush=True)

    pred = trainer.predict(val_ds)
    preds = pred.predictions.argmax(-1)
    val_ids = [s.id for s in val_ds.samples]
    y_true = [ID_TO_CLASS[CLASS_TO_ID[s.action]] for s in val_ds.samples]
    import csv

    with (out / "oof.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "true", "pred"])
        for i, p, t in zip(val_ids, preds, y_true):
            w.writerow([i, t, ID_TO_CLASS[int(p)]])

    (out / "metrics.json").write_text(
        json.dumps(
            {
                "model": args.model,
                "val_fold": args.val_fold,
                "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
                "acc": float(metrics.get("eval_acc", 0.0)),
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] macro_f1={metrics.get('eval_macro_f1'):.4f} -> {model_dir}", flush=True)


if __name__ == "__main__":
    main()
