"""접근 A/B 학습 — 인코더(mDeBERTa)/디코더(Qwen) 공용 14-way seq-cls.

공통 로직은 ``ai_challenge.models.common``. 이 파일은 CLI + 오케스트레이션만 담당한다.

사용:
    uv run python -m ai_challenge.method.train --model microsoft/mdeberta-v3-base --out runs/A --bf16
    uv run python -m ai_challenge.method.train --model Qwen/Qwen2.5-0.5B --out runs/B --bf16 --no-class-weight
"""

from __future__ import annotations

import argparse
from pathlib import Path

from transformers import DataCollatorWithPadding

from ai_challenge.datasets import SERIALIZE_PRESETS
from ai_challenge.models.common import (
    WeightedTrainer,
    build_datasets,
    build_model,
    build_tokenizer,
    build_training_args,
    compute_class_weights,
    compute_metrics,
    get_folds,
    load_train_records,
    save_submission_model,
    write_metrics,
    write_oof,
)


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
                    help="검증셋 없이 전체 70k 학습(최종 제출용). OOF 미산출.")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--serialize", default="base", choices=list(SERIALIZE_PRESETS),
                    help="입력 직렬화 프리셋 (입력 신호 실험용)")
    ap.add_argument("--seed", type=int, default=42, help="앙상블 다양성용 시드")
    ap.add_argument("--optim", default="adamw_torch",
                    help="옵티마이저. 큰 모델은 paged_adamw_8bit (bitsandbytes)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] {vars(args)}", flush=True)

    records = load_train_records()
    fold = get_folds(records)
    tok = build_tokenizer(args.model)
    train_ds, val_ds = build_datasets(
        records, fold, tok, args.max_length,
        val_fold=args.val_fold, all_data=args.all_data,
        serialize_kwargs=SERIALIZE_PRESETS[args.serialize],
    )
    print(f"[data] train={len(train_ds)} val={len(val_ds) if val_ds else 0}", flush=True)

    class_weights = None if args.no_class_weight else compute_class_weights(train_ds.samples)

    model = build_model(args.model, tok, grad_checkpoint=args.grad_checkpoint)
    targs = build_training_args(
        out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        grad_accum=args.grad_accum, warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay, bf16=args.bf16, num_workers=args.num_workers,
        grad_checkpoint=args.grad_checkpoint, all_data=args.all_data,
        seed=args.seed, optim=args.optim,
    )
    trainer = WeightedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics, class_weights=class_weights,
    )
    trainer.train()
    save_submission_model(trainer, tok, out, args.max_length)

    if args.all_data:
        print(f"[done] all-data 학습 완료 -> {out}/model", flush=True)
        return

    metrics = trainer.evaluate()
    print(f"[eval] {metrics}", flush=True)
    preds = trainer.predict(val_ds).predictions.argmax(-1)
    write_oof(out, val_ds.samples, preds)
    write_metrics(out, {
        "model": args.model, "val_fold": args.val_fold,
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "acc": float(metrics.get("eval_acc", 0.0)),
        "n_train": len(train_ds), "n_val": len(val_ds),
    })
    print(f"[done] macro_f1={metrics.get('eval_macro_f1'):.4f} -> {out}/model", flush=True)


if __name__ == "__main__":
    main()
