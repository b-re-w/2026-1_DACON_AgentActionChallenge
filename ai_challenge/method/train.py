"""단일 fold(또는 all-data) 학습 CLI — seq-cls 14-way.

`models/common.py` 를 오케스트레이션한다. 산출물(runs/<name>/):
  model/            제출용 저장 모델(id2label 포함, infer_config)
  oof.csv           검증셋 (id, true, pred)  — all-data 면 생략
  oof_logits.npy    검증셋 로짓 (N, 14)      — threshold/ensemble 용
  oof_ids.json      검증셋 id 순서
  metrics.json      macro_f1 / acc / 설정

사용:
    uv run python -m ai_challenge.method.train \
        --model microsoft/mdeberta-v3-base --preset base \
        --fold 0 --epochs 3 --bs 32 --lr 2e-5 --max-length 512 \
        --name mdeberta_base_f0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import DataCollatorWithPadding

from ai_challenge.datasets import CLASS_TO_ID, ID_TO_CLASS, SERIALIZE_PRESETS
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="seq-cls 14-way 단일 fold 학습")
    ap.add_argument("--model", default="microsoft/mdeberta-v3-base")
    ap.add_argument("--preset", default="base", choices=list(SERIALIZE_PRESETS),
                    help="직렬화 프리셋(학습·추론 동일해야 함)")
    ap.add_argument("--fold", type=int, default=0, help="검증 fold 번호")
    ap.add_argument("--all-data", action="store_true", help="전체 데이터 학습(val 없음)")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup-ratio", type=float, default=0.06)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--class-weight", action="store_true", help="balanced class-weighted CE")
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42, help="앙상블 다양성용 시드")
    ap.add_argument("--optim", default="adamw_torch",
                    help="7B 급은 paged_adamw_8bit(bitsandbytes)로 옵티마이저 메모리 절약")
    ap.add_argument("--qlora", action="store_true",
                    help="4bit+LoRA 학습(20B+ 이질 teacher 용). 저장 시 merge 병합")
    ap.add_argument("--lora", action="store_true",
                    help="bf16+LoRA(양자화 없음) — mxfp4 네이티브 모델(gpt-oss)용")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--no-inloop-eval", action="store_true",
                    help="에폭 eval/저장 끄기 (MoE 등 eval 배치 OOM 회피용 — OOF 는 종료 후 별도)")
    ap.add_argument("--fp16", action="store_true", help="bf16 대신 fp16(구형 GPU)")
    ap.add_argument("--name", default=None, help="runs/<name>. 미지정 시 자동 생성")
    ap.add_argument("--out-root", default="runs")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    name = args.name or (
        f"{args.model.split('/')[-1]}_{args.preset}"
        + ("_all" if args.all_data else f"_f{args.fold}")
        + f"_s{args.seed}"
    )
    out_dir = Path(args.out_root) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    serialize_kwargs = SERIALIZE_PRESETS[args.preset]

    print(f"[train] {name}  model={args.model} preset={args.preset} "
          f"fold={'all' if args.all_data else args.fold} bs={args.bs} lr={args.lr} "
          f"epochs={args.epochs} max_len={args.max_length} cw={args.class_weight}")

    records = load_train_records()
    fold = None if args.all_data else get_folds(records)
    tok = build_tokenizer(args.model)
    model = build_model(args.model, tok, grad_checkpoint=args.grad_checkpoint,
                        qlora=args.qlora, lora=args.lora, lora_r=args.lora_r)

    train_ds, val_ds = build_datasets(
        records, fold, tok, args.max_length,
        val_fold=args.fold, all_data=args.all_data,
        serialize_kwargs=serialize_kwargs,
    )

    class_weights = compute_class_weights(train_ds.samples) if args.class_weight else None

    targs = build_training_args(
        out_dir,
        epochs=args.epochs, batch_size=args.bs, lr=args.lr, grad_accum=args.grad_accum,
        warmup_ratio=args.warmup_ratio, weight_decay=args.weight_decay,
        bf16=not args.fp16, num_workers=args.num_workers,
        grad_checkpoint=args.grad_checkpoint, all_data=args.all_data, seed=args.seed,
        optim=args.optim, inloop_eval=not args.no_inloop_eval,
    )

    trainer = WeightedTrainer(
        model=model, args=targs,
        train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )
    trainer.train()

    # (Q)LoRA 는 어댑터를 본체에 병합해 일반 모델로 저장(predict_logits 호환)
    if args.qlora or args.lora:
        trainer.model = trainer.model.merge_and_unload()
    # 제출용 모델 저장 + infer_config 에 preset 기록(pack 이 추론 직렬화에 사용)
    model_dir = save_submission_model(trainer, tok, out_dir, args.max_length)
    icfg = json.loads((model_dir / "infer_config.json").read_text())
    icfg["serialize"] = args.preset
    (model_dir / "infer_config.json").write_text(json.dumps(icfg))

    payload: dict = {
        "name": name, "model": args.model, "preset": args.preset,
        "fold": "all" if args.all_data else args.fold,
        "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
        "max_length": args.max_length, "class_weight": args.class_weight, "seed": args.seed,
    }

    if val_ds is not None and not args.no_inloop_eval:
        pred = trainer.predict(val_ds)
        logits = np.asarray(pred.predictions, dtype=np.float32)
        preds = logits.argmax(-1)
        write_oof(out_dir, val_ds.samples, preds)
        np.save(out_dir / "oof_logits.npy", logits)
        (out_dir / "oof_ids.json").write_text(
            json.dumps([s.id for s in val_ds.samples]), encoding="utf-8"
        )
        from sklearn.metrics import accuracy_score, f1_score
        y = np.array([CLASS_TO_ID[s.action] for s in val_ds.samples])
        payload["macro_f1"] = float(f1_score(y, preds, average="macro"))
        payload["acc"] = float(accuracy_score(y, preds))
        # per-class F1 (병목 진단용)
        per = f1_score(y, preds, average=None, labels=list(range(len(ID_TO_CLASS))))
        payload["per_class_f1"] = {ID_TO_CLASS[i]: float(per[i]) for i in range(len(per))}
        print(f"[result] macro_f1={payload['macro_f1']:.5f} acc={payload['acc']:.5f}")

    write_metrics(out_dir, payload)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
