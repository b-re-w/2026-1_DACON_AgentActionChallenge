"""공통 학습 빌딩 블록 — `method/` 의 train·distill·tune_threshold 가 공유한다.

여기에 모델/토크나이저 생성, TrainingArguments 구성, 지표, class weight, fold,
저장, 추론(logits) 등 반복 로직을 모은다. **커스텀 모델(헤드 변경·아키텍처 수정 등)이
필요해지면 그 모델 코드도 이 `models/` 패키지에 둔다.**
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from ai_challenge.datasets import (
    ACTION_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    SPECIAL_TOKENS,
    ActionDataset,
    assign_folds,
    build_fold_datasets,
    load_folds,
    save_folds,
    serialize_sample,
)

DATA_DIR = Path("data")
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
FOLDS_CSV = DATA_DIR / "folds.csv"


# --------------------------------------------------------------------------- #
# 모델 / 토크나이저
# --------------------------------------------------------------------------- #
def build_tokenizer(model_name: str):
    """토크나이저 로드 + pad 토큰(디코더용) + 섹션 특수 토큰 추가."""
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:  # Qwen 등 디코더는 pad 토큰이 없음
        tok.pad_token = tok.eos_token
    tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def build_model(model_name: str, tokenizer, grad_checkpoint: bool = False):
    """14-way seq-cls 모델 로드 (id2label 부여 → 저장 모델이 클래스명 self-describe)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label={i: c for i, c in ID_TO_CLASS.items()},
        label2id=dict(CLASS_TO_ID),
    )
    model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if grad_checkpoint:
        model.config.use_cache = False
    return model


# --------------------------------------------------------------------------- #
# 데이터 / 지표 / 가중치
# --------------------------------------------------------------------------- #
def load_train_records():
    from ai_challenge.datasets import load_records

    return load_records(TRAIN_JSONL, TRAIN_LABELS)


def get_folds(records, n_splits: int = 5, seed: int = 42):
    """folds.csv 캐시를 재사용(없으면 생성) → 모든 method 가 동일 분할 공유."""
    if FOLDS_CSV.exists():
        return load_folds(FOLDS_CSV, records)
    fold = assign_folds(records, n_splits=n_splits, seed=seed)
    save_folds(records, fold, FOLDS_CSV)
    return fold


def build_datasets(records, fold, tokenizer, max_length, val_fold=0, all_data=False,
                   serialize_kwargs=None):
    """(train_ds, val_ds) 생성. all_data 면 전체로 학습(val 없음).

    serialize_kwargs: 직렬화 프리셋(입력 신호 실험용). None 이면 기본 표현.
    """
    if all_data:
        return ActionDataset(records, tokenizer, max_length=max_length,
                             serialize_kwargs=serialize_kwargs), None
    return build_fold_datasets(
        records, fold, val_fold, tokenizer=tokenizer, max_length=max_length,
        serialize_kwargs=serialize_kwargs,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.asarray(logits).argmax(-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "acc": accuracy_score(labels, preds),
    }


def compute_class_weights(samples) -> torch.Tensor:
    """balanced class weight (train fold 기준)."""
    y = np.array([CLASS_TO_ID[s.action] for s in samples])
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float64)
    w = counts.sum() / (NUM_CLASSES * np.maximum(counts, 1.0))
    return torch.tensor(w, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# TrainingArguments / Trainer
# --------------------------------------------------------------------------- #
def build_training_args(
    out_dir,
    *,
    epochs: float,
    batch_size: int,
    lr: float,
    grad_accum: int = 1,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    bf16: bool = True,
    num_workers: int = 4,
    grad_checkpoint: bool = False,
    all_data: bool = False,
    remove_unused_columns: bool = True,
    seed: int = 42,
    optim: str = "adamw_torch",
) -> TrainingArguments:
    """train·distill 공용 TrainingArguments (all_data 면 eval/save 끔).

    distill 은 데이터셋에 teacher_logits 컬럼을 실어 보내므로 remove_unused_columns=False
    로 호출해야 한다(기본 True 면 collator 전에 제거되어 KeyError).
    seed: 앙상블 다양성용. optim: 큰 모델은 "paged_adamw_8bit"(bitsandbytes)로 메모리 절약.
    """
    return TrainingArguments(
        remove_unused_columns=remove_unused_columns,
        seed=seed,
        optim=optim,
        output_dir=str(Path(out_dir) / "hf"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=max(batch_size, 64),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=bf16,
        fp16=not bf16,
        eval_strategy="no" if all_data else "epoch",
        save_strategy="no" if all_data else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not all_data,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=num_workers,
        gradient_checkpointing=grad_checkpoint,
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
    )


class WeightedTrainer(Trainer):
    """class-weighted CrossEntropy Trainer. class_weights=None 이면 일반 CE."""

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


# --------------------------------------------------------------------------- #
# 저장 / OOF / 추론
# --------------------------------------------------------------------------- #
def save_submission_model(trainer, tokenizer, out_dir, max_length: int) -> Path:
    """제출용 model/ 디렉터리 저장 (가중치 + 토크나이저 + infer_config)."""
    model_dir = Path(out_dir) / "model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    (model_dir / "infer_config.json").write_text(
        json.dumps({"max_length": max_length}), encoding="utf-8"
    )
    return model_dir


def write_oof(out_dir, val_samples, preds) -> None:
    with (Path(out_dir) / "oof.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "true", "pred"])
        for s, p in zip(val_samples, preds):
            w.writerow([s.id, s.action, ID_TO_CLASS[int(p)]])


def write_metrics(out_dir, payload: dict) -> None:
    (Path(out_dir) / "metrics.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


@torch.inference_mode()
def predict_logits(model_dir, samples, max_length: int = 512, batch_size: int = 128,
                   serialize_kwargs=None):
    """저장된 모델로 샘플들의 14-class logits 계산 (길이정렬 배칭).

    tune_threshold(OOF logits)·distill(teacher soft-label) 이 공유한다.
    serialize_kwargs: 직렬화 프리셋(teacher/student 입력 일치가 중요).
    """
    sk = serialize_kwargs or {}
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = (
        AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        .cuda()
        .half()
        .eval()
    )
    texts = [serialize_sample(s, **sk) for s in samples]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = np.zeros((len(texts), NUM_CLASSES), dtype=np.float32)
    for st in range(0, len(order), batch_size):
        chunk = order[st : st + batch_size]
        enc = tok(
            [texts[i] for i in chunk],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to("cuda")
        logit = model(**enc).logits.float().cpu().numpy()
        for pos, i in enumerate(chunk):
            out[i] = logit[pos]
    del model
    torch.cuda.empty_cache()
    return out


__all__ = [
    "DATA_DIR", "TRAIN_JSONL", "TRAIN_LABELS", "FOLDS_CSV",
    "build_tokenizer", "build_model", "load_train_records", "get_folds",
    "build_datasets", "compute_metrics", "compute_class_weights",
    "build_training_args", "WeightedTrainer",
    "save_submission_model", "write_oof", "write_metrics", "predict_logits",
]
