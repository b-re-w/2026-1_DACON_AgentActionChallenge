"""AI Agent action 예측 데이터셋 패키지.

계층: io(로딩) → serialize(직렬화) → splits(StratifiedGroupKFold) → action_dataset(torch Dataset).

기본 사용 예시
--------------
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from ai_challenge.datasets import load_records, assign_folds, build_fold_datasets

    ds = AgentActionDataset(root=".", split="train", download=True)  # 없으면 자동 다운로드
    records = ds.samples                                   # list[ActionSample]
    fold = assign_folds(records, n_splits=5, seed=42)      # StratifiedGroupKFold

    tok = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base")
    train_ds, val_ds = build_fold_datasets(records, fold, val_fold=0,
                                           tokenizer=tok, max_length=512)
    collator = DataCollatorWithPadding(tok)                # 동적 패딩(표준)
"""

from __future__ import annotations

from .action_dataset import ActionDataset, AgentActionDataset, build_fold_datasets
from .io import ActionSample, load_jsonl, load_labels, load_records
from .schema import (
    ACTION_CLASSES,
    CLASS_TO_ID,
    ID_TO_CLASS,
    NUM_CLASSES,
    session_id_from_id,
)
from .serialize import SPECIAL_TOKENS, serialize_sample
from .splits import (
    assign_folds,
    load_folds,
    save_folds,
    train_val_indices,
    verify_no_leakage,
)

__all__ = [
    # schema
    "ACTION_CLASSES", "NUM_CLASSES", "CLASS_TO_ID", "ID_TO_CLASS", "session_id_from_id",
    # io
    "ActionSample", "load_records", "load_jsonl", "load_labels",
    # serialize
    "serialize_sample", "SPECIAL_TOKENS",
    # splits
    "assign_folds", "train_val_indices", "verify_no_leakage", "save_folds", "load_folds",
    # dataset
    "AgentActionDataset", "ActionDataset", "build_fold_datasets",
]
