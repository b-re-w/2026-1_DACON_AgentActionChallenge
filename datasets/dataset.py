"""torch.utils.data.Dataset 구현 + transformers 연동 헬퍼.

동적 패딩(transformers.DataCollatorWithPadding)을 쓰기 위해 __getitem__ 은
패딩하지 않은 토큰 시퀀스를 반환한다(HF 표준 패턴).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from torch.utils.data import Dataset

from .io import ActionSample
from .schema import CLASS_TO_ID
from .serialize import serialize_sample


class ActionDataset(Dataset):
    """AI Agent action 예측용 PyTorch Dataset.

    __getitem__ 은 {input_ids, attention_mask, (token_type_ids), labels?} dict 반환.
    labels 는 라벨이 있을 때만 포함(test 는 생략).
    """

    def __init__(
        self,
        samples: list[ActionSample],
        tokenizer,
        max_length: int = 512,
        serialize_fn: Callable[..., str] = serialize_sample,
        serialize_kwargs: dict | None = None,
        include_labels: bool = True,
    ) -> None:
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.serialize_fn = serialize_fn
        self.serialize_kwargs = serialize_kwargs or {}
        self.include_labels = include_labels

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        text = self.serialize_fn(sample, **self.serialize_kwargs)
        enc = self.tokenizer(text, truncation=True, max_length=self.max_length)
        item = {k: enc[k] for k in enc}
        if self.include_labels and sample.action is not None:
            item["labels"] = CLASS_TO_ID[sample.action]
        return item


def build_fold_datasets(
    samples: list[ActionSample],
    fold: np.ndarray,
    val_fold: int,
    tokenizer,
    **dataset_kwargs,
) -> tuple[ActionDataset, ActionDataset]:
    """fold 배열과 val_fold 로 (train_ds, val_ds) 생성."""
    from .splits import train_val_indices

    train_idx, val_idx = train_val_indices(fold, val_fold)
    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    train_ds = ActionDataset(train_samples, tokenizer, **dataset_kwargs)
    val_ds = ActionDataset(val_samples, tokenizer, **dataset_kwargs)
    return train_ds, val_ds
