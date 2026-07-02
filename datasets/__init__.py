"""AI Agent action 예측 데이터셋 패키지.

계층: io(로딩) → serialize(직렬화) → splits(StratifiedGroupKFold) → dataset(torch Dataset).

주의: 이 패키지 이름은 HuggingFace `datasets` 라이브러리와 겹친다. 프로젝트 루트를
sys.path 최상단에 둔 채 `import datasets` 하면 이 로컬 패키지가 우선한다. 본 프로젝트
스택(torch + transformers 분류 파인튜닝)은 HF datasets 를 요구하지 않으므로 문제되지
않지만, HF datasets 를 함께 쓰려면 실행 위치를 조정하거나 패키지명을 바꿀 것.

기본 사용 예시
--------------
    from transformers import AutoTokenizer, DataCollatorWithPadding
    from datasets import load_records, assign_folds, build_fold_datasets

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
