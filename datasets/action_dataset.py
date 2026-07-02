"""torch.utils.data.Dataset 구현 + transformers 연동 헬퍼.

- ``AgentActionDataset`` : torchvision 규격(root/split/download/transform)의 원천
  데이터셋. DACON open.zip 을 자동 내려받아 ``data/`` 로 풀고 ActionSample 을 로드한다.
- ``ActionDataset`` : transformers Trainer 용 토크나이즈 어댑터. 동적 패딩
  (transformers.DataCollatorWithPadding)을 쓰기 위해 __getitem__ 은 패딩하지 않은
  토큰 시퀀스를 반환한다(HF 표준 패턴).
"""

from __future__ import annotations

import hashlib
import os
import urllib.request
import zipfile
from collections.abc import Callable

import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from .io import ActionSample, load_records
from .schema import CLASS_TO_ID
from .serialize import serialize_sample


class AgentActionDataset(Dataset):
    """AI Agent action 예측 원천 데이터셋 (torchvision 규격).

    torchvision 데이터셋 관례를 따른다: ``root`` 아래에 데이터가 없으면
    ``download=True`` 로 DACON open.zip 을 내려받아 무결성(MD5) 검증 후 압축을 풀고,
    ``__getitem__`` 은 ``(sample, target)`` 튜플을 반환한다. target 은 라벨이 있으면
    클래스 정수 id, 라벨이 없는 test split 은 -1.

    Parameters
    ----------
    root : 데이터 루트. 압축 해제 시 ``root/data/*.jsonl`` 로 풀린다(기본 ".").
    split : "train" 또는 "test".
    download : True 면 데이터가 없을 때 자동으로 내려받아 압축 해제한다.
    transform : sample(ActionSample)에 적용할 변환(예: 직렬화·토크나이즈).
    target_transform : target(int)에 적용할 변환.

    예시
    ----
        ds = AgentActionDataset(root=".", split="train", download=True)
        records = ds.samples                      # list[ActionSample]
        fold = assign_folds(records, n_splits=5)  # StratifiedGroupKFold
    """

    url: str = "https://cfiles.dacon.co.kr/competitions/236694/open.zip"
    # 배포본 무결성 해시(2026-06-30 기준). 서버 ETag == 콘텐츠 MD5 임을 확인.
    md5: str = "ac1738d0b81cc044f6fa31f16cf7a962"
    filename: str = "open.zip"

    # split -> (jsonl 파일명, labels 파일명 or None)
    _FILES: dict[str, tuple[str, str | None]] = {
        "train": ("train.jsonl", "train_labels.csv"),
        "test": ("test.jsonl", None),
    }

    def __init__(
        self,
        root: str | os.PathLike = ".",
        split: str = "train",
        download: bool = False,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        if split not in self._FILES:
            raise ValueError(
                f"split must be one of {list(self._FILES)}, got {split!r}"
            )
        self.root = os.path.expanduser(str(root))
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError(
                "Dataset not found. Use download=True to download it, or place the "
                f"DACON open.zip contents so that {self.data_dir!r} exists."
            )

        jsonl_name, labels_name = self._FILES[split]
        labels_path = (
            os.path.join(self.data_dir, labels_name) if labels_name else None
        )
        self.samples: list[ActionSample] = load_records(
            os.path.join(self.data_dir, jsonl_name), labels_path
        )

    # ------------------------------------------------------------------ #
    # torchvision-style Dataset 인터페이스
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        target = CLASS_TO_ID[sample.action] if sample.action is not None else -1
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(root={self.root!r}, split={self.split!r}, "
            f"n={len(self)})"
        )

    @property
    def data_dir(self) -> str:
        """압축 해제된 데이터 디렉터리 (``root/data``)."""
        return os.path.join(self.root, "data")

    # ------------------------------------------------------------------ #
    # 다운로드 / 무결성 검증 (torchvision.datasets.utils 규격 최소 구현)
    # ------------------------------------------------------------------ #
    def _check_exists(self) -> bool:
        """현재 split 이 필요로 하는 파일이 모두 존재하는지 확인한다."""
        jsonl_name, labels_name = self._FILES[self.split]
        needed = [jsonl_name] + ([labels_name] if labels_name else [])
        return all(
            os.path.isfile(os.path.join(self.data_dir, n)) for n in needed
        )

    def download(self) -> None:
        """open.zip 을 내려받아(무결성 검증) ``root`` 에 압축 해제한다.

        이미 데이터가 있으면 아무 것도 하지 않는다(멱등).
        """
        if self._check_exists():
            print("Files already downloaded and verified")
            return
        os.makedirs(self.root, exist_ok=True)
        archive = os.path.join(self.root, self.filename)

        if self._check_integrity(archive, self.md5):
            print(f"Using downloaded and verified file: {archive}")
        else:
            print(f"Downloading {self.url} to {archive}")
            self._urlretrieve(self.url, archive)
            if not self._check_integrity(archive, self.md5):
                raise RuntimeError(
                    f"File corrupted after download (md5 mismatch): {archive}"
                )

        print(f"Extracting {archive} to {self.root}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(self.root)

    @staticmethod
    def _calculate_md5(fpath: str, chunk_size: int = 1024 * 1024) -> str:
        md5 = hashlib.md5()
        with open(fpath, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    @classmethod
    def _check_integrity(cls, fpath: str, md5: str | None = None) -> bool:
        if not os.path.isfile(fpath):
            return False
        if md5 is None:
            return True
        return cls._calculate_md5(fpath) == md5

    @staticmethod
    def _urlretrieve(url: str, fpath: str, chunk: int = 1024 * 32) -> None:
        """진행률 표시와 함께 url 을 fpath 로 스트리밍 다운로드한다."""
        req = urllib.request.Request(
            url, headers={"User-Agent": "agent-action/0.1"}
        )
        with urllib.request.urlopen(req) as resp:  # noqa: S310 (신뢰된 대회 서버)
            total = int(resp.headers.get("Content-Length", 0)) or None
            with open(fpath, "wb") as fh, tqdm(
                total=total, unit="B", unit_scale=True, unit_divisor=1024,
                desc=os.path.basename(fpath),
            ) as pbar:
                while True:
                    data = resp.read(chunk)
                    if not data:
                        break
                    fh.write(data)
                    pbar.update(len(data))


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
