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
def build_tokenizer(model_name: str, add_special: bool = True):
    """토크나이저 로드 + pad 토큰(디코더용) + (선택) 섹션 특수 토큰 추가.

    add_special=False 는 QLoRA 경로용 — 4bit 고정 임베딩에 새 토큰 행을 학습시키지
    않기 위해 special token 을 붙이지 않는다(섹션 마커는 서브워드로 처리).
    """
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:  # Qwen 등 디코더는 pad 토큰이 없음
        tok.pad_token = tok.eos_token
    if add_special:
        tok.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})
    return tok


def _propagate_pad_token(model, tokenizer):
    """pad_token_id 를 config + 중첩 config 에 전파 (Qwen 등 디코더/멀티모달 대응)."""
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    # 멀티모달/중첩 config(예: qwen3_5 는 text_config·vision_config 보유) 에서는
    # seq-cls forward 가 하위 text_config.pad_token_id 를 읽으므로 거기에도 전파한다.
    for sub_name in ("text_config", "llm_config", "language_config"):
        sub = getattr(model.config, sub_name, None)
        if sub is not None and getattr(sub, "pad_token_id", None) is None:
            sub.pad_token_id = model.config.pad_token_id


def build_model(model_name: str, tokenizer, grad_checkpoint: bool = False):
    """14-way seq-cls 모델 로드 (id2label 부여 → 저장 모델이 클래스명 self-describe)."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_CLASSES,
        id2label={i: c for i, c in ID_TO_CLASS.items()},
        label2id=dict(CLASS_TO_ID),
    )
    model.resize_token_embeddings(len(tokenizer))
    _propagate_pad_token(model, tokenizer)
    if grad_checkpoint:
        model.config.use_cache = False
    return model


# LoRA 기본 대상 — Qwen2.5/Llama-3/Gemma-2 계열 공통 proj 이름.
QLORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]


def build_qlora_model(model_name: str, tokenizer, *, lora_r: int = 16,
                      lora_alpha: int = 32, lora_dropout: float = 0.05,
                      target_modules=None, grad_checkpoint: bool = False,
                      attn_implementation=None):
    """대형 teacher 용 QLoRA 모델: 4bit(nf4) 고정 base + 학습가능 LoRA 어댑터.

    핵심: seq-cls 의 ``score`` head 는 새로 초기화되므로 **반드시 학습·저장**해야 한다
    (modules_to_save=["score"]). 이걸 빠뜨리면 base 는 잘 학습돼도 head 가 랜덤이라
    soft-label 이 전부 노이즈가 된다.

    special token 은 붙이지 않는다(build_tokenizer(add_special=False) 로 호출) — 4bit
    고정 임베딩에 새 행을 학습시키지 않기 위해서다. 섹션 마커([META] 등)는 일반
    서브워드로 토크나이즈되어 그대로 작동한다(표현이 base teacher 와 약간 달라 앙상블
    다양성에도 기여).
    """
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    fp_kwargs = dict(
        num_labels=NUM_CLASSES,
        id2label={i: c for i, c in ID_TO_CLASS.items()}, label2id=dict(CLASS_TO_ID),
        quantization_config=bnb, torch_dtype=torch.bfloat16, device_map={"": 0},
    )
    if attn_implementation:  # Gemma-2 는 soft-capping 때문에 'eager' 권장
        fp_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **fp_kwargs)
    _propagate_pad_token(model, tokenizer)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=grad_checkpoint)
    lora = LoraConfig(
        task_type="SEQ_CLS", r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
        target_modules=target_modules or QLORA_TARGET_MODULES, modules_to_save=["score"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
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
    inloop_eval: bool = True,
    seed: int = 42,
    optim: str = "adamw_torch",
    fsdp: str = "",
    fsdp_layer_cls: str = "Qwen2DecoderLayer",
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
        eval_strategy="no" if (all_data or not inloop_eval) else "epoch",
        save_strategy="no" if (all_data or not inloop_eval) else "epoch",
        save_total_limit=1,
        load_best_model_at_end=not (all_data or not inloop_eval),
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        dataloader_num_workers=num_workers,
        # FSDP 에서는 TrainingArguments 의 gradient_checkpointing 이 backward 에 전체 AllGather 를
        #유발(OOM)하므로 끄고, fsdp_config 의 activation_checkpointing 을 사용한다.
        gradient_checkpointing=grad_checkpoint and not fsdp,
        logging_steps=50,
        report_to=[],
        disable_tqdm=False,
        **({"fsdp": fsdp,
            "fsdp_config": {"transformer_layer_cls_to_wrap": [fsdp_layer_cls],
                            "activation_checkpointing": grad_checkpoint,
                            "backward_prefetch": "backward_pre",
                            "use_orig_params": False}}
           if fsdp else {}),
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
def measure_inference_latency(model_dir, samples, max_length: int = 512, batch_size: int = 128,
                              serialize_kwargs=None, warmup_batches: int = 2):
    """제출 추론경로(길이정렬 배칭 + half)와 동일한 방식으로 추론 지연을 측정.

    반환: {n, total_s, ms_per_sample, proj_30k_s, device, batch_size}.
    로컬 GPU(예: A100) 기준 상대지표 — 제출 환경(T4)의 절대값과는 다르지만, 모델·입력
    프리셋 간 **상대 비교**(base vs rich 등)에 쓴다. proj_30k_s 는 test 30k 추론 예산(10분)
    대비 여유를 가늠하기 위한 선형 투영값.
    """
    import gc
    import time

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
    batches = [order[st : st + batch_size] for st in range(0, len(order), batch_size)]

    def _run(chunk):
        enc = tok(
            [texts[i] for i in chunk],
            truncation=True, max_length=max_length, padding=True, return_tensors="pt",
        ).to("cuda")
        model(**enc)

    for chunk in batches[:warmup_batches]:  # warmup (커널 컴파일·캐시)
        _run(chunk)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for chunk in batches:
        _run(chunk)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    n = len(texts)
    res = {
        "n": n,
        "total_s": round(elapsed, 3),
        "ms_per_sample": round(elapsed / n * 1000.0, 3),
        "proj_30k_s": round(elapsed / n * 30000.0, 1),
        "device": torch.cuda.get_device_name(0),
        "batch_size": batch_size,
    }
    model.cpu()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return res


def load_seqcls_model(model_dir, *, load_in_4bit: bool = False):
    """저장된 seq-cls 모델을 추론용(eval, GPU)으로 로드.

    - 일반 저장모델: fp16 로 로드해 .cuda().half().
    - PEFT 어댑터(adapter_config.json 존재): base 를 (선택적 4bit nf4) 로드 후 어댑터
      결합. 대형 QLoRA teacher 는 load_in_4bit=True 로 4bit base 위에서 바로 추론
      (fp16 merge 불필요 → 70B 도 안전).
    """
    md = Path(model_dir)
    if (md / "adapter_config.json").exists():
        from peft import PeftModel
        base = json.loads((md / "adapter_config.json").read_text())["base_model_name_or_path"]
        kw = dict(num_labels=NUM_CLASSES,
                  id2label={i: c for i, c in ID_TO_CLASS.items()}, label2id=dict(CLASS_TO_ID))
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
            kw["device_map"] = {"": 0}
        else:
            kw["torch_dtype"] = torch.float16
        base_model = AutoModelForSequenceClassification.from_pretrained(base, **kw)
        if base_model.config.pad_token_id is None:
            tk = AutoTokenizer.from_pretrained(str(md))
            base_model.config.pad_token_id = tk.pad_token_id or tk.eos_token_id
        model = PeftModel.from_pretrained(base_model, str(md))
        if not load_in_4bit:
            model = model.cuda()
        return model.eval()
    return AutoModelForSequenceClassification.from_pretrained(str(md)).cuda().half().eval()


@torch.inference_mode()
def predict_logits(model_dir, samples, max_length: int = 512, batch_size: int = 128,
                   serialize_kwargs=None, load_in_4bit: bool = False):
    """저장된 모델로 샘플들의 14-class logits 계산 (길이정렬 배칭).

    tune_threshold(OOF logits)·distill(teacher soft-label)·gen_softlabels 가 공유한다.
    serialize_kwargs: 직렬화 프리셋(teacher/student 입력 일치가 중요).
    load_in_4bit: QLoRA 대형 teacher 어댑터를 4bit base 위에서 추론.
    """
    sk = serialize_kwargs or {}
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = load_seqcls_model(model_dir, load_in_4bit=load_in_4bit)
    device = next(model.parameters()).device
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
        ).to(device)
        logit = model(**enc).logits.float().cpu().numpy()
        for pos, i in enumerate(chunk):
            out[i] = logit[pos]
    # 앙상블에서 teacher 를 여러 개 순차 로드하므로 확실히 해제(누수 방지)
    import gc

    model.cpu()
    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------- #
# Teacher soft-label 재사용 store
# --------------------------------------------------------------------------- #
# teacher 1개당 **전체 train 레코드**의 14-class raw logits 를 1회 캐시한다. 이후
# 증류(distill)는 이 store 들에서 임의의 teacher 부분집합을 sample.id 로 조회·평균해
# 쓰므로, fold/all_data/student/α/T/앙상블조합 을 바꾸는 실험이 teacher 재추론 없이
# 돈다. logits 는 softmax 전(raw) 이라 distill 시점의 temperature/alpha 스윕이 무료.
#   파일: runs/_teacher_logits/{tag}.npz  (logits:(N,14) f32, ids:(N,) str)
#         runs/_teacher_logits/{tag}.json (메타)
# id 로 조회하므로 레코드 순서가 바뀌어도 안전 — 없는 id 는 조기 실패시킨다.
TEACHER_LOGITS_DIR = Path("runs/_teacher_logits")


def resolve_teacher_logits_path(tag_or_path) -> Path:
    """bare tag('q25_3b_base') → runs/_teacher_logits/q25_3b_base.npz. .npz 경로면 그대로."""
    p = Path(tag_or_path)
    return p if p.suffix == ".npz" else TEACHER_LOGITS_DIR / f"{tag_or_path}.npz"


def save_teacher_logits(tag, ids, logits, meta=None) -> Path:
    """teacher store 저장 (전체 레코드 logits + id 인덱스 + 메타)."""
    TEACHER_LOGITS_DIR.mkdir(parents=True, exist_ok=True)
    path = TEACHER_LOGITS_DIR / f"{tag}.npz"
    np.savez(path, logits=np.asarray(logits, np.float32),
             ids=np.asarray(list(ids), dtype=object))
    (TEACHER_LOGITS_DIR / f"{tag}.json").write_text(
        json.dumps({"tag": tag, "n": len(logits), **(meta or {})}, indent=2),
        encoding="utf-8",
    )
    return path


def gather_teacher_logits(tags_or_paths, samples, weights=None) -> np.ndarray:
    """지정 teacher store 들을 samples 순서로 정렬·(가중)평균한 (N,14) logits 반환.

    각 store 는 전체 레코드 logits 이므로 여기서 sample.id 로 조회해 현재
    fold/all_data 하위집합만 뽑는다. 요청 id 가 store 에 없으면 조기 실패(불일치 감지).
    weights: store별 곱가중(합으로 정규화). None 이면 동등평균. 이질 teacher up-weight 용.
    """
    if not samples:
        return np.zeros((0, NUM_CLASSES), np.float32)
    if weights is None:
        weights = [1.0] * len(tags_or_paths)
    if len(weights) != len(tags_or_paths):
        raise ValueError(f"weights({len(weights)}) != teachers({len(tags_or_paths)})")
    want = [s.id for s in samples]
    acc = np.zeros((len(samples), NUM_CLASSES), np.float32)
    for t, w in zip(tags_or_paths, weights):
        path = resolve_teacher_logits_path(t)
        if not path.exists():
            raise FileNotFoundError(
                f"teacher-logits store 없음: {path} — gen_softlabels 로 먼저 생성"
            )
        z = np.load(path, allow_pickle=True)
        row = {rid: i for i, rid in enumerate(z["ids"].tolist())}
        try:
            idx = [row[i] for i in want]
        except KeyError as e:
            raise KeyError(
                f"{path.name} 에 없는 레코드 id={e.args[0]} — teacher store 가 현재 "
                f"데이터와 불일치(재생성 필요)"
            ) from None
        acc += float(w) * z["logits"][idx]
    acc /= float(sum(weights))
    return acc


__all__ = [
    "DATA_DIR", "TRAIN_JSONL", "TRAIN_LABELS", "FOLDS_CSV",
    "build_tokenizer", "build_model", "load_train_records", "get_folds",
    "build_datasets", "compute_metrics", "compute_class_weights",
    "build_training_args", "WeightedTrainer",
    "save_submission_model", "write_oof", "write_metrics", "predict_logits",
    "measure_inference_latency", "load_seqcls_model", "build_qlora_model",
    "TEACHER_LOGITS_DIR", "resolve_teacher_logits_path",
    "save_teacher_logits", "gather_teacher_logits",
]
