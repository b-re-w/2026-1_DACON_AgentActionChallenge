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
import hashlib
import json
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
    SERIALIZE_PRESETS,
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
    gather_teacher_logits,
    measure_inference_latency,
    predict_logits,
    save_submission_model,
    write_metrics,
)


class KDDataset(Dataset):
    """student 입력 + teacher soft-label(14-dim) 을 함께 반환."""

    def __init__(self, samples, t_logits, tokenizer, max_length, serialize_kwargs=None):
        self.samples = samples
        self.t_logits = t_logits
        self.tok = tokenizer
        self.max_length = max_length
        self.serialize_kwargs = serialize_kwargs or {}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        enc = self.tok(serialize_sample(s, **self.serialize_kwargs),
                       truncation=True, max_length=self.max_length)
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


def dkd_loss(s, t, labels, T, dkd_alpha, dkd_beta):
    """Decoupled KD (Zhao et al. 2022): KL 을 TCKD(정답 vs 비정답 이분) + NCKD(비정답 분포)로 분리.

    NCKD 가 dark knowledge 의 핵심(teacher 과확신 완화)이라 dkd_beta 로 강하게 준다.
    이 태스크의 병목(탐색도구 4개 상호 애매성)이 비정답 클래스 구조라 NCKD 강화가 직접 공략.
    """
    gt = F.one_hot(labels, num_classes=s.size(1)).bool()   # (B,C) 정답 마스크
    other = ~gt
    ps = F.softmax(s / T, dim=1)
    pt = F.softmax(t / T, dim=1)

    # TCKD: [정답확률, 비정답합] 2-class 이분 분포의 KL
    def binmask(p):
        return torch.cat([(p * gt).sum(1, keepdim=True),
                          (p * other).sum(1, keepdim=True)], dim=1)
    bs, bt = binmask(ps), binmask(pt)
    tckd = F.kl_div(bs.clamp_min(1e-8).log(), bt, reduction="batchmean") * (T * T)

    # NCKD: 정답 클래스를 마스킹(-1000) 후 비정답끼리 재정규화한 분포의 KL
    t_nc = F.softmax(t / T - 1000.0 * gt, dim=1)
    s_nc = F.log_softmax(s / T - 1000.0 * gt, dim=1)
    nckd = F.kl_div(s_nc, t_nc, reduction="batchmean") * (T * T)

    return dkd_alpha * tckd + dkd_beta * nckd


class KDTrainer(Trainer):
    def __init__(self, *args, temperature=3.0, alpha=0.5,
                 dkd=False, dkd_alpha=1.0, dkd_beta=8.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.T = temperature
        self.alpha = alpha
        self.dkd = dkd
        self.dkd_alpha = dkd_alpha
        self.dkd_beta = dkd_beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        teacher = inputs.pop("teacher_logits")
        labels = inputs.pop("labels")
        out = model(**inputs)
        s = out.logits
        ce = F.cross_entropy(s, labels)
        T = self.T
        t = teacher.to(s.device)
        if self.dkd:
            kd = dkd_loss(s, t, labels, T, self.dkd_alpha, self.dkd_beta)
        else:
            kd = F.kl_div(
                F.log_softmax(s / T, dim=-1),
                F.softmax(t / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)
        loss = self.alpha * ce + (1.0 - self.alpha) * kd
        return (loss, out) if return_outputs else loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", nargs="+", default=None,
                    help="teacher model 디렉터리(들). 여러 개면 soft-label 평균(앙상블). "
                         "매번 teacher forward 발생.")
    ap.add_argument("--teacher-logits", nargs="+", default=None,
                    help="gen_softlabels 로 만든 재사용 store 태그/경로(들). 여러 개면 "
                         "부분집합 평균(앙상블). teacher forward 0회 → 빠른 스윕. "
                         "--teacher-dir 와 택일.")
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
    ap.add_argument("--dkd", action="store_true",
                    help="Decoupled KD 사용 (KL 을 TCKD+NCKD 로 분리, NCKD 강화). vanilla KD 대체.")
    ap.add_argument("--dkd-alpha", type=float, default=1.0, help="TCKD(정답 이분) 가중")
    ap.add_argument("--dkd-beta", type=float, default=8.0, help="NCKD(비정답 분포) 가중 — DKD 핵심 레버")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--serialize", default="base", choices=list(SERIALIZE_PRESETS),
                    help="student 입력 직렬화 프리셋 (teacher 미지정 시 teacher soft-label 에도 동일 적용)")
    ap.add_argument("--teacher-serialize", default=None, choices=list(SERIALIZE_PRESETS),
                    help="teacher soft-label 생성 전용 프리셋. 미지정 시 --serialize 와 동일(기존 동작). "
                         "예: --teacher-serialize rich --serialize base → rich teacher → base student(비대칭 KD).")
    args = ap.parse_args()
    if not args.teacher_dir and not args.teacher_logits:
        ap.error("--teacher-dir 또는 --teacher-logits 중 하나는 필요합니다")
    if args.teacher_dir and args.teacher_logits:
        ap.error("--teacher-dir 와 --teacher-logits 는 동시 사용 불가")
    sk = SERIALIZE_PRESETS[args.serialize]                       # student 입력
    t_serialize = args.teacher_serialize or args.serialize
    sk_teacher = SERIALIZE_PRESETS[t_serialize]                  # teacher soft-label 입력

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

    # teacher soft-label 확보 — 두 경로:
    #  (A) --teacher-logits: gen_softlabels 로 만든 재사용 store 에서 teacher 부분집합을
    #      id 기준으로 조회·평균. teacher forward 0회 → α/T/student/앙상블조합 스윕이 즉시.
    #  (B) --teacher-dir: 기존 방식(모델 로드 후 forward). 결과는 아래 공유 캐시에 저장.
    if args.teacher_logits:
        print(f"[teacher] 재사용 store {args.teacher_logits} 에서 soft-label 로드 "
              f"(teacher forward 없음)", flush=True)
        tl_train = gather_teacher_logits(args.teacher_logits, train_samples)
        tl_val = gather_teacher_logits(args.teacher_logits, val_samples)
    else:
        # teacher soft-label 공유 캐시 — teacher soft-label 은 (teacher 모델들 × teacher 입력
        # 프리셋 × fold × all_data × max_length) 에만 의존하고 student·α·T 와 무관하므로,
        # 이 키로 runs/_softlabels/ 에 저장해 다른 student/하이퍼 실험이 자동 재사용한다.
        cache_key = "|".join([
            ",".join(sorted(args.teacher_dir)),
            f"ser={t_serialize}", f"fold={args.val_fold}",
            f"all={int(args.all_data)}", f"ml={args.max_length}",
        ])
        digest = hashlib.md5(cache_key.encode()).hexdigest()[:16]
        shared_dir = Path("runs/_softlabels")
        shared_dir.mkdir(parents=True, exist_ok=True)
        cache = shared_dir / f"{digest}.npz"
        legacy = out / "teacher_cache.npz"  # 이전 per-out 캐시 하위호환
        src = cache if cache.exists() else (legacy if legacy.exists() else None)
        if src is not None:
            z = np.load(src)
            tl_train, tl_val = z["train"], z["val"]
            print(f"[teacher] soft-label 캐시 재사용: {src}", flush=True)
            if not cache.exists():  # legacy → 공유 위치로 승격
                np.savez(cache, train=tl_train, val=tl_val)
        else:
            print(f"[teacher] soft-label 생성 ({len(args.teacher_dir)}개 teacher 평균, "
                  f"serialize={t_serialize}) → 저장 {cache}", flush=True)
            tl_train = np.zeros((len(train_samples), NUM_CLASSES), np.float32)
            tl_val = np.zeros((len(val_samples), NUM_CLASSES), np.float32)
            for td in args.teacher_dir:
                tl_train += predict_logits(td, train_samples, max_length=args.max_length, serialize_kwargs=sk_teacher)
                if val_samples:
                    tl_val += predict_logits(td, val_samples, max_length=args.max_length, serialize_kwargs=sk_teacher)
            tl_train /= len(args.teacher_dir)  # logit 평균(앙상블)
            tl_val /= len(args.teacher_dir)
            np.savez(cache, train=tl_train, val=tl_val)
            (shared_dir / f"{digest}.json").write_text(
                json.dumps({"key": cache_key, "teacher_dir": args.teacher_dir,
                            "teacher_serialize": t_serialize, "val_fold": args.val_fold,
                            "all_data": args.all_data, "max_length": args.max_length}, indent=2),
                encoding="utf-8",
            )
    if val_samples:
        yv = np.array([CLASS_TO_ID[s.action] for s in val_samples])
        print(f"[teacher] val argmax macro_f1 = {f1_score(yv, tl_val.argmax(1), average='macro'):.4f}", flush=True)

    tok = build_tokenizer(args.student)
    train_ds = KDDataset(train_samples, tl_train, tok, args.max_length, serialize_kwargs=sk)
    val_ds = (KDDataset(val_samples, tl_val, tok, args.max_length, serialize_kwargs=sk)
              if val_samples else None)

    model = build_model(args.student, tok, grad_checkpoint=args.grad_checkpoint)
    targs = build_training_args(
        out, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        grad_accum=args.grad_accum, bf16=args.bf16, num_workers=args.num_workers,
        grad_checkpoint=args.grad_checkpoint, all_data=args.all_data,
        remove_unused_columns=False,  # teacher_logits 컬럼 보존 (KDCollator 가 사용)
    )
    trainer = KDTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=KDCollator(tok), compute_metrics=compute_metrics,
        temperature=args.temperature, alpha=args.alpha,
        dkd=args.dkd, dkd_alpha=args.dkd_alpha, dkd_beta=args.dkd_beta,
    )
    trainer.train()
    model_dir = save_submission_model(trainer, tok, out, args.max_length)

    if args.all_data:
        print(f"[done] all-data KD 학습 완료 -> {out}/model", flush=True)
        return

    metrics = trainer.evaluate()
    print(f"[eval] {metrics}", flush=True)

    # 추론시간 측정 — student 입력(sk)으로 제출 추론경로와 동일 배칭. 제출 속도 배점 대비용.
    latency = measure_inference_latency(
        model_dir, val_samples, max_length=args.max_length, serialize_kwargs=sk,
    )
    print(f"[latency] {latency['ms_per_sample']:.3f} ms/sample, "
          f"30k 투영 {latency['proj_30k_s']:.1f}s ({latency['device']}, n={latency['n']})", flush=True)

    write_metrics(out, {
        "student": args.student, "teacher": args.teacher_dir or args.teacher_logits,
        "teacher_source": "logits_store" if args.teacher_logits else "model_dir",
        "val_fold": args.val_fold,
        "temperature": args.temperature, "alpha": args.alpha,
        "serialize": args.serialize,
        "teacher_serialize": ("store" if args.teacher_logits else t_serialize),
        "macro_f1": float(metrics.get("eval_macro_f1", 0.0)),
        "acc": float(metrics.get("eval_acc", 0.0)),
        "latency": latency,
    })
    print(f"[done] student KD macro_f1={metrics.get('eval_macro_f1'):.4f} -> {out}/model", flush=True)


if __name__ == "__main__":
    main()
