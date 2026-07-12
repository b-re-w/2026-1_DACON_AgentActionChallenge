"""지식증류(KD) — teacher 앙상블 soft-label → student(Qwen0.5B 등) 학습.

팀 최고 기록(LB 0.787, kd_w6_all) 레시피 재현+개선:
  loss = alpha * CE(hard) + (1-alpha) * T^2 * KL(student/T || teacher_blend/T)
  기본 T=3, alpha=0.25 (팀 검증값)

teacher 로짓은 run 디렉터리별로 전체 train 70k 에 대해 계산 후
`runs/<teacher>/train_logits.npy` 에 캐시한다(7B teacher 재계산 방지).
여러 teacher 는 --weights 로 가중 soft-blend (팀의 GPT5x2 가중과 동형).

사용:
    uv run python -m ai_challenge.method.distill \
        --teachers runs/qwen3b_f0/model runs/qwen7b_f0/model --weights 1 2 \
        --student Qwen/Qwen2.5-0.5B --preset base --student-preset base \
        --fold 0 --epochs 3 --bs 32 --lr 1e-5 --name kd_7b3b_f0
    # all-data 학습(제출용): --all-data (fold0 은 proxy 평가로만 사용)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Trainer

from ai_challenge.datasets import (
    CLASS_TO_ID,
    ID_TO_CLASS,
    SERIALIZE_PRESETS,
    ActionDataset,
)
from ai_challenge.models.common import (
    build_datasets,
    build_model,
    build_tokenizer,
    build_training_args,
    compute_metrics,
    get_folds,
    load_train_records,
    predict_logits,
    save_submission_model,
    write_metrics,
    write_oof,
)


# --------------------------------------------------------------------------- #
# teacher 로짓 — 중앙 저장소(teacher_logits/) 캐시
# --------------------------------------------------------------------------- #
def teacher_train_logits(teacher: str | Path, records, preset: str,
                         max_length: int = 512, batch_size: int = 64) -> np.ndarray:
    """teacher 의 전체 train 로짓 (N,14).

    teacher 는 (a) run 의 model/ 디렉터리 경로 또는 (b) teacher_logits 저장소 key.
    계산 결과는 `teacher_logits/<run>__<preset>.npy` 에 저장(레지스트리+TEACHERS.md 갱신).
    """
    from ai_challenge.models.teacher_store import (
        load_teacher_logits, logits_path, make_key, save_teacher_logits,
    )

    # (b) 저장소 key 직접 지정 (예: team_qgptoss). Path 로 들어와도 인식.
    key_str = str(teacher)
    if logits_path(key_str).exists():
        arr = load_teacher_logits(key_str)
        if arr is not None and len(arr) == len(records):
            print(f"[teacher] store hit (key): {key_str}")
            return arr

    model_dir = Path(teacher)
    run_name = model_dir.parent.name if model_dir.name == "model" else model_dir.name
    key = make_key(run_name, preset)

    arr = load_teacher_logits(key)
    if arr is not None and len(arr) == len(records):
        print(f"[teacher] store hit: {key}")
        return arr

    print(f"[teacher] predicting 70k logits: {model_dir} (preset={preset})")
    arr = predict_logits(model_dir, records, max_length=max_length,
                         batch_size=batch_size,
                         serialize_kwargs=SERIALIZE_PRESETS[preset])

    # run 의 metrics.json 에서 백본/OOF 점수 메타 자동 수집
    backbone, oof, fold = "?", None, "?"
    mpath = model_dir.parent / "metrics.json"
    if mpath.exists():
        m = json.loads(mpath.read_text())
        backbone = m.get("model") or m.get("student", "?")
        oof = m.get("macro_f1")
        fold = m.get("fold", "?")
    save_teacher_logits(key, arr, backbone=backbone, preset=preset, fold=fold,
                        model_dir=str(model_dir), oof_macro_f1=oof)
    return arr


def blend_teachers(logit_list: list[np.ndarray], weights: list[float], T: float) -> np.ndarray:
    """teacher 별 softmax(logits/T) 가중평균 → soft target 확률 (N,14)."""
    total = None
    wsum = sum(weights)
    for logits, w in zip(logit_list, weights):
        t = torch.from_numpy(logits) / T
        p = F.softmax(t, dim=-1).numpy() * (w / wsum)
        total = p if total is None else total + p
    return total.astype(np.float32)


# --------------------------------------------------------------------------- #
# KD 데이터셋 / Trainer
# --------------------------------------------------------------------------- #
class KDDataset(ActionDataset):
    """ActionDataset + soft_targets (teacher blend 확률) 를 함께 반환."""

    def __init__(self, samples, tokenizer, soft_targets: np.ndarray, **kw):
        super().__init__(samples, tokenizer, **kw)
        self.soft_targets = soft_targets

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item["soft_targets"] = self.soft_targets[idx].tolist()
        return item


class KDCollator:
    """동적 패딩 + soft_targets 스택. eval 배치(soft 없음)도 허용."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        has_soft = "soft_targets" in features[0]
        soft = (torch.tensor([f.pop("soft_targets") for f in features], dtype=torch.float32)
                if has_soft else None)
        batch = self.tokenizer.pad(features, return_tensors="pt")
        if soft is not None:
            batch["soft_targets"] = soft
        return batch


class KDTrainer(Trainer):
    """alpha*CE + (1-alpha)*T^2*KL(student/T || teacher_soft).

    eval 배치는 soft_targets 가 없으므로 plain CE 로 loss 계산(지표에만 사용).
    class_weights(14,) 지정 시 샘플 손실을 정답 클래스 가중으로 재가중(macro-F1 정합).
    """

    def __init__(self, *args, kd_alpha: float = 0.25, kd_T: float = 3.0,
                 class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.kd_alpha = kd_alpha
        self.kd_T = kd_T
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        soft = inputs.pop("soft_targets", None)
        outputs = model(**inputs)
        logits = outputs.logits.float()
        if soft is None:  # eval 경로
            loss = F.cross_entropy(logits, labels)
        else:
            ce_i = F.cross_entropy(logits, labels, reduction="none")
            log_p = F.log_softmax(logits / self.kd_T, dim=-1)
            kl_i = F.kl_div(log_p, soft.to(log_p.device), reduction="none").sum(-1)
            li = self.kd_alpha * ce_i + (1.0 - self.kd_alpha) * (self.kd_T ** 2) * kl_i
            if self.class_weights is not None:
                w = self.class_weights.to(li.device)[labels]
                loss = (li * w).sum() / w.sum()
            else:
                loss = li.mean()
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="teacher 앙상블 KD → student 학습")
    ap.add_argument("--teachers", nargs="+", required=True, help="teacher model/ 디렉터리들")
    ap.add_argument("--weights", nargs="+", type=float, default=None, help="teacher 가중치(기본 균등)")
    ap.add_argument("--teacher-preset", default=None,
                    help="teacher 직렬화 프리셋(기본: 각 model/infer_config.json 의 serialize)")
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--preset", default="base", choices=list(SERIALIZE_PRESETS),
                    help="student 직렬화 프리셋(제출 추론과 일치)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--kd-alpha", type=float, default=0.25)
    ap.add_argument("--kd-T", type=float, default=3.0)
    ap.add_argument("--eval-steps", type=int, default=500,
                    help="fold 런 스텝단위 검증 주기(베스트 체크포인트 선택). 0=에폭단위")
    ap.add_argument("--target-bias", default=None,
                    help="bias json 경로 — soft-target 확률에 bias 를 더해(클립·재정규화) 증류")
    ap.add_argument("--class-weight", default=None, choices=[None, "balanced", "sqrt"],
                    help="약클래스 가중 KD (샘플 손실을 정답 클래스 빈도 역수로 가중)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-checkpoint", action="store_true")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="runs")
    args = ap.parse_args()

    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = args.weights or [1.0] * len(args.teachers)
    assert len(weights) == len(args.teachers), "--weights 개수가 teacher 와 달라요"

    records = load_train_records()

    # 1) teacher 로짓(캐시) → soft blend
    logit_list = []
    for tdir in args.teachers:
        tdir = Path(tdir)
        preset = args.teacher_preset
        if preset is None:
            icfg = tdir / "infer_config.json"
            preset = json.loads(icfg.read_text()).get("serialize", "base") if icfg.exists() else "base"
        logit_list.append(teacher_train_logits(tdir, records, preset, args.max_length))
    soft = blend_teachers(logit_list, weights, args.kd_T)
    if args.target_bias:
        bvec = np.array(json.loads(Path(args.target_bias).read_text())["bias_vector"],
                        dtype=np.float32)
        soft = np.clip(soft + bvec[None, :], 1e-8, None)
        soft = soft / soft.sum(axis=1, keepdims=True)
        print(f"[target-bias] soft-target 에 bias 적용: {args.target_bias}")

    # teacher blend 자체의 train 정확도(상한 참고치)
    y_all = np.array([CLASS_TO_ID[s.action] for s in records])
    from sklearn.metrics import f1_score
    blend_macro = f1_score(y_all, soft.argmax(1), average="macro")
    print(f"[teacher-blend] train macro_f1={blend_macro:.5f} (KD 상한 참고치)")

    # 2) student 데이터셋 (KD)
    tok = build_tokenizer(args.student)
    model = build_model(args.student, tok, grad_checkpoint=args.grad_checkpoint)
    fold = None if args.all_data else get_folds(records)
    sk = SERIALIZE_PRESETS[args.preset]

    if args.all_data:
        train_samples, val_samples = records, None
        train_soft = soft
    else:
        from ai_challenge.datasets import train_val_indices
        tr_idx, va_idx = train_val_indices(fold, args.fold)
        train_samples = [records[i] for i in tr_idx]
        val_samples = [records[i] for i in va_idx]
        train_soft = soft[tr_idx]

    train_ds = KDDataset(train_samples, tok, train_soft, max_length=args.max_length,
                         serialize_kwargs=sk)
    val_ds = (ActionDataset(val_samples, tok, max_length=args.max_length, serialize_kwargs=sk)
              if val_samples else None)

    targs = build_training_args(
        out_dir, epochs=args.epochs, batch_size=args.bs, lr=args.lr,
        grad_accum=args.grad_accum, all_data=args.all_data, seed=args.seed,
        grad_checkpoint=args.grad_checkpoint,
        remove_unused_columns=False,  # soft_targets 유지 필수
        # fold 런은 스텝단위 검증으로 macro-F1 피크 체크포인트를 선택(load_best).
        inloop_eval=not args.all_data,
        eval_steps=args.eval_steps or None,
    )
    cw = None
    if args.class_weight:
        counts = np.bincount([CLASS_TO_ID[s.action] for s in train_samples], minlength=14)
        w = counts.sum() / (14 * np.maximum(counts, 1))
        if args.class_weight == "sqrt":
            w = np.sqrt(w)
        cw = torch.tensor(w / w.mean(), dtype=torch.float32)
        print(f"[cw] {args.class_weight}: min={cw.min():.2f} max={cw.max():.2f}")

    trainer = KDTrainer(
        model=model, args=targs, train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=KDCollator(tok),  # eval 배치(soft 없음)도 처리
        compute_metrics=compute_metrics if val_ds is not None else None,
        kd_alpha=args.kd_alpha, kd_T=args.kd_T, class_weights=cw,
    )
    trainer.train()

    # 검증 곡선/베스트 스텝 기록 (all-data 이식·과적합 진단용)
    eval_curve = [
        {"step": h["step"], "macro_f1": h["eval_macro_f1"]}
        for h in trainer.state.log_history if "eval_macro_f1" in h
    ]
    best_step = None
    if eval_curve:
        best = max(eval_curve, key=lambda e: e["macro_f1"])
        best_step = best["step"]
        steps_per_epoch = max(1, len(train_ds) // (args.bs * args.grad_accum))
        print(f"[ckpt] best step={best_step} ({best_step/steps_per_epoch:.2f}ep) "
              f"macro={best['macro_f1']:.5f} | last macro={eval_curve[-1]['macro_f1']:.5f}")

    model_dir = save_submission_model(trainer, tok, out_dir, args.max_length)
    icfg = json.loads((model_dir / "infer_config.json").read_text())
    icfg["serialize"] = args.preset
    (model_dir / "infer_config.json").write_text(json.dumps(icfg))

    payload = {
        "name": args.name, "student": args.student, "teachers": args.teachers,
        "weights": weights, "preset": args.preset,
        "fold": "all" if args.all_data else args.fold,
        "kd_alpha": args.kd_alpha, "kd_T": args.kd_T,
        "epochs": args.epochs, "bs": args.bs, "lr": args.lr,
        "teacher_blend_train_macro_f1": float(blend_macro),
        "eval_curve": eval_curve, "best_step": best_step,
    }

    # 3) fold 평가(비 all-data 시): student val 로짓 저장 → threshold 튜닝에 사용
    if val_ds is not None:
        logits = predict_logits(model_dir, val_ds.samples, max_length=args.max_length,
                                serialize_kwargs=sk)
        preds = logits.argmax(1)
        y = np.array([CLASS_TO_ID[s.action] for s in val_ds.samples])
        payload["macro_f1"] = float(f1_score(y, preds, average="macro"))
        per = f1_score(y, preds, average=None)
        payload["per_class_f1"] = {ID_TO_CLASS[i]: float(per[i]) for i in range(len(per))}
        write_oof(out_dir, val_ds.samples, preds)
        np.save(out_dir / "oof_logits.npy", logits.astype(np.float32))
        (out_dir / "oof_ids.json").write_text(json.dumps([s.id for s in val_ds.samples]))
        print(f"[result] student macro_f1={payload['macro_f1']:.5f}")

    write_metrics(out_dir, payload)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
