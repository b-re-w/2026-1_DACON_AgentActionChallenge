"""듀얼헤드 KD — 공유 백본 + 탐색4 전용 4-way 전문가 헤드.

문제: 탐색4(read/grep/list/glob, id 0~3)가 서로만 잡아먹는 상호혼동이 macro-F1 병목.
아이디어: 14-way 주 헤드는 그대로 KD 학습하되, 같은 백본 풀링 위에 4-way 전문가
헤드를 추가로 달아 탐색4 정답 샘플에서만 CE 로 공동 학습한다. 추론에서 주 헤드가
"탐색4 중 하나"라고 판단하면 넷 중 택일은 전문가 헤드가 맡는다.
14-way 에 희석되지 않은 경계 학습 → 4-way 판별 정밀화가 가설. 파라미터 +3.6k
(hidden×4)라 제출 용량·시간 무영향.

사용 (fold0 검증):
    uv run python -m ai_challenge.method.distill_dual \
        --teachers team_qgptoss team_qw6 qwen7b_base_f0__base --weights 1 1 1 \
        --fold 0 --epochs 2.86 --name kdd_f0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai_challenge.datasets import CLASS_TO_ID, ID_TO_CLASS, SERIALIZE_PRESETS, ActionDataset
from ai_challenge.method.distill import (
    KDCollator,
    KDDataset,
    KDTrainer,
    blend_teachers,
    teacher_train_logits,
)
from ai_challenge.models.common import (
    build_model,
    build_tokenizer,
    build_training_args,
    compute_metrics,
    get_folds,
    load_train_records,
    save_submission_model,
    write_metrics,
)

N_EXPL = 4  # 탐색4 = 클래스 id 0..3 (schema canonical 순서)


def pool_last_token(hidden, attention_mask):
    """Qwen2ForSequenceClassification 과 동일한 마지막 유효토큰 풀링."""
    seq_len = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.size(0), device=hidden.device), seq_len]


class DualKDTrainer(KDTrainer):
    """KD(주 헤드) + 탐색4 CE(전문가 헤드). expl_head 는 model 에 부착돼 있어야 함."""

    def __init__(self, *args, expl_beta: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.expl_beta = expl_beta

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        soft = inputs.pop("soft_targets", None)
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits.float()
        ce = F.cross_entropy(logits, labels)
        if soft is None:
            loss = ce
        else:
            log_p = F.log_softmax(logits / self.kd_T, dim=-1)
            kl = F.kl_div(log_p, soft.to(log_p.device), reduction="batchmean")
            loss = self.kd_alpha * ce + (1 - self.kd_alpha) * (self.kd_T ** 2) * kl
        # 전문가 헤드: 탐색4 정답 샘플에서만 4-way CE
        mask = labels < N_EXPL
        if mask.any():
            pooled = pool_last_token(outputs.hidden_states[-1], inputs["attention_mask"])
            expl_logits = model.expl_head(pooled.float())
            loss = loss + self.expl_beta * F.cross_entropy(expl_logits[mask], labels[mask])
        return (loss, outputs) if return_outputs else loss


@torch.inference_mode()
def predict_dual(model, tok, samples, max_length, sk, batch_size=96):
    """(주 14-way logits, 전문가 4-way logits) 동시 추론."""
    from ai_challenge.datasets import serialize_sample
    texts = [serialize_sample(s, **sk) for s in samples]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out14 = np.zeros((len(texts), 14), dtype=np.float32)
    out4 = np.zeros((len(texts), N_EXPL), dtype=np.float32)
    model.eval()
    for st in range(0, len(order), batch_size):
        chunk = order[st:st + batch_size]
        enc = tok([texts[i] for i in chunk], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(model.device)
        o = model(**enc, output_hidden_states=True)
        pooled = pool_last_token(o.hidden_states[-1], enc["attention_mask"])
        e = model.expl_head(pooled.float())
        for pos_, i in enumerate(chunk):
            out14[i] = o.logits[pos_].float().cpu().numpy()
            out4[i] = e[pos_].float().cpu().numpy()
    return out14, out4


def main() -> None:
    ap = argparse.ArgumentParser(description="듀얼헤드 KD (탐색4 전문가)")
    ap.add_argument("--teachers", nargs="+", required=True)
    ap.add_argument("--weights", nargs="+", type=float, required=True)
    ap.add_argument("--student", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--preset", default="base")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--all-data", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.86)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-length", type=int, default=640)
    ap.add_argument("--kd-alpha", type=float, default=0.25)
    ap.add_argument("--kd-T", type=float, default=3.0)
    ap.add_argument("--expl-beta", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    out_dir = Path("runs") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    sk = SERIALIZE_PRESETS[args.preset]
    records = load_train_records()

    logit_list = [teacher_train_logits(t, records, "base", args.max_length)
                  for t in args.teachers]
    soft = blend_teachers(logit_list, args.weights, args.kd_T)

    tok = build_tokenizer(args.student)
    model = build_model(args.student, tok)
    model.expl_head = nn.Linear(model.config.hidden_size, N_EXPL)

    fold = None if args.all_data else get_folds(records)
    if args.all_data:
        train_samples, val_samples, train_soft = records, None, soft
    else:
        from ai_challenge.datasets import train_val_indices
        tr_idx, va_idx = train_val_indices(fold, args.fold)
        train_samples = [records[i] for i in tr_idx]
        val_samples = [records[i] for i in va_idx]
        train_soft = soft[tr_idx]

    train_ds = KDDataset(train_samples, tok, train_soft, max_length=args.max_length,
                         serialize_kwargs=sk)
    targs = build_training_args(
        out_dir, epochs=args.epochs, batch_size=args.bs, lr=args.lr,
        all_data=args.all_data, seed=args.seed,
        remove_unused_columns=False, inloop_eval=False,
    )
    trainer = DualKDTrainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=KDCollator(tok),
        kd_alpha=args.kd_alpha, kd_T=args.kd_T, expl_beta=args.expl_beta,
    )
    trainer.train()

    model_dir = save_submission_model(trainer, tok, out_dir, args.max_length)
    icfg = json.loads((model_dir / "infer_config.json").read_text())
    icfg["serialize"] = args.preset
    icfg["dual_head"] = True
    (model_dir / "infer_config.json").write_text(json.dumps(icfg))
    torch.save(model.expl_head.state_dict(), model_dir / "expl_head.pt")

    payload = {"name": args.name, "teachers": args.teachers, "weights": args.weights,
               "expl_beta": args.expl_beta, "epochs": args.epochs, "seed": args.seed,
               "fold": "all" if args.all_data else args.fold}

    if val_samples is not None:
        from sklearn.metrics import f1_score
        model.cuda().half(); model.expl_head.float().cuda()
        l14, l4 = predict_dual(model, tok, val_samples, args.max_length, sk)
        y = np.array([CLASS_TO_ID[s.action] for s in val_samples])
        p_main = l14.argmax(1)
        p_dual = p_main.copy()
        m = p_main < N_EXPL
        p_dual[m] = l4[m].argmax(1)
        f_main = f1_score(y, p_main, average="macro")
        f_dual = f1_score(y, p_dual, average="macro")
        # 탐색4 진단: 정답이 탐색4인 샘플에서 4-way 정확도
        em = y < N_EXPL
        acc_main = float((p_main[em] == y[em]).mean())
        acc_dual = float((p_dual[em] == y[em]).mean())
        payload.update({
            "macro_f1_main": float(f_main), "macro_f1_dual": float(f_dual),
            "expl_acc_main": acc_main, "expl_acc_dual": acc_dual,
        })
        np.save(out_dir / "oof_logits.npy", l14)
        np.save(out_dir / "oof_expl_logits.npy", l4)
        (out_dir / "oof_ids.json").write_text(json.dumps([s.id for s in val_samples]))
        print(f"[result] main={f_main:.5f} dual={f_dual:.5f} "
              f"| 탐색4 acc: main={acc_main:.4f} dual={acc_dual:.4f}")

    write_metrics(out_dir, payload)
    print(f"[done] {out_dir}")


if __name__ == "__main__":
    main()
