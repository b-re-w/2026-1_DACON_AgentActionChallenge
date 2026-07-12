"""Teacher 앙상블 조합 초고속 스크리닝 (GPU 불필요).

캐시된 teacher soft-label(.npz)을 가중평균 → OOF(fold 0) argmax macro-F1 을 즉시 계산.
수백 조합/가중치를 초 단위로 훑어 순위화한다. 이는 **teacher 앙상블 자체 품질**의 프록시로,
student KD 결과와 완전 일치하진 않지만 유망 후보를 값싸게 걸러내는 용도(그 뒤 상위만 실제 KD).

출력: docs/exp_logs/others/ensemble_screen.csv (combo, oof_macro_f1, oof_acc, n_teachers)
"""
from __future__ import annotations

import csv
import itertools
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from ai_challenge.datasets import CLASS_TO_ID
from ai_challenge.models.common import (
    TEACHER_LOGITS_DIR,
    gather_teacher_logits,
    get_folds,
    load_train_records,
)


def parse_spec(spec: str):
    """'qw6:5,qdebertaL:1' → (['qw6','qdebertaL'], [5.0,1.0])."""
    tags, wts = [], []
    for part in spec.split(","):
        if ":" in part:
            t, w = part.split(":")
            tags.append(t); wts.append(float(w))
        else:
            tags.append(part); wts.append(1.0)
    return tags, wts


def available_tags() -> set[str]:
    return {p.stem for p in TEACHER_LOGITS_DIR.glob("*.npz")}


def build_combos(avail: set[str]) -> list[str]:
    """스크리닝할 조합 스펙(콤마구분 tag[:weight]) 목록 생성."""
    combos: list[str] = []

    def add(spec):
        tags, _ = parse_spec(spec)
        if all(t in avail for t in tags):
            combos.append(spec)

    # 풀 정의
    seeds = ["q3bb", "q3b43", "q3b44", "q3b45"]            # 3B full-FT 시드
    ladder = ["q3b", "q14b", "q32b"]                        # QLoRA 크기사다리
    diversity = ["qgptoss", "qgpt5"]                        # Qwen3-4B distill
    nonqwen = ["qmistral24b", "qdebertaL", "qmodernbertL"]  # 진짜 이질(Mistral/encoder)
    encoders = ["qdebertaL", "qmodernbertL"]
    best = "qw6"

    # 1) 단일 teacher 전부
    for t in sorted(avail):
        add(t)

    # 2) qw6 + 단일 X (가중 그리드) — 비-Qwen/다양성 up
    for x in nonqwen + diversity + ["q32b"] + seeds:
        for w in (3, 5, 8):
            add(f"{best}:{w},{x}:1")

    # 3) qw6 + 인코더 가중 세밀 (핵심 가설: encoder 다양성)
    for d in encoders:
        for dw in (1, 2, 3):
            for w in (4, 6):
                add(f"{best}:{w},{d}:{dw}")

    # 4) qw6 + 두 개 비-Qwen 조합
    for a, b in itertools.combinations(nonqwen, 2):
        for w in (4, 6):
            add(f"{best}:{w},{a}:1,{b}:1")

    # 5) qw6 + 인코더 둘 다
    for w in (4, 6, 8):
        add(f"{best}:{w},qdebertaL:1,qmodernbertL:1")

    # 6) qw6 + 다양성(gptoss/gpt5) + 인코더
    add(f"{best}:6,qgptoss:1,qdebertaL:1")
    add(f"{best}:6,qgpt5:1,qdebertaL:1")
    add(f"{best}:8,qgptoss:1,qgpt5:1,qdebertaL:1,qmistral24b:1")

    # 7) 비-Qwen only (순수 이질 앙상블)
    add("qmistral24b:1,qdebertaL:1")
    add("qmistral24b:1,qdebertaL:1,qmodernbertL:1")
    add("qgptoss:1,qgpt5:1,qmistral24b:1,qdebertaL:1")

    # 8) 다양성 코어(qw6 없이) 여러개
    add("q3bb:1,qgptoss:1,qgpt5:1,qmistral24b:1")
    add("q3bb:1,q3b43:1,q3b44:1,q3b45:1,qgptoss:2")            # w6 근사 재현
    add("q3bb:1,q3b43:1,q3b44:1,q3b45:1,qgptoss:2,qdebertaL:1")

    # 9) big-tent (전체 균등 / 다양성만)
    add(",".join(sorted(avail)))
    add(",".join(seeds + diversity + nonqwen))

    # 중복 제거(순서 유지)
    seen, uniq = set(), []
    for c in combos:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def main() -> None:
    records = load_train_records()
    fold = get_folds(records)
    val_idx = np.where(fold == 0)[0]
    val_samples = [records[i] for i in val_idx]
    y_true = np.array([CLASS_TO_ID[val_samples[i].action] for i in range(len(val_samples))])
    avail = available_tags()
    print(f"[screen] val(fold0)={len(val_samples)}  available teachers={sorted(avail)}", flush=True)

    combos = build_combos(avail)
    print(f"[screen] 조합 {len(combos)}개 평가 시작", flush=True)

    rows = []
    for i, spec in enumerate(combos):
        tags, wts = parse_spec(spec)
        logits = gather_teacher_logits(tags, val_samples, weights=wts)
        pred = logits.argmax(1)
        macro = f1_score(y_true, pred, average="macro")
        acc = accuracy_score(y_true, pred)
        rows.append((spec, macro, acc, len(tags)))
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(combos)}", flush=True)

    rows.sort(key=lambda r: r[1], reverse=True)
    out = Path("docs/exp_logs/others/ensemble_screen.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["combo", "oof_macro_f1", "oof_acc", "n_teachers"])
        for spec, macro, acc, n in rows:
            w.writerow([spec, f"{macro:.4f}", f"{acc:.4f}", n])

    print("\n=== TOP 30 (OOF fold0 argmax macro-F1) ===", flush=True)
    for spec, macro, acc, n in rows[:30]:
        print(f"  {macro:.4f}  acc={acc:.4f}  n={n}  {spec}", flush=True)
    print(f"\n[done] 전체 {len(rows)}개 → {out}", flush=True)


if __name__ == "__main__":
    main()
