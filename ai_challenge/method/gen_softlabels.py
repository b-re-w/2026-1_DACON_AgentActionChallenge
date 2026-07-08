"""Teacher soft-label 사전 생성·캐시 — 증류 실험을 재추론 없이 반복하기 위한 store 생성.

teacher 1개당 **전체 train 레코드**의 14-class raw logits 를 1회 계산해
runs/_teacher_logits/{tag}.npz 에 저장한다. 이후 distill 은 --teacher-logits {tag ...}
로 임의의 teacher 부분집합을 즉시 평균(앙상블)해 사용 → teacher forward 0회.

설계 포인트:
  - logits 는 softmax 전(raw) 저장 → distill 시점의 temperature/alpha 스윕이 무료.
  - fold 무관(전체 레코드) 저장 → 어떤 val_fold/all_data 실험이든 같은 store 재사용.
  - sample.id 인덱스 동봉 → distill 이 id 로 조회·정렬(레코드 순서 변화에 안전).
  - serialize 프리셋을 태그·메타에 남긴다(증류시 teacher-serialize 와 일치가 중요).

사용:
    uv run python -m ai_challenge.method.gen_softlabels \
        --teacher-dir runs/t3b_base/model --tag q25_3b_base --serialize base
그 뒤 증류(재추론 없음):
    uv run python -m ai_challenge.method.distill \
        --teacher-logits q25_3b_base --student Qwen/Qwen2.5-0.5B --out runs/kd --bf16
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import f1_score

from ai_challenge.datasets import CLASS_TO_ID, SERIALIZE_PRESETS, load_records
from ai_challenge.models.common import (
    TRAIN_JSONL,
    TRAIN_LABELS,
    predict_logits,
    save_teacher_logits,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-dir", required=True, help="teacher model 디렉터리(1개)")
    ap.add_argument("--tag", required=True,
                    help="store 이름 → runs/_teacher_logits/{tag}.npz")
    ap.add_argument("--serialize", default="base", choices=list(SERIALIZE_PRESETS),
                    help="teacher 입력 직렬화 프리셋 (증류시 --teacher-serialize 와 일치해야 함)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA 대형 teacher 어댑터를 4bit base 위에서 추론(merge 불필요)")
    args = ap.parse_args()

    sk = SERIALIZE_PRESETS[args.serialize]
    records = load_records(TRAIN_JSONL, TRAIN_LABELS)
    print(f"[gen] teacher={args.teacher_dir} tag={args.tag} "
          f"serialize={args.serialize} 4bit={args.load_in_4bit} N={len(records)}", flush=True)

    logits = predict_logits(args.teacher_dir, records, max_length=args.max_length,
                            batch_size=args.batch_size, serialize_kwargs=sk,
                            load_in_4bit=args.load_in_4bit)

    path = save_teacher_logits(
        args.tag, [s.id for s in records], logits,
        meta={"teacher_dir": args.teacher_dir, "serialize": args.serialize,
              "max_length": args.max_length},
    )

    # 참고용: 저장한 soft-label 자체 argmax 의 전체-train macro-f1 (라벨 존재 시)
    if records and records[0].action is not None:
        y = np.array([CLASS_TO_ID[s.action] for s in records])
        print(f"[gen] train argmax macro_f1={f1_score(y, logits.argmax(1), average='macro'):.4f}",
              flush=True)
    print(f"[done] 저장 {path} (shape={logits.shape})", flush=True)


if __name__ == "__main__":
    main()
