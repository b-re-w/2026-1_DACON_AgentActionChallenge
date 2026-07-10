"""학습된 run 의 teacher 로짓(70k)을 중앙 저장소(teacher_logits/)로 내보내기.

distill 을 돌리지 않아도 teacher 가 완성되는 즉시 soft-label 을 확보/보관한다.

사용:
    uv run python -m ai_challenge.method.export_teacher --runs runs/qwen3b_cueshist_f0 [...]
    # preset 은 기본적으로 각 run 의 model/infer_config.json 의 serialize 를 따른다
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ai_challenge.method.distill import teacher_train_logits
from ai_challenge.models.common import load_train_records
from ai_challenge.models.teacher_store import update_entry, make_key


def main() -> None:
    ap = argparse.ArgumentParser(description="teacher 로짓 → teacher_logits/ 저장소")
    ap.add_argument("--runs", nargs="+", required=True, help="run 디렉터리(내부에 model/ 필요)")
    ap.add_argument("--preset", default=None, help="직렬화 프리셋(기본: infer_config.serialize)")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--note", default="", help="TEACHERS.md 비고")
    args = ap.parse_args()

    records = load_train_records()
    for run in args.runs:
        run = Path(run)
        model_dir = run / "model" if (run / "model").is_dir() else run
        preset = args.preset
        if preset is None:
            icfg = model_dir / "infer_config.json"
            preset = (json.loads(icfg.read_text()).get("serialize", "base")
                      if icfg.exists() else "base")
        teacher_train_logits(model_dir, records, preset, max_length=args.max_length)
        if args.note:
            run_name = model_dir.parent.name if model_dir.name == "model" else model_dir.name
            update_entry(make_key(run_name, preset), note=args.note)
    print("[export] done")


if __name__ == "__main__":
    main()
