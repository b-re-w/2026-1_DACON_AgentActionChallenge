"""제출 zip 을 평가 서버처럼 로컬에서 실행·검증 (제출 전 사전 점검).

평가 서버 동작을 모사한다: zip 을 임시폴더에 풀고 → data/ 를 채워 넣고 →
그 안에서 `python script.py` 를 실행 → output/submission.csv 를 검증한다.
제출은 일 10회 한도를 소모하므로, 이 스크립트로 먼저 오프라인 정상동작을 확인한다.

사용:
    uv run python -m ai_challenge.local_eval --zip submit_A.zip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from datasets import ACTION_CLASSES

DATA_DIR = Path("data")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--data", default=str(DATA_DIR))
    ap.add_argument("--workdir", default=".local_eval")
    args = ap.parse_args()

    data = Path(args.data)
    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    # 1) zip 풀기 (model/ script.py requirements.txt)
    with zipfile.ZipFile(args.zip) as z:
        z.extractall(work)

    # 2) 평가 서버가 넣어주는 data/ 구성 (test.jsonl + sample_submission.csv)
    (work / "data").mkdir(exist_ok=True)
    shutil.copy(data / "test.jsonl", work / "data" / "test.jsonl")
    shutil.copy(data / "sample_submission.csv", work / "data" / "sample_submission.csv")

    # 3) script.py 실행 (cwd=work → ./data ./model ./output 상대경로 해석)
    t0 = time.time()
    proc = subprocess.run(
        [sys.executable, "script.py"], cwd=work, capture_output=True, text=True
    )
    dt = time.time() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:\n", proc.stderr[-3000:])
        raise SystemExit(f"❌ script.py 실패 (exit {proc.returncode})")

    # 4) 출력 검증
    out_csv = work / "output" / "submission.csv"
    if not out_csv.exists():
        raise SystemExit("❌ output/submission.csv 미생성")

    import csv

    with out_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_bad = sum(1 for r in rows if r.get("action") not in ACTION_CLASSES)
    preds = [r["action"] for r in rows]
    from collections import Counter

    print(f"✅ rows={len(rows)}  time={dt:.1f}s  invalid_class={n_bad}")
    print("   pred dist:", dict(Counter(preds)))
    if n_bad:
        raise SystemExit(f"❌ 유효하지 않은 클래스 {n_bad}건")


if __name__ == "__main__":
    main()
