"""제출 도구 통합 — 추론 스크립트(script.py) + zip 패키징 + 로컬 검증.

- ``SCRIPT_PY`` : 제출 zip 에 ``script.py`` 로 들어가는 self-contained 추론 코드
  (오프라인 T4, 저장소 패키지 의존 없음). 직렬화는 ai_challenge/datasets/serialize.py 와 동일.
- ``pack()``   : 학습된 model/ → ``build/<name>.zip`` (fp16 변환, 1GB 한도 검증).
- ``local_eval()`` : 평가 서버를 모사해 zip 을 실제 실행·검증(제출 전 점검, 일 10회 한도 절약).

CLI:
    uv run python -m ai_challenge.utils.pack pack --model-dir runs/B/model --name submit_B
    uv run python -m ai_challenge.utils.pack eval --zip build/submit_B.zip
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter
from pathlib import Path

from ai_challenge.datasets import ACTION_CLASSES

MAX_BYTES = 1024 ** 3  # 1GB
BUILD_DIR = Path("build")
DATA_DIR = Path("data")
# 평가 서버 기본 설치 패키지(transformers==4.46.3, torch, sentencepiece 등)를 중복 명시하면
# 설치 충돌 위험 → requirements.txt 는 비운다(주석만).
REQUIREMENTS = "# 추가 패키지 없음 (transformers/torch/sentencepiece 는 평가 서버 기본 설치)\n"
# 토크나이저/설정 등 model.safetensors 외 함께 넣어야 하는 파일들
_TOK_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "spm.model", "vocab.json", "merges.txt",
    "infer_config.json", "generation_config.json",
)

# 제출 zip 에 script.py 로 기록되는 추론 코드 (self-contained).
SCRIPT_PY = r'''
"""[제출용 추론 코드] AutoModelForSequenceClassification 14-way 분류.

평가 서버가 zip 을 풀고 이 파일을 `script.py` 로 실행한다. 오프라인·저장소 패키지
없음 → **완전 self-contained**. 직렬화는 ai_challenge/datasets/serialize.py 와 동일 로직을
raw dict 기준으로 재현한다(학습/추론 표현 일치가 정확도에 필수).

I/O 계약:
  입력  ./data/test.jsonl, ./data/sample_submission.csv  (읽기전용)
  모델  ./model/ (save_pretrained 산출물, id2label 포함)
  출력  ./output/submission.csv  (sample_submission 의 id 순서/컬럼)
"""

import csv
import json
import os

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 직렬화 (datasets/serialize.py 와 동일; raw dict 버전) ---
TOK_META, TOK_LAST, TOK_HISTORY, TOK_PROMPT = "[META]", "[LAST_ACTION]", "[HISTORY]", "[PROMPT]"


def _truncate(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _last_action(history):
    for turn in reversed(history):
        if turn.get("role") == "assistant_action" and turn.get("name"):
            return turn["name"]
    return None


def _format_meta(sample):
    sm = sample.get("session_meta", {}) or {}
    ws = sm.get("workspace", {}) or {}
    open_files = ws.get("open_files") or []
    parts = [
        f"tier={sm.get('user_tier', 'na')}",
        f"lang={sm.get('language_pref', 'na')}",
        f"turn={sm.get('turn_index', 0)}",
        f"budget={sm.get('budget_tokens_remaining', 0)}",
        f"ci={ws.get('last_ci_status', 'none')}",
        f"git={'dirty' if ws.get('git_dirty') else 'clean'}",
        f"open_files={len(open_files)}",
        f"loc={ws.get('loc', 0)}",
    ]
    lang_mix = ws.get("language_mix") or {}
    if lang_mix:
        top = sorted(lang_mix.items(), key=lambda kv: -kv[1])[:3]
        parts.append("codemix=" + ",".join(f"{k}:{v:.2f}" for k, v in top))
    return " ".join(parts)


def _format_history_turn(turn, text_limit):
    if turn.get("role") == "user":
        return "U: " + _truncate(turn.get("content", ""), text_limit)
    name = turn.get("name", "?")
    args = turn.get("args") or {}
    arg_str = ",".join(f"{k}={_truncate(v, 40)}" for k, v in list(args.items())[:4])
    return f"A: {name}({arg_str}) -> {_truncate(turn.get('result_summary', ''), text_limit)}"


def serialize(sample, max_history_turns=12, prompt_limit=512, history_text_limit=160):
    chunks = [f"{TOK_META} {_format_meta(sample)}",
              f"{TOK_LAST} {_last_action(sample.get('history', []) or []) or 'none'}"]
    history = sample.get("history", []) or []
    if history:
        recent = history[-max_history_turns:]
        lines = [_format_history_turn(t, history_text_limit) for t in recent]
        chunks.append(TOK_HISTORY + " " + " ".join(lines))
    chunks.append(f"{TOK_PROMPT} {_truncate(sample.get('current_prompt', ''), prompt_limit)}")
    return " ".join(chunks)


# --- I/O ---
def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main():
    TEST_PATH = "./data/test.jsonl"
    SAMPLE_SUB = "./data/sample_submission.csv"
    MODEL_DIR = "./model"
    OUT_PATH = "./output/submission.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_path = os.path.join(MODEL_DIR, "infer_config.json")
    max_length = 512
    if os.path.exists(cfg_path):
        max_length = int(json.load(open(cfg_path)).get("max_length", 512))
    batch_size = 64

    print(f"Load model on {device} (max_length={max_length})...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device).eval()
    if device == "cuda":
        model.half()
    id2label = model.config.id2label

    samples = load_jsonl(TEST_PATH)
    ids = [s.get("id", "") for s in samples]
    texts = [serialize(s, prompt_limit=max_length) for s in samples]
    print(f"samples={len(samples)}")

    # 길이 정렬 배칭: 비슷한 길이끼리 묶어 동적 패딩 낭비를 줄인다(정확도 무영향, 속도 ↑).
    # 예측은 원래 순서(preds_by_idx)에 되돌려 넣는다.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    preds_by_idx = [None] * len(texts)
    with torch.inference_mode():
        for s in range(0, len(order), batch_size):
            chunk = order[s:s + batch_size]
            batch = [texts[i] for i in chunk]
            enc = tok(batch, truncation=True, max_length=max_length,
                      padding=True, return_tensors="pt").to(device)
            idx = model(**enc).logits.argmax(-1).tolist()
            for pos, i in enumerate(chunk):
                preds_by_idx[i] = id2label[idx[pos]]
    pred_map = dict(zip(ids, preds_by_idx))

    with open(SAMPLE_SUB, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    for row in rows:
        p = pred_map.get(row["id"])
        if p is not None:
            row["action"] = p

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {OUT_PATH} (rows={len(rows)})")


if __name__ == "__main__":
    main()
'''


# --------------------------------------------------------------------------- #
# 패키징
# --------------------------------------------------------------------------- #
def to_fp16_dir(model_dir: Path, dst: Path) -> None:
    """model_dir 를 fp16 로 재저장(가중치 절반 → 1GB 한도 대응) + 토크나이저 동봉."""
    import torch
    from transformers import AutoModelForSequenceClassification

    dst.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.float16
    )
    model.save_pretrained(dst)
    for name in _TOK_FILES:
        src = model_dir / name
        if src.exists():
            shutil.copy(src, dst / name)


def pack(model_dir, out=None, name=None, fp16: bool = True) -> Path:
    """학습된 model/ 를 제출 zip 으로 패키징한다. out 미지정 시 build/<name>.zip."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"model dir 없음: {model_dir}")
    if out is None:
        stem = name or model_dir.parent.name  # 예: runs/B/model → "B"
        out = BUILD_DIR / f"{stem}.zip"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp = None
    if fp16:
        tmp = Path(tempfile.mkdtemp(prefix="pack_fp16_"))
        print(f"[fp16] {model_dir} → {tmp} 재저장(fp16)...")
        to_fp16_dir(model_dir, tmp)
        src_dir = tmp
    else:
        src_dir = model_dir

    skip = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "rng_state.pth"}
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.name not in skip]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"model/{p.relative_to(src_dir).as_posix()}")
        z.writestr("script.py", SCRIPT_PY)
        z.writestr("requirements.txt", REQUIREMENTS)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)

    size = out.stat().st_size
    print(f"[zip] {out}  ({size / 1e6:.1f} MB)")
    if size > MAX_BYTES:
        raise SystemExit(f"❌ 1GB 초과: {size / 1e6:.1f} MB")
    print(f"✅ 1GB 한도 OK ({size / MAX_BYTES * 100:.1f}%)")
    return out


# --------------------------------------------------------------------------- #
# 로컬 검증 (평가 서버 모사)
# --------------------------------------------------------------------------- #
def local_eval(zip_path, data=DATA_DIR, workdir=None) -> None:
    """zip 을 임시폴더에 풀고 data/ 를 채운 뒤 script.py 를 실행·검증한다."""
    data = Path(data)
    work = Path(workdir) if workdir else BUILD_DIR / "local_eval"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)

    with zipfile.ZipFile(zip_path) as z:
        z.extractall(work)
    (work / "data").mkdir(exist_ok=True)
    shutil.copy(data / "test.jsonl", work / "data" / "test.jsonl")
    shutil.copy(data / "sample_submission.csv", work / "data" / "sample_submission.csv")

    import time

    t0 = time.time()
    proc = subprocess.run([sys.executable, "script.py"], cwd=work, capture_output=True, text=True)
    dt = time.time() - t0
    print(proc.stdout)
    if proc.returncode != 0:
        print("STDERR:\n", proc.stderr[-3000:])
        raise SystemExit(f"❌ script.py 실패 (exit {proc.returncode})")

    out_csv = work / "output" / "submission.csv"
    if not out_csv.exists():
        raise SystemExit("❌ output/submission.csv 미생성")
    with out_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    n_bad = sum(1 for r in rows if r.get("action") not in ACTION_CLASSES)
    print(f"✅ rows={len(rows)}  time={dt:.1f}s  invalid_class={n_bad}")
    print("   pred dist:", dict(Counter(r["action"] for r in rows)))
    if n_bad:
        raise SystemExit(f"❌ 유효하지 않은 클래스 {n_bad}건")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="제출 zip 패키징/검증")
    sub = ap.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("pack", help="model/ → build/<name>.zip (fp16)")
    pp.add_argument("--model-dir", required=True)
    pp.add_argument("--name", default=None, help="출력 이름(build/<name>.zip). 미지정 시 runs 이름")
    pp.add_argument("--out", default=None, help="출력 경로 직접 지정")
    pp.add_argument("--no-fp16", action="store_true")

    pe = sub.add_parser("eval", help="제출 전 로컬 검증(평가 서버 모사)")
    pe.add_argument("--zip", required=True)
    pe.add_argument("--data", default=str(DATA_DIR))

    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.model_dir, out=args.out, name=args.name, fp16=not args.no_fp16)
    elif args.cmd == "eval":
        local_eval(args.zip, data=args.data)


if __name__ == "__main__":
    main()
