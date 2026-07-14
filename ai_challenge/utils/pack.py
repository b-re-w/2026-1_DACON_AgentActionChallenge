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
import json
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
import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- 직렬화 (datasets/serialize.py 와 동일; raw dict 버전) ---
TOK_META, TOK_LAST, TOK_HISTORY, TOK_PROMPT = "[META]", "[LAST_ACTION]", "[HISTORY]", "[PROMPT]"
TOK_CUES = "[CUES]"

# 탐색도구 판별 힌트(serialize.py 와 동일 정규식)
_FILE_RE = re.compile(
    r"[\w./-]*\.(?:py|js|ts|tsx|jsx|go|rs|java|c|cc|cpp|h|hpp|rb|php|cs|kt|swift|scala|sh|"
    r"ya?ml|json|toml|cfg|ini|md|txt|sql|html|css|xml|lock|mod|sum|proto|gradle|env)",
    re.I,
)
_WILDCARD_RE = re.compile(r"[*?]|\ball\b[^.]{0,20}\bfiles?\b|모든[^.]{0,10}파일|전부[^.]{0,10}파일|\*\.\w+", re.I)
_DIR_RE = re.compile(r"[\w./-]+/(?:\s|$)|폴더|디렉터리|디렉토리|structure|구조|안에\s|어떤.{0,6}있|목록|contents of|ls\b|dir\b", re.I)
_SEARCH_RE = re.compile(r"\bgrep\b|\bsearch\b|\bfind\b|찾|검색|어디서|어디에|\bwhere\b|사용.{0,4}곳|참조|references?\b|usages?\b", re.I)
_READ_RE = re.compile(r"\bread\b|\bshow\b|\bopen\b|보여|열어|까보|내용|살펴|어떻게.{0,4}생겼|들여다", re.I)
_LIST_RE = re.compile(r"\blist\b|\bls\b|목록|무엇이.{0,4}있|뭐가.{0,4}있|어떤.{0,4}파일|나열|tree\b", re.I)


def _format_cues(sample):
    prompt = sample.get("current_prompt", "") or ""
    ws = (sample.get("session_meta", {}) or {}).get("workspace", {}) or {}
    file_names = [m.group(0) for m in _FILE_RE.finditer(prompt)]
    open_base = {os.path.basename(str(p)).lower() for p in (ws.get("open_files") or [])}
    ment_in_open = int(any(os.path.basename(f).lower() in open_base for f in file_names))
    parts = [
        f"nfile={len(file_names)}",
        f"fopen={ment_in_open}",
        f"wild={int(bool(_WILDCARD_RE.search(prompt)))}",
        f"dir={int(bool(_DIR_RE.search(prompt)))}",
        f"search={int(bool(_SEARCH_RE.search(prompt)))}",
        f"read={int(bool(_READ_RE.search(prompt)))}",
        f"list={int(bool(_LIST_RE.search(prompt)))}",
    ]
    if file_names:
        parts.append("names=" + ",".join(os.path.basename(f) for f in file_names[:4]))
    return " ".join(parts)


def _truncate(text, limit):
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + "…"


def _last_action(history):
    for turn in reversed(history):
        if turn.get("role") == "assistant_action" and turn.get("name"):
            return turn["name"]
    return None


def _format_meta(sample, open_files_names=0):
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
    if open_files_names and open_files:
        names = [os.path.basename(str(p)) for p in open_files[:open_files_names]]
        parts.append("openf=" + ",".join(names))
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


def serialize(sample, max_history_turns=12, prompt_limit=512, history_text_limit=160,
              include_cues=False, open_files_names=0, include_trail=False, trail_k=5, trail_fail=False, include_nopen=False, include_lastres=False):
    chunks = [f"{TOK_META} {_format_meta(sample, open_files_names=open_files_names)}",
              f"{TOK_LAST} {_last_action(sample.get('history', []) or []) or 'none'}"]
    if include_cues:
        chunks.append(f"{TOK_CUES} {_format_cues(sample)}")
    if include_trail:
        _FAIL = re.compile(r"fail|error|denied|not found|exception|실패", re.I)
        acts = []
        for t in (sample.get("history") or []):
            if t.get("role") == "assistant_action":
                nm = t.get("name", "?")
                if trail_fail and _FAIL.search(str(t.get("result_summary", ""))):
                    nm += "!"
                acts.append(nm)
        chunks.append("[TRAIL] " + (">".join(acts[-trail_k:]) if acts else "none"))
    if include_nopen:
        _ws = (sample.get("session_meta") or {}).get("workspace", {}) or {}
        chunks.append("[NOPEN] " + str(len(_ws.get("open_files") or [])))
    if include_lastres:
        _F = re.compile(r"fail|error|denied|not found|exception|실패", re.I)
        _A = [t for t in (sample.get("history") or []) if t.get("role") == "assistant_action"]
        if _A:
            _rs = str(_A[-1].get("result_summary", ""))
            _b = "fail" if _F.search(_rs) else ("empty" if not _rs.strip() else ("num" if re.search(r"\d", _rs) else "ok"))
        else:
            _b = "none"
        chunks.append("[LASTRES] " + _b)
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
    serialize_mode = "base"
    if os.path.exists(cfg_path):
        _cfg = json.load(open(cfg_path))
        max_length = int(_cfg.get("max_length", 512))
        serialize_mode = _cfg.get("serialize", "base")
    include_cues = serialize_mode in ("cues", "cues_hist")
    include_trail = serialize_mode in ("btrail", "btrail3", "btrail8", "bthist", "btrailf", "btrailn", "btrailr")
    trail_k = {"btrail8": 8, "btrail3": 3}.get(serialize_mode, 5)
    trail_fail = serialize_mode == "btrailf"
    inc_nopen = serialize_mode == "btrailn"
    inc_lastres = serialize_mode == "btrailr"
    hist_turns = 16 if serialize_mode == "bthist" else 12
    hist_lim = 280 if serialize_mode == "bthist" else 160
    open_files_names = 8 if serialize_mode in ("cues", "cues_hist", "paths", "rich") else 0
    batch_size = 64

    print(f"Load model on {device} (max_length={max_length})...")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    # 저장된 config 에 quantization_config 가 있으면 bitsandbytes 4bit 로 자동 로드됨.
    _mcfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    quantized = "quantization_config" in _mcfg
    if quantized:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, device_map=device).eval()
    else:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        model.to(device).eval()
        if device == "cuda":
            model.half()
    id2label = model.config.id2label

    # per-class bias(threshold) — bias.json 있으면 (logits+bias) argmax 로 macro-F1 최적화.
    # 없으면 기존 argmax 그대로(하위호환). tune_threshold.py 산출물.
    bias_t = None
    _bias_path = os.path.join(MODEL_DIR, "bias.json")
    if os.path.exists(_bias_path):
        _b = json.load(open(_bias_path)).get("bias")
        if _b is not None:
            bias_t = torch.tensor(_b, dtype=torch.float32, device=device)
            print(f"[bias] per-class bias 적용 (bias.json)")

    samples = load_jsonl(TEST_PATH)
    ids = [s.get("id", "") for s in samples]
    texts = [serialize(s, prompt_limit=max_length, include_cues=include_cues,
                       open_files_names=open_files_names, include_trail=include_trail, trail_k=trail_k, trail_fail=trail_fail, include_nopen=inc_nopen, include_lastres=inc_lastres,
                       max_history_turns=hist_turns, history_text_limit=hist_lim) for s in samples]
    print(f"samples={len(samples)} serialize={serialize_mode}")

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
            logits = model(**enc).logits
            if bias_t is not None:
                logits = logits + bias_t.to(logits.dtype)
            idx = logits.argmax(-1).tolist()
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


def to_int4_dir(model_dir: Path, dst: Path) -> None:
    """model_dir 를 bitsandbytes nf4(4bit) 로 재저장(1.5B 등 큰 모델 1GB 대응)."""
    import torch
    from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

    dst.mkdir(parents=True, exist_ok=True)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, quantization_config=bnb, device_map="cuda")
    model.save_pretrained(dst)
    for name in _TOK_FILES:
        src = model_dir / name
        if src.exists():
            shutil.copy(src, dst / name)


def pack(model_dir, out=None, name=None, fp16: bool = True, serialize_mode="base",
         quant="fp16") -> Path:
    """학습된 model/ 를 제출 zip 으로 패키징한다. out 미지정 시 build/<name>.zip.

    serialize_mode: 추론 직렬화 프리셋(학습과 일치해야 함).
    quant: fp16(기본) | int4(bitsandbytes nf4, 큰 모델용 → requirements 에 bitsandbytes).
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise SystemExit(f"model dir 없음: {model_dir}")
    if out is None:
        stem = name or model_dir.parent.name  # 예: runs/B/model → "B"
        out = BUILD_DIR / f"{stem}.zip"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tmp = None
    requirements = REQUIREMENTS
    if quant == "int4":
        tmp = Path(tempfile.mkdtemp(prefix="pack_int4_"))
        print(f"[int4] {model_dir} → {tmp} nf4 재저장...")
        to_int4_dir(model_dir, tmp)
        src_dir = tmp
        requirements = "bitsandbytes\n"  # 4bit 추론에 필요
        fp16 = False
    elif fp16:
        tmp = Path(tempfile.mkdtemp(prefix="pack_fp16_"))
        print(f"[fp16] {model_dir} → {tmp} 재저장(fp16)...")
        to_fp16_dir(model_dir, tmp)
        src_dir = tmp
    else:
        src_dir = model_dir

    # infer_config 에 serialize 모드 주입(추론이 학습과 같은 직렬화를 쓰도록).
    # tmp(우리 소유)면 거기에 쓰고, 원본 model_dir 이면 fresh tmp 를 만들어 훼손을 피한다.
    if tmp is None:
        tmp = Path(tempfile.mkdtemp(prefix="pack_cfg_"))
        for p in src_dir.iterdir():
            if p.is_file():
                shutil.copy(p, tmp / p.name)
        src_dir = tmp
    icfg_path = src_dir / "infer_config.json"
    icfg = json.loads(icfg_path.read_text()) if icfg_path.exists() else {}
    icfg["serialize"] = serialize_mode
    icfg_path.write_text(json.dumps(icfg))

    # bias.json 은 fp16/int4 재저장(save_pretrained) 시 복사되지 않으므로 명시 복사.
    # (script.py 가 model/bias.json 을 읽어 (logits+bias).argmax — 56k best-ckpt+bias70k 표준)
    bias_src = model_dir / "bias.json"
    if bias_src.exists() and not (src_dir / "bias.json").exists():
        shutil.copy(bias_src, src_dir / "bias.json")
        print("[bias] bias.json 포함")

    skip = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "rng_state.pth"}
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.name not in skip]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"model/{p.relative_to(src_dir).as_posix()}")
        z.writestr("script.py", SCRIPT_PY)
        z.writestr("requirements.txt", requirements)
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
    pp.add_argument("--serialize", default="base",
                    help="추론 직렬화 프리셋(학습과 일치). cues 등")
    pp.add_argument("--quant", default="fp16", choices=["fp16", "int4"],
                    help="int4 = bitsandbytes nf4(큰 모델 1GB 대응)")

    pe = sub.add_parser("eval", help="제출 전 로컬 검증(평가 서버 모사)")
    pe.add_argument("--zip", required=True)
    pe.add_argument("--data", default=str(DATA_DIR))

    args = ap.parse_args()
    if args.cmd == "pack":
        pack(args.model_dir, out=args.out, name=args.name, fp16=not args.no_fp16,
             serialize_mode=args.serialize, quant=args.quant)
    elif args.cmd == "eval":
        local_eval(args.zip, data=args.data)


if __name__ == "__main__":
    main()
