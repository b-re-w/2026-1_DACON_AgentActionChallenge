"""학습된 model/ 를 DACON 제출 zip 으로 패키징.

submit.zip 구조 (README 규격):
    model/            <- save_pretrained 산출물 전체
    script.py         <- ai_challenge/submit_script.py (self-contained 추론)
    requirements.txt  <- 비움 (transformers/torch/sentencepiece 는 평가 서버 기본 설치)

1GB 한도를 검증하고 넘으면 에러를 낸다.

사용:
    uv run python -m ai_challenge.pack --model-dir runs/A/model --out submit_A.zip
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

MAX_BYTES = 1024 ** 3  # 1GB
SCRIPT_SRC = Path(__file__).with_name("submit_script.py")
# 토크나이저/설정 등 model.safetensors 외 함께 넣어야 하는 파일들
_TOK_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "spm.model", "vocab.json", "merges.txt",
    "infer_config.json", "generation_config.json",
)


def to_fp16_dir(model_dir: Path, dst: Path) -> None:
    """model_dir 를 fp16 로 다시 저장(가중치 절반 → 1GB 한도 대응)한 뒤 토크나이저 동봉."""
    import torch
    from transformers import AutoModelForSequenceClassification

    dst.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, torch_dtype=torch.float16
    )
    model.save_pretrained(dst)  # config.json + model.safetensors(fp16)
    for name in _TOK_FILES:
        src = model_dir / name
        if src.exists():
            shutil.copy(src, dst / name)
# 평가 서버 기본 설치 패키지(transformers==4.46.3, torch, sentencepiece 등)를
# 중복 명시하면 설치 충돌 위험 → requirements.txt 는 비운다(주석만).
REQUIREMENTS = "# 추가 패키지 없음 (transformers/torch/sentencepiece 는 평가 서버 기본 설치)\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-fp16", action="store_true", help="fp16 변환 없이 원본 그대로 패키징")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    out = Path(args.out)
    if not model_dir.is_dir():
        raise SystemExit(f"model dir 없음: {model_dir}")
    if not SCRIPT_SRC.exists():
        raise SystemExit(f"script 원본 없음: {SCRIPT_SRC}")

    tmp = None
    if not args.no_fp16:
        tmp = Path(tempfile.mkdtemp(prefix="pack_fp16_"))
        print(f"[fp16] {model_dir} → {tmp} 로 재저장(fp16)...")
        to_fp16_dir(model_dir, tmp)
        src_dir = tmp
    else:
        src_dir = model_dir

    # 체크포인트/옵티마이저 등 추론에 불필요한 큰 파일 제외
    skip = {"optimizer.pt", "scheduler.pt", "trainer_state.json", "training_args.bin", "rng_state.pth"}
    files = [p for p in sorted(src_dir.rglob("*")) if p.is_file() and p.name not in skip]

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, arcname=f"model/{p.relative_to(src_dir).as_posix()}")
        z.writestr("script.py", SCRIPT_SRC.read_text(encoding="utf-8"))
        z.writestr("requirements.txt", REQUIREMENTS)
    if tmp is not None:
        shutil.rmtree(tmp, ignore_errors=True)

    size = out.stat().st_size
    print(f"[zip] {out}  ({size / 1e6:.1f} MB)")
    print("[contents]")
    with zipfile.ZipFile(out) as z:
        for info in z.infolist():
            print(f"  {info.file_size:>12,}  {info.filename}")
    if size > MAX_BYTES:
        raise SystemExit(f"❌ 1GB 초과: {size / 1e6:.1f} MB > 1024 MB")
    print(f"✅ 1GB 한도 OK ({size / MAX_BYTES * 100:.1f}%)")


if __name__ == "__main__":
    main()
