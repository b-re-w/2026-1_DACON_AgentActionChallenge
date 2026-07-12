"""프론티어 API teacher 파일럿 — fold0 표본에 단일글자+logprobs 로 14-way 분포 추출.

사용:
    OPENAI_API_KEY=sk-... PYTHONPATH=. uv run python scripts/api_teacher_pilot.py \
        --model gpt-5 --n 2000 [--out runs/api_pilot_gpt5]

출력: <out>/logits.npy (n,14 log-prob), ids.json, score.json (argmax·온도보정 macro)
게이트: 보정 후 fold0 macro ≥ 0.78 이면 70k 확장 가치.
"""
import argparse
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ai_challenge.datasets import ACTION_CLASSES, CLASS_TO_ID, load_folds, load_records, serialize_sample

LETTERS = "ABCDEFGHIJKLMN"
L2ID = {c: i for i, c in enumerate(LETTERS)}
ACTION_MENU = "\n".join(f"{LETTERS[i]}) {a}" for i, a in enumerate(ACTION_CLASSES))
SYSTEM = (
    "You predict the NEXT action of an AI coding agent given its session state.\n"
    "The 14 possible actions:\n" + ACTION_MENU + "\n"
    "Rules: read_file=open a specific file, grep_search=search text pattern, "
    "list_directory=list folder contents, glob_pattern=find files by name pattern, "
    "edit_file=modify existing file, write_file=create new file, apply_patch=apply a diff, "
    "run_bash=shell command, run_tests=execute tests, lint_or_typecheck=linter/type check, "
    "ask_user=ask a clarifying question, plan_task=write a plan, web_search=search the web, "
    "respond_only=answer without tools.\n"
    "Answer with ONLY the single letter of the most likely next action."
)


def call_one(api_key: str, model: str, text: str, retries: int = 4):
    body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": text}],
        "max_completion_tokens": 4,
        "logprobs": True,
        "top_logprobs": 20,
        "temperature": 0,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                out = json.loads(r.read())
            content = out["choices"][0]["logprobs"]["content"]
            # 첫 글자 토큰의 top_logprobs 에서 14글자 질량 수집
            vec = np.full(14, -20.0)
            for tok in content[:2]:  # 첫 토큰(공백 변형 대비 두 개까지)
                for cand in tok.get("top_logprobs", []):
                    t = cand["token"].strip().upper()
                    if t in L2ID:
                        vec[L2ID[t]] = max(vec[L2ID[t]], cand["logprob"])
                if vec.max() > -20:
                    break
            return vec
        except Exception as e:
            if attempt == retries - 1:
                return np.full(14, np.nan)
            time.sleep(2 ** attempt + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    api_key = os.environ["OPENAI_API_KEY"]
    out = Path(args.out or f"runs/api_pilot_{args.model.replace('/','_').replace('.','')}")
    out.mkdir(parents=True, exist_ok=True)

    recs = load_records("data/train.jsonl", "data/train_labels.csv")
    fold = load_folds("data/folds.csv", recs)
    val = [r for r, f in zip(recs, fold) if f == args.fold][: args.n]
    texts = [serialize_sample(r) for r in val]
    y = np.array([CLASS_TO_ID[r.action] for r in val])

    logits = np.full((len(val), 14), np.nan)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(call_one, api_key, args.model, t): i for i, t in enumerate(texts)}
        done = 0
        for fu in as_completed(futs):
            logits[futs[fu]] = fu.result()
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(val)}  ({time.time()-t0:.0f}s)", flush=True)

    ok = ~np.isnan(logits).any(1)
    print(f"성공 {ok.sum()}/{len(val)}")
    from sklearn.metrics import f1_score
    m_arg = f1_score(y[ok], logits[ok].argmax(1), average="macro")
    # 온도 보정(간단 스윕) — logprob/T 후 argmax 는 동일하므로 클래스 오프셋만 스윕
    from ai_challenge.method.tune_threshold import coordinate_ascent, softmax
    half = ok.nonzero()[0][: ok.sum() // 2]
    rest = ok.nonzero()[0][ok.sum() // 2:]
    bias, _ = coordinate_ascent(y[half], softmax(logits[half]))
    m_cal = f1_score(y[rest], (softmax(logits[rest]) + bias).argmax(1), average="macro")
    print(f"[{args.model}] fold0 표본 {len(val)}: argmax macro={m_arg:.4f} | 반분보정 macro={m_cal:.4f}")
    np.save(out / "logits.npy", logits)
    (out / "ids.json").write_text(json.dumps([r.id for r in val]))
    (out / "score.json").write_text(json.dumps(
        {"model": args.model, "n": len(val), "ok": int(ok.sum()),
         "macro_argmax": float(m_arg), "macro_calibrated_half": float(m_cal)}))


if __name__ == "__main__":
    main()
