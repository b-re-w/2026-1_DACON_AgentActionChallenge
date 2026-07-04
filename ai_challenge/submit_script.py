"""[제출용 추론 코드] AutoModelForSequenceClassification 14-way 분류.

평가 서버가 zip 을 풀고 이 파일을 `script.py` 로 실행한다. 오프라인·저장소 패키지
없음 → **완전 self-contained**. 직렬화는 datasets/serialize.py 와 동일 로직을
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
