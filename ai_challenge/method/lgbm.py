"""구조 피처 기반 LightGBM (접근 3 — tabular). 신경망과 이질적이라 앙상블 다양성이 큼.

데이터가 시뮬레이터 산출물이라 규칙 기반 신호(직전 action 전이, open_files=empty,
ci_status, git_dirty, current_prompt 키워드)가 강하다. 이를 명시적 피처로 뽑아
LightGBM(14-class)을 학습한다. OOF 확률을 저장해 KD student 와 앙상블한다.

사용:
    uv run python -m ai_challenge.method.lgbm --val-fold 0 --out runs/lgbm
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ai_challenge.datasets import ACTION_CLASSES, CLASS_TO_ID
from ai_challenge.models.common import get_folds, load_train_records

DATA_DIR = Path("data")

# current_prompt 키워드 (IDEA EDA: write/create→write_file, plan→plan_task, grep→grep 등)
KEYWORDS = [
    "write", "create", "add", "new", "plan", "grep", "search", "find", "run",
    "patch", "diff", "test", "tests", "fix", "bug", "error", "read", "show",
    "list", "ls", "dir", "edit", "change", "update", "modify", "delete", "remove",
    "check", "lint", "type", "typecheck", "refactor", "implement", "glob",
    "shell", "bash", "command", "web", "google", "docs", "ask", "confirm", "?",
    "file", "function", "class", "config", "install", "build", "commit",
]

ACTION_SET = set(ACTION_CLASSES)


def _last_two_actions(history):
    acts = [t.get("name") for t in history if t.get("role") == "assistant_action" and t.get("name")]
    last = acts[-1] if acts else "none"
    second = acts[-2] if len(acts) >= 2 else "none"
    return last, second


def extract_features(records) -> tuple[pd.DataFrame, list[str]]:
    """ActionSample 리스트 → 피처 DataFrame + categorical 컬럼명."""
    rows = []
    for s in records:
        sm = s.session_meta or {}
        ws = s.workspace
        hist = s.history or []
        open_files = ws.get("open_files") or []
        last, second = _last_two_actions(hist)
        lang_mix = ws.get("language_mix") or {}
        top_lang = max(lang_mix.items(), key=lambda kv: kv[1])[0] if lang_mix else "none"
        prompt = (s.current_prompt or "").lower()

        # history 내 action 종류별 카운트 (전이 신호)
        act_counts = {f"h_{a}": 0 for a in ACTION_CLASSES}
        n_user = 0
        for t in hist:
            if t.get("role") == "assistant_action" and t.get("name") in ACTION_SET:
                act_counts[f"h_{t['name']}"] += 1
            elif t.get("role") == "user":
                n_user += 1

        feat = {
            # categorical
            "user_tier": sm.get("user_tier", "na"),
            "language_pref": sm.get("language_pref", "na"),
            "last_ci_status": ws.get("last_ci_status", "none"),
            "last_action": last,
            "second_last_action": second,
            "top_lang": top_lang,
            # numeric
            "turn_index": sm.get("turn_index", 0) or 0,
            "budget": sm.get("budget_tokens_remaining", 0) or 0,
            "elapsed_sec": sm.get("elapsed_session_sec", 0) or 0,
            "loc": ws.get("loc", 0) or 0,
            "open_files_count": len(open_files),
            "open_files_empty": int(len(open_files) == 0),
            "git_dirty": int(bool(ws.get("git_dirty"))),
            "history_len": len(hist),
            "n_user_turns": n_user,
            "prompt_len": len(prompt),
            "prompt_words": len(prompt.split()),
            "n_langs": len(lang_mix),
            "py_ratio": float(lang_mix.get("py", 0.0)),
        }
        feat.update(act_counts)
        for kw in KEYWORDS:
            feat[f"kw_{kw}"] = int(kw in prompt)
        rows.append(feat)

    df = pd.DataFrame(rows)
    cat_cols = ["user_tier", "language_pref", "last_ci_status",
                "last_action", "second_last_action", "top_lang"]
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return df, cat_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-fold", type=int, default=0)
    ap.add_argument("--out", default="runs/lgbm")
    ap.add_argument("--n-estimators", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import lightgbm as lgb

    records = load_train_records()
    fold = get_folds(records)
    y = np.array([CLASS_TO_ID[s.action] for s in records])
    X, cat_cols = extract_features(records)
    print(f"[data] {len(records)} samples, {X.shape[1]} features ({len(cat_cols)} categorical)", flush=True)

    tr = fold != args.val_fold
    va = fold == args.val_fold
    model = lgb.LGBMClassifier(
        objective="multiclass", num_class=len(ACTION_CLASSES),
        n_estimators=args.n_estimators, learning_rate=args.lr,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(X[tr], y[tr], categorical_feature=cat_cols,
              eval_set=[(X[va], y[va])], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(50, verbose=False)])

    proba = model.predict_proba(X[va])
    pred = proba.argmax(1)
    macro = f1_score(y[va], pred, average="macro")
    acc = accuracy_score(y[va], pred)
    print(f"[lgbm] OOF macro_f1 = {macro:.4f}  acc = {acc:.4f}  (best_iter={model.best_iteration_})", flush=True)

    # OOF 확률 저장 (앙상블용) — 해당 fold 샘플 id 순서
    val_ids = [s.id for s in records if True]  # 전체 순서 유지용
    np.savez(out / "oof.npz", proba=proba, y=y[va],
             ids=np.array([s.id for s, v in zip(records, va) if v]))
    # 피처 중요도 상위
    imp = sorted(zip(X.columns, model.feature_importances_), key=lambda kv: -kv[1])[:15]
    print("[top features]", ", ".join(f"{k}={v}" for k, v in imp), flush=True)


if __name__ == "__main__":
    main()
