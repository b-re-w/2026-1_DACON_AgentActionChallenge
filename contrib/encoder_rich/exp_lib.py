# -*- coding: utf-8 -*-
"""실험 공용 라이브러리 — 데이터 로드 · 피처 구성 · 손실 · 평가.
train.py 가 이 모듈을 import 해서 사용한다. (제출용 files/script.py 는 규칙상
standalone 이어야 하므로 이 모듈을 쓰지 않고 동일 로직을 자체 복제한다.)
"""
import os, json, csv
import numpy as np
import torch, torch.nn as nn
from sklearn.metrics import f1_score, classification_report

ALL_CLASSES = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]
LABEL2ID = {c: i for i, c in enumerate(ALL_CLASSES)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


# ---------------- 데이터 ----------------
def load_data(data_dir="data", n_limit=0):
    import pandas as pd
    rows = [json.loads(l) for l in open(os.path.join(data_dir, "train.jsonl"), encoding="utf-8") if l.strip()]
    labels = {r["id"]: r["action"]
              for r in csv.DictReader(open(os.path.join(data_dir, "train_labels.csv"), encoding="utf-8"))}
    df = pd.DataFrame(rows)
    df["action"] = df["id"].map(labels)
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    df["group"] = df["id"].str.rsplit("-step", n=1).str[0]
    if n_limit:
        df = df.iloc[:n_limit].reset_index(drop=True)
    return df


# ---------------- 피처(입력 텍스트) 구성 ----------------
def flatten_meta(m):
    if not isinstance(m, dict):
        return ""
    ws = m.get("workspace", {}) or {}
    mix = ws.get("language_mix", {}) or {}
    top_lang = max(mix, key=mix.get) if mix else "?"
    open_files = ws.get("open_files", []) or []
    exts = sorted({f.rsplit(".", 1)[-1] for f in open_files if "." in f})
    budget = m.get("budget_tokens_remaining")
    bud = "low" if isinstance(budget, (int, float)) and budget < 30000 else "ok"
    return (f"tier={m.get('user_tier','?')} lang={m.get('language_pref','?')} "
            f"turn={m.get('turn_index','?')} budget={bud} "
            f"git_dirty={ws.get('git_dirty','?')} ci={ws.get('last_ci_status','?')} "
            f"toplang={top_lang} openext={','.join(exts) if exts else 'none'}")


def flatten_history(h, keep=6):
    if not isinstance(h, list) or not h:
        return "none"
    parts = []
    for turn in reversed(h[-keep:]):
        role = turn.get("role")
        if role == "user":
            parts.append("U:" + str(turn.get("content", ""))[:120])
        elif role == "assistant_action":
            parts.append(f"A:{turn.get('name','?')}(" + str(turn.get("result_summary", ""))[:40] + ")")
    return " | ".join(parts)


def last_action(h):
    if isinstance(h, list):
        for turn in reversed(h):
            if turn.get("role") == "assistant_action":
                return turn.get("name", "none")
    return "none"


# ---------- 'rich' 피처 (max_len 320 전제 — 맥락 절단 최소화 + 신호 강화) ----------
def _meta_fields(m):
    """meta 필드 dict (ablation 재사용용)."""
    if not isinstance(m, dict):
        return {}
    ws = m.get("workspace", {}) or {}
    mix = ws.get("language_mix", {}) or {}
    top2 = sorted(mix.items(), key=lambda x: -x[1])[:2]
    mixs = " ".join(f"{k}:{v:.0%}" for k, v in top2) if top2 else "?"
    open_files = ws.get("open_files", []) or []
    exts = sorted({f.rsplit(".", 1)[-1] for f in open_files if "." in f})
    b = m.get("budget_tokens_remaining")
    bud = ("verylow" if b < 10000 else "low" if b < 30000 else "mid" if b < 80000 else "high") \
        if isinstance(b, (int, float)) else "?"
    loc = ws.get("loc")
    locb = ("small" if loc < 2000 else "med" if loc < 15000 else "large") \
        if isinstance(loc, (int, float)) else "?"
    el = m.get("elapsed_session_sec")
    elb = ("new" if el < 120 else "short" if el < 600 else "long") \
        if isinstance(el, (int, float)) else "?"
    return {"tier": m.get('user_tier', '?'), "lang": m.get('language_pref', '?'),
            "turn": m.get('turn_index', '?'), "budget": bud, "sess": elb, "loc": locb,
            "git_dirty": ws.get('git_dirty', '?'), "ci": ws.get('last_ci_status', '?'),
            "nopen": len(open_files), "openext": ','.join(exts) if exts else 'none', "mix": mixs}


_META_ORDER = ["tier", "lang", "turn", "budget", "sess", "loc",
               "git_dirty", "ci", "nopen", "openext", "mix"]


def flatten_meta_rich(m, drop=()):
    """drop 에 든 필드는 제외하고 평문화 (ablation: drop={tier,lang,budget} 등)."""
    f = _meta_fields(m)
    if not f:
        return ""
    return " ".join(f"{k}={f[k]}" for k in _META_ORDER if k not in drop)


def flatten_history_rich(h, keep=10):
    if not isinstance(h, list) or not h:
        return "none"
    parts = []
    for turn in reversed(h[-keep:]):
        role = turn.get("role")
        if role == "user":
            parts.append("U:" + str(turn.get("content", ""))[:160])
        elif role == "assistant_action":
            parts.append(f"A:{turn.get('name','?')}(" + str(turn.get("result_summary", ""))[:60] + ")")
    return " | ".join(parts)


def action_trail(h, k=5):
    """직전 K개 행동 이름 시퀀스 (시간의존/전이 신호 명시)."""
    if not isinstance(h, list):
        return "none"
    acts = [t.get("name", "?") for t in h if t.get("role") == "assistant_action"]
    return ">".join(acts[-k:]) if acts else "none"


_META_DROP = {  # meta ablation: 어떤 필드를 제외할지
    "rich": (),                                              # 전체(기준)
    "rich_leanmeta": {"tier", "lang", "budget"},             # 쓸모없는 것만 제거(MI≈0)
    "rich_wsmeta": {"tier", "lang", "turn", "budget", "sess", "loc"},  # 워크스페이스 상태만
}


def build_text(row, features="full"):
    """features: prompt_only|prompt_meta|prompt_hist|full|rich|rich_nometa|rich_leanmeta|rich_wsmeta"""
    prompt = str(row.get("current_prompt", "") or "")
    if features == "prompt_only":
        return f"[PROMPT] {prompt}"
    if features == "rich_nometa":  # meta 완전 제거 (meta 총 기여 측정용)
        h = row.get("history")
        return (f"[PROMPT] {prompt} [LAST] {last_action(h)} [TRAIL] {action_trail(h)} "
                f"[HIST] {flatten_history_rich(h)}")
    if features in _META_DROP:  # rich / rich_leanmeta / rich_wsmeta
        h = row.get("history")
        return (f"[PROMPT] {prompt} [LAST] {last_action(h)} [TRAIL] {action_trail(h)} "
                f"[META] {flatten_meta_rich(row.get('session_meta'), drop=_META_DROP[features])} "
                f"[HIST] {flatten_history_rich(h)}")
    parts = [f"[PROMPT] {prompt}"]
    if features in ("prompt_hist", "full"):
        parts.append(f"[LAST] {last_action(row.get('history'))}")
    if features in ("prompt_meta", "full"):
        parts.append(f"[META] {flatten_meta(row.get('session_meta'))}")
    if features in ("prompt_hist", "full"):
        parts.append(f"[HIST] {flatten_history(row.get('history'))}")
    return " ".join(parts)


# ---------------- 손실 ----------------
class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, logits, target):
        ce = nn.functional.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss(kind, class_w, device):
    w = torch.tensor(class_w, dtype=torch.float, device=device)
    if kind == "focal":
        return FocalLoss(weight=w, gamma=2.0)
    return nn.CrossEntropyLoss(weight=w)   # ce (weighted, balanced)


# ---------------- 평가 (여러 post-hoc 변형을 한 번에) ----------------
def _macro(y, pred):
    return f1_score(y, pred, labels=range(len(ALL_CLASSES)), average="macro", zero_division=0)


def tune_bias(val_logits, val_y, rounds=8):
    n = len(ALL_CLASSES)
    bias = np.zeros(n, dtype=np.float32)
    best = _macro(val_y, val_logits.argmax(1))
    for _ in range(rounds):
        improved = False
        for c in range(n):
            for d in (-0.5, -0.25, -0.1, 0.1, 0.25, 0.5):
                trial = bias.copy(); trial[c] += d
                s = _macro(val_y, (val_logits + trial).argmax(1))
                if s > best + 1e-5:
                    best, bias, improved = s, trial, True
        if not improved:
            break
    return bias, best


def logit_adjust(val_logits, val_y, class_prior):
    """post-hoc logit adjustment (Menon et al. 2021): argmax(logit - tau*log(prior)). tau 그리드 탐색."""
    logp = np.log(class_prior + 1e-12)
    best_tau, best = 0.0, _macro(val_y, val_logits.argmax(1))
    for tau in (0.5, 1.0, 1.5, 2.0):
        s = _macro(val_y, (val_logits - tau * logp).argmax(1))
        if s > best:
            best, best_tau = s, tau
    return best_tau, best


def evaluate_all(val_logits, val_y, class_prior):
    """세 가지 변형의 macro-F1 을 dict 로 반환 + best 예측의 per-class."""
    none = _macro(val_y, val_logits.argmax(1))
    bias, f_bias = tune_bias(val_logits, val_y)
    tau, f_la = logit_adjust(val_logits, val_y, class_prior)
    variants = {"none": round(none, 4), "bias_tune": round(f_bias, 4), "logit_adjust": round(f_la, 4)}
    best_name = max(variants, key=variants.get)
    if best_name == "bias_tune":
        best_pred = (val_logits + bias).argmax(1)
    elif best_name == "logit_adjust":
        best_pred = (val_logits - tau * np.log(class_prior + 1e-12)).argmax(1)
    else:
        best_pred = val_logits.argmax(1)
    report = classification_report(val_y, best_pred, labels=range(len(ALL_CLASSES)),
                                   target_names=ALL_CLASSES, digits=3, zero_division=0, output_dict=True)
    per_class = {c: round(report[c]["f1-score"], 3) for c in ALL_CLASSES}
    return variants, best_name, {"bias": bias.tolist(), "tau": tau}, per_class, best_pred
