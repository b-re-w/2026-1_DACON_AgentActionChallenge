# %% [코랩 T4 추론 벤치마크 — DACON 236694 평가서버 모사]
# 사용법: Google Colab (런타임: T4 GPU) 에서 이 파일 전체를 셀 하나에 붙여넣고 실행.
# 목적: 제출 슬롯을 태우지 않고 후보 구성의 30k 추론 시간을 실측.
#   - 0.5B fp16 bs64  : 캘리브레이션 기준 (DACON 실측 7분50초와 비율 계산)
#   - 0.5B fp16 bs256 : 배치 최적화 여지
#   - 1.5B nf4 bs64/256, max_len 512/640 : 1.5B 단독 재도전 가능성 판정
import os, time, json, zipfile, urllib.request, re

# ── 0) 환경 ──────────────────────────────────────────────────────────
os.system("pip -q install bitsandbytes 2>/dev/null")
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
assert torch.cuda.is_available(), "GPU 런타임(T4)으로 변경 필요"
print("GPU:", torch.cuda.get_device_name(0))

# ── 1) 데이터: DACON open.zip → train.jsonl 앞 30,000건을 테스트 대용으로 ──
if not os.path.exists("data/train.jsonl"):
    print("downloading open.zip ...")
    urllib.request.urlretrieve("https://cfiles.dacon.co.kr/competitions/236694/open.zip", "open.zip")
    with zipfile.ZipFile("open.zip") as z: z.extractall(".")
samples = []
with open("data/train.jsonl", encoding="utf-8") as f:
    for line in f:
        samples.append(json.loads(line))
        if len(samples) >= 30000: break
print("samples:", len(samples))

# ── 2) 직렬화 (제출 script.py 와 동일 로직, base 프리셋) ─────────────────
def _tr(t, lim):
    t = " ".join(str(t).split()); return t if len(t) <= lim else t[:lim] + "…"
def _last(h):
    for t in reversed(h):
        if t.get("role") == "assistant_action" and t.get("name"): return t["name"]
def _meta(s):
    sm = s.get("session_meta", {}) or {}; ws = sm.get("workspace", {}) or {}
    of = ws.get("open_files") or []
    p = [f"tier={sm.get('user_tier','na')}", f"lang={sm.get('language_pref','na')}",
         f"turn={sm.get('turn_index',0)}", f"budget={sm.get('budget_tokens_remaining',0)}",
         f"ci={ws.get('last_ci_status','none')}", f"git={'dirty' if ws.get('git_dirty') else 'clean'}",
         f"open_files={len(of)}", f"loc={ws.get('loc',0)}"]
    lm = ws.get("language_mix") or {}
    if lm: p.append("codemix=" + ",".join(f"{k}:{v:.2f}" for k, v in sorted(lm.items(), key=lambda kv: -kv[1])[:3]))
    return " ".join(p)
def _turn(t, lim):
    if t.get("role") == "user": return "U: " + _tr(t.get("content",""), lim)
    a = ",".join(f"{k}={_tr(v,40)}" for k, v in list((t.get("args") or {}).items())[:4])
    return f"A: {t.get('name','?')}({a}) -> {_tr(t.get('result_summary',''), lim)}"
def serialize(s, mlen):
    ch = [f"[META] {_meta(s)}", f"[LAST_ACTION] {_last(s.get('history',[]) or []) or 'none'}"]
    h = s.get("history", []) or []
    if h: ch.append("[HISTORY] " + " ".join(_turn(t, 160) for t in h[-12:]))
    ch.append(f"[PROMPT] {_tr(s.get('current_prompt',''), mlen)}")
    return " ".join(ch)

# ── 3) 벤치 러너 (길이정렬 배칭 — 제출 script.py 동일) ────────────────────
@torch.inference_mode()
def bench(model_id, quant, bs, mlen, n=30000):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    t0 = time.time()
    if quant == "nf4":
        bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                 bnb_4bit_compute_dtype=torch.float16)
        m = AutoModelForSequenceClassification.from_pretrained(
            model_id, num_labels=14, quantization_config=bnb, device_map="cuda")
    else:
        m = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=14)
        m.to("cuda").half()
    m.config.pad_token_id = tok.pad_token_id
    m.eval()
    load_s = time.time() - t0

    texts = [serialize(s, mlen) for s in samples[:n]]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    t1 = time.time()
    done = 0
    for st in range(0, len(order), bs):
        chunk = [texts[i] for i in order[st:st+bs]]
        enc = tok(chunk, truncation=True, max_length=mlen, padding=True, return_tensors="pt").to("cuda")
        _ = m(**enc).logits.float().argmax(-1)
        done += len(chunk)
        if st // bs % 40 == 0:
            el = time.time() - t1
            print(f"  {done}/{n}  {el:.0f}s  (proj {el/max(done,1)*n:.0f}s)", end="\r")
    infer_s = time.time() - t1
    del m; torch.cuda.empty_cache()
    total = load_s + infer_s
    print(f"\n[{model_id} {quant} bs{bs} ml{mlen}] load {load_s:.0f}s + infer {infer_s:.0f}s = {total:.0f}s ({total/60:.1f}분)")
    return total

# ── 4) 실행 ─────────────────────────────────────────────────────────
results = {}
results["05b_fp16_bs64_640"]  = bench("Qwen/Qwen2.5-0.5B", "fp16", 64, 640)   # 기준(달콘 7'50")
results["05b_fp16_bs256_640"] = bench("Qwen/Qwen2.5-0.5B", "fp16", 256, 640)
results["15b_nf4_bs64_640"]   = bench("Qwen/Qwen2.5-1.5B", "nf4", 64, 640)
results["15b_nf4_bs256_640"]  = bench("Qwen/Qwen2.5-1.5B", "nf4", 256, 640)
results["15b_nf4_bs256_512"]  = bench("Qwen/Qwen2.5-1.5B", "nf4", 256, 512)

# ── 5) 달콘 환산 (0.5B bs64 = 470s 기준 스케일) ──────────────────────────
DACON_BASE = 470.0
scale = DACON_BASE / results["05b_fp16_bs64_640"]
print("\n===== DACON 환산 예상 (10분=600s 한도) =====")
for k, v in results.items():
    est = v * scale
    print(f"{k:22s} colab {v:6.0f}s → DACON 예상 {est:6.0f}s ({est/60:.1f}분) {'✅' if est < 570 else '❌'}")
