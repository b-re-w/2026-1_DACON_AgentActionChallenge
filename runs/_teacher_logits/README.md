# Teacher soft-label store (공유용)

`gen_softlabels` 로 만든 teacher 별 **전체 train 레코드(70,000)의 14-class raw logits** 캐시.
`distill --teacher-logits <tag>` 로 teacher forward 0회 증류에 바로 쓴다.

- 포맷: `{tag}.npz` — `logits:(70000,14) float32`, `ids:(70000,) str` / `{tag}.json` — 메타
- **레코드 id 기준**이라 teacher 의 tokenizer·아키텍처·max_length 가 달라도 앙상블 호환된다.
- 단, `serialize` 프리셋은 반드시 일치해야 한다(전부 `base`).

이 디렉터리에는 로컬 실험용 store 가 여럿 있지만, **git 에는 아래 것만 커밋한다**
(나머지는 메인 서버 복사본이거나 폐기 대상 — `.gitignore` 에서 파일명 단위로 허용).

---

## 커밋된 store

### `qdebertaL2` — DeBERTa-v3-large (encoder 이질 teacher) ✅ 권장

| 항목 | 값 |
|---|---|
| 모델 | `microsoft/deberta-v3-large` (435M, disentangled attention) |
| **OOF Macro-F1 (fold0)** | **0.7415** (acc 0.7479) |
| train argmax | 0.7763 |
| 레시피 | full-ft, ep8(early-stop patience 2 → ep6 best), lr 2e-5, batch 64, ML 512, bf16, grad-ckpt, `--no-class-weight`, serialize=base |
| 재현 | `scripts/train_deberta_long.sh` |

**왜 쓰나 — 이질성(계열 다양성).** 기존 teacher 는 전부 decoder LLM(Qwen 사다리·GPT·OSS)이다.
이건 **최초의 encoder 계열 teacher**라 아키텍처·tokenizer 가 완전히 달라 오류 상관이 낮다.

**앙상블 기여 (fold0 val, teacher argmax 기준)**

| 구성 | Macro-F1 |
|---|---|
| ALL-decoders(11) baseline | 0.7826 |
| + qdebertaL2 **weight 2** | **0.7846** (+0.0020) |
| mistral 제거 + qdebertaL2 weight 2 | **0.7860** (+0.0034) |

**사용법 — weight 2 권장** (`tag:weight` 문법):

```bash
uv run python -m ai_challenge.method.distill \
  --teacher-logits q3b q3bb q3b43 q3b44 q3b45 q14b q32b qw6 qgpt5 qgptoss qdebertaL2:2 \
  --student Qwen/Qwen2.5-0.5B --out runs/kd_hetero --bf16 \
  --temperature 3 --alpha 0.25 --max-length 640 --serialize base
```

- weight 1 → +0.0018, **weight 2 → +0.0020(최적)**, weight 3~4 → +0.0019 (내성 있음, 급락 안 함).
- 개별 성능이 앙상블(0.783)보다 낮으므로 **과도한 up-weight 는 무의미**하다.

---

## 알려진 문제 (중요)

### ⚠️ `qmistral24b` — 학습 실패, 앙상블에서 **빼야 함** (커밋 안 함)
- `train_hetero.sh` 의 `--early-stop-patience 1` 때문에 **2 epoch 만에 조기중단**
  (ep1=0.6672 → ep2=0.6220 하락 → 즉시 중단, ep1 복원). 사실상 1 epoch 학습.
- OOF 0.667 로 모든 decoder teacher(0.764~0.784)보다 크게 낮다. 24B 의 실력이 아니라 **미학습**.
- **앙상블에서 제거하면 +0.0026** (0.7826 → 0.7852). 넣어두면 오히려 깎아먹는다.
- 재학습 시 `--early-stop-patience 2~3` + lr 하향(1.5e-4 는 24B 에 과했을 가능성) 권장.
- 같은 스크립트를 쓰는 **gemma/llama 도 동일한 조기중단 위험**에 노출된다 — patience 를 올릴 것.

### ⚠️ `qphi4` — 학습이 완료되지 않음 (store 없음)
- `runs/t_qphi4/hf/` 가 비어 있음(체크포인트 0). step ~100(epoch 0.11)에서 중단.
- 재학습 필요(14B QLoRA, 약 10~15h).

### `qdebertaL` (구버전, OOF 0.7267) — 커밋 안 함
- `qdebertaL2` 로 대체됨. **둘을 동시에 넣으면 오히려 손해**(+0.0003) — 서로 상관이 높아
  약한 구버전이 발목을 잡는다. **신버전만** 쓸 것.

### `qmodernbertL` (OOF 0.5977) — 커밋 안 함
- lr 2e-5 가 ModernBERT 에 너무 낮아 undertrain(ep5 까지 계속 상승 중이었음).
- 현 상태로는 앙상블 기여 ~0. 재학습(lr 5e-5, ep8) 후 재평가 필요.

---

## store 재생성

```bash
uv run python -m ai_challenge.method.gen_softlabels \
  --teacher-dir runs/t_<tag>/model --tag <tag> --serialize base --max-length <ML>
# QLoRA 대형 teacher 는 --load-in-4bit 추가
```
