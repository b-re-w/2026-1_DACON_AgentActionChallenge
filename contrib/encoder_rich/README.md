# 인코더 라인 실험 (exp01~17) — `rich` 직렬화로 인코더 teacher 강화

> 작성: 임동연 · 2026-07-13 · 독립 서버(로컬 24GB)에서 진행한 인코더 중심 실험 전체 기록
>
> **팀에 바로 쓰이는 결론 한 줄:
> `rich` 직렬화가 인코더를 +0.05~0.06 끌어올린다. → 현재 최고 다양성 멤버인 `qdebertaL`(0.7264)을
> 더 강하게 만들 여지가 있다.**

---

## 1. 왜 이게 지금 중요한가

브랜치 로그 기준, **인코더 teacher가 트리오의 유일한 상승 멤버**:
- `qdebertaL 0.7264` — 단독으론 모든 decoder teacher(0.764~0.784)보다 약함
- 그런데 **`qdebertaL2:1 = 0.7863`** → 앵커(0.7847)를 넘긴 **유일한** 조합, **LB 0.79049**
- 이유: 인코더는 디코더들과 **아키텍처·토크나이저가 완전 이질** → decorrelation 최대

**즉 "약하지만 이질적인" 인코더가 앙상블을 올린다.**
그렇다면 **같은 이질성을 유지하면서 인코더 자체를 더 강하게** 만들면 더 오를 여지가 있다.
아래 실험이 그 방법(=`rich` 직렬화)을 보여준다.

---

## 2. 핵심 발견 — `rich` 직렬화가 인코더를 크게 올림

같은 모델·같은 fold에서 **직렬화만** 바꾼 비교:

| 모델 | `full`(기본) | **`rich`** | Δ |
|---|---|---|---|
| xlm-roberta-base | 0.6954 | **0.7496** | **+0.054** |
| mdeberta-v3-base | 0.6937 | **0.7555** | **+0.062** |
| klue/roberta-base | 0.6681 | **0.7478** | **+0.080** |
| deberta-v3-large | — | **0.7583** | (rich만 수행) |

**3개 인코더 전부 +0.05~0.08.** 단발 우연이 아님.

### `rich` 가 뭐가 다른가
```
[PROMPT] {prompt}
[LAST]   {직전 action 1개}
[TRAIL]  {직전 K개 action 시퀀스}      ← ★ 팀 base 에는 없는 것
[META]   {tier lang turn budget sess loc git ci nopen openext mix}  (버킷화)
[HIST]   {최근 10턴, 요약 60자}
```
- ⭐ **`[TRAIL]` = 직전 3~5개 행동 이름 시퀀스**를 명시 토큰으로 노출 (예: `read_file>edit_file>run_tests`)
- 별도 MI 분석 결과 **action_trail(직전3행동)이 단일 최강 신호** (타깃 엔트로피의 **23.2%**;
  last_action 단독은 10.1%) — 팀 `base` 는 history 안에 묻어두기만 하고 압축 trail 토큰이 없음
- META 를 원시값이 아니라 **버킷**(loc/elapsed/budget 구간화)으로 제공

구현: `exp_lib.py` 의 `build_text(row, features="rich")`, `action_trail()`, `flatten_meta_rich()`.

---

## 3. 실험 전체 그리드

| exp | 모델 | 피처 | 손실 | fold0 macro-F1 | 메모 |
|---|---|---|---|---|---|
| exp01 | klue/roberta-base | full | ce | 0.6681 | |
| exp02 | mdeberta-v3-base | full | ce | 0.6937 | |
| exp03 | xlm-roberta-base | full | ce | 0.6954 | |
| exp04 | klue/roberta-base | **prompt_only** | ce | **0.4484** | 맥락 제거 시 폭락 → history/meta 필수 |
| exp05 | klue/roberta-base | full | **focal** | 0.6616 | **focal < ce** (focal 무익) |
| exp06 | xlm-roberta-base | full | ce | 0.6926 | 모델저장용 재현 |
| **exp07** | xlm-roberta-base | **rich**512 | ce | **0.7496** | |
| **exp08** | mdeberta-v3-base | **rich**512 | ce | **0.7555** | |
| **exp09** | klue/roberta-base | **rich**512 | ce | **0.7478** | |
| **exp10** | **deberta-v3-large** | **rich**512 | ce | **0.7583** | 인코더 최고 |
| exp11 | Qwen2.5-0.5B | rich512 | ce | 0.7600 | 디코더 대조군 |
| **exp12** | mdeberta-v3-base | rich512 | **KD** | **0.7721** | 인코더앙상블→mdeberta 증류, **LB 0.7601** |
| exp13 | Qwen2.5-Coder-0.5B | rich512 | ce | 0.7594 | Coder ≈ 범용(0.7600), 동률 |
| exp14 | Qwen2.5-Coder-0.5B | rich512 | KD | 0.7716 | |
| exp17 | Qwen2.5-Coder-7B | rich512 | QLoRA | ep1 0.7483 | **4bit QLoRA 7B는 부진**(팀 full-FT 7B 0.776과 대조) |

---

## 4. ⚠️ 한계 — 숫자를 그대로 비교하면 안 되는 이유

1. **class-weighted CE 사용**: `exp_lib.make_loss` 가 balanced weight를 **항상** 적용.
   팀이 나중에 밝힌 **no-class-weight(+0.013)** 레버를 우리는 못 썼음 → **우리 절대수치는 ~0.013 저평가**.
2. **fold 분할이 다름**: 자체 GroupKFold(session) 사용, 팀 `data/folds.csv` 와 불일치.
   → 팀 수치와 **직접 비교 금지**. 단 **같은 셋업 안에서의 Δ(rich 효과 +0.05~0.08)는 유효**.
3. exp17 은 **QLoRA 4bit**(메모리 제약) — 팀의 full-FT 7B(0.776)와 다른 조건. 4bit가 원인일 수 있음.

**→ 즉 "rich 가 인코더를 +0.05~0.08 올린다"는 상대효과만 취하면 됨.**

---

## 5. 제안 액션 (겹치지 않는 것)

분담표 #1(`trio + qdebertaL2 가중 스윕`)에 직접 연결:

1. **`qdebertaL` 을 rich 스타일로 재학습** — 특히 **`[TRAIL]` 토큰 추가** + META 버킷화.
   현재 0.7264 → 우리 실측 추세상 **0.75+ 기대**. no-cw 유지 시 더.
2. 강해진 인코더 teacher 로 **trio + qdebertaL_rich 가중 스윕**(:0.5/:1/:1.5) 재측정.
   - 근거: 인코더는 **유일하게 앵커를 넘긴 다양성 멤버**. 이질성은 유지되고 성능만 오르면 순이득 가능.
   - ⚠️ 단 "강해지면 오히려 decorrelation 이 줄어" 이득이 사라질 수도 → **실측 필요**(2~3 fold 평균).
3. (부가) `focal` 은 인코더에서 ce 보다 나빴음(exp05) — 팀에서 재시도 불필요.
4. (부가) `prompt_only` 0.4484 → **맥락(history/meta) 이 신호의 대부분**. 입력 축소 실험은 비추천.

---

## 6. 코드 사용법

이 폴더는 **독립 실행형**(팀 `ai_challenge` 패키지에 의존하지 않음). `data/` 만 있으면 됨.

```bash
# 학습 (config 로 지정)
python train.py configs/exp10_debertalarge_rich512_ce.json

# teacher soft-label 생성 (저장된 모델들 앙상블 → teacher_soft.npy)
python distill.py

# 증류 학습: config 에 "teacher_soft": "...npy", "kd_alpha": 0.5 넣고
python train.py configs/exp12_distill_mdeberta_kd.json

# OOF 재생성 / 앙상블 / 제출 패키징
python gen_oof.py <run_name>
python ensemble.py
python make_submit.py <run_name>
```

**핵심 파일**
- `exp_lib.py` — **`rich` 직렬화 정의**(`build_text`, `action_trail`, `flatten_meta_rich`), 손실, 평가(bias_tune/logit_adjust)
- `train.py` — 학습 + KD(`teacher_soft`, `kd_alpha`; **T=1**, 팀의 T=3 와 다름) + OOF 저장
- `train_qlora.py` — QLoRA teacher(exp17)

**주의**: 이 코드의 KD 는 **T=1**(온도 없음). 팀 `distill.py` 의 **T=3 + T²** 가 표준이니 그쪽을 쓸 것.

---

## 7. 파일

```
contrib/encoder_rich/
├── README.md          ← 이 문서
├── exp_lib.py         rich 직렬화 · 손실 · 평가
├── train.py           학습(+KD, OOF 저장)
├── train_qlora.py     QLoRA teacher
├── distill.py         teacher soft-label 생성
├── ensemble.py  gen_oof.py  make_submit.py  run.py
├── configs/           exp01~exp17 설정
└── metrics/           각 exp 의 metrics.json (per-class F1 포함, 모델 가중치는 제외)
```
