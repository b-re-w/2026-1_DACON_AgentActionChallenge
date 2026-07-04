# AI Agent Action 예측 챌린지 — 접근 계획 (IDEA)

> 2026 AI·SW중심대학 디지털 경진대회 AI부문 / DACON 236694
> 태스크: AI 코딩 에이전트의 다음 action을 **14클래스**로 예측 · 지표 **Macro-F1**

---

## 0. 대회 요약 & 제약 (설계에 직접 영향)

- **지표**: Macro-F1 (14클래스 균등 가중 → **희소 클래스가 승부처**)
- **코드 제출 대회**: `submit.zip = model/ + script.py + requirements.txt`
- **평가 환경**: T4 16GB / 3 vCPU / 12GB RAM, **추론 ≤10분, 설치 ≤10분, zip ≤1GB, 완전 오프라인**, Python 3.11 / CUDA 12.8 / torch 2.7.1
- **본선 배점**: 예선 50% + **추론속도 10%** + 전문가 발표 40% → 경량·빠른 모델 + 신규성이 유리
- **데이터**: train 70,000건 / 9,429 세션(평균 7.4 step) / test 30k 비공개. id `sess_sim_...` → **합성(sim) 데이터**
- **현재 baseline**: RoBERTa로 LB(Public) **0.6853** → **넘어야 할 목표치**

---

## 1. 데이터 & 검증 전략 (모든 접근의 공통 토대)

### 1-1. 핵심 전제 — 데이터는 합성이다
- id가 `sess_sim_*` → train/test 모두 같은 시뮬레이터 산출물로 추정.
- ⇒ **목표는 "진짜 에이전트 행동"이 아니라 "이 시뮬레이터의 결정 함수"를 재현하는 것.**
- ⇒ train을 SWE/외부데이터로 **대체 금지**(분포 shift). 외부는 희소 클래스 소량 증강만.

### 1-2. 검증 = StratifiedGroupKFold (필수, 확정)
- **group = 세션 id**, **stratify = action**.
- 누수 근거: step_N의 정답 action이 **step_N+1의 history 마지막**에 그대로 들어감 + 연속 step은 맥락 거의 동일. 랜덤 split 시 val이 train 정답을 커닝 → 점수 뻥튀기.
- LB(비공개 30k = 처음 보는 세션) ↔ GroupKFold OOF(처음 보는 세션)는 **같은 조건** → 로컬 OOF가 신뢰 가능한 나침반이 됨.
- 구현: `ai_challenge/datasets` 패키지(작성 완료, 위치 이전 예정). 5-fold, 누수 자동 검증 통과.

### 1-3. 불균형 & 강한 신호 (EDA)
- 분포: `edit_file` 16% ↔ `web_search` 1.8% (약 8.8배). 희소: web_search, write_file, lint_or_typecheck, plan_task, ask_user.
- 강한 신호: **직전 action 전이**(edit→run_tests 등), **open_files=[]**(write_file lift 2.83, 최강 단일 신호), **last_ci_status**, **git_dirty**, **current_prompt 키워드**(write/create→write_file lift 10배).
- 직렬화에서 이 신호들을 `[META] [LAST_ACTION] [HISTORY] [PROMPT]` 토큰으로 명시.

---

## 2. 세 가지 접근

### 접근 A — Encoder-based (주력 베이스라인)
**아이디어**: 인코더로 어떻게든 뽑아낸다. 모든 팀이 시도할 강한 베이스라인이므로 **먼저 이겨야 할 대상**.

- **모델**: 다국어 필수 — `microsoft/mdeberta-v3-base` 또는 `xlm-roberta-base`. (프롬프트가 ko/en/mixed라 **영어 RoBERTa는 반쪽짜리** → 교체만으로 상승 여지)
- **입력**: 우리 직렬화(meta + last_action + history + current_prompt), max_length 512.
- **구조 피처 융합**: [CLS] 임베딩 + 명시적 구조 피처(open_files 수, ci_status one-hot, git_dirty, turn_index, last_action one-hot) concat → 분류 헤드. (또는 전부 텍스트 직렬화로 흡수)
- **불균형 처리**: class-weighted CrossEntropy 또는 focal loss. **OOF 기반 per-class threshold 튜닝**(Macro-F1 직접 최적화).
- **희소 클래스 구제**: 키워드 규칙/피처(write·create, plan, grep, patch...).
- **장점**: T4/10분/1GB 여유, 본선 속도배점 유리, 학습 안정.
- **목표**: 0.685 돌파 후 개선 누적.

### 접근 B — Decoder-based SLM
**아이디어**: 작은 LM 파인튜닝(next-action = next-token 프레이밍).

- **모델**: Qwen3-0.6B 등 소형 다국어 LM.
- **방식**: (a) 분류 헤드 부착 or (b) constrained decoding으로 14개 라벨 토큰만 생성.
- **위치**: 단일 라벨 분류는 양방향 인코더가 더 sample-efficient·고속 → **주력보다는 앙상블 다양성 멤버**.
- **1GB**: 0.6B를 int8/int4로 누르면 제출 가능.
- **속도 예산**: 30k/10분 = 20ms/샘플. 단일 forward(생성 최소)면 여유.

### 접근 C — Speculative Decoding / MatFormer (Gemma 3n)
**프레이밍(핵심)**: 이 대회 태스크 = **LLM 앞단의 경량 draft 모델**(speculative execution의 draft 역할). train이 대형 LM 산출물(sim)이므로 **action 예측 = 대형 모델의 결정 모방**. MatFormer는 대형 모델의 **중첩 서브모델을 재학습 없이 추출** → draft-target 일치성을 공짜로 확보하는 원리적 방법.

- **왜 큰 걸로 학습 후 추출?** 작은 모델은 언어/긴 맥락 이해가 약함(툴콜 이름 1개 선택 자체는 소형도 가능하나, 한국어·history 추론에서 큰 모델이 유리). 큰 MatFormer로 학습하면 서브모델이 부모 역량을 상속.
- **추출 범위**: Gemma 3n은 E2B ↔ E4B 사이 Mix-n-Match(FFN dim 8192~16384 + layer skip)로 임의 크기 추출. **바닥 = E2B(~5B raw / 2B eff)**.
- **제약(결정적)**: E2B도 int4로 ~2.5GB → **1GB 제출 한도 초과 → 제출물로는 불가**. ⇒ **teacher / distillation / 파일럿 전용.**
- **파일럿(다음 스텝)**: 로컬 GroupKFold 홀드아웃에서 **사전학습 E2B zero-shot**(constrained decoding, 파인튜닝 0) Macro-F1 측정.
  - 이미 쓸 만함 → 데이터가 대형 LM 산출물이라는 강한 신호(프레이밍 검증) + teacher 상한 추정.
  - 형편없음 → 큰 모델 루트 접고 인코더로 회귀.
- **활용 경로**: Gemma 3n(teacher) → soft-label/rationale distillation → **1GB 이하 경량 student(접근 A/B)** 제출.

---

## 3. 앙상블 & 마무리
- **A(인코더) + B(SLM) + (선택) LightGBM(구조 피처 전용)** OOF 앙상블.
- LightGBM은 구조 신호가 강해 저비용 고성능 멤버 후보(속도 사실상 공짜).
- OOF 기반 가중치 + per-class threshold로 Macro-F1 최종 최적화.

---

## 4. 실행 로드맵 (순서)
1. **[진행 중] 리더보드 모니터링 파이프라인** — 순위·점수 자동 확인.
2. `ai_challenge/datasets`로 데이터 패키지 이전(상단 `datasets/` 이름 충돌 회피).
3. **접근 A**: mDeBERTa-v3 + 우리 직렬화 + GroupKFold **OOF 베이스라인** → 0.685와 대조(로컬↔LB 상관 확정).
4. 개선 누적: 불균형(class weight/focal), per-class threshold, 구조 피처 융합, 키워드 → 제출.
5. **접근 C 파일럿**: E2B zero-shot OOF → distillation go/no-go.
6. **접근 B** + 앙상블 → 최종 제출.

---

## 5. 리스크 & 환경 메모
- **폴더명 충돌**: 최상위 `datasets/`는 HF `datasets`(4.0.0 설치됨)와 shadowing → Trainer 크래시 위험. ⇒ 코드는 `ai_challenge/datasets`로 둘 것.
- **sentencepiece 미설치**(로컬): mDeBERTa/XLM-R 토크나이저에 필요.
- **제출 제약**: 1GB / 추론 10분 / 설치 10분 / 오프라인. 사전학습 가중치는 zip에 동봉.
- 로컬: torch 2.7.0+cu128, transformers 4.57.3, accelerate 1.9.0, sklearn 1.6.1.

---

## 6. 실험 로그 규칙 (에이전트 지침 — 상세는 `LOGGING.md`)

모든 실험 에이전트는 **실험이 끝나면 반드시** 결과를 기록한다.

- **상세 기록**: `docs/exp_logs/<컨셉>/<YYYY-MM-DD>.md` — 컨셉별 폴더 · 날짜별 파일(형식 `YYYY-MM-DD` 통일), 같은 날은 시각 섹션으로 이어붙임.
- **요약 인덱스**: `docs/exp_logs/LOG.md` — 실험마다 한 줄(날짜·컨셉·타이틀·핵심 수치·링크).
- **손으로 쓰지 말고 헬퍼 호출**: `ai_challenge.utils.experiment_log.log_experiment(concept, title, metrics, notes)`.

**컨셉별 로그 폴더 (각 컨셉 폴더 따로 존재)**:
| 폴더 | 컨셉 |
|------|------|
| `docs/exp_logs/encoder_team/` | 접근 A · Encoder-based |
| `docs/exp_logs/decoder_team/` | 접근 B · Decoder-based SLM |
| `docs/exp_logs/speculative_team/` | 접근 C · Speculative/MatFormer |
| `docs/exp_logs/others/` | 기타(데이터·검증·앙상블·전처리) |

원칙: 실패 실험도 기록 · LB 점수는 설정과 함께 기록 · 포맷은 `log_experiment` 로 통일.
