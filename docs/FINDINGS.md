# 실험 종합 & 전략 핸드오프 (2026-07-08)

새 세션에서 전략을 다시 세우기 위한 **전체 실험 기록·결론·미탐색 방향** 정리.

## 1. 현재 위치
- **최고 LB: 0.78740** (submit_kd_w6_all: 4×Qwen2.5-3B + GPT5-Qwen3-4B×2 가중 앙상블 → Qwen2.5-0.5B student KD, 전체70k)
- **순위: 11위 / 199팀** (2026-07-08 기준). 1위 Vertex 0.79795, 상위권 0.789~0.798.
- Metric: **Macro-F1**, 14-class action 분류. 제출: code-submission(model+script.py+requirements, ≤1GB zip, T4 16GB, 추론≤10분, 오프라인, transformers 4.46.3).

## 2. 최적 파이프라인 (현재 best)
1. **Teacher**: Qwen2.5-3B 직접 파인튜닝 4개 seed(42~45) + GPT5-Qwen3-4B(Jackrong/GPT-5-Distill-Qwen3-4B-Instruct). no-class-weight, base 직렬화.
2. **앙상블**: 5개 teacher 소프트라벨 평균, 단 **GPT5에 가중치 ×2**((4×3B합 + 2×GPT5)/6).
3. **Student**: Qwen2.5-0.5B, KD(T=3, α=0.25), 3에폭, 전체 70k(all-data).
4. **제출**: fp16 패키징(~880MB), pack.py의 self-contained script.py.

## 3. fold0 OOF ↔ LB 관계 (방법론 — 신뢰 가능)
**LB ≈ (student fold0 OOF) + 0.0042** (편차 ±0.0004, 매우 안정적).
- 원인: fold0는 56k 학습, 제출은 70k 학습(all-data 부스트 = +0.004). 노이즈 아님.
- 실측 쌍: 단일3B(0.7801→0.7839, +0.0038), 4-teacher(0.7810→0.78519, +0.0042), 5-teacher(0.7816→0.78621, +0.0046).
- **teacher-argmax 프록시**는 student OOF보다 ~+0.002 높음(teacher가 student보다 강함) — 프록시로 스크리닝 시 감안.
- **함의**: top(LB 0.79+) = student fold0 OOF ~0.786 필요. 우리 최고 student fold0 = **~0.782**. → fold0에서 **+0.004**를 만드는 레버가 관건인데 아직 못 찾음.

## 4. 시도한 레버 전체 (✅효과 / ◐마진 / ❌실패)

### 모델 축
- ✅ **디코더 > 인코더**: Qwen(디코더)이 mDeBERTa/xlm-r(인코더, 0.60~0.64)를 압도.
- ✅ **teacher 크기 sweet spot = 3B**: 1.5B(0.757) < 3B(0.7745) > 7B(0.7688, 더 큰데 나쁨) > Qwen3-4B(0.7727). 14B는 PCIe FSDP라 146h 비현실.
- ✅ **student = Qwen2.5-0.5B 최적**: gemma-3-270m(0.776, 5.x필요), Qwen3-0.6B(0.779 동률, 1.5GB), bloomz(0.70), SmolLM2(0.53) 다 미달/동률.
- ❌ **distilled teacher(R1/GPT-oss/Claude)**: R1-1.5B(0.7533)·R1-7B(0.7655) 해로움, GPT5-4B(0.7741) 동률, Claude-Qwen3.5-4B(0.7719) 미달. "Claude가설(시뮬레이터가 Claude산출)" 기각.

### 앙상블 축
- ✅ **이질 teacher 다양성이 유일하게 통한 레버**: 4×3B(0.7810) → +GPT5-4B(0.78621) → +GPT5×2가중(0.78740). **+0.0012씩 실측 상승**.
- ✅ **가중치**: GPT5에 ×1.5~2가 최적(proxy). ×2.5, ×3은 하락.
- ❌ **약한 이질 teacher 추가는 희석**: Qwen3-4B(0.7727)·Claude(0.7719) 넣으면 앙상블 하락. **≥0.774짜리만 유효**한데 GPT5-4B가 유일.
- ◐ **같은-arch 더 많은 3B seed**: 4개→마진.

### 입력/피처 축
- ◐ **rich 직렬화**(open_files/args): 직접학습 +0.006, **KD에선 무익**(teacher가 이미 흡수).
- ❌ **cues([CUES] 탐색도구 힌트: 파일명/와일드카드/디렉터리/검색어)**: teacher +0.0015(수렴속도 착시), **student-cues LB 0.78272(-0.0047 해로움)**. 탐색도구4개(read/grep/list/glob) 혼동은 **본질적 애매함** 확정.
- ❌ **LightGBM 구조피처**: 0.42(신호는 텍스트에 있음).

### 용량/양자화 축
- ◐ **1.5B student**: fold0 +0.0005(0.7821 vs 0.5B 0.7816), int4 손실 -0.0008(거의 무손실). **하지만 int4 zip=1.047GB → 제출 FAILED**(1GB 한도 엄격 or 4bit추론 실패). 용량 레버 dead end.

### student 레시피 축
- ✅ **3에폭 최적**: 5에폭 과적합(0.7787 < 0.7810). 2에폭은 undertrained(-0.003~6, 오늘 변형 실패 원인).
- ✅ **class-weight 제거가 최대 단일 레버**(-제거 시 LB +0.013, 초기).
- ✅ **KD α=0.25, T=3** 최적(스윕 완료).
- ❌ per-class threshold 튜닝: +0.0002 무효.

### 데이터 축
- ✅ 데이터 synthetic(sess_sim_*), in-distribution 학습이 정답. test-set 누수 없음(테스트는 1-step-per-session).
- **train fit 천장 ~0.87** = 입력 자체의 내재적 애매성 → val ~0.78대가 사실상 천장으로 추정.

## 5. 확립된 결론 (새 전략의 전제)
1. **우리 접근(다양성 가중 앙상블 KD)의 천장 = LB 0.78740.** 크기·distill·입력·용량·레시피 다 최적화/소진.
2. **fold0 OOF는 신뢰 가능**(LB = OOF+0.004). fold0 0.782가 우리 한계.
3. **탐색도구 4개 혼동이 macro-F1 병목**(F1 0.53~0.67), 상당부분 본질적 애매함.
4. **상위권(0.79~0.798)은 우리가 못 찾은 코어 방법**이 있음. 그들은 39~50회 제출로 집요하게 반복.

## 6. 아직 안 해봤거나 재검증 필요한 방향 (새 전략 후보)
- **가중치 변형 3에폭 재실험**: 오늘 건 2에폭이라 무효. ×1.5(proxy 0.7842)를 제대로 → +0.001 가능성.
- **더 많은 3B seed(8~12) 대규모 앙상블** + 가중: 저비용, 마진이지만 미검증.
- **강한 이질 teacher 추가 발굴**: GPT5류 ≥0.774 다른 소스(현재 후보들은 다 약함).
- **외부 데이터 신중 활용**: in-distribution 유사 데이터 증강(SWE-bench는 OOD라 위험, but 필터링 가능성). ← **미탐색, 상위권 격차 설명 후보 1순위**.
- **완전히 다른 입력 표현**: cues는 실패했으나 다른 featurization은 미탐색.
- **student를 teacher-앙상블 소프트라벨이 아닌 다른 타깃으로**: rationale distillation, multi-task 등.
- **1GB 안에 드는 더 큰 student**: vocab pruning으로 1.5B 임베딩 축소(1.047GB→<1GB) 또는 다른 소vocab 모델.
- **추론측 TTA**: 미착수(제출로 검증 가능).

## 7. 자산 (코드/모델/캐시)
- **코드**: ai_challenge/{datasets,models,method,utils}. train.py(--serialize/--fsdp/--manual-oof/--keep-columns), distill.py(--teacher-dir 다중, 캐시 재사용), pack.py(--serialize/--quant int4). 5.x 호환 레이어 완성.
- **teacher(gpu4)**: runs/t3b_base, t3b_s43~45(4×3B), t_gpt5q3_4b(GPT5-4B). 그 외 다수(약함).
- **all-data 캐시(재사용 가능)**: runs/kd_ens4_all(4×3B평균), kd_div5_all(5-teacher평균), kd_w6_all(GPT5×2가중). → 새 student 즉석 distill 가능.
- **fold0 캐시**: runs/kd_div5_f0. **val 로짓**: runs/_vallogits/*.npy(4×3B+GPT5+Qwen3-4B → 앙상블 조합 학습없이 proxy 평가 가능).
- **제출 이력**: docs/exp_logs/LOG.md, DACON app.dacon.io(user_id 501433).

## 8. 인프라 노트
- gpu4(4×A100, PCIe — 다중GPU FSDP 비현실적), local(1×A100). HF_HOME=/data/hf, 토큰 /data/hf/token(gated 접근).
- bitsandbytes는 venv 직접설치(`uv pip install` 후 `.venv/bin/python`; `uv run --with`은 torch를 CPU로 되돌림).
- 대형모델 단일GPU: `--optim adafactor`(bitsandbytes 없이 메모리 절약) + grad-ckpt.
- 제출: gpu4에서(아웃바운드 비용), 메모 앞에 `(채운) `.
