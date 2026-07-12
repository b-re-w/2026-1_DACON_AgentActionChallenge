# 트리오 기반 전환 핸드오프 (다른 서버용, 2026-07-12)

> qw6 기반 → **트리오 기반**으로 갈아탈 것. 근거·재료·분담안.

## 왜 트리오인가 (확정 근거)
- **트리오 = qgptoss + qw6 + q7bteam (1:1:1)** → student KD. 팀원 kd3n LB **0.79195** > qw6_all640 **0.7905**.
- 5-fold 평균(신뢰지표): 트리오 **0.78561** > qw6 0.78501 — LB 순서와 일치.
  (⚠ fold0 단독은 역전함: qw6 f0 0.7847 > 트리오 f0 0.7821 인데 LB는 반대. **fold0 단독 판단 금지, 다중 fold 평균으로**)
- qw6 에 뭘 더해도(qw6+GLM −0.0024, qw6+gemma −0.0013) 하락. **트리오에 더하면 상승** (아래).

## 트리오 + 이질 1개 (fold0, 기준 trio f0=0.78205)
| 추가 | OOF | Δ |
|---|---|---|
| +gemma:0.5 | 0.78388 | +0.0018 (f4 +0.0018, f2 ±0 → 3fold 평균 +0.0012) |
| +GLM:0.5 | 0.78331 | +0.0013 |
| 32B 스왑(7B→32B) | 0.78391 | +0.0019 (f2 −0.0008) |
| +gem+glm(+32b) 중첩 | ~0.7838 | **중첩 무효 — 이질은 1개가 최적** |

## 표준 파이프라인 (팀원 신기록 0.79371 구조 — 필수 채택)
- **신기록 = ①fold0(56k) 모델을 --eval-steps 500 step피크 선택 + ②70k-OOF bias**. 
- **bias 를 all-data 모델에 얹으면 무효**(팀원 실측 동률·우리 정직검증 음수). 반드시 ①+② 짝으로.
- distill.py 에 `--eval-steps` 추가됨(이 브랜치). 파이프라인: fold0~4 학습(--eval-steps 500) → 5fold 70k OOF → coordinate_ascent bias → fold0-best 모델에 bias.json → pack.

## 재료 (이 브랜치에 전부 공유됨, git pull)
`q7bteam`(팀7B, 0.776) `qgemma9b`(0.7624) `qglm32b`(0.7605, bf16 재생성판) + 기존 qw6/qgptoss/q3b*/q14b/q32b/qgpt5, qdebertaL2(그쪽 것).
- ⚠ GLM 4bit 추론은 NaN 남 — bf16 풀로드로 쓸 것(store 는 이미 정상).

## 분담 (중복 회피)
**우리(4GPU) 진행중/예정:**
- 표준 파이프라인 × trio_gem05 (fold0~4 --eval-steps, 오늘 저녁 완료)
- trio_soup (트리오 시드 42/43/44 가중평균)
- trio + qdebertaL2 :0.5/:0.25 (fold0)
- 내일: 72B store → 트리오+72B/스왑 → 표준 파이프라인

**다른 서버 제안 (트리오 기반, 겹치지 않는 것):**
1. **trio + qdebertaL2 가중 스윕 확장** (:1, :1.5 — 우리는 0.25/0.5만) + trio + ModernBERT(있으면)
2. **표준 파이프라인 × mega_trio(트리오+gem:0.5+glm:0.5)** — fold0 0.78383, 2fold +0.0011 로 tg05와 동급인데 우리 큐엔 없음
3. **트리오 teacher 자체 강화**: 새 이질 teacher 학습 (Mistral-24B 등 train_hetero.sh, 목표 OOF ≥0.77 — "0.774+ 만 유효" 룰상 gemma/deberta 급은 한계)
4. 트리오 hyper 스윕 (T/α — qw6 에선 T3/α0.25 최적이었으나 트리오에선 미검증)

## 판단 원칙
- fold0 단독 ✗ → **2~3 fold 평균**으로 승격 판단
- 제출은 슬롯 아끼며 사용자 결정. LB만이 최종 심판 (OOF↑↛LB↑ 사례: oss2g1 OOF 0.7878→LB 0.7895 < qw6 0.7905)
