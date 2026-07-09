# Feature Engineering — 탐색도구(read/grep/list/glob) 애매성 공략 기록

> 병목: macro-F1의 대부분이 탐색도구 4개(read_file/grep_search/list_directory/glob_pattern)의
> **상호 혼동**에서 깎임 (그 4개 F1 0.53~0.67 vs 나머지 0.95~0.99). 시뮬레이터가 이들을
> 사실상 교체가능하게 뽑는 **aleatoric(환원 불가) 라벨 애매성**이 원인.

## 1. [CUES] — 이미 해본 것 (결론: 실패, OOF·LB 둘 다 하락)

### 무엇을 했나
`ai_challenge/datasets/serialize.py` 의 `_format_cues()` 가 prompt·workspace에서 **탐색도구를
시사하는 표층 단서를 정규식으로 뽑아** `[CUES]` 토큰으로 명시화. 직렬화 프리셋 `cues`, `cues_hist`.

주입되는 신호(각 샘플):
| 필드 | 의미 | 정규식(요지) |
|---|---|---|
| `nfile` | prompt 내 파일명 개수 | `\.(py\|js\|md\|...)` |
| `fopen` | 그 파일이 open_files에 있나 | basename 매칭 |
| `wild` | 와일드카드/"모든 파일" | `[*?]`, `모든.*파일`, `\*\.\w+` |
| `dir` | 디렉터리/구조 언급 | `폴더\|디렉터리\|structure\|ls\|목록\|contents of` |
| `search` | 검색 의도 | `grep\|search\|find\|찾\|검색\|references\|usages` |
| `read` | 읽기 의도 | `read\|show\|open\|보여\|열어\|까보\|내용\|살펴` |
| `list` | 나열 의도 | `list\|ls\|목록\|어떤.*파일\|나열\|tree` |
| `names` | 파일 basename 목록 | 상위 4개 |

가설: read vs grep vs list vs glob를 가르는 표층 단서를 모델이 쉽게 집게 하면 혼동↓.

### 결과 (하락)
| 실험 | 지표 | 값 | base 대비 |
|---|---|---|---|
| t3b_base (teacher) | OOF macro_f1 | **0.7745** | 기준 |
| t3b_cues (teacher) | OOF macro_f1 | 0.7693 | **−0.0052** |
| t3b_cueshist (teacher) | OOF macro_f1 | 0.7666 | −0.0079 |
| kd_cues_stu (student=cues, target=w6) | **LB** | **0.782716** | w6 0.787396 대비 **−0.0047** |

**OOF·LB 둘 다 하락.** (주의: 초기에 "+소폭"이라 오기록된 적 있으나 실제로는 하락. 그 +0.006은
`rich` 프리셋의 *direct-training* 값이었고 그것도 KD에선 무익이었음 — 별개 사안.)

### 왜 실패했나 (분석)
1. **애매성은 입력이 아니라 라벨 생성 과정에 있음.** 시뮬레이터가 애매 맥락에서 4개를 랜덤에
   가깝게 뽑으므로, 아무리 좋은 입력 단서도 **정답을 결정하지 못함**. 단서는 train의
   spurious correlation을 학습시켜 **과적합/오도**.
2. **정규식 단서가 서로 겹침.** 한 prompt가 read·search·list 정규식에 동시 매칭(예: "README
   목록 좀 보여줘")되는 경우가 많아, 노이즈 피처로 작용.
3. **중복성.** 모델은 이미 raw prompt에서 이 표층 단서를 추출함. 명시 토큰은 잉여이거나,
   길이만 늘려 실제 신호(current_prompt, last_action, open_files=0)를 희석.

### 교훈
> **탐색도구 애매성에 대한 hand-crafted 판별 피처는 (이 데이터에선) 역효과.**
> 병목이 aleatoric라, "정보를 더 준다"가 아니라 "노이즈를 더 준다"가 됨.

---

## 2. 관련 직렬화 실험 (참고)

`SERIALIZE_PRESETS`: base / paths(open_files basename) / hist(긴 history) / rich(paths+hist).
- **direct training**: base 0.76 < paths 0.7646 < hist 0.7652 < rich 0.7658 (단조↑, rich +0.006)
- **KD context**: rich 무익 — teacher soft-label이 이미 그 정보를 담아 student에 rich 입력 잉여.
  오히려 teacher를 rich로 학습하면 하락(0.7745→0.7676).
- **결론**: teacher·student 모두 `base`가 정답. 입력 풍부화는 KD에서 안 통함.

---

## 3. 더 해볼 만한 것 (냉정한 기대값)

병목이 aleatoric라 **대부분 기대값 낮음.** 그래도 미탐색 각도:

| 아이디어 | 내용 | 예상 | 리스크 |
|---|---|---|---|
| **깔끔한 단일 단서** | 겹치는 정규식 대신 배타적 신호 1~2개만(예: 와일드카드 유무→glob, dir 키워드→list) | 낮음 | CUES가 이미 하락 |
| **history 액션 시퀀스** | 최근 N개 action type을 명시 토큰(정책 패턴 포착) | 낮음~중 | 길이↑ |
| **open_files 구조** | 열린 파일 확장자/개수/경로깊이 (open_files=0은 이미 최강신호로 [META]에 있음) | 낮음 | 대부분 base에 존재 |
| **prompt 임베딩 클러스터** | prompt를 사전 임베딩→클러스터 id를 피처로 | 낮음 | 복잡·과적합 |
| **teacher 불확실성 피처** | teacher 엔트로피/top2 gap을 student에 힌트 | 낮음 | 순환적 |

**공통 한계**: 어떤 피처든 "애매 맥락에서 정답이 랜덤"이라는 벽을 못 넘음.
피처가 도움되려면 **약하지만 실재하는 판별 신호**가 있어야 하는데, CUES 실패가
"그런 신호가 거의 없다"는 증거.

## 4. 실측이 말하는 진짜 레버 (feature 아님)
- ✗ 크기 키우기 (14B student 0.7732 < 3B 0.7752)
- ✗ threshold/bias 튜닝 (student −0.0019~−0.0030)
- ✗ DKD (−0.0114)
- ✗ feature engineering (CUES −0.005)
- ✓ **이질(계열 다양성) 앙상블** (w6 0.787 = 4×3B+GPT5×2). **유일하게 통함.**

## 5. ⚠️ 방법론 노트 — OOF vs LB
- 로컬 OOF(fold0 val)와 LB(숨김 test 30k)는 **대체로 일치**(train==test 같은 시뮬레이터).
- 단 **작은 차이(±0.001~0.003)는 OOF↔LB 순위가 뒤집힐 수 있음** → 최종 판단은 LB로.
- CUES는 OOF에서 이미 하락이라 LB 볼 것도 없이 기각됐어야 했음(실제 LB도 하락 확인).
- **교훈: 새 아이디어는 먼저 OOF로 거르고, OOF에서 이득 있을 때만 LB 슬롯 사용.**
