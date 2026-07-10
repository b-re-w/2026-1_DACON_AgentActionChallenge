# Teacher Soft-Label Stores (재사용 가능 · 서버 간 공유)

KD teacher들의 soft-label(logit)을 **record id 키**로 저장한 `.npz` 파일들.
14-class라 파일당 **~6.6MB** (70000 × 14 logits + ids). transformers 버전·머신 무관하게 재사용 가능.

- **위치**: `runs/_teacher_logits/{tag}.npz` (+ `{tag}.json` 메타)
- **쓰는 법**: `distill.py --teacher-logits <tag> [<tag>:<weight> ...]` → teacher forward **0회**로 임의 조합 앙상블
- **가중**: `tag:2` = 평균에서 2배 (예: `qw6:5 q14b:1` = (5·qw6 + 1·q14b)/6)
- **다른 서버에서 새로 만들면**: `.npz`만 이 폴더에 넣으면 됨 (id만 train.jsonl과 일치하면 호환)

---

## store 일람

### ① 베이스 — Qwen2.5-3B full-FT × 4시드 (승리레시피 w6의 뼈대)
| tag | 모델 | 설명 | ml | 단일 |
|---|---|---|---|---|
| `q3bb` | Qwen2.5-3B (seed 42) | 3B full-FT 기본 | 640 | ~0.764 |
| `q3b43` | Qwen2.5-3B (seed 43) | 같은 3B, 시드 43 | 640 | ~0.764 |
| `q3b44` | Qwen2.5-3B (seed 44) | 같은 3B, 시드 44 | 640 | ~0.764 |
| `q3b45` | Qwen2.5-3B (seed 45) | 같은 3B, 시드 45 | 640 | ~0.764 |

→ 같은 모델 seed 앙상블. 서로 상관 높음(같은 계열).

### ② 다양성 재료 — 다른 계열 distill (★ 다양성의 원천)
| tag | 모델 | 설명 | ml | 단일 |
|---|---|---|---|---|
| `qgptoss` | Qwen3-4B-gpt-oss-distill | **gpt-oss** 출력으로 증류된 공개모델 | 640 | 0.7810 |
| `qgpt5` | GPT-5-Qwen3-4B | **GPT-5** 출력으로 증류된 공개모델 | 640 | 0.7741 |

→ Qwen3-4B 베이스(3B와 다른 계열) → 다양성 기여. 단 둘은 서로 같은 Qwen3-4B라 상관 있음.

### ③ 큰 Qwen — 같은 계열, 크기만 큼 (QLoRA)
| tag | 모델 | 설명 | ml | 단일 |
|---|---|---|---|---|
| `q3b` | Qwen2.5-3B (QLoRA) | 크기사다리용 (full-FT q3bb와 별개) | 640 | 0.7639 |
| `q14b` | Qwen2.5-14B (QLoRA) | 3B와 같은 계열, 14B | 640 | 0.7651 |
| `q32b` | Qwen2.5-32B (QLoRA) | 같은 계열, 32B | 640 | 0.7675 |

→ 같은 Qwen2.5라 다양성 약함. **크기↑ 무익** 확인됨(단일·앙상블 둘 다).

### ④ 뭉친 승리 레시피
| tag | 구성 | 설명 | ml | student |
|---|---|---|---|---|
| `qw6` | (4×3B + GPT5×2) 평균 | ①4개 + ②qgpt5×2 를 하나로 뭉침 | 512 | **0.7847 → LB 0.790490** |

→ `--teacher-logits qw6` 하나로 w6 앙상블 전체를 부름. **현재 팀 최고 레시피**.
→ **주의**: qw6은 teacher 추론이 512로 계산됨. `qw6_all640` = qw6 타깃 + student max_length **640** 재학습 = LB 0.7905.

---

## 핵심 결론 (2026-07-11 기준)
- **student 640 학습이 최대 이득**(+0.003 LB). teacher를 640으로 **재추론**하면 오히려 손해(외삽, `w6r_g20`=0.7816 < qw6 0.7847).
- **같은 Qwen 계열(3B seed·14B·32B) 추가는 희석** — 다양성 병목. **비-Qwen(gemma/GLM/72B/Phi-4) teacher가 진짜 돌파구**(학습 중).
- gpt-oss가 640-teacher서 GPT5보다 우위(`w6r_oss20`=0.7847 > `w6r_g20`=0.7816) → **gpt-oss@512** 검증 중.

## 다른 서버에서 non-Qwen teacher 만들기
1. `git pull` (이 stores + 코드)
2. QLoRA 학습: `train.py --model <hf_id> --qlora --serialize base --max-length 640 ...` (Gemma는 `--attn-implementation eager`)
3. soft-label 저장: `gen_softlabels.py --teacher-dir runs/<model>/model --tag <tag> --max-length 640`
4. 생성된 `runs/_teacher_logits/<tag>.npz` 를 커밋 → 메인 서버에서 `--teacher-logits qw6 <tag>:1` 로 앙상블
