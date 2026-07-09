# 🤝 친구용 가이드 — Gemma & Llama teacher 학습 (PRO 6000 서버)

이 문서만 따라 하면 됩니다. **명령어를 위에서부터 순서대로 복사해서 붙여넣기** 하세요.
막히면 어디서 막혔는지(어떤 명령어에서 무슨 메시지가 나왔는지)만 캡처해서 보내주면 됩니다.

목표: Gemma-2-27B → Llama-3.3-70B 를 학습해서 결과 파일(`.npz`) 2개를 만들어 돌려주는 것.
GPU 하나로 **차례대로** 돌기 때문에 오래 걸립니다(대략 **2~4일**). 중간에 컴퓨터/터미널이
꺼지지 않게만 해주세요(아래 `nohup` 방법 참고).

---

## 0. 미리 받아둘 것 (메인 담당자에게 요청)
- **HF 토큰** 하나 (`hf_` 로 시작하는 문자열)
- **데이터 파일 2개**: `train.jsonl`, `train_labels.csv`

## 1. 코드 내려받기
```bash
git clone https://github.com/b-re-w/agent_action.git
cd agent_action
git checkout big_model_lim
uv sync --extra dev
```
> `uv` 가 없다고 나오면 먼저 설치: `curl -LsSf https://astral.sh/uv/install.sh | sh` 후 터미널 새로 열기.

## 2. 데이터 넣기
받은 `train.jsonl`, `train_labels.csv` 를 `agent_action/data/` 폴더 안에 넣습니다.
```bash
mkdir -p data
# 받은 파일을 data/ 로 복사 (파일 위치는 상황에 맞게)
# 예: cp ~/받은폴더/train.jsonl ~/받은폴더/train_labels.csv data/
```

## 3. HF 설정 (2줄 — 큰 따옴표 안 값만 바꾸세요)
```bash
export HF_HOME="$HOME/hf"                 # 모델이 저장될 곳. 디스크 여유 많은 경로로.
export HF_TOKEN="hf_여기에_받은_토큰"        # 메인 담당자에게 받은 토큰
```
> ⚠️ Llama-70B 는 다운로드가 ~140GB 입니다. `HF_HOME` 은 **여유 200GB 이상** 되는 디스크로 정하세요.
> 매번 새 터미널에서 다시 export 하기 귀찮으면, 위 2줄을 `~/.bashrc` 맨 아래에 추가해 두세요.

## 4. 준비 상태 점검 (문제 미리 찾기)
```bash
bash scripts/preflight_hetero.sh
```
- 전부 ✅ 이고 마지막에 **"준비 완료!"** 가 뜨면 다음 단계로.
- ❌ 가 있으면 그 줄에 적힌 안내대로 고친 뒤 다시 실행하세요.
- `bitsandbytes` 관련 ❌ 가 뜨면 (PRO 6000 은 최신 GPU라 종종 그럴 수 있음):
  ```bash
  uv pip install -U bitsandbytes
  ```
  후 다시 점검. 그래도 안 되면 그 메시지를 캡처해서 보내주세요.
- Llama 가 **"접근 불가"(⚠️)** 로 나오면, 아직 라이선스 승인 대기중일 수 있습니다.
  Gemma 만 먼저 시작해도 됩니다(6-B 참고).

## 5. 라이선스 (이미 처리돼 있을 수 있음)
Gemma/Llama 는 HF 계정으로 **한 번 라이선스 수락**을 해야 받아집니다.
보통 메인 담당자가 미리 해둡니다. 4번 점검에서 둘 다 "접근 가능 ✅" 면 넘어가세요.

## 6. 실행 🚀

### 6-A. Gemma → Llama 둘 다 (기본, 권장)
컴퓨터를 며칠 켜둘 수 있으면 이 한 줄이면 됩니다:
```bash
nohup bash scripts/run_hetero_all.sh > logs/run_all.log 2>&1 &
```
- `nohup ... &` 덕분에 터미널을 닫아도 계속 돕니다.
- 시작하면 자동으로 점검 → Gemma 학습 → Llama 학습 순으로 진행됩니다.

### 6-B. (Llama 승인 대기 등으로) Gemma 만 먼저
```bash
nohup bash scripts/run_hetero_all.sh gemma > logs/run_all.log 2>&1 &
```
나중에 Llama 승인되면:
```bash
nohup bash scripts/run_hetero_all.sh llama > logs/run_llama.log 2>&1 &
```

## 7. 진행 상황 보기
```bash
tail -f logs/run_all.log
```
- 숫자 막대(예: `1500/8750`)가 조금씩 올라가면 정상입니다. (보는 걸 멈추려면 `Ctrl+C` — 학습은 안 멈춥니다.)
- GPU 가 실제로 일하는지 보고 싶으면 다른 터미널에서: `nvidia-smi` (사용률이 높으면 정상).
- 개별 모델 로그는 `logs/qgemma27b.log`, `logs/qllama70b.log` 에도 쌓입니다.

## 8. 끝났는지 확인 & 결과 전달
다 끝나면 아래에 파일이 생깁니다:
```bash
ls -la runs/_teacher_logits/
# qgemma27b.npz / qgemma27b.json / qllama70b.npz / qllama70b.json
```
이 **`.npz` 와 `.json` 파일들(개당 ~7MB)** 을 메인 담당자에게 전달하면 끝입니다.
scp 를 쓸 수 있으면:
```bash
scp runs/_teacher_logits/qgemma27b.{npz,json} 담당자서버:/data/agent_action/runs/_teacher_logits/
```
scp 가 어려우면 그냥 이 파일들을 압축해서(드라이브/메신저 등으로) 보내도 됩니다.

---

## ❓ 자주 나오는 문제
| 증상 | 해결 |
|---|---|
| `uv: command not found` | 1번의 uv 설치 후 터미널 새로 열기 |
| `bitsandbytes` 오류 / CUDA 오류 | `uv pip install -U bitsandbytes` (필요시 torch 도) 후 점검 재실행 |
| Llama `접근 불가` / 403 | HF 라이선스 승인 대기중 — Gemma 먼저(6-B), 승인되면 Llama 실행 |
| `HF_TOKEN 미설정` | 3번 export 를 다시 실행 (새 터미널에선 매번 필요) |
| 디스크 부족 | `HF_HOME` 을 더 큰 디스크로 바꾸고 다시 실행 |
| 중간에 멈춤/에러 | 다시 `run_hetero_all.sh` 실행 — 이미 끝난 모델은 자동으로 건너뜁니다 |

문제가 계속되면 **막힌 명령어 + 화면 메시지 마지막 20줄**을 캡처해서 메인 담당자에게 보내주세요.
