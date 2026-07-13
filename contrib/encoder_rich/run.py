# -*- coding: utf-8 -*-
"""순차 실험 러너 (GPU 경합 방지) — config 들을 하나씩 실행하고 로그를 저장.

사용:
  python experiments/run.py                       # configs/ 의 모든 *.json 순차 실행
  python experiments/run.py exp02 exp04           # 이름(부분일치)만 골라 실행
  python experiments/run.py --list                # 실험 목록/현재 리더보드만 출력
각 실험 로그는 experiments/runs/<name>/train.log 에 저장된다.
"""
import os, sys, glob, subprocess, time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "configs")
RUNS = os.path.join(HERE, "runs")
PY = os.path.join(os.path.dirname(HERE), ".venv", "Scripts", "python.exe")
PROJ = os.path.dirname(HERE)


def show_leaderboard():
    lb = os.path.join(HERE, "leaderboard.csv")
    if not os.path.exists(lb):
        print("(아직 leaderboard.csv 없음)"); return
    print("\n===== leaderboard.csv =====")
    print(open(lb, encoding="utf-8").read())


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if "--list" in sys.argv:
        for c in sorted(glob.glob(os.path.join(CFG_DIR, "*.json"))):
            print(" -", os.path.basename(c))
        show_leaderboard(); return

    cfgs = sorted(glob.glob(os.path.join(CFG_DIR, "*.json")))
    if args:  # 이름 부분일치 필터
        cfgs = [c for c in cfgs if any(a in os.path.basename(c) for a in args)]
    if not cfgs:
        print("실행할 config 없음"); return

    print(f"순차 실행 예정 ({len(cfgs)}개):")
    for c in cfgs:
        print("  -", os.path.basename(c))

    for i, c in enumerate(cfgs, 1):
        name = os.path.splitext(os.path.basename(c))[0]
        os.makedirs(os.path.join(RUNS, name), exist_ok=True)
        log_path = os.path.join(RUNS, name, "train.log")
        print(f"\n[{i}/{len(cfgs)}] ▶ {name}  (log: runs/{name}/train.log)", flush=True)
        t0 = time.time()
        env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        with open(log_path, "w", encoding="utf-8") as logf:
            proc = subprocess.Popen([PY, os.path.join(HERE, "train.py"), c],
                                    cwd=PROJ, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=env, text=True, encoding="utf-8", errors="replace", bufsize=1)
            for line in proc.stdout:
                sys.stdout.write(line); logf.write(line); logf.flush()
            proc.wait()
        status = "OK" if proc.returncode == 0 else f"FAIL(code={proc.returncode})"
        print(f"[{i}/{len(cfgs)}] {status}  {name}  ({time.time()-t0:.0f}s)", flush=True)

    show_leaderboard()


if __name__ == "__main__":
    main()
