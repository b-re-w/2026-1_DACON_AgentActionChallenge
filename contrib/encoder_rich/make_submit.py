# -*- coding: utf-8 -*-
"""저장된 실험(runs/<name>/model/) 을 제출 zip 으로 패키징.

사용:
  python experiments/make_submit.py exp07_xlmr_rich512_ce
  python experiments/make_submit.py --list        # 제출가능(저장된) 실험 목록

동작:
  runs/<name>/model/  +  files/script.py(feature-aware)  +  submit/requirements.txt
    → submit_<name>/  구성 → submit_<name>.zip (top-level: model/, script.py, requirements.txt)
  · infer_config.json 의 features/max_len 을 script.py 가 읽어 학습과 동일 추론
  · 1GB 초과·구조 이상이면 경고
"""
import os, sys, glob, json, shutil, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
RUNS = os.path.join(HERE, "runs")
SCRIPT_SRC = os.path.join(PROJ, "files", "script.py")
REQ_SRC = os.path.join(PROJ, "submit", "requirements.txt")
LIMIT = 1024 ** 3  # 1GB


def list_saved():
    out = []
    for d in sorted(glob.glob(os.path.join(RUNS, "*"))):
        if os.path.exists(os.path.join(d, "model", "config.json")):
            out.append(os.path.basename(d))
    return out


def make(name):
    model_dir = os.path.join(RUNS, name, "model")
    if not os.path.exists(os.path.join(model_dir, "config.json")):
        # 부분일치 허용
        cand = [n for n in list_saved() if name in n]
        if len(cand) == 1:
            name, model_dir = cand[0], os.path.join(RUNS, cand[0], "model")
        else:
            print(f"저장된 모델 없음: {name}. 제출가능 목록: {list_saved()}"); return

    ic = json.load(open(os.path.join(model_dir, "infer_config.json"))) \
        if os.path.exists(os.path.join(model_dir, "infer_config.json")) else {}
    print(f"패키징: {name}  (features={ic.get('features','full')}, max_len={ic.get('max_len','?')})")

    out_dir = os.path.join(PROJ, f"submit_{name}")
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "model"))
    for f in os.listdir(model_dir):
        shutil.copy(os.path.join(model_dir, f), os.path.join(out_dir, "model", f))
    shutil.copy(SCRIPT_SRC, os.path.join(out_dir, "script.py"))
    shutil.copy(REQ_SRC, os.path.join(out_dir, "requirements.txt"))

    zip_path = os.path.join(PROJ, f"submit_{name}.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, fs in os.walk(out_dir):
            for f in fs:
                p = os.path.join(root, f)
                z.write(p, os.path.relpath(p, out_dir))

    size = os.path.getsize(zip_path)
    tops = {n.split("/")[0] for n in zipfile.ZipFile(zip_path).namelist()}
    ok_struct = tops == {"model", "script.py", "requirements.txt"}
    print(f"  → {zip_path}")
    print(f"  크기: {size/1e6:.0f} MB  ({'✅ ≤1GB' if size <= LIMIT else '❌ 1GB 초과!'})")
    print(f"  top-level: {sorted(tops)}  ({'✅ 구조 정상' if ok_struct else '❌ 구조 이상'})")
    if size > LIMIT or not ok_struct:
        print("  ⚠️ 위 문제 해결 전 제출 금지 (하루 제출 기회 소모 방지)")


def main():
    if "--list" in sys.argv:
        print("제출가능(저장된) 실험:", list_saved() or "(없음)")
        return
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        print("사용: python experiments/make_submit.py <run이름>  |  --list")
        print("제출가능:", list_saved() or "(없음)")
        return
    for n in args:
        make(n)


if __name__ == "__main__":
    main()
