"""DACON 236694 제출 & 내 제출기록 조회.

인증: 최초 1회 브라우저(Playwright)로 로그인하면, newapi 요청의 Bearer 토큰과
submission_list 요청 URL을 그대로 가로채(추측 제거) `.env`(토큰) 및
`.dacon_auth.json`(세션 storage_state + 캡처된 엔드포인트)에 저장한다. 이후엔 저장된
토큰으로 재사용한다.

읽기(내 기록)는 requests 로, 실제 제출은 저장된 세션의 Playwright 로 DACON UI 를
그대로 구동해 처리한다(파일 업로드가 tus 프로토콜이라 DACON 자체 로직을 재사용).

CLI:
    python -m ai_challenge.utils.submission login
    python -m ai_challenge.utils.submission list [--page 1]
    python -m ai_challenge.utils.submission submit <zip> --memo "설명" --yes
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# 프로젝트 루트(= .env 위치) 기준 경로
_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = _ROOT / ".env"
AUTH_PATH = _ROOT / ".dacon_auth.json"

COMP_ID = 236694
BASE_API = "https://newapi.dacon.io"
MYSUB_URL = f"https://dacon.io/competitions/official/{COMP_ID}/mysubmission"
DEFAULT_LIST_URL = f"{BASE_API}/competition/submission_list?page=1&data_type=1&cpt_id={COMP_ID}"
# 실제 브라우저와 동일한 완전한 User-Agent (로그인 시 실제 UA 로 덮어씀).
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Whale/4.38.386.14 Safari/537.36"
)
MAX_ZIP_BYTES = 1024 * 1024 * 1024  # 제출 용량 한도 1GB


# --------------------------------------------------------------------------- #
# .env / 인증 상태 입출력
# --------------------------------------------------------------------------- #
def load_env(path: Path = ENV_PATH) -> dict[str, str]:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def set_env_var(key: str, value: str, path: Path = ENV_PATH) -> None:
    """.env 의 key=value 를 갱신(없으면 추가). 주석/다른 줄은 보존."""
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    out, done = [], False
    for line in lines:
        if line.strip().startswith(f"{key}=") or line.strip().startswith(f"{key} ="):
            out.append(f"{key}={value}")
            done = True
        else:
            out.append(line)
    if not done:
        out.append(f"{key}={value}")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def load_auth() -> dict:
    return json.loads(AUTH_PATH.read_text(encoding="utf-8")) if AUTH_PATH.exists() else {}


def _auth_headers() -> dict:
    """DACON newapi 요청 헤더 구성.

    인증은 표준 Authorization/Cookie 가 아니라 **커스텀 `token` 헤더**로 JWT 를 전달한다
    (프론트엔드 getAxios 가 `headers:{token: jwt}` 로 붙임).
    """
    auth = load_auth()
    token = load_env().get("DACON_TOKEN") or auth.get("token")
    if not token:  # 쿠키로 로그인한 세션이면 storage_state 의 'token' 쿠키에서 회수
        for c in (auth.get("storage_state") or {}).get("cookies", []):
            if c.get("name") == "token":
                token = c.get("value")
                break
    if not token:
        raise RuntimeError("인증 토큰이 없습니다. `.env` 의 DACON_TOKEN 을 설정하세요.")
    return {
        "User-Agent": auth.get("user_agent") or DEFAULT_UA,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://dacon.io",
        "Referer": "https://dacon.io/",
        "token": token,  # newapi 커스텀 인증 헤더
    }


# --------------------------------------------------------------------------- #
# 로그인 (Playwright, 최초 1회) — 토큰 & 엔드포인트 캡처
# --------------------------------------------------------------------------- #
def _looks_like_jwt(value: str) -> bool:
    """JWT 형태(점 2개로 구분된 긴 토큰) 판별 — 로그인 감지용."""
    return isinstance(value, str) and value.count(".") == 2 and len(value) > 80 and " " not in value


def login(timeout_sec: int = 300) -> None:
    """브라우저를 띄워 로그인시키고, 세션(쿠키)·Bearer 토큰·실제 User-Agent·
    submission_list URL 을 캡처·저장한다.

    로그인 감지는 쿠키/localStorage 에 JWT 형태 토큰이 나타나는지로 판단한다
    (응답 본문을 이벤트 콜백에서 읽지 않아 연결이 끊기지 않도록).
    인증 방식(Bearer/쿠키)에 무관하게 동작한다.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError("playwright 가 필요합니다:  uv run playwright install chromium") from e

    cap = {"token": None, "list_url": None}

    def on_request(req):  # 헤더만 읽음(본문 X) → 안전
        if "newapi.dacon.io" in req.url:
            tok = req.headers.get("token", "")  # newapi 커스텀 인증 헤더
            if _looks_like_jwt(tok):
                cap["token"] = tok
            auth = req.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                cap["token"] = auth.split(" ", 1)[1]
            if "submission_list" in req.url:
                cap["list_url"] = req.url

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        ctx = browser.new_context()
        ctx.on("request", on_request)  # 컨텍스트 레벨: 팝업/새 탭도 포함
        page = ctx.new_page()
        user_agent = page.evaluate("() => navigator.userAgent")  # 실제 브라우저 UA 캡처

        print(">> 로그인 창에서 DACON 에 로그인해 주세요.")
        print("   (완료되면 자동 진행 · 창을 직접 닫지 마세요)")
        page.goto(MYSUB_URL)

        deadline = time.time() + timeout_sec
        logged_in = False
        while time.time() < deadline and not logged_in:
            time.sleep(1.0)
            try:
                cookies = ctx.cookies()  # Playwright 호출 → 대기 중 이벤트도 함께 처리
            except Exception:
                break  # 컨텍스트/브라우저가 닫힘
            if cap["token"] or any(_looks_like_jwt(c.get("value", "")) for c in cookies):
                logged_in = True
                break
            try:  # localStorage 의 JWT 도 확인(bearer-in-localStorage 대비)
                vals = page.evaluate("() => Object.values(window.localStorage)")
                if any(_looks_like_jwt(str(v)) for v in vals):
                    logged_in = True
                    break
            except Exception:
                pass

        if not logged_in:
            try:
                browser.close()
            except Exception:
                pass
            raise RuntimeError("로그인을 감지하지 못했습니다(시간 초과 또는 창 닫힘). 다시 시도하세요.")

        # 로그인 후 submission_list 요청을 유도해 정확한 URL 캡처
        try:
            page.goto(MYSUB_URL)
            page.wait_for_timeout(2500)
        except Exception:
            pass

        try:
            storage_state = ctx.storage_state()
        finally:
            try:
                browser.close()
            except Exception:
                pass

    auth_mode = "bearer" if cap["token"] else "cookie"
    AUTH_PATH.write_text(
        json.dumps(
            {
                "token": cap["token"],
                "auth_mode": auth_mode,
                "user_agent": user_agent,
                "list_url": cap["list_url"] or DEFAULT_LIST_URL,
                "storage_state": storage_state,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    if cap["token"]:
        set_env_var("DACON_TOKEN", cap["token"])
    n_cookies = len(storage_state.get("cookies", []))
    print(f"[OK] 인증 저장 (mode={auth_mode}, 쿠키 {n_cookies}개, UA 캡처) → {AUTH_PATH.name}")

    # requests 로 즉시 검증(브라우저 밖, 안전)
    try:
        data = get_my_submissions(1)
        ok = "ClientError" not in json.dumps(data, ensure_ascii=False)
        print("[검증] 제출목록 조회:", "성공 ✅" if ok else "실패(ClientError) — 재로그인 필요")
    except Exception as e:
        print("[검증] 제출목록 조회 실패:", str(e)[:150])


# --------------------------------------------------------------------------- #
# 내 제출기록 조회 (requests, 읽기전용)
# --------------------------------------------------------------------------- #
def get_my_submissions(page: int = 1) -> dict:
    """내 제출 목록(JSON)을 반환. 저장된 인증(UA + 쿠키/토큰)과 캡처 URL 사용."""
    import re

    list_url = load_auth().get("list_url") or DEFAULT_LIST_URL
    if "page=" in list_url:  # 캡처된 URL 의 page 파라미터를 요청 page 로 교체
        url = re.sub(r"page=\d+", f"page={page}", list_url)
    else:
        sep = "&" if "?" in list_url else "?"
        url = f"{list_url}{sep}page={page}"

    req = urllib.request.Request(url, headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def print_my_submissions(page: int = 1) -> None:
    data = get_my_submissions(page)
    # 응답 안에서 리스트를 찾아 표로 출력(스키마가 바뀌어도 견고하게)
    rows = None
    if isinstance(data, dict):
        for v in data.values():
            if isinstance(v, list):
                rows = v
                break
    elif isinstance(data, list):
        rows = data

    if not rows:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:2000])
        return
    for i, r in enumerate(rows, 1):
        if isinstance(r, dict):
            score = r.get("score") or r.get("public_score") or ""
            memo = r.get("memo") or r.get("submission_memo") or ""
            ctime = r.get("c_time") or r.get("created") or ""
            print(f"{i:>3}. score={score}  time={ctime}  memo={memo}")
        else:
            print(f"{i:>3}. {r}")


# --------------------------------------------------------------------------- #
# 제출 (Playwright, 저장된 세션으로 DACON UI 구동)
# --------------------------------------------------------------------------- #
def submit(zip_path: str | Path, memo: str = "", confirm: bool = False, headless: bool = False) -> None:
    """submit.zip 을 실제 제출. confirm=True 필수(되돌릴 수 없고 일일 한도 소모)."""
    zip_path = Path(zip_path).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(zip_path)
    size = zip_path.stat().st_size
    if size > MAX_ZIP_BYTES:
        raise ValueError(f"용량 초과: {size/1e9:.2f}GB > 1GB 한도")
    if not confirm:
        raise RuntimeError("실제 제출입니다. confirm=True(CLI --yes)로 명시해야 진행합니다.")

    auth = load_auth()
    if not auth.get("storage_state"):
        raise RuntimeError("저장된 세션이 없습니다. 먼저 login 을 실행하세요.")

    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright 가 필요합니다:  pip install playwright && playwright install chromium"
        ) from e

    print(f">> 제출 시작: {zip_path.name} ({size/1e6:.1f}MB) memo='{memo}'")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(storage_state=auth["storage_state"])
        page = ctx.new_page()
        page.goto(MYSUB_URL)
        page.wait_for_load_state("networkidle")

        # 파일 첨부
        page.set_input_files("input[type=file]", str(zip_path))
        # 메모 입력(있으면)
        if memo:
            box = page.query_selector("textarea") or page.query_selector("input[type=text]")
            if box:
                box.fill(memo)
        # 제출 버튼(텍스트 '제출')
        page.get_by_role("button", name="제출").first.click()
        # 업로드/처리 대기 — 성공 표식이 뜰 때까지 여유있게 대기
        page.wait_for_timeout(3000)
        page.wait_for_load_state("networkidle")
        print("[OK] 제출 동작 완료. mysubmission 에서 반영 여부를 확인하세요.")
        page.wait_for_timeout(2000)
        browser.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="DACON 236694 제출/기록 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("login", help="브라우저 로그인 후 토큰 저장(최초 1회)")

    pl = sub.add_parser("list", help="내 제출기록 조회")
    pl.add_argument("--page", type=int, default=1)

    ps = sub.add_parser("submit", help="submit.zip 실제 제출")
    ps.add_argument("zip", help="제출할 zip 경로")
    ps.add_argument("--memo", default="", help="제출 메모")
    ps.add_argument("--yes", action="store_true", help="실제 제출 확인(필수)")
    ps.add_argument("--headless", action="store_true")

    args = ap.parse_args()
    if args.cmd == "login":
        login()
    elif args.cmd == "list":
        print_my_submissions(args.page)
    elif args.cmd == "submit":
        submit(args.zip, memo=args.memo, confirm=args.yes, headless=args.headless)


if __name__ == "__main__":
    main()
