"""DACON 236694 내 제출기록 조회 & (예정) 제출 — 브라우저 없이 API 직접 호출.

인증
----
newapi 는 표준 Authorization/Cookie 가 아니라 **커스텀 `token` 헤더**로 JWT 를 받는다
(프론트 getAxios 가 `headers:{token: jwt}` 로 붙임). 로그인된 브라우저 쿠키의 `token=`
값(JWT)을 수동으로 뽑아 `.env` 의 DACON_TOKEN 에 넣는다. (Google 로그인은 자동화 브라우저를
차단하므로 Playwright 자동 로그인은 쓰지 않는다.)

토큰 2종
--------
- DACON_TOKEN    : 세션 JWT(쿠키 `token`). 제출기록 조회(`list`)에 사용.
- DACON_API_TOKEN: 마이페이지>계정관리의 개인 API 토큰. 제출(`submit`)에 사용.

CLI
---
    python -m ai_challenge.utils.submission token <JWT>            # 세션 JWT 저장 + 검증
    python -m ai_challenge.utils.submission list [--page 1]        # 내 제출기록
    python -m ai_challenge.utils.submission submit <zip> --memo m --yes   # 실제 제출
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = _ROOT / ".env"

COMP_ID = 236694
BASE_API = "https://newapi.dacon.io"
LIST_URL = f"{BASE_API}/competition/submission_list?page=1&data_type=1&cpt_id={COMP_ID}"
# 실제 브라우저와 동일한 완전한 User-Agent (파이썬 클라이언트로 식별되지 않도록).
DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/148.0.0.0 Whale/4.38.386.14 Safari/537.36"
)


# --------------------------------------------------------------------------- #
# .env 입출력
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


def _looks_like_jwt(value: str) -> bool:
    return isinstance(value, str) and value.count(".") == 2 and len(value) > 80 and " " not in value


# --------------------------------------------------------------------------- #
# 인증 헤더 & 토큰 저장
# --------------------------------------------------------------------------- #
def _auth_headers() -> dict:
    """DACON newapi 요청 헤더: 완전한 UA + 커스텀 `token` 헤더(JWT)."""
    token = load_env().get("DACON_TOKEN", "")
    if not token:
        raise RuntimeError("인증 토큰이 없습니다. `... submission token <JWT>` 로 저장하세요.")
    return {
        "User-Agent": DEFAULT_UA,
        "Accept": "application/json, text/plain, */*",
        "Origin": "https://dacon.io",
        "Referer": "https://dacon.io/",
        "token": token,  # newapi 커스텀 인증 헤더
    }


def save_token(token: str) -> None:
    """붙여넣은 JWT 를 .env 에 저장하고 즉시 조회로 검증한다.

    토큰 얻는 법: 로그인된 브라우저에서 F12 → Application → Cookies → dacon.io 의
    `token` 값(또는 Network 의 newapi 요청 헤더 `token`)을 복사.
    """
    token = token.strip()
    if not _looks_like_jwt(token):
        raise ValueError("JWT 형태가 아닙니다(점 2개로 구분된 긴 토큰이어야 함).")
    set_env_var("DACON_TOKEN", token)
    print(f"[OK] .env 에 DACON_TOKEN 저장 → {ENV_PATH}")
    try:
        data = get_my_submissions(1)
        ok = "ClientError" not in json.dumps(data, ensure_ascii=False)
        print("[검증] 제출기록 조회:", "성공 ✅" if ok else "실패(ClientError) — 토큰 확인 필요")
    except Exception as e:
        print("[검증] 실패:", str(e)[:150])


# --------------------------------------------------------------------------- #
# 내 제출기록 조회 (읽기전용)
# --------------------------------------------------------------------------- #
def get_my_submissions(page: int = 1) -> dict:
    """내 제출 목록(JSON dict)을 반환. `{"message":..,"data":[...]}` 형태."""
    url = re.sub(r"page=\d+", f"page={page}", LIST_URL)
    req = urllib.request.Request(url, headers=_auth_headers())
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def print_my_submissions(page: int = 1) -> None:
    data = get_my_submissions(page)
    rows = data.get("data") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        print(json.dumps(data, ensure_ascii=False, indent=2)[:1500])
        return
    for i, r in enumerate(rows, 1):
        score = r.get("score", "")
        ctime = r.get("c_time", "")
        memo = r.get("memo") or r.get("submission_memo") or ""
        fin = " [FINAL]" if r.get("use_final") in (1, "1", True) else ""
        print(f"{i:>3}. score={score}  time={ctime}{fin}  memo={memo}")


# --------------------------------------------------------------------------- #
# 제출 (DACON 공식 코드제출 API 사용)
# --------------------------------------------------------------------------- #
def submit(zip_path: str, memo: str = "", confirm: bool = False) -> dict:
    """submit.zip 을 DACON 공식 코드제출 API 로 제출한다.

    내부적으로 dacon_submit_api.post_code_submission_file 을 호출한다
    (validate → tus 업로드 dus.dacon.io → app.dacon.io code-submission/submit).

    ⚠️ 비가역이며 하루 10회 제출 한도를 소모한다 → confirm=True(CLI --yes) 필수.
    필요: `.env` 의 DACON_API_TOKEN(마이페이지>계정관리 개인 토큰), DACON_TEAM_NAME.
    파일명 제약: .zip, 30자 이내, URL-safe 문자만.
    """
    zp = Path(zip_path)
    name = zp.name
    if not zp.exists():
        raise FileNotFoundError(zp)
    if not name.endswith(".zip"):
        raise ValueError(".zip 파일만 제출 가능합니다.")
    if len(name) > 30:
        raise ValueError("파일명은 30자 이내여야 합니다.")
    if not re.match(r"^[A-Za-z0-9._~-]+$", name):
        raise ValueError("파일명은 URL-safe 문자(A-Za-z0-9._~-)만 사용해야 합니다.")

    env = load_env()
    api_token = env.get("DACON_API_TOKEN", "")
    team = env.get("DACON_TEAM_NAME", "")
    if not api_token:
        raise RuntimeError("DACON_API_TOKEN 이 .env 에 없습니다 (마이페이지>계정관리에서 발급).")
    if not team:
        raise RuntimeError("DACON_TEAM_NAME 이 .env 에 없습니다 (대회 팀 페이지의 팀명).")
    if not confirm:
        raise RuntimeError("실제 제출입니다(비가역·일 10회 한도). CLI --yes 로 확인하세요.")

    from dacon_submit_api import dacon_submit_api

    result = dacon_submit_api.post_code_submission_file(
        str(zp), api_token, str(COMP_ID), team, memo
    )
    print(result)
    return result


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

    pt = sub.add_parser("token", help="브라우저 쿠키에서 뽑은 JWT 를 .env 에 저장")
    pt.add_argument("jwt", help="token 쿠키(JWT) 값")

    pl = sub.add_parser("list", help="내 제출기록 조회")
    pl.add_argument("--page", type=int, default=1)

    ps = sub.add_parser("submit", help="submit.zip 실제 제출(공식 API)")
    ps.add_argument("zip", help="제출할 zip 경로")
    ps.add_argument("--memo", default="", help="제출 메모")
    ps.add_argument("--yes", action="store_true", help="실제 제출 확인(필수)")

    args = ap.parse_args()
    if args.cmd == "token":
        save_token(args.jwt)
    elif args.cmd == "list":
        print_my_submissions(args.page)
    elif args.cmd == "submit":
        submit(args.zip, memo=args.memo, confirm=args.yes)


if __name__ == "__main__":
    main()
