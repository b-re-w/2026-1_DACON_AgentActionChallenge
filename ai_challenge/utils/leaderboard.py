"""DACON 236694 퍼블릭 리더보드 조회.

로그인 없이 리더보드 페이지 HTML의 `window.__NUXT__`(SSR 임베디드 상태)에서
순위 데이터를 추출한다. Nuxt 페이로드는 minify 된 함수 래핑 형태라 정규식 대신
Node(vm 샌드박스)로 안전하게 평가해 JSON 으로 뽑는다.

사용:
    python -m ai_challenge.utils.leaderboard --top 20 --team "우리팀명"
또는 import 해서 get_leaderboard() / find_team() 사용.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import urllib.request
from dataclasses import dataclass

COMP_ID = 236694
LEADERBOARD_URL = f"https://dacon.io/competitions/official/{COMP_ID}/leaderboard"
# 실제 브라우저(Chrome)와 동일한 완전한 User-Agent — 파이썬 클라이언트로 식별되지 않도록.
_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36"
    )
}

# Node: stdin 으로 받은 `window.__NUXT__` 원본을 vm 샌드박스에서 평가하고,
# team_name+score 를 가진 가장 큰 배열(=리더보드)을 찾아 JSON 으로 출력한다.
_NODE_EVAL = r"""
const vm = require("vm");
let chunks = [];
process.stdin.on("data", d => chunks.push(d));
process.stdin.on("end", () => {
  const raw = Buffer.concat(chunks).toString("utf8");
  const sandbox = { window: {} };
  vm.createContext(sandbox);
  vm.runInContext("window.__NUXT__=" + raw, sandbox, { timeout: 8000 });
  const state = sandbox.window.__NUXT__;
  let best = [];
  (function walk(o, depth) {
    if (!o || typeof o !== "object" || depth > 8) return;
    if (Array.isArray(o)) {
      if (o.length && o[0] && o[0].team_name !== undefined && o[0].score !== undefined) {
        if (o.length > best.length) best = o;
      }
      for (const v of o) walk(v, depth + 1);
    } else {
      for (const k of Object.keys(o)) walk(o[k], depth + 1);
    }
  })(state, 0);
  process.stdout.write(JSON.stringify(best));
});
"""


@dataclass
class LBRow:
    ranking: int
    team_name: str
    score: float
    submission_cnt: int
    c_time: str
    team_id: int

    @classmethod
    def from_raw(cls, d: dict) -> "LBRow":
        return cls(
            ranking=int(d.get("ranking", 0)),
            team_name=str(d.get("team_name", "")),
            score=float(d.get("score", 0.0)),
            submission_cnt=int(d.get("submission_cnt", 0)),
            c_time=str(d.get("c_time", "")),
            team_id=int(d.get("team_id", 0)),
        )


def fetch_html(url: str = LEADERBOARD_URL, timeout: int = 20) -> str:
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def extract_nuxt_raw(html: str) -> str:
    """HTML 에서 `window.__NUXT__=...;</script>` 원본 표현식을 추출."""
    m = re.search(r"window\.__NUXT__\s*=\s*(.*?);?\s*</script>", html, re.S)
    if not m:
        raise RuntimeError("window.__NUXT__ 를 찾지 못했습니다 (페이지 구조 변경?).")
    return m.group(1)


def _node_eval(raw_nuxt: str) -> list[dict]:
    """Node vm 샌드박스로 Nuxt 페이로드를 평가해 리더보드 배열을 반환."""
    try:
        proc = subprocess.run(
            ["node", "-e", _NODE_EVAL],
            input=raw_nuxt.encode("utf-8"),
            capture_output=True,
            timeout=30,
        )
    except FileNotFoundError as e:
        raise RuntimeError("node 실행 파일이 필요합니다 (Node.js 설치 확인).") from e
    if proc.returncode != 0:
        raise RuntimeError("Nuxt 평가 실패: " + proc.stderr.decode("utf-8", "replace")[:500])
    return json.loads(proc.stdout.decode("utf-8", "replace") or "[]")


def get_leaderboard(url: str = LEADERBOARD_URL) -> list[LBRow]:
    """퍼블릭 리더보드를 순위 오름차순 LBRow 리스트로 반환."""
    rows = [LBRow.from_raw(d) for d in _node_eval(extract_nuxt_raw(fetch_html(url)))]
    rows.sort(key=lambda r: r.ranking)
    return rows


def find_team(rows: list[LBRow], team_name: str) -> LBRow | None:
    """팀명(대소문자 무시, 정확일치 우선 후 부분일치)으로 행 검색."""
    key = team_name.strip().lower()
    for r in rows:
        if r.team_name.lower() == key:
            return r
    for r in rows:  # 부분일치 fallback
        if key in r.team_name.lower():
            return r
    return None


def format_table(rows: list[LBRow], top: int | None = None, highlight: str | None = None) -> str:
    shown = rows[:top] if top else rows
    lines = [f"{'#':>4}  {'score':>12}  {'subs':>4}  {'c_time':<19}  team"]
    lines.append("-" * 72)
    hl = highlight.strip().lower() if highlight else None
    for r in shown:
        mark = " *" if hl and hl in r.team_name.lower() else "  "
        lines.append(
            f"{r.ranking:>4}  {r.score:>12.8f}  {r.submission_cnt:>4}  {r.c_time:<19} {mark}{r.team_name}"
        )
    return "\n".join(lines)


def main() -> None:
    # Windows 콘솔(cp949)에서 한글 팀명이 깨지지 않도록 UTF-8 출력 고정
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="DACON 236694 퍼블릭 리더보드 조회")
    ap.add_argument("--top", type=int, default=20, help="상위 N개 출력 (0=전체)")
    ap.add_argument("--team", type=str, default=None, help="이 팀의 순위를 함께 표시")
    args = ap.parse_args()

    rows = get_leaderboard()
    print(f"총 {len(rows)}팀 · 1위 {rows[0].team_name} {rows[0].score:.8f}\n")
    print(format_table(rows, top=(args.top or None), highlight=args.team))

    if args.team:
        me = find_team(rows, args.team)
        if me:
            print(f"\n[내 팀] {me.team_name}: {me.ranking}위 / {len(rows)}팀 "
                  f"· score {me.score:.8f} · 제출 {me.submission_cnt}회")
        else:
            print(f"\n[내 팀] '{args.team}' 를 상위 {len(rows)}팀에서 찾지 못했습니다.")


if __name__ == "__main__":
    main()
