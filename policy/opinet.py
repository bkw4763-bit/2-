# 오피넷 API로 오늘의 전국 평균 판매가(휘발유·경유·실내등유)를 받습니다. 3·4단계 물가 기여도의 분모(판매가 기준값)로 씁니다.
# API: https://www.opinet.co.kr/api/avgAllPrice.do?out=json&code=키  (전국 주유소 평균가격, 하루 300건까지)
#   - 키는 `code` 변수로 보냅니다(`certkey` 로 보내면 빈 목록).
#   - 키는 환경변수 OPINET_KEY 또는 저장소 루트 .env 의 OPINET_KEY 에서 읽습니다. 코드와 커밋에 넣지 않습니다.
# 실패하면(키 없음, 인터넷 끊김, 빈 응답) 마지막 저장본 → 주간 판매가 파일 순서로 대체합니다.
# 단독 실행: python policy/opinet.py
import json
import os
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd

import config as C

URL = "https://www.opinet.co.kr/api/avgAllPrice.do?out=json&code={key}"
PRODCD = {"gasoline": "B027", "diesel": "D047", "kerosene": "C004"}   # 보통휘발유, 자동차용경유, 실내등유
CACHE_PATH = C.PROJECT_DIR / "data" / "opinet_latest.csv"             # 마지막 성공 응답 저장본 (.gitignore 의 data/)




# 1. 키를 환경변수 → .env 순서로 읽습니다. 없으면 None.
def load_key():
    key = os.environ.get("OPINET_KEY", "").strip()
    env = C.PROJECT_DIR / ".env"
    if not key and env.exists():
        for line in env.read_text(encoding="utf-8-sig").splitlines():
            name, sep, value = line.strip().partition("=")
            if sep and name.strip() == "OPINET_KEY":
                key = value.strip().strip("'\"")
    return key or None


# 2. API 응답(JSON)에서 유종별 가격과 기준일을 꺼냅니다. 세 유종이 다 없으면 ValueError.
# 응답 모양: {"RESULT": {"OIL": [{"TRADE_DT": "20261010", "PRODCD": "B027", "PRICE": "1857.29", ...}, ...]}}
def parse(payload):
    rows = {r.get("PRODCD"): r for r in payload.get("RESULT", {}).get("OIL", [])}
    missing = [code for code in PRODCD.values() if code not in rows]
    if missing:
        raise ValueError(f"응답에 유종 코드가 없습니다: {missing}")
    prices = {f: float(rows[code]["PRICE"]) for f, code in PRODCD.items()}
    return prices, rows[PRODCD["gasoline"]].get("TRADE_DT", "")


# 3. API를 부르고 성공하면 저장본을 갱신합니다. 반환: (가격 사전, 기준일, 출처)
def fetch(timeout=10):
    key = load_key()
    if key is None:
        raise ValueError("OPINET_KEY 가 없습니다(.env 확인).")
    with urlopen(URL.format(key=key), timeout=timeout) as resp:
        prices, day = parse(json.load(resp))
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"trade_dt": day, "fetched_at": datetime.now().isoformat(timespec="seconds"), **prices}]).to_csv(
        CACHE_PATH, index=False, encoding="utf-8-sig")
    return prices, day, "오피넷 API"


# 4. 판매가 기준값을 돌려줍니다. API → 저장본 → 주간 판매가 파일 순서로 시도합니다.
# 반환: (가격 사전, 기준일, 출처). 출처는 화면·로그에 그대로 찍어 어떤 값을 썼는지 밝힙니다.
def latest_retail():
    try:
        return fetch()
    except (HTTPError, URLError, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        reason = str(error)
    if CACHE_PATH.exists():
        row = pd.read_csv(CACHE_PATH, encoding="utf-8-sig", dtype={"trade_dt": str}).iloc[0]
        return {f: float(row[f]) for f in C.FUELS}, row["trade_dt"], f"오피넷 저장본 (API 실패: {reason})"
    prices, week = C.weekly_retail_last()
    return prices, week, f"주간 판매가 파일 (API 실패: {reason})"


if __name__ == "__main__":
    prices, day, source = latest_retail()
    print(f"기준일 {day}, 출처 {source}")
    for f in C.FUELS:
        print(f"  {C.KO[f]}: {prices[f]:,.2f}원/리터")
