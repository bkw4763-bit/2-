# 1. 필요한 기능을 불러옵니다. 기본 라이브러리만 사용하므로 추가 설치가 필요 없습니다.
import argparse
import csv
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen




# 2. ECOS 원·달러 환율의 통계 코드와 파일 위치를 지정합니다.
# 731Y001: 주요국 통화의 대원화환율 / 0000001: 원·미국달러 / D: 일별
# 환율 값은 1달러당 원화 금액입니다.
# API 안내: https://ecos.bok.or.kr/api/
BASE_DIR = Path(__file__).resolve().parent
STAT_CODE = "731Y001"
ITEM_CODE = "0000001"
PAGE_SIZE = 1000




# 3. 환경변수 또는 같은 폴더의 .env에서 ECOS 인증키를 읽습니다.
# 인증키를 코드에 직접 적거나 화면에 출력하지 않습니다.
def load_api_key():
    key = os.environ.get("ECOS_API_KEY", "").strip()
    env_path = BASE_DIR / ".env"
    if not key and env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            name, separator, value = line.strip().partition("=")
            if separator and name.strip() == "ECOS_API_KEY":
                value = value.strip()
                if value.startswith(("'", '"')):
                    end = value.find(value[0], 1)
                    if end < 0:
                        raise ValueError(".env의 ECOS_API_KEY 따옴표를 확인하세요.")
                    key = value[1:end].strip()
                else:
                    key = value.split(" #", 1)[0].strip()
                break
    if not key:
        raise ValueError(".env 파일에 ECOS_API_KEY=발급받은인증키를 입력하세요.")
    return key




# 4. 조회 날짜를 확인합니다. 날짜는 YYYYMMDD 형식으로 입력합니다.
def parse_date(value):
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("날짜는 YYYYMMDD 형식이어야 합니다.")
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError("실제로 존재하는 날짜를 입력하세요.") from None




# 5. ECOS에 HTTPS로 연결하여 일별 원·달러 환율을 가져옵니다.
# 1,000건을 넘으면 다음 페이지도 조회하여 전체 기간의 자료를 모읍니다.
# 주말·공휴일 등 ECOS에 자료가 없는 날짜는 결과에 포함되지 않습니다.
def fetch_exchange_rates(api_key, start_date, end_date):
    rows = []
    first = 1
    while True:
        url = (
            "https://ecos.bok.or.kr/api/StatisticSearch/"
            f"{quote(api_key, safe='')}/json/kr/{first}/{first + PAGE_SIZE - 1}/"
            f"{STAT_CODE}/D/{start_date:%Y%m%d}/{end_date:%Y%m%d}/{ITEM_CODE}"
        )
        try:
            with urlopen(url, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8-sig"))
        except HTTPError as error:
            raise RuntimeError(f"ECOS HTTP 오류: {error.code}") from None
        except (URLError, TimeoutError, OSError):
            raise RuntimeError("ECOS 연결 실패: 인터넷 연결과 서버 상태를 확인하세요.") from None
        except (ValueError, UnicodeError):
            raise RuntimeError("ECOS 응답을 JSON으로 읽을 수 없습니다.") from None

        if not isinstance(payload, dict):
            raise RuntimeError("ECOS 응답 형식이 올바르지 않습니다.")
        if "RESULT" in payload:
            result = payload["RESULT"]
            code = result.get("CODE", "알 수 없음")
            if code == "INFO-200":
                raise RuntimeError("조회 기간에 환율 자료가 없습니다.")
            # 서버 메시지 대신 오류 코드만 표시하여 인증키 노출을 방지합니다.
            raise RuntimeError(f"ECOS API 오류: {code}. 인증키와 조회 조건을 확인하세요.")

        try:
            data = payload["StatisticSearch"]
            page = data["row"]
            total = int(data["list_total_count"])
            if not isinstance(page, list) or not page:
                raise ValueError
            rows.extend(page)
        except (KeyError, TypeError, ValueError):
            raise RuntimeError("ECOS 환율 자료 형식이 올바르지 않습니다.") from None
        if len(rows) >= total:
            return sorted(rows, key=lambda row: row["TIME"])
        first += len(page)




# 6. 조회한 환율을 엑셀에서 열 수 있는 UTF-8 CSV 파일로 저장합니다.
# 생성된 자료는 현재 프로젝트의 csv 폴더에 모아 저장합니다.
def save_csv(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["날짜", "원/달러 환율", "단위"])
        for row in rows:
            day = datetime.strptime(row["TIME"], "%Y%m%d").date()
            writer.writerow([day.isoformat(), row["DATA_VALUE"], row.get("UNIT_NAME", "원")])




# 7. 실행 옵션을 읽고 인증키 로딩 → API 조회 → CSV 저장 순서로 진행합니다.
# 기본값은 한국 시간 기준 최근 30일이며, 기간과 저장 위치를 변경할 수 있습니다.
# 실행 예: python "환율 api 불러오기.py" --start 20260101 --end 20260131
def main():
    today = datetime.now(timezone(timedelta(hours=9))).date()
    parser = argparse.ArgumentParser(description="ECOS 일별 원·달러 환율을 CSV로 저장합니다.")
    parser.add_argument("--start", type=parse_date, default=today - timedelta(days=29))
    parser.add_argument("--end", type=parse_date, default=today)
    parser.add_argument("--output", type=Path, help="CSV 저장 경로")
    args = parser.parse_args()
    if args.start > args.end:
        parser.error("시작일은 종료일보다 늦을 수 없습니다.")
    output_path = args.output or BASE_DIR / "csv" / f"원달러환율_{args.start:%Y%m%d}_{args.end:%Y%m%d}.csv"
    try:
        rows = fetch_exchange_rates(load_api_key(), args.start, args.end)
        save_csv(rows, output_path)
    except (RuntimeError, ValueError, KeyError, OSError) as error:
        print(f"오류: {error}", file=sys.stderr)
        return 1
    print(f"환율 {len(rows)}건 저장 완료: {output_path}")
    return 0




# 8. 이 파일을 직접 실행할 때만 프로그램을 시작합니다.
if __name__ == "__main__":
    sys.exit(main())
