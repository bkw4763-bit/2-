"""오피넷에서 내려받은 원자료(input)를 합쳐 회귀분석용 일간 자료(data/opinet_daily.csv)를 만든다.

원자료 (오피넷에서 받은 그대로이며 수정하지 않는다. CSV 인코딩은 cp949)
  retail_daily.csv       국내유가통계 > 주유소 > 평균판매가격 (일간, 원/리터)
  intl_daily_krw.csv     유가관련정보 > 국제유가 > 석유제품 (일간, 단위 원 = 원/리터)
  tax_gasoline.xls, tax_diesel.xls   유가관련정보 > 유류세 > 세금변동추이

사용법
  python build_dataset.py
"""
import io
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input"                      # 오피넷 원본 (저장소에 올린다)
OUT_FILE = HERE / "data" / "opinet_daily.csv"   # 합친 결과 (실행하면 다시 만들어진다)

# 원자료의 열 이름 → 분석용 열 이름
RETAIL_COLUMNS = {"보통휘발유": "gasoline_retail", "자동차용경유": "diesel_retail", "실내등유": "kerosene_retail"}
INTL_COLUMNS = {"휘발유(92RON)": "gasoline_intl", "경유(0.001%)": "diesel_intl", "등유": "kerosene_intl"}


def parse_date(text):
    """오피넷의 날짜 표기를 날짜 값으로 바꾼다. ('2015년01월01일', '15년01월02일' 모두 처리)

    파일마다 연도를 네 자리 또는 두 자리로 적기 때문에, 합치기 전에 같은 날짜 값으로 맞춘다.
    """
    year, month, day = re.match(r"(\d{2,4})년(\d{2})월(\d{2})일", text).groups()
    return pd.Timestamp(f"{year if len(year) == 4 else '20' + year}-{month}-{day}")


def read_daily(file_name, columns):
    """오피넷 일간 CSV 하나를 읽어, 필요한 열만 분석용 이름으로 바꿔 돌려준다.

    첫 열(날짜 표기)을 날짜 값으로 바꿔 인덱스로 쓴다. 그래야 두 파일을 날짜 기준으로 나란히 합칠 수 있다.
    """
    table = pd.read_csv(INPUT_DIR / file_name, encoding="cp949")
    table.index = table.iloc[:, 0].map(parse_date)
    return table[list(columns)].rename(columns=columns)


def read_tax(file_name):
    """세금변동추이 표에서 변동일자와 유류세 합계(원/리터, 부가세 제외)를 읽는다.

    오피넷의 xls 는 실제로는 HTML 표라서 read_html 로 읽는다.
    """
    html = (INPUT_DIR / file_name).read_text(encoding="utf-8")
    table = pd.read_html(io.StringIO(html))[0]
    table.columns = ["date", "excise", "transport", "education", "driving", "total", "levy"]
    table["date"] = pd.to_datetime(table["date"], format="%Y년%m월%d일")
    return table[["date", "total"]].sort_values("date")


def daily_tax(tax, dates):
    """날짜마다 그날 적용된 유류세를 붙인다.

    세금변동추이에는 세율이 바뀐 날짜만 있으므로, 각 날짜에 대해 그 이전의 가장 최근 변동값을 찾는다.
    """
    by_date = tax.set_index("date")["total"]
    return by_date.reindex(dates, method="ffill").values


def main():
    """판매가격·국제가격·유류세를 날짜 기준으로 합쳐 data/opinet_daily.csv 로 저장한다."""
    retail = read_daily("retail_daily.csv", RETAIL_COLUMNS)
    intl = read_daily("intl_daily_krw.csv", INTL_COLUMNS)

    # 판매가격은 매일 있어야 한다. 빠진 날이 있으면 조용히 넘어가지 않고 멈춘다.
    days = pd.date_range(retail.index.min(), retail.index.max())
    if len(retail) != len(days) or retail.isna().any().any():
        raise ValueError("판매가격 자료에 빠진 날짜나 빈 값이 있습니다.")

    # 국제가격은 영업일에만 있다. 주말·휴일에는 직전 영업일 값을 쓴다. (그날 알려진 가장 최근 가격)
    data = retail.join(intl.reindex(days).ffill())
    data["gasoline_tax"] = daily_tax(read_tax("tax_gasoline.xls"), days)
    data["diesel_tax"] = daily_tax(read_tax("tax_diesel.xls"), days)
    data.index.name = "date"

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_FILE, encoding="utf-8-sig")
    print(f"{OUT_FILE.name}: {len(data)}일 ({data.index[0].date()} ~ {data.index[-1].date()})")


if __name__ == "__main__":
    main()
