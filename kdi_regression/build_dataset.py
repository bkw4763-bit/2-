"""오피넷에서 내려받은 원자료(input)를 합쳐 회귀분석용 주간 자료(data/opinet_weekly.csv)를 만든다.

원자료 (오피넷에서 받은 그대로이며 수정하지 않는다. 인코딩은 cp949)
  retail_weekly.csv      국내유가통계 > 주유소 > 평균판매가격 (주간, 원/리터)
  intl_weekly_krw.csv    유가관련정보 > 국제유가 > 석유제품 (주간, 원/리터)
  intl_weekly_usd.csv    같은 화면의 달러/배럴 값 (참고용)
  tax_gasoline.xls, tax_diesel.xls   유가관련정보 > 유류세 > 세금변동추이

사용법
  python build_dataset.py
"""
import datetime as dt
import io
import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE / "input"                       # 오피넷 원본 (저장소에 올린다)
OUT_FILE = HERE / "data" / "opinet_weekly.csv"   # 합친 결과 (실행하면 다시 만들어진다)

# 원자료의 열 이름 → 분석용 열 이름
RETAIL_COLUMNS = {"보통휘발유": "gasoline_retail", "자동차용경유": "diesel_retail", "실내등유": "kerosene_retail"}
INTL_COLUMNS = {"휘발유(92RON)": "gasoline_intl", "경유(0.001%)": "diesel_intl", "등유": "kerosene_intl"}


def week_label(text):
    """오피넷의 주차 표기를 '연-월-주' 꼴로 통일한다. ('2015년01월1주', '15년01월1주' → '2015-01-1')

    파일마다 연도를 네 자리 또는 두 자리로 적기 때문에, 합치기 전에 같은 꼴로 맞춘다.
    """
    year, month, nth = re.match(r"(\d{2,4})년(\d{2})월(\d)주", text).groups()
    return f"{year if len(year) == 4 else '20' + year}-{month}-{nth}"


def week_wednesday(label):
    """주차('연-월-주')의 수요일 날짜를 돌려준다.

    오피넷의 'M월 N주'는 그 달의 N번째 수요일이 든 일~토 주간이다.
    수요일을 알면 그 주의 시작일(일요일)과 종료일(토요일)을 구할 수 있다.
    """
    year, month, nth = map(int, label.split("-"))
    day = dt.date(year, month, 1)
    while day.weekday() != 2:  # 2 = 수요일
        day += dt.timedelta(days=1)
    return day + dt.timedelta(weeks=nth - 1)


def read_weekly(file_name, columns, suffix=""):
    """오피넷 주간 CSV 하나를 읽어, 필요한 열만 분석용 이름으로 바꿔 돌려준다.

    첫 열(주차 표기)을 '연-월-주' 로 바꿔 인덱스로 쓴다. 그래야 세 파일을 주차 기준으로 나란히 합칠 수 있다.
    """
    table = pd.read_csv(INPUT_DIR / file_name, encoding="cp949")
    table["week"] = table.iloc[:, 0].map(week_label)
    renamed = {old: new + suffix for old, new in columns.items()}
    return table.set_index("week")[list(columns)].rename(columns=renamed)


def read_tax(file_name):
    """세금변동추이 표에서 변동일자와 유류세 합계(원/리터, 부가세 제외)를 읽는다.

    오피넷의 xls 는 실제로는 HTML 표라서 read_html 로 읽는다.
    """
    html = (INPUT_DIR / file_name).read_text(encoding="utf-8")
    table = pd.read_html(io.StringIO(html))[0]
    table.columns = ["date", "excise", "transport", "education", "driving", "total", "levy"]
    table["date"] = pd.to_datetime(table["date"], format="%Y년%m월%d일").dt.date
    return table[["date", "total"]].sort_values("date")


def weekly_tax(tax, wednesday):
    """한 주(일~토) 7일에 적용된 유류세의 평균을 구한다.

    세금변동추이에는 세율이 바뀐 날짜만 있다. 주중에 세율이 바뀐 주는
    바뀌기 전과 후의 세액이 일수만큼 섞이도록 하루씩 찾아 평균한다.
    """
    days = [wednesday + dt.timedelta(days=offset) for offset in range(-3, 4)]
    return sum(tax.loc[tax["date"] <= day, "total"].iloc[-1] for day in days) / 7


def main():
    """원자료 다섯 개를 주차 기준으로 합쳐 data/opinet_weekly.csv 로 저장한다."""
    data = pd.concat(
        [
            read_weekly("retail_weekly.csv", RETAIL_COLUMNS),
            read_weekly("intl_weekly_krw.csv", INTL_COLUMNS),
            read_weekly("intl_weekly_usd.csv", INTL_COLUMNS, suffix="_usd"),
        ],
        axis=1,
    )
    # 세 파일의 조회 기간이 다르면 빈칸이 생긴다. 조용히 넘어가지 않고 멈춘다.
    if data.isna().any().any():
        raise ValueError("원자료의 기간이 서로 다르거나 빈 값이 있습니다.")

    wednesdays = [week_wednesday(week) for week in data.index]
    data.insert(0, "week_start", [day - dt.timedelta(days=3) for day in wednesdays])
    data.insert(1, "week_end", [day + dt.timedelta(days=3) for day in wednesdays])

    gasoline_tax, diesel_tax = read_tax("tax_gasoline.xls"), read_tax("tax_diesel.xls")
    data["gasoline_tax"] = [round(weekly_tax(gasoline_tax, day), 2) for day in wednesdays]
    data["diesel_tax"] = [round(weekly_tax(diesel_tax, day), 2) for day in wednesdays]

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_FILE, encoding="utf-8-sig")
    print(f"{OUT_FILE.name}: {len(data)}주 ({data.index[0]} ~ {data.index[-1]})")


if __name__ == "__main__":
    main()
