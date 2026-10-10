# 1. 석유공사 페트로넷의 "제품별수요(국내출하량)" 월별 표를 2단계(재정 부담) 입력 형식으로 바꿉니다.
# 원본은 "25년 01월" 다음부터 "02월"처럼 연도를 생략하고, 각 월 아래에 비중(%) 행, 맨 끝에 합계 행을 둡니다.
# 결과: csv/fuel_sales_monthly.csv (월, 휘발유, 경유, 등유, 출처). 단위는 원본 그대로 천 배럴/월입니다.
# 원본에 단위가 적혀 있지 않아 페트로넷 수요 통계의 기본 단위(천 배럴)로 봅니다. 크기도 맞습니다.
# 예: 2025년 휘발유 96,691천 배럴 = 하루 약 26만 배럴. 천 kL나 천 톤이면 국내 전체 석유 소비보다 커집니다.
# 실행: python "preprocessing/preprocess_fuel_demand.py"  (원본 지정: --input "csv/파일.csv")
import argparse
import csv
import io
import re
from pathlib import Path




# 2. 입력·출력 위치와 유종 열입니다. 원본 파일을 지정하지 않으면 csv 폴더의 가장 최근 원본을 씁니다.
CSV_DIR = Path(__file__).resolve().parent.parent / "csv"
OUTPUT = CSV_DIR / "fuel_sales_monthly.csv"
FUELS = ["휘발유", "경유", "등유"]




# 3. UTF-8 또는 CP949로 저장된 원본을 읽습니다. 페트로넷에서 받은 파일은 둘 중 하나입니다.
# 페트로넷 파일은 BOM이 두 번 붙어 있고 "25년 01월" 사이에 줄바꿈 없는 공백(\xa0)을 써서 둘 다 정리합니다.
def read_rows(path):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"원본의 문자 인코딩을 읽을 수 없습니다: {path.name}")
    text = text.lstrip("﻿").replace("\xa0", " ")
    rows = list(csv.DictReader(io.StringIO(text)))
    missing = {"월", *FUELS} - set(rows[0] if rows else {})
    if missing:
        raise ValueError(f"원본에 필요한 열이 없습니다: {sorted(missing)}")
    return rows




# 4. 월 행만 골라 연도를 채웁니다. 연도가 생략된 행은 바로 앞에 나온 연도를 이어 씁니다.
# 비중(%) 행은 월 칸이 비어 있어 건너뛰고, 합계 행은 검산에 씁니다. 월이 거꾸로 가거나 겹치면 중단합니다.
def parse_months(rows):
    months, total, year, previous = [], None, None, None
    for row in rows:
        label = (row["월"] or "").strip()
        if not label:
            continue
        if label.replace(" ", "") == "합계":
            total = {fuel: int(row[fuel].replace(",", "")) for fuel in FUELS}
            continue
        match = re.fullmatch(r"(?:(\d{2})년\s*)?(\d{1,2})월", label)
        if not match:
            raise ValueError(f"월 형식이 잘못되었습니다: {label}")
        if match.group(1):
            year = 2000 + int(match.group(1))
        if year is None:
            raise ValueError(f"첫 행에 연도가 없습니다: {label}")
        key = (year, int(match.group(2)))
        if not 1 <= key[1] <= 12 or (previous and key <= previous):
            raise ValueError(f"월이 순서대로가 아니거나 잘못되었습니다: {label}")
        previous = key
        months.append({"월": f"{key[0]:04d}-{key[1]:02d}",
                       **{fuel: int(row[fuel].replace(",", "")) for fuel in FUELS}})
    if not months:
        raise ValueError("월 자료가 없습니다.")
    return months, total




# 5. 월별 값을 더해 원본의 합계 행과 같은지 확인합니다. 다르면 행을 잘못 읽은 것이므로 저장하지 않습니다.
def check_total(months, total):
    if total is None:
        return "합계 행 없음 (검산 생략)"
    sums = {fuel: sum(month[fuel] for month in months) for fuel in FUELS}
    if sums != total:
        raise ValueError(f"월별 합이 원본 합계와 다릅니다: {sums} vs {total}")
    return "월별 합 = 원본 합계 (" + ", ".join(f"{fuel} {total[fuel]:,}" for fuel in FUELS) + ")"




# 6. 2단계 입력 형식으로 저장합니다. 출처 열에 원본 파일 이름을 남깁니다.
def save(months, source, path=OUTPUT):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["월", *FUELS, "출처"])
        writer.writeheader()
        for month in months:
            writer.writerow({**month, "출처": f"페트로넷 제품별수요(국내출하량) {source}, 천 배럴"})




# 7. 원본 선택 → 읽기 → 월 정리 → 검산 → 저장을 실행합니다.
def main():
    parser = argparse.ArgumentParser(description="페트로넷 제품별 국내출하량을 월별 판매량 표로 바꿉니다.")
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    try:
        source = args.input or max(CSV_DIR.glob("fuel_demand_monthly_*.csv"), default=None)
        if source is None:
            raise ValueError("csv 폴더에 fuel_demand_monthly_*.csv 원본이 없습니다.")
        months, total = parse_months(read_rows(source))
        message = check_total(months, total)
        save(months, source.name)
    except (OSError, ValueError, KeyError, csv.Error) as error:
        parser.exit(1, f"오류: {error}\n")
    print(f"{source.name} → {OUTPUT.name}: {months[0]['월']} ~ {months[-1]['월']} ({len(months)}개월)")
    print(message)




# 8. 직접 실행할 때만 변환합니다.
if __name__ == "__main__":
    main()
