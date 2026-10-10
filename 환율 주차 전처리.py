# 1. CSV 읽기, 날짜 계산, 정확한 평균 계산에 필요한 기본 라이브러리입니다.
import argparse
import csv
import re
from collections import defaultdict
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path




# 2. 일별 환율과 기존 주간 CSV의 위치를 지정합니다.
# 기본 실행: python "환율 주차 전처리.py"
CSV_DIR = Path(__file__).resolve().parent / "csv"
REFERENCE = CSV_DIR / "주유소_평균판매가격_제품별(주간).csv"




# 3. 기존 CSV의 주차 목록을 읽습니다. UTF-8과 오피넷 CP949 파일을 지원합니다.
# 두 자리 연도(15년)도 네 자리 연도(2015년)로 통일합니다.
def read_week_labels(path):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("기준 CSV의 문자 인코딩을 확인하세요.")
    reader = csv.reader(text.splitlines())
    next(reader, None)
    labels = []
    for row in reader:
        if not row:
            continue
        match = re.fullmatch(r"(\d{2}|\d{4})년(\d{1,2})월([1-5])주", row[0].strip())
        if not match:
            raise ValueError(f"기준 CSV의 주차 형식이 잘못되었습니다: {row[0]}")
        year, month, week = map(int, match.groups())
        year = year + 2000 if year < 100 else year
        date(year, month, 1)
        labels.append(f"{year:04d}년{month:02d}월{week}주")
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("기준 CSV의 주차가 없거나 중복되어 있습니다.")
    return labels




# 4. 일별 날짜를 주차로 변환합니다.
# 가정: 한 주는 일요일~토요일이며, 그 주 수요일이 속한 달의 주차입니다.
# 기존 CSV의 전체 월별 주차 수가 이 수요일 기준과 일치함을 확인했습니다.
# 예: 2015-01-04~01-10 → 2015년01월1주, 01-25~01-31 → 01월4주.
# 이 날짜 경계는 기존 파일의 주차 이름으로 추정한 기준입니다.
def week_label(day):
    sunday = day - timedelta(days=(day.weekday() + 1) % 7)
    wednesday = sunday + timedelta(days=3)
    week = (wednesday.day - 1) // 7 + 1
    return f"{wednesday.year:04d}년{wednesday.month:02d}월{week}주"




# 5. 같은 주의 일별 환율을 모읍니다.
# 주말·공휴일의 미발표 값은 채우지 않고, 실제 발표된 값만 동일한 비중으로 평균냅니다.
# 날짜 중복이나 잘못된 환율은 오류로 처리하여 평균이 왜곡되지 않게 합니다.
def group_daily_rates(path):
    groups = defaultdict(list)
    seen = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not {"날짜", "원/달러 환율"}.issubset(reader.fieldnames or []):
            raise ValueError("일별 CSV에 날짜와 원/달러 환율 열이 필요합니다.")
        for row in reader:
            day = date.fromisoformat(row["날짜"])
            rate = Decimal(row["원/달러 환율"].replace(",", "").strip())
            if day in seen or not rate.is_finite() or rate <= 0:
                raise ValueError(f"중복 날짜 또는 잘못된 환율: {day}")
            seen.add(day)
            groups[week_label(day)].append(rate)
    if not groups:
        raise ValueError("일별 환율 자료가 없습니다.")
    return groups




# 6. 기존 CSV와 같은 주차 순서로 주간 대표값을 저장합니다.
# 산술평균 = 일별 환율 합계 / 발표 일수이며 소수 둘째 자리까지 반올림합니다.
# 기준 파일에 없는 주차는 제외합니다. 자료가 없는 주차는 0 대신 빈칸으로 남깁니다.
def save_weekly_rates(groups, labels, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["구분", "원/달러 환율"])
        for label in labels:
            values = groups.get(label, [])
            if values:
                average = (sum(values) / len(values)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                writer.writerow([label, format(average, ".2f")])
            else:
                writer.writerow([label, ""])
                missing += 1
    return missing




# 7. 가장 긴 기간의 일별 파일을 기본 입력으로 골라 전처리를 실행합니다.
# 직접 지정: python "환율 주차 전처리.py" --input "csv/일별파일.csv"
# --reference로 다른 주간 CSV, --output으로 결과 저장 위치를 지정할 수 있습니다.
def main():
    parser = argparse.ArgumentParser(description="일별 원·달러 환율을 기존 CSV 주차별 평균으로 변환합니다.")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--output", type=Path, default=CSV_DIR / "원달러환율_주간평균.csv")
    args = parser.parse_args()
    try:
        input_path = args.input
        if input_path is None:
            candidates = []
            for path in CSV_DIR.glob("원달러환율_*.csv"):
                match = re.fullmatch(r"원달러환율_(\d{8})_(\d{8})", path.stem)
                if match:
                    start, end = (date(int(s[:4]), int(s[4:6]), int(s[6:])) for s in match.groups())
                    candidates.append(((end - start).days, end, path))
            if not candidates:
                raise ValueError("csv 폴더에 일별 원달러환율 파일이 없습니다.")
            input_path = max(candidates, key=lambda item: item[:2])[2]
        if args.output.resolve() in {input_path.resolve(), args.reference.resolve()}:
            raise ValueError("출력 경로는 원본 파일과 달라야 합니다.")
        labels = read_week_labels(args.reference)
        groups = group_daily_rates(input_path)
        missing = save_weekly_rates(groups, labels, args.output)
    except (OSError, ValueError, InvalidOperation) as error:
        parser.exit(1, f"오류: {error}\n")
    excluded = sum(len(values) for label, values in groups.items() if label not in set(labels))
    print(f"주간 평균 {len(labels)}주 저장 완료: {args.output}")
    print(f"자료 없는 주차: {missing}주 / 기준 주차 밖의 일별 자료: {excluded}건 제외")




# 8. 이 파일을 직접 실행할 때만 전처리를 시작합니다.
if __name__ == "__main__":
    main()
