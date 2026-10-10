# 1. CSV 처리와 주차 형식 변환에 필요한 기본 라이브러리입니다.
import argparse
import csv
import io
import re
from datetime import date
from pathlib import Path




# 2. csv 폴더의 주간 자료를 합쳐 같은 폴더에 저장합니다.
# 일별 환율 원본은 제외하고, 전처리된 주간 평균 환율을 사용합니다.
# 실행: python "csv 파일 합치기.py"
CSV_DIR = Path(__file__).resolve().parent / "csv"
DEFAULT_OUTPUT = CSV_DIR / "전체_주간통합.csv"




# 3. UTF-8 또는 CP949로 저장된 CSV를 읽습니다.
def read_csv(path):
    for encoding in ("utf-8-sig", "cp949"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"CSV 인코딩을 읽을 수 없습니다: {path.name}")
    rows = [row for row in csv.reader(io.StringIO(text)) if any(cell.strip() for cell in row)]
    if not rows or len(rows[0]) < 2:
        raise ValueError(f"CSV에 주차와 값 열이 필요합니다: {path.name}")
    return rows[0], rows[1:]




# 4. 15년01월1주와 2015년01월1주를 같은 주차로 변환합니다.
# 연도·월·주차를 숫자로 정렬하여 자료가 시간순으로 저장되게 합니다.
def parse_week(value):
    match = re.fullmatch(r"(\d{2}|\d{4})년(\d{1,2})월([1-5])주", value.strip())
    if not match:
        raise ValueError(f"주차 형식이 잘못되었습니다: {value}")
    year, month, week = map(int, match.groups())
    year = year + 2000 if year < 100 else year
    date(year, month, 1)
    return year, month, week




# 5. 각 주간 CSV의 값을 주차별로 읽고 열 이름 앞에 자료 이름을 붙입니다.
# 예: 정유사_주간공급가격_제품별(세전)_보통휘발유
# 같은 이름의 유종이라도 자료 출처와 세전·세후를 구분할 수 있습니다.
# 중복 주차나 열 개수 오류는 자동으로 덮어쓰지 않고 중단합니다.
def load_weekly_table(path, header, rows):
    if header[0].strip() not in {"구분", "기간", "주차"}:
        raise ValueError(f"주차 열을 찾을 수 없습니다: {path.name}")
    prefix = re.sub(r"\d{7,8}_\d{6,8}", "", path.stem).replace(" ", "")
    columns = [f"{prefix}_{name.strip()}" for name in header[1:]]
    if any(not name.strip() for name in header[1:]) or len(columns) != len(set(columns)):
        raise ValueError(f"빈 열 이름 또는 중복 열 이름: {path.name}")
    table = {}
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f"열 개수가 맞지 않습니다: {path.name}")
        week = parse_week(row[0])
        if week in table:
            raise ValueError(f"중복 주차: {path.name}, {row[0]}")
        table[week] = [value.strip() for value in row[1:]]
    return columns, table




# 6. 모든 파일에 등장하는 주차를 유지하는 방식으로 가로 병합합니다.
# 일부 파일에 해당 주차가 없으면 빈칸으로 남겨 관측값과 결측을 구분합니다.
# 일별 환율과 이미 생성된 통합 파일은 입력에서 제외합니다.
def merge_csv_files(input_dir, output):
    tables = []
    columns = ["주차"]
    weeks = set()
    skipped = []
    for path in sorted(input_dir.glob("*.csv")):
        if path.resolve() in {output.resolve(), (input_dir / DEFAULT_OUTPUT.name).resolve()}:
            continue
        header, rows = read_csv(path)
        if header[0].strip() == "날짜":
            skipped.append(path.name)
            continue
        if header[0].strip() == "주차":
            # 이 프로그램이 만든 통합 자료를 재실행 때 다시 합치지 않습니다.
            skipped.append(path.name)
            continue
        names, table = load_weekly_table(path, header, rows)
        if set(names) & set(columns):
            raise ValueError(f"다른 파일과 열 이름이 겹칩니다: {path.name}")
        columns.extend(names)
        tables.append((names, table))
        weeks.update(table)
    if not tables or not weeks:
        raise ValueError("합칠 주간 CSV 자료가 없습니다.")
    output.parent.mkdir(parents=True, exist_ok=True)
    missing = 0
    with output.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)
        for week in sorted(weeks):
            year, month, number = week
            row = [f"{year:04d}년{month:02d}월{number}주"]
            for names, table in tables:
                values = table.get(week, [""] * len(names))
                missing += sum(value == "" for value in values)
                row.extend(values)
            writer.writerow(row)
    return len(tables), len(weeks), len(columns), missing, skipped




# 7. 실행 옵션을 읽고 통합 결과를 안내합니다. 원본 CSV는 수정하지 않습니다.
# 저장 위치 변경: python "csv 파일 합치기.py" --output "csv/통합.csv"
def main():
    parser = argparse.ArgumentParser(description="주간 CSV를 주차 기준으로 가로 병합합니다.")
    parser.add_argument("--input-dir", type=Path, default=CSV_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        if not args.input_dir.is_dir():
            raise ValueError("입력 폴더가 존재하지 않습니다.")
        if args.output.exists():
            header, _ = read_csv(args.output)
            if header[0].strip() != "주차":
                raise ValueError("출력 경로에 원본 자료가 있습니다. 다른 경로를 지정하세요.")
        files, weeks, columns, missing, skipped = merge_csv_files(args.input_dir, args.output)
    except (OSError, ValueError, csv.Error) as error:
        parser.exit(1, f"오류: {error}\n")
    print(f"주간 CSV {files}개 통합 완료: {weeks}주, {columns}열 → {args.output}")
    print(f"결측값: {missing}칸 / 일별·기존 통합 파일 제외: {len(skipped)}개")




# 8. 이 파일을 직접 실행할 때만 CSV 병합을 시작합니다.
if __name__ == "__main__":
    main()
