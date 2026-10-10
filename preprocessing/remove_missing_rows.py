# 1. CSV 저장 기능과 기존 결측 분석 파일의 읽기·점검·판별 기능을 불러옵니다.
# 실행: python "preprocessing/remove_missing_rows.py"
import csv

from check_missing_values import (
    INPUT_PATH,
    PROJECT_DIR,
    check_structure,
    identify_missing,
    load_integrated_csv,
)




# 2. 결측 행을 제거한 결과를 별도 CSV로 저장합니다.
# 원본 merged_weekly_data.csv는 유지합니다.
OUTPUT_PATH = PROJECT_DIR / "csv" / "merged_weekly_data_clean.csv"




# 3. 숫자 열 중 하나라도 결측이면 해당 주차의 행 전체를 제외합니다.
# 기존 분석과 동일한 결측 기준을 사용하며, 숫자 0은 제거하지 않습니다.
# 남는 행의 순서와 값, 열 이름은 원본 그대로 유지합니다.
def remove_missing_rows(columns, rows):
    missing_mask = identify_missing(columns, rows)
    kept = []
    removed = []
    for row, flags in zip(rows, missing_mask, strict=True):
        if any(flags.values()):
            removed.append(row)
        else:
            kept.append(row)
    return kept, removed




# 4. 엑셀에서도 한글 열 이름을 읽을 수 있도록 UTF-8 BOM 형식으로 저장합니다.
def save_clean_csv(columns, rows, path=OUTPUT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)




# 5. 원본 불러오기 → 구조 점검 → 결측 행 제거 → 결과 저장을 진행합니다.
# 구조 오류가 있는 자료는 저장하지 않고 오류 위치를 안내합니다.
def main():
    try:
        columns, rows = load_integrated_csv(INPUT_PATH)
        errors = check_structure(columns, rows)
        if errors:
            raise ValueError("구조 점검 실패:\n" + "\n".join(errors))
        kept, removed = remove_missing_rows(columns, rows)
        save_clean_csv(columns, kept)
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise SystemExit(f"결측 행 제거 실패: {error}") from None
    print(f"전체 {len(rows)}행 → 제거 {len(removed)}행 → 저장 {len(kept)}행")
    for row in removed:
        print(f"제거된 주차: {row['주차']}")
    print(f"저장 완료: {OUTPUT_PATH}")




# 6. 직접 실행할 때만 결측 행을 제거하고 저장합니다.
if __name__ == "__main__":
    main()
