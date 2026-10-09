"""수집 → 전처리 실행 스크립트.

준비
  1) 오피넷에서 내려받은 CSV 네 개를 input 폴더에 넣는다.
     retail_daily.csv, supply_weekly_pretax.csv, supply_weekly_aftertax.csv, intl_daily.csv
  2) API 키를 환경변수나 .env 파일에 넣는다. (ECOS_API_KEY, KOSIS_API_KEY, OPINET_API_KEY)

사용법
  python run_pipeline.py             파일을 읽고 API 를 호출한다
  python run_pipeline.py --offline   API 를 호출하지 않고 저장본만 쓴다 (와이파이 장애 대비)
"""
import argparse

import pandas as pd

import collect
import config as cfg
import prepare


def main():
    parser = argparse.ArgumentParser(description="석유 최고가격제 분석용 자료 수집·전처리")
    parser.add_argument("--offline", action="store_true", help="API 를 호출하지 않고 저장본만 사용")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print("[1] 수집 (last_date = 실제 마지막 관측일, lag_days = 오늘과의 차이)")
    raw, report = collect.collect_all(offline=args.offline)
    print(report.to_string(index=False))
    failed = report[report["status"] == "FAIL"]["name"].tolist()
    if failed:
        print(f"  ※ 받지 못한 자료: {failed} → 해당 열은 빈칸으로 남는다")

    print("\n[2] 전처리")
    tables = prepare.prepare_all(raw)
    report.to_csv(cfg.OUT_DIR / "data_status.csv", index=False, encoding="utf-8-sig")
    print("  " + ", ".join(f"{name} {table.shape}" for name, table in tables.items()))
    print(f"  저장 위치: {cfg.OUT_DIR}")

    print("\n[3] 유종별 최신값 (기준일 포함)")
    print(tables["latest"].to_string(index=False))

    print("\n[4] 직접 채워야 하는 값")
    for name, table in (("물가 가중치", cfg.CPI_WEIGHTS), ("하루 판매량", cfg.DAILY_VOLUME_L)):
        empty = [cfg.FUELS[f]["ko"] for f, v in table.items() if v is None]
        if empty:
            print(f"  {name}: {', '.join(empty)} 미입력 (config.py)")


if __name__ == "__main__":
    main()
