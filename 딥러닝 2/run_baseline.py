"""직전 판매가격 유지 기준선. 기본은 검증만 평가하며 최종 테스트는 명시적으로 실행한다."""
import argparse

import numpy as np

import settings as cfg
from evaluation import score_predictions, score_folds
from make_windows import load_experiment, result_dir
from ai_protocol import load_locked_settings
from rolling_origin import run_rolling_origin


def predict_last_price(context, known_future, metadata):
    """기준일 마지막 판매가를 이후 모든 날짜의 예측값으로 유지한다."""
    return np.repeat(context.supply_pretax.iloc[-1], len(known_future))


def main():
    """공통 예측 형식과 지표 파일을 만들어 이후 AI 결과와 비교한다."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    args = parser.parse_args()
    data, windows, protocol = load_experiment()
    windows = windows.loc[(windows.split == args.split) & (windows.context_years == 1)].copy()
    if args.split == "test":
        load_locked_settings(result_dir(protocol)/"timesfm3"/"locked_settings.json", protocol)
    out = result_dir(protocol)/"baseline"/args.split
    if args.split == "test" and out.exists():
        parser.error("최종 테스트 기준선 결과가 이미 있습니다.")
    predictions = run_rolling_origin(data, windows, predict_last_price, "last_price", args.split)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    score_predictions(predictions).to_csv(out / "metrics.csv", index=False, encoding="utf-8-sig")
    score_folds(predictions).to_csv(out / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"기준선 {args.split} 예측·지표 저장: {out}")


if __name__ == "__main__":
    main()
