# 1. 딥러닝 모형(TimesFM 등)에 넣을 학습셋을 제품별로 불러옵니다.
# 자료 처리·변수 구성·분할 규칙은 regression 폴더("regression/full_sample/counterfactual.py",
# "regression/train_test/split_regression.py")에 정의된 것을 그대로 가져와 사용합니다. 회귀 쪽에서 바꾸면 여기도 바뀝니다.
#   - 자료: 클린파일을 달력상 연속한 주별 표로 만듭니다. 결측 행 제거로 빠진 주(2017-10-07)는 빈칸으로 두어,
#           그 주가 시차에 걸리는 행은 자동으로 빠집니다(빠진 주를 건너뛰어 시차를 붙이면 "1주 전"이 2주 전이 되기 때문).
#   - 설명변수: 해당 제품 국제가(원/리터)의 같은 주·1주 전·2주 전 3개. 국제가가 이미 원화이므로 환율은 쓰지 않습니다.
#   - 학습 대상: 정책(최고가격제, 2026-03-13) 이전 주만. 정책 기간은 "policy" 구간으로 따로 받습니다.
#   - 분할: 정책 이전 학습 행을 시간순으로 세어 앞 80%가 학습, 뒤 20%가 평가 (split_regression.py와 같은 경계).
# 실행: python 딥러닝/load_training_set.py
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
REGRESSION_DIR = PROJECT_DIR / "regression" / "full_sample"
OUTPUT_DIR = PROJECT_DIR / "csv"
if not (REGRESSION_DIR / "counterfactual.py").is_file():
    raise FileNotFoundError(f"회귀분석 파일이 없습니다: {REGRESSION_DIR / 'counterfactual.py'}")
sys.path.insert(0, str(REGRESSION_DIR))
import counterfactual  # noqa: E402




# 2. 회귀분석과 공유하는 설정입니다. 값은 전부 counterfactual.py에서 가져옵니다.
#   INPUT_PATH     데이터 클린파일 경로
#   LAGS           국제가 시차 개수(3이면 같은 주, 1주 전, 2주 전)
#   POLICY_START   최고가격제 시행일. 이 날이 든 주부터 학습·평가에서 제외
#   TRAIN_SHARE    시간순 앞쪽 학습셋 비율 (split_regression.py와 같은 0.8)
#   MODELS         제품(한글) → 회귀 쪽 유종 키, 타겟 열, 국제가 열
INPUT_PATH = counterfactual.INPUT_PATH
KEY_COLUMN = "주차"
LAGS = counterfactual.LAGS
POLICY_START = counterfactual.POLICY_START
TRAIN_SHARE = 0.8
MODELS = {
    spec["ko"]: {"fuel": fuel, "타겟": spec["pretax"], "국제가격": spec["intl"]}
    for fuel, spec in counterfactual.FUELS.items()
}
REGRESSION_MODELS = counterfactual.MODELS          # {"baseline": "11년 전체 회귀", "recent_margin": "최근 마진"}
REGRESSION_CHOSEN = counterfactual.CHOSEN          # "recent_margin"
SPLITS = ("train", "test", "all", "policy")

# regression 폴더의 기점 이동 백테스트 설정입니다. 딥러닝 교차검증을 같은 격자로 돌릴 때 씁니다.
#   BACKTEST_START    첫 기점 날짜(그 앞은 최소 학습 기간)
#   BACKTEST_STEP     기점 간격(주)
#   BACKTEST_HORIZON  기점마다 맞혀 보는 주 수
BACKTEST_START = counterfactual.BACKTEST_START
BACKTEST_STEP = counterfactual.BACKTEST_STEP
BACKTEST_HORIZON = counterfactual.HORIZON


def feature_names(product):
    intl = MODELS[product]["국제가격"]
    return [f"{intl} (t)" if k == 0 else f"{intl} (t-{k})" for k in range(LAGS)]




# 3. 주별 표를 읽고 점검합니다. 세 유종의 학습 행(정책 이전 + 시차·타겟 모두 있음)이 같은지 확인해
# 같은 경계로 나눌 수 있게 합니다. split_regression.split_date와 같은 규칙입니다.
def load_weekly_table(path=INPUT_PATH):
    weekly = counterfactual.load_weekly(path)
    indexes = [counterfactual.training_data(weekly, fuel)[0].index for fuel in counterfactual.FUELS]
    if any(not index.equals(indexes[0]) for index in indexes):
        raise ValueError("유종마다 학습 주가 달라 같은 경계로 나눌 수 없습니다")
    return weekly


def split_date(weekly):
    index = counterfactual.training_data(weekly, next(iter(counterfactual.FUELS)))[0].index
    return index[round(len(index) * TRAIN_SHARE) - 1]




# 4. 제품 하나의 학습셋을 만듭니다.
# 설계행렬은 counterfactual.lagged_intl(국제가 lag0·lag1·lag2), 타겟은 세전 공급가입니다.
# 반환값은 딥러닝 모형에 바로 넣을 수 있는 numpy 배열과 주차·변수 이름을 담은 사전입니다.
#   "y": 세전 공급가격 (표본수,)            → 예측 대상 시계열
#   "X": 설명변수 행렬 (표본수, 3)         → 국제가 같은 주·1주 전·2주 전, 공변량으로 사용
#   "주차": 각 행의 주차 표기, "날짜": 주의 끝(토요일) 날짜
# split: "train" 학습 80%, "test" 평가 20%, "all" 정책 이전 전체, "policy" 정책 기간(상한이 걸린 실제 공급가가 y)
def load_training_set(product, weekly=None, split="train"):
    if product not in MODELS:
        raise ValueError(f"제품은 {', '.join(MODELS)} 중 하나여야 합니다: {product}")
    if split not in SPLITS:
        raise ValueError(f"split은 {', '.join(SPLITS)} 중 하나여야 합니다: {split}")
    if weekly is None:
        weekly = load_weekly_table()
    fuel = MODELS[product]["fuel"]
    X_all = counterfactual.lagged_intl(weekly, fuel)
    y_all = weekly[f"{fuel}_supply_pretax"]
    complete = X_all.notna().all(axis=1) & y_all.notna()
    before = complete & ~weekly["in_policy"]
    train_end = split_date(weekly)
    masks = {
        "train": before & (weekly.index <= train_end),
        "test": before & (weekly.index > train_end),
        "all": before,
        "policy": complete & weekly["in_policy"],
    }
    selected = masks[split]
    if not selected.any():
        raise ValueError(f"{product}: '{split}' 구간에 자료가 없습니다")
    X = X_all[selected].to_numpy(dtype=np.float32)
    y = y_all[selected].to_numpy(dtype=np.float32)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{product} 자료에 결측 또는 무한대가 있습니다")
    labels = weekly.loc[selected, KEY_COLUMN].tolist()
    return {
        "제품": product,
        "구분": split,
        "타겟": MODELS[product]["타겟"],
        "설명변수": feature_names(product),
        "시차": LAGS,
        "주차": labels,
        "날짜": [day.date().isoformat() for day in weekly.index[selected]],
        "기간": (labels[0], labels[-1]),
        "학습경계": train_end.date().isoformat(),
        "X": X,
        "y": y,
    }


def load_all_training_sets(split="train"):
    weekly = load_weekly_table()
    return {product: load_training_set(product, weekly, split) for product in MODELS}




# 5. 비교용 회귀모형입니다. counterfactual.fit/predict를 numpy 배열로 감싼 것이라 회귀 쪽과 결과가 같습니다.
#   baseline       OLS 상수·계수 그대로
#   recent_margin  계수를 합 1로 정규화하고 상수는 학습 끝 52주 마진으로 다시 잡음 (regression 채택 모형)
def fit_regression(X, y, model=REGRESSION_CHOSEN):
    frame = pd.DataFrame(np.asarray(X, dtype=float), columns=[f"lag{k}" for k in range(LAGS)])
    return counterfactual.fit(frame, pd.Series(np.asarray(y, dtype=float)), model)


def predict_regression(params, X):
    return [float(v) for v in counterfactual.predict(params, pd.DataFrame(np.asarray(X, dtype=float)))]




# 6. 평가 지표입니다. 이전 보고서와 같은 정의(R²·RMSE·MAE·MAPE·최대오차)에 편향(예측 − 실제 평균)을 더했습니다.
def evaluate(actual, predicted):
    count = len(actual)
    if count == 0:
        return {"n": 0}
    mean = sum(actual) / count
    residuals = [a - p for a, p in zip(actual, predicted, strict=True)]
    sse = sum(r * r for r in residuals)
    sst = sum((a - mean) ** 2 for a in actual)
    absolute = [abs(r) for r in residuals]
    percentage = [abs(r) / abs(a) for r, a in zip(residuals, actual, strict=True) if a]
    return {
        "n": count,
        "R2": 1.0 - sse / sst if sst else float("nan"),
        "RMSE": math.sqrt(sse / count),
        "MAE": sum(absolute) / count,
        "MAPE": 100.0 * sum(percentage) / len(percentage) if percentage else float("nan"),
        "최대오차": max(absolute),
        "편향": -sum(residuals) / count,
    }




# 7. 제품별 학습셋을 CSV로 저장해 어떤 행·변수가 들어갔는지 눈으로 확인할 수 있게 합니다.
def save_training_set(result, directory=OUTPUT_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"training_set_{result['제품']}.csv"
    fieldnames = [KEY_COLUMN, "날짜", *result["설명변수"], result["타겟"]]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for label, day, features, target in zip(result["주차"], result["날짜"], result["X"], result["y"], strict=True):
            writer.writerow([label, day, *(f"{value:.2f}" for value in features), f"{target:.2f}"])
    return path




# 8. 불러오기 → 점검 → 회귀와 같은 시차·분할 → 제품별 요약 출력·저장 순서로 진행합니다.
def main():
    try:
        weekly = load_weekly_table()
        results = {product: load_training_set(product, weekly, "train") for product in MODELS}
        counts = {split: len(load_training_set(next(iter(MODELS)), weekly, split)["y"]) for split in SPLITS}
        paths = [save_training_set(result) for result in results.values()]
    except (OSError, UnicodeError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"학습셋 불러오기 실패: {error}") from None

    first = next(iter(results.values()))
    print(f"자료: {INPUT_PATH.relative_to(PROJECT_DIR).as_posix()} → 달력 연속 주별 표 (regression/full_sample/counterfactual.py 방식)")
    print(f"변수: 제품 국제가(원/L) 시차 {LAGS}개 (같은 주·1주 전·2주 전), 환율 없음. 정책({POLICY_START.date()}) 이전 주만 학습·평가")
    print(f"분할: 정책 이전 {counts['all']}주 중 앞 {TRAIN_SHARE:.0%} 학습 {counts['train']}주 (~{first['학습경계']}) / 평가 {counts['test']}주, 정책 기간 {counts['policy']}주 별도")
    print(f"학습 기간: {first['기간'][0]} ~ {first['기간'][1]}")
    print()
    for result in results.values():
        X, y = result["X"], result["y"]
        print(f"[{result['제품']}] 타겟: {result['타겟']}")
        print(f"  설명변수 {len(result['설명변수'])}개: " + ", ".join(result["설명변수"]))
        print(f"  X 크기 {X.shape}, y 크기 {y.shape}")
        print(f"  y 요약: 평균 {y.mean():.1f} / 최소 {y.min():.1f} / 최대 {y.max():.1f} 원/L")
        for name, column in zip(result["설명변수"], X.T, strict=True):
            print(f"   - {name:<44} 평균 {column.mean():>9.2f}  최소 {column.min():>9.2f}  최대 {column.max():>9.2f}")
        print()
    for path in paths:
        print(f"학습셋 저장: {path.relative_to(PROJECT_DIR).as_posix()}")




# 9. 직접 실행할 때만 학습셋을 만듭니다. 다른 파일에서는 load_training_set / load_all_training_sets를 import해 씁니다.
if __name__ == "__main__":
    main()
