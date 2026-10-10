# 1. regression 폴더의 회귀모형(최근 마진·학습 구간 전체 회귀)을 TimesFM과 같은 테스트셋에 적용합니다.
# 딥러닝 로더(딥러닝/load_training_set.py)가 regression 방식(정책 이전만, 국제가 3개 시차, 80/20)을 그대로 쓰므로
# 테스트셋은 split_regression.py의 평가 구간(2023-12 ~ 2026-03, 정책 이전 116주)과 같습니다.
# 모형·자료 처리는 full_sample/counterfactual.py의 함수를 그대로 써서 regression 회귀와 TimesFM을 같은 주에서 비교합니다.
# ols 111 OLS(국제가 t-1 + 환율, 다른 분할) 예측 파일은 테스트 주가 모두 들어 있을 때만 비교에 넣습니다.
# 실행: python "regression/train_test/apply_common_test.py"
import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "regression" / "full_sample"))
from counterfactual import (  # noqa: E402
    CHOSEN, FUELS, LAGS, MODELS, POLICY_START, fit, lagged_intl, load_weekly, predict, week_end,
)




# 2. 출력 위치와 비교 대상 파일입니다.
OUT_DIR = PROJECT_DIR / "results" / "regression" / "train_test"
PREDICTION_PATH = OUT_DIR / "common_test_predictions.csv"
MODEL_PATH = OUT_DIR / "common_test_models.csv"
REPORT_PATH = OUT_DIR / "common_test_report.md"
CSV_DIR = PROJECT_DIR / "csv"
NAMES = {"recent_margin": "regression 최근 마진", "baseline": "regression 전체 회귀",
         "ols111": "ols 111 OLS (국제가 t-1 + 환율)", "timesfm3": "TimesFM 3.0", "naive": "직전 가격 유지"}




# 3. 딥러닝 쪽 분할 경계를 가져옵니다. 폴더 이름에 한글이 있어 경로로 모듈을 읽습니다.
# 학습 마지막 주와 테스트 주차 목록만 쓰고, 설명변수 구성은 regression 모형 것을 따릅니다.
def load_common_split():
    path = PROJECT_DIR / "딥러닝" / "load_training_set.py"
    spec = importlib.util.spec_from_file_location("load_training_set", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    weekly = module.load_weekly_table()
    product = next(iter(module.MODELS))
    train = module.load_training_set(product, weekly, split="train")
    test = module.load_training_set(product, weekly, split="test")
    return week_end(train["기간"][1]), [week_end(label) for label in test["주차"]], test["주차"]




# 4. 한 유종에 대해 학습 경계 이전 자료로 두 regression 모형을 학습하고 테스트 주를 예측합니다.
# 학습 행: 경계 이전 + 정책 이전 + 시차 국제가·세전가 모두 있음 (counterfactual.training_data와 같은 조건)
# 테스트 행: 경계 이후 + 시차 국제가·세전가 모두 있음. 정책 기간도 포함해 실제 국제가만 넣어 예측합니다.
def run_fuel(weekly, fuel, train_end, test_weeks):
    X_all = lagged_intl(weekly, fuel)
    y_all = weekly[f"{fuel}_supply_pretax"]
    complete = X_all.notna().all(axis=1) & y_all.notna()
    train = complete & (weekly.index <= train_end) & ~weekly["in_policy"]
    test = complete & weekly.index.isin(test_weeks)
    X_train, y_train = X_all[train], y_all[train]
    table = pd.DataFrame({"주차": weekly.loc[test, "주차"], "fuel": fuel, "in_policy": weekly.loc[test, "in_policy"],
                          "actual": y_all[test], "naive": y_train.iloc[-1]})
    params = {}
    for model in MODELS:
        params[model] = fit(X_train, y_train, model)
        table[model] = predict(params[model], X_all[test])
    return table, params, (X_train.index[0], X_train.index[-1], len(X_train))




# 5. 다른 두 모형의 테스트 예측을 읽어 같은 주에 붙입니다. 파일이 없으면 그 열은 비워 둡니다.
def attach_predictions(table, fuel, model, filename):
    path = CSV_DIR / filename
    if not path.is_file():
        table[model] = np.nan
        return False
    other = pd.read_csv(path, encoding="utf-8-sig")
    other = other[other["구분"] == "test"].set_index("주차")["예측_세전가격"]
    mapped = table["주차"].map(other)
    if mapped.isna().any():
        # 다른 분할로 만든 예측 파일이라 테스트 주가 다 없으면 비교에서 뺍니다(일부 주만 채점하면 공정하지 않음).
        table[model] = np.nan
        return False
    table[model] = mapped.to_numpy()
    return True




# 6. 평가 지표입니다. ols 111·TimesFM 보고서와 같은 정의(R²·RMSE·MAE·MAPE)에 편향을 더했습니다.
def metrics(actual, predicted):
    pair = pd.DataFrame({"a": actual, "p": predicted}).dropna()
    if pair.empty:
        return None
    error = pair["p"] - pair["a"]
    sst = ((pair["a"] - pair["a"].mean()) ** 2).sum()
    return {"n": len(pair), "R2": 1 - (error ** 2).sum() / sst if sst else np.nan,
            "RMSE": np.sqrt((error ** 2).mean()), "MAE": error.abs().mean(), "bias": error.mean(),
            "MAPE": (error.abs() / pair["a"].abs()).mean() * 100, "max_error": error.abs().max()}




# 7. 유종·모형·구간(전체 / 정책 이전 / 정책 기간)별 지표표를 만듭니다.
def score_table(table, model_columns):
    rows = []
    segments = {"전체": slice(None), "정책 이전": ~table["in_policy"], "정책 기간": table["in_policy"]}
    for segment, mask in segments.items():
        part = table if segment == "전체" else table[mask]
        for model in model_columns:
            score = metrics(part["actual"], part[model])
            if score:
                rows.append({"fuel": table["fuel"].iloc[0], "segment": segment, "model": model,
                             "name": NAMES[model], **score})
    return rows




# 8. 보고서를 마크다운으로 정리합니다.
def save_report(scores, fits, train_info, test_labels, path=REPORT_PATH):
    ko = {fuel: spec["ko"] for fuel, spec in FUELS.items()}
    lines = ["# regression 폴더 회귀모형을 공통 테스트셋에 적용한 결과", "",
             "## 1. 설계", "",
             "- 자료: `csv/merged_weekly_data_clean.csv`",
             "- 모형: `regression/full_sample/counterfactual.py`의 최근 마진 모형(채택)과 전체 회귀. 국제가 같은 주·1주 전·2주 전 3개 시차, 환율은 쓰지 않음(국제가가 이미 원/리터)",
             f"- 분할: 딥러닝(`딥러닝/load_training_set.py`)과 같은 시간순 80/20, 정책 이전만. 학습 ~ {train_info[1].date()} ({train_info[2]}주), 테스트 {test_labels[0]} ~ {test_labels[-1]} ({len(test_labels)}주)",
             f"- 테스트셋은 정책 이전({POLICY_START.date()} 이전) 주만이며 `regression/train_test/split_report.md`의 평가 구간과 같습니다. 정책 기간은 평가에서 제외했습니다.",
             "- 모형은 학습 구간에서 한 번 고정하고 테스트 구간은 실제 국제가만 넣어 예측했습니다. 마진을 다시 잡지 않았습니다.",
             "- 비교 대상: `csv/timesfm3_predictions_*.csv`(TimesFM 3.0, regression 방식 자료·변수). `csv/regression_predictions_*.csv`(ols 111 OLS)는 테스트 주가 모두 있을 때만 포함. 모두 같은 주에서 채점했습니다.",
             "", "## 2. 학습 결과 (regression 최근 마진 모형)", "",
             "| 유종 | 비중 (같은 주 / 1주 전 / 2주 전) | 마진 (학습 마지막 52주) |", "| --- | --- | --- |"]
    for fuel, params in fits.items():
        w = params[CHOSEN]["weights"]
        lines.append(f"| {ko[fuel]} | {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} | {params[CHOSEN]['const']:.1f}원 |")
    frame = pd.DataFrame(scores)
    for segment in ["전체", "정책 이전", "정책 기간"]:
        part = frame[frame["segment"] == segment]
        if part.empty:
            continue
        lines += ["", f"## 3. 테스트 성적 — {segment} (세전 원/리터)", "",
                  "| 유종 | 모형 | 주 | R² | RMSE | MAE | 편향 | MAPE |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for _, r in part.iterrows():
            lines.append(f"| {ko[r['fuel']]} | {r['name']} | {r['n']} | {r['R2']:.3f} | {r['RMSE']:.1f} | "
                         f"{r['MAE']:.1f} | {r['bias']:+.1f} | {r['MAPE']:.2f}% |")
    lines += ["", "편향 = 예측 − 실제의 평균. 양수면 높게 예측한 것입니다.", "",
              "## 4. 읽는 법", "",
              "- 테스트셋이 정책 이전 주만이므로 '전체'와 '정책 이전'은 같은 표입니다. regression 폴더 `split_report.md` 3절의 최근 마진 MAE와 같은 주·같은 모형이라 직접 비교할 수 있습니다.",
              "- 정책 기간(상한제)은 세전 공급가가 상한에 묶여 국제가만으로 맞힐 수 없으므로 평가에서 제외했습니다. 그 구간의 격차는 regression 폴더의 '없었다면' 분석이 다룹니다.",
              "- 학습 경계에서 한 번 고정한 조건이라, 창마다 다시 적합하는 학습셋 안 교차검증(`results/timesfm_cv/cv_report.md`)과는 성격이 다릅니다.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 9. 분할 → 유종별 학습·예측 → 다른 모형 예측 결합 → 채점 → 저장 → 요약 출력을 실행합니다.
def main():
    try:
        weekly = load_weekly()
        train_end, test_weeks, test_labels = load_common_split()
        tables, fits, scores, train_info = [], {}, [], None
        found = set()
        for fuel, spec in FUELS.items():
            table, fits[fuel], train_info = run_fuel(weekly, fuel, train_end, test_weeks)
            if attach_predictions(table, fuel, "ols111", f"regression_predictions_{spec['ko']}.csv"):
                found.add("ols111")
            if attach_predictions(table, fuel, "timesfm3", f"timesfm3_predictions_{spec['ko']}.csv"):
                found.add("timesfm3")
            tables.append(table)
            scores += score_table(table, [CHOSEN, "baseline", *sorted(found & {"ols111", "timesfm3"}), "naive"])
        predictions = pd.concat(tables)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(PREDICTION_PATH, encoding="utf-8-sig", float_format="%.2f")
        pd.DataFrame(scores).to_csv(MODEL_PATH, index=False, encoding="utf-8-sig", float_format="%.4f")
        report = save_report(scores, fits, train_info, list(test_labels))
    except (OSError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"공통 테스트셋 적용 실패: {error}") from None

    print(f"학습: {train_info[0].date()} ~ {train_info[1].date()} ({train_info[2]}주) / "
          f"테스트: {test_labels[0]} ~ {test_labels[-1]} ({len(test_labels)}주, 정책 이전만)")
    if "timesfm3" not in found:
        print("주의: TimesFM 예측 파일이 없어 비교에서 빠졌습니다. 먼저 python 딥러닝/run_timesfm.py 를 실행하세요.")
    pd.set_option("display.width", 200)
    frame = pd.DataFrame(scores)
    frame["fuel"] = frame["fuel"].map({f: s["ko"] for f, s in FUELS.items()})
    for segment in ["전체", "정책 이전", "정책 기간"]:
        part = frame[frame["segment"] == segment]
        if part.empty:
            continue
        print(f"\n[{segment}] 세전 원/리터")
        print(part[["fuel", "name", "n", "R2", "RMSE", "MAE", "bias", "MAPE"]].round(3).to_string(index=False))
    print(f"\n저장: {PREDICTION_PATH.relative_to(PROJECT_DIR).as_posix()}, {MODEL_PATH.name}, {report.name}")




# 10. 직접 실행할 때만 분석을 시작합니다.
if __name__ == "__main__":
    main()
