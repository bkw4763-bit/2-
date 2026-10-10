# 1. 학습·평가 분할 분석의 근거 점검입니다. full_sample의 점검을 학습 구간(앞 80%)만으로 다시 하고, 평가 구간에서 확인합니다.
# ① 정상성·공적분·자기상관, ② 차분 회귀: 학습 구간만 씁니다.
# ③ 시차 수: 학습 구간 안 기점 이동 검증 MAE로 고르고, 평가 구간 MAE를 함께 적어 그 선택이 평가에서도 맞는지 봅니다.
# ④ 구간 적중률: split_regression.py가 평가 구간에서 계산합니다. (학습 구간 오차로 만든 구간 → 평가 구간 적중률)
# ⑤ 민감도: 비중 방식 {수준, 차분} × 마진 기간 {26, 52, 104주} 6가지 사양의 평가 구간 MAE와 정책 기간 격차를 봅니다.
#    채택 사양(수준, 52주)은 평가 전에 정했습니다. 평가 성적을 보고 사양을 바꾸면 평가 구간을 학습에 쓴 셈이 되므로 바꾸지 않습니다.
# 실행: python "regression/train_test/split_check.py"
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "full_sample"))
from counterfactual import FUELS, HORIZON, LAGS, VAT, fit, lagged_intl, load_weekly, predict, recent_margin, normalize_weights, select_lags
from counterfactual_check import (
    CHECK_WEEK, MARGIN_OPTIONS, MAX_LAGS, diff_data, diff_regression, lag_count_checks, unit_root_checks, wide_table,
)
from split_regression import OUT_DIR, split_data, split_date, train_panel




# 2. ③ 시차 수를 평가 구간에서 확인합니다. 학습 구간으로 학습한 시차 1~5개 모형의 평가 구간 MAE입니다.
# 시차 5개가 모두 있는 주만 쓰므로 학습 행이 split_regression.py보다 2주 적습니다.
def test_lag_counts(weekly, fuel, train_end):
    X_train, y_train, X_test, y_test = split_data(weekly, fuel, train_end, lags=MAX_LAGS)
    scores = {}
    for lags in range(1, MAX_LAGS + 1):
        params = fit(select_lags(X_train, lags), y_train)
        scores[("③ 시차 수", f"평가 구간 MAE: 시차 {lags}개")] = np.abs(predict(params, select_lags(X_test, lags)) - y_test).mean()
    return scores




# 3. ⑤ 민감도: 시차 비중을 학습 구간의 수준 회귀와 차분 회귀 중 어디서 가져오는지, 마진을 몇 주로 잡는지를 바꿉니다.
# 사양마다 평가 구간 성적과 정책 기간 격차("없었다면" 세후 − 실제 세후 공급가)를 계산합니다.
def sensitivity(weekly, fuel, train_end):
    X_train, y_train, X_test, y_test = split_data(weekly, fuel, train_end)
    diff_coef = diff_regression(*diff_data(train_panel(weekly, train_end), fuel)).params.to_numpy()
    sources = {"수준": fit(X_train, y_train)["weights"], "차분": normalize_weights(diff_coef, LAGS)}
    policy = weekly[weekly["in_policy"]]
    X_policy = lagged_intl(weekly, fuel)[weekly["in_policy"]]
    first = np.arange(len(y_test)) < HORIZON
    rows = []
    for source, weights in sources.items():
        for margin_weeks in MARGIN_OPTIONS:
            params = {"const": recent_margin(X_train, y_train, weights, margin_weeks), "weights": weights}
            error = predict(params, X_test) - y_test.to_numpy()
            gap = (predict(params, X_policy) * (1 + VAT) + policy[f"{fuel}_tax"]
                   - policy[f"{fuel}_supply_aftertax"])
            rows.append({"fuel": fuel, "비중": source, "마진 기간(주)": margin_weeks,
                         "채택": source == "수준" and margin_weeks == 52, "마진": params["const"],
                         "평가 MAE": np.abs(error).mean(), "평가 1~32주 MAE": np.abs(error[first]).mean(),
                         "평가 편향": error.mean(), "정책 기간 평균 격차": gap.mean(),
                         f"{CHECK_WEEK.date()} 격차": gap[CHECK_WEEK], f"{policy.index[-1].date()} 격차": gap.iloc[-1]})
    return rows




# 4. 점검을 실행하고 results/regression/train_test 폴더에 저장합니다.
def main():
    try:
        weekly = load_weekly()
        train_end = split_date(weekly)
        train = train_panel(weekly, train_end)
        checks = {fuel: {**unit_root_checks(train, fuel), **lag_count_checks(train, fuel),
                         **test_lag_counts(weekly, fuel, train_end)} for fuel in FUELS}
        table = wide_table(checks)
        sens = pd.DataFrame([row for fuel in FUELS for row in sensitivity(weekly, fuel, train_end)])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_DIR / "split_checks.csv", encoding="utf-8-sig", float_format="%.4f")
        sens.to_csv(OUT_DIR / "split_sensitivity.csv", index=False, encoding="utf-8-sig", float_format="%.3f")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"분할 근거 점검 실패: {error}") from None
    pd.set_option("display.width", 220)
    print(f"[근거 점검 ①~③] 학습 구간 ~ {train_end.date()}, 평가 구간 MAE는 학습 구간 모형으로 계산")
    print(table.round(3).to_string())
    print("\n[⑤ 민감도] 6가지 사양 (원/리터)")
    print(sens.round(1).to_string(index=False))
    print(f"\n저장 완료: {OUT_DIR}")




# 5. 직접 실행할 때만 근거 점검을 시작합니다.
if __name__ == "__main__":
    main()
