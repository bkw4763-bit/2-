# 1. 1단계 회귀분석의 근거 점검과 그림입니다. counterfactual.py 모형이 왜 타당한지 확인합니다.
# ① 정상성·공적분·자기상관: 수준 회귀의 높은 R²가 근거가 될 수 있는지 봅니다.
# ② 차분 회귀: 단위근 문제 없이도 국제가 변화가 공급가에 1:1로 반영되는지 봅니다. (비중 정규화의 근거)
# ③ 시차 수: 시차 1~5개의 기점 이동 검증 오차를 비교합니다.
# ④ 구간 적중률: 2021년까지 끝나는 기점의 오차로 만든 90% 구간이 2022년 이후 기점에서 맞는 비율을 봅니다.
# ⑤ 민감도: 비중 방식 {수준, 차분} × 마진 기간 {26, 52, 104주} 6가지 사양의 정책 기간 격차 범위를 봅니다.
# ⑥ 외부 대조: 정부 설명(2026-09-18)과 9월 2주 격차를 비교합니다.
# ⑦ 그림: 위 = 실제·"없었다면"·90% 구간, 아래 = 정책 기간 격차와 90% 구간
# 원유(브렌트) 비교는 이 저장소에 브렌트유 자료가 없어 하지 않습니다.
# 실행: python "regression/full_sample/counterfactual_check.py"
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint

from counterfactual import (
    FUELS, HORIZON, LAGS, OUT_DIR, VAT,
    backtest, build_counterfactual, cutoffs, fit, lagged_intl, load_weekly, predict, recent_margin, training_data,
)




# 2. 점검 매개변수와 그림 색입니다.
# 유종 색은 유종에 고정합니다. 실제 공급가는 회색, 정책 시작은 점선으로 그립니다.
SPLIT_DATE = pd.Timestamp("2022-01-01")
MARGIN_OPTIONS = (26, 52, 104)
MAX_LAGS = 5
NW_LAGS = 4                                  # Newey-West 시차. 주별 자료이므로 약 한 달
CHECK_WEEK = pd.Timestamp("2026-09-12")      # 정부 설명(9/18) 직전에 공급가가 공개된 주
GOV_CLAIM = {"gasoline": "100원대 후반", "diesel": "300원대 중반 이상", "kerosene": "언급 없음"}
COLORS = {"gasoline": "#2a78d6", "diesel": "#eb6834", "kerosene": "#1baf7a"}
INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
FIG_START = pd.Timestamp("2025-07-01")




# 3. 그림 공통 서식입니다. 맑은 고딕에는 수학 기호 '−'가 없어 일반 하이픈으로 표시합니다.
def apply_plot_style():
    plt.rcParams.update({
        "font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 9,
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 1.6,
    })




# 4. 차분 자료: Δ세전 공급가 ~ Δ국제가(같은 주·1주 전·2주 전). 상수 없이 Newey-West 표준오차로 추정합니다.
# 달력상 연속인 주별 표에서 차분하므로, 빠진 주에 걸린 차분은 빈칸이 되어 학습에서 빠집니다.
def diff_data(weekly, fuel, lags=LAGS):
    dy = weekly[f"{fuel}_supply_pretax"].diff()
    dX = lagged_intl(weekly, fuel, lags).diff()
    ok = ~weekly["in_policy"] & dX.notna().all(axis=1) & dy.notna()
    return dX[ok], dy[ok]


def diff_regression(dX, dy):
    return sm.OLS(dy, dX).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})


# 상수 없는 회귀의 statsmodels R²는 평균을 빼지 않은 값이므로, 평균을 뺀 R²를 따로 계산합니다.
def centered_r2(result, dy):
    return 1 - result.ssr / ((dy - dy.mean()) ** 2).sum()


# ADF 검정의 p값입니다. statsmodels 0.15의 반환 형식 변경 안내는 결과와 무관해 숨깁니다.
def adf_p(series):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return adfuller(series, autolag="AIC")[1]




# 5. ① 정상성·공적분·자기상관과 ② 차분 회귀 결과를 한 유종씩 계산합니다.
# ADF p가 크면 단위근(비정상)입니다. 이때 수준 회귀의 R²는 허구적 회귀일 수 있어 근거로 쓰지 않습니다.
# Engle-Granger p가 크면 "고정 마진의 장기 관계"가 확인되지 않은 것입니다. 마진을 최근 값으로 다시 잡는 이유입니다.
def unit_root_checks(weekly, fuel):
    X, y = training_data(weekly, fuel)
    pre = weekly[~weekly["in_policy"]]
    supply, intl = pre[f"{fuel}_supply_pretax"], pre[f"{fuel}_intl_krw_l"]
    weights = fit(X, y)["weights"]
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef, ssr = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[:2]
    dX, dy = diff_data(weekly, fuel)
    result = diff_regression(dX, dy)
    total = result.t_test((np.ones(LAGS), 1.0))
    return {
        ("① 정상성", "ADF p: 공급가 수준"): adf_p(supply.dropna()),
        ("① 정상성", "ADF p: 공급가 차분"): adf_p(supply.diff().dropna()),
        ("① 정상성", "ADF p: 국제가 수준"): adf_p(intl.dropna()),
        ("① 정상성", "ADF p: 국제가 차분"): adf_p(intl.diff().dropna()),
        ("① 공적분", "Engle-Granger p: 공급가 대 가중 국제가"): coint(y, X.to_numpy() @ weights)[1],
        ("① 자기상관", "수준 회귀 Durbin-Watson"): durbin_watson(y.to_numpy() - A @ coef),
        ("① 자기상관", "수준 회귀 R² (근거로 쓰지 않음)"): 1 - ssr[0] / ((y - y.mean()) ** 2).sum(),
        **{("② 차분 회귀", f"계수 lag{k}"): result.params.iloc[k] for k in range(LAGS)},
        ("② 차분 회귀", "계수 합"): result.params.sum(),
        ("② 차분 회귀", "계수 합 표준오차 (Newey-West)"): float(np.ravel(total.sd)[0]),
        ("② 차분 회귀", "'계수 합 = 1' p"): float(total.pvalue),
        ("② 차분 회귀", "R² (평균 제거)"): centered_r2(result, dy),
        ("② 차분 회귀", "관측 주"): len(dy),
    }




# 6. ③ 시차 수: 시차 1~5개 모형을 같은 학습 행(시차 5개가 모두 있는 주)으로 검증해 MAE를 비교합니다.
def lag_count_checks(weekly, fuel):
    X, y = training_data(weekly, fuel, lags=MAX_LAGS)
    return {("③ 시차 수", f"기점 이동 MAE: 시차 {lags}개"): np.abs(backtest(X.iloc[:, :lags], y, fit)).mean()
            for lags in range(1, MAX_LAGS + 1)}




# 7. ④ 구간 적중률: 구간을 만든 오차와 확인하는 오차를 시기로 나눕니다. (같은 오차로 재면 순환 논리)
# 앞 = 32주 예측 구간이 2021년 안에 끝나는 기점, 뒤 = 2022년 이후에 시작하는 기점이라 예측 대상 주가 겹치지 않습니다.
def coverage_check(weekly, fuel):
    X, y = training_data(weekly, fuel)
    errors = backtest(X, y, fit)
    points = cutoffs(X)
    early = np.array([X.index[c + HORIZON - 1] < SPLIT_DATE for c in points])
    late = np.array([X.index[c] >= SPLIT_DATE for c in points])
    q05, q95 = np.quantile(errors[early].ravel(), [0.05, 0.95])
    hit = (errors[late] >= q05) & (errors[late] <= q95)
    return {("④ 구간 적중률", "2021년까지 끝나는 기점 수"): early.sum(),
            ("④ 구간 적중률", "2022년 이후 기점 수"): late.sum(),
            ("④ 구간 적중률", "앞 시기 오차 q05"): q05,
            ("④ 구간 적중률", "앞 시기 오차 q95"): q95,
            ("④ 구간 적중률", "뒤 시기 적중률 (목표 0.90)"): hit.mean()}




# 8. ⑤ 민감도: 시차 비중을 수준 회귀와 차분 회귀 중 어디서 가져오는지, 마진을 몇 주로 잡는지를 바꿉니다.
# 정책 기간 주마다 6가지 사양의 격차("없었다면" 세후 − 실제 세후 공급가)를 계산합니다.
def sensitivity(weekly, fuel):
    X, y = training_data(weekly, fuel)
    X_policy = lagged_intl(weekly, fuel)[weekly["in_policy"]]
    policy = weekly[weekly["in_policy"]]
    dX, dy = diff_data(weekly, fuel)
    diff_coef = diff_regression(dX, dy).params.to_numpy()
    sources = {"수준": fit(X, y)["weights"], "차분": diff_coef / diff_coef.sum()}
    gaps = {}
    for source, weights in sources.items():
        for margin_weeks in MARGIN_OPTIONS:
            params = {"const": recent_margin(X, y, weights, margin_weeks), "weights": weights}
            aftertax = predict(params, X_policy) * (1 + VAT) + policy[f"{fuel}_tax"]
            gaps[f"{source}_{margin_weeks}주"] = aftertax - policy[f"{fuel}_supply_aftertax"]
    return pd.DataFrame(gaps)




# 9. ⑦ 그림: 2행 3열. 위 = 2025년 7월부터 실제 세후 공급가·"없었다면" 세후 공급가·90% 구간, 아래 = 정책 기간 격차.
def plot_counterfactual(cf, path):
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    shown = cf[cf.index >= FIG_START]
    policy = cf[cf["in_policy"]]
    start = policy.index[0]
    for col, (fuel, spec) in enumerate(FUELS.items()):
        color = COLORS[fuel]
        ax = axes[0, col]
        ax.fill_between(shown.index, shown[f"{fuel}_cf_low"], shown[f"{fuel}_cf_high"], color=color, alpha=0.15,
                        linewidth=0, label="90% 구간")
        ax.plot(shown.index, shown[f"{fuel}_cf_aftertax"], color=color, label="\"없었다면\" 세후 공급가")
        ax.plot(shown.index, shown[f"{fuel}_supply_aftertax"], color=INK_2, label="실제 세후 공급가")
        ax.axvline(start, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(f"{spec['ko']}: 세후 공급가 (원/리터)", loc="left")
        ax.legend(loc="upper left", fontsize=8)
        ax = axes[1, col]
        ax.fill_between(policy.index, policy[f"{fuel}_gap_low"], policy[f"{fuel}_gap_high"], color=color, alpha=0.15,
                        linewidth=0, label="90% 구간")
        ax.plot(policy.index, policy[f"{fuel}_gap"], color=color, marker="o", markersize=3, label="격차")
        ax.axhline(0, color=AXIS, linewidth=1)
        last = policy.index[-1]
        ax.annotate(f"{policy[f'{fuel}_gap'].iloc[-1]:.0f}원", (last, policy[f"{fuel}_gap"].iloc[-1]),
                    textcoords="offset points", xytext=(-4, 8), ha="right", color=INK, fontsize=8)
        ax.set_title(f"{spec['ko']}: 격차 = \"없었다면\" - 실제 (원/리터)", loc="left")
        ax.legend(loc="upper right", fontsize=8)
        for ax in axes[:, col]:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y.%m"))
    fig.savefig(path, dpi=150)
    plt.close(fig)




# 10. 유종별 점검 결과를 (점검, 항목) 행 × 유종 열의 표로 모읍니다.
def wide_table(per_fuel):
    table = pd.DataFrame({FUELS[fuel]["ko"]: pd.Series(values) for fuel, values in per_fuel.items()})
    table.index = pd.MultiIndex.from_tuples(table.index, names=["점검", "항목"])
    return table




# 11. 점검 ①~⑦을 실행하고 results/regression/full_sample 폴더에 저장합니다.
def main():
    try:
        weekly = load_weekly()
        cf, _ = build_counterfactual(weekly)
        checks = {fuel: {**unit_root_checks(weekly, fuel), **lag_count_checks(weekly, fuel),
                         **coverage_check(weekly, fuel)} for fuel in FUELS}
        table = wide_table(checks)
        sens = {fuel: sensitivity(weekly, fuel) for fuel in FUELS}
        summary = pd.DataFrame({"주차": cf.loc[cf["in_policy"], "주차"]})
        for fuel, gaps in sens.items():
            summary[f"{fuel}_gap_adopted"] = gaps["수준_52주"]
            summary[f"{fuel}_gap_min"] = gaps.min(axis=1)
            summary[f"{fuel}_gap_max"] = gaps.max(axis=1)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_DIR / "counterfactual_checks.csv", encoding="utf-8-sig", float_format="%.4f")
        summary.to_csv(OUT_DIR / "counterfactual_sensitivity.csv", encoding="utf-8-sig", float_format="%.2f")
        plot_counterfactual(cf, OUT_DIR / "counterfactual.png")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"근거 점검 실패: {error}") from None
    pd.set_option("display.width", 200)
    print("[근거 점검 ①~④]")
    print(table.round(3).to_string())
    print("\n[⑤ 민감도] 6가지 사양의 격차 범위 (원/리터)")
    for week in summary.index[-4:]:
        row = summary.loc[week]
        print(f"  {row['주차']}: " + " / ".join(
            f"{FUELS[f]['ko']} {row[f'{f}_gap_adopted']:.0f} ({row[f'{f}_gap_min']:.0f}~{row[f'{f}_gap_max']:.0f})"
            for f in FUELS))
    print(f"\n[⑥ 외부 대조] {CHECK_WEEK.date()} 주 격차 vs 정부 설명(2026-09-18)")
    row = cf.loc[CHECK_WEEK]
    for fuel, spec in FUELS.items():
        print(f"  {spec['ko']}: {row[f'{fuel}_gap']:.0f}원 (90% 구간 {row[f'{fuel}_gap_low']:.0f}~"
              f"{row[f'{fuel}_gap_high']:.0f}) / 정부: {GOV_CLAIM[fuel]}")
    print(f"\n저장 완료: {OUT_DIR}")




# 12. 직접 실행할 때만 근거 점검을 시작합니다.
if __name__ == "__main__":
    main()
