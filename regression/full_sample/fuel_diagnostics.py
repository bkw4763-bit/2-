# 1. 유종별 예측 난이도 진단입니다. 1단계 회귀의 성적이 유종마다 다른 이유를 여섯 기준으로 확인합니다.
# ① 자기상관(ACF): 휘발유의 과거 가격 의존성이 더 강한가?
# ② 교차상관(CCF): 국제가격과 국내 공급가격의 반영 시차가 제품별로 다른가?
# ③ 변동률의 표준편차: 경유·등유가 실제로 더 불규칙한가?
# ④ 잔차 분석: 경유·등유에서 특정 시기에 오차가 집중되는가?
# ⑤ 변수 중요도: 휘발유에서 사용한 입력변수가 더 높은 설명력을 가지는가?
# ⑥ Naive 모델 비교: 단순히 가격이 이전 값과 비슷해서 휘발유 예측이 쉬운 것인가?
# 모든 계산은 정책 이전(2026-03-07까지) 자료만 씁니다. 가격 수준은 단위근이 있어 상관이 모두 1에 가까우므로,
# 의존성·시차·설명력은 주간 변화(차분)로 봅니다. 수준 값은 참고로만 함께 적습니다.
# 실행: python "regression/full_sample/fuel_diagnostics.py"
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2, kurtosis
from statsmodels.stats.diagnostic import het_arch

from counterfactual import FUELS, HORIZON, OUT_DIR, cutoffs, fit, load_weekly, predict, training_data
from counterfactual_check import (
    AXIS, COLORS, INK, INK_2, MUTED, apply_plot_style, centered_r2, diff_data, diff_regression, wide_table,
)




# 2. 진단 매개변수입니다.
ACF_LAGS = 8              # 자기상관을 볼 최대 시차(주)
CCF_LAGS = range(-4, 9)   # 교차상관 시차. 음수는 국내 공급가가 국제가보다 먼저 움직이는 방향
PASS_LAGS = 7             # 누적 반영률을 볼 차분 회귀 시차 수 (같은 주 ~ 6주 전)
ARCH_LAGS = 4             # 오차 변동성 군집 검정 시차
PERMUTATIONS = 30         # 변수 중요도 순열 반복 횟수
TOP_SHARE = 0.10          # 오차 집중도: 오차가 큰 상위 10% 주가 전체 절대오차에서 차지하는 비중
LAG_NAMES = ["같은 주", "1주 전", "2주 전"]




# 3. 정책 이전 주간 변화(차분)입니다. 달력상 연속인 표에서 차분하므로 빠진 주에 걸린 변화는 빈칸입니다.
def pre_policy_changes(weekly, fuel):
    pre = ~weekly["in_policy"]
    supply = weekly[f"{fuel}_supply_pretax"]
    intl = weekly[f"{fuel}_intl_krw_l"]
    return {"supply": supply.where(pre), "intl": intl.where(pre),
            "d_supply": supply.diff().where(pre), "d_intl": intl.diff().where(pre),
            "pct_supply": (supply.pct_change(fill_method=None) * 100).where(pre),
            "pct_intl": (intl.pct_change(fill_method=None) * 100).where(pre)}


# k주 전 값과의 상관계수입니다. 빈칸이 있는 쌍은 빼고 계산합니다.
def lag_corr(a, b, lag):
    return a.corr(b.shift(lag))


# Ljung-Box 검정: 시차 1~m 자기상관이 모두 0인지 봅니다. p가 작으면 과거 변화가 다음 변화와 관련이 있습니다.
def ljung_box_p(series, lags=ACF_LAGS):
    n = series.notna().sum()
    q = n * (n + 2) * sum(lag_corr(series, series, k) ** 2 / (n - k) for k in range(1, lags + 1))
    return chi2.sf(q, lags)




# 4. 기점 이동 검증을 한 줄에 한 예측씩 펼친 표로 만듭니다. (counterfactual.backtest와 같은 기점·같은 학습)
# naive = 기점 직전 주의 실제 공급가를 32주 동안 그대로 쓰는 예측입니다. 국제가 정보를 쓰지 않습니다.
def backtest_frame(X, y):
    frames = []
    for c in cutoffs(X):
        params = fit(X.iloc[:c], y.iloc[:c])
        window = slice(c, c + HORIZON)
        frames.append(pd.DataFrame({
            "cutoff": X.index[c], "h": np.arange(1, HORIZON + 1), "target": X.index[window],
            "actual": y.iloc[window].to_numpy(), "pred": predict(params, X.iloc[window]), "naive": y.iloc[c - 1],
        }))
    frame = pd.concat(frames, ignore_index=True)
    frame["error"] = frame["pred"] - frame["actual"]
    frame["naive_error"] = frame["naive"] - frame["actual"]
    return frame




# 5. ① 자기상관: 공급가 변화가 자기 과거 변화와 얼마나 관련 있는지 봅니다.
# 국제가 변화를 이미 넣은 차분 회귀에 "지난주 공급가 변화"를 더했을 때 설명력이 얼마나 느는지도 봅니다.
# 이 증가분이 작으면 공급가의 자기상관은 국제가를 시차를 두고 따라가며 생긴 것입니다.
def acf_checks(weekly, fuel, changes):
    d_supply, supply = changes["d_supply"], changes["supply"]
    dX, dy = diff_data(weekly, fuel)
    own = weekly[f"{fuel}_supply_pretax"].diff().shift(1).reindex(dX.index)
    ok = own.notna()
    base = centered_r2(diff_regression(dX[ok], dy[ok]), dy[ok])
    with_own = centered_r2(diff_regression(dX[ok].assign(own=own[ok]), dy[ok]), dy[ok])
    return {
        ("① 자기상관", "수준 ACF 1주 (참고)"): lag_corr(supply, supply, 1),
        ("① 자기상관", "수준 ACF 52주 (참고)"): lag_corr(supply, supply, 52),
        **{("① 자기상관", f"변화 ACF {k}주"): lag_corr(d_supply, d_supply, k) for k in (1, 2, 3, 4)},
        ("① 자기상관", f"변화 Ljung-Box p (1~{ACF_LAGS}주)"): ljung_box_p(d_supply),
        ("① 자기상관", "국제가 변화 ACF 1주"): lag_corr(changes["d_intl"], changes["d_intl"], 1),
        ("① 자기상관", "지난주 공급가 변화 추가 시 R² 증가"): with_own - base,
    }


def acf_series(changes):
    d_supply = changes["d_supply"]
    return pd.Series({k: lag_corr(d_supply, d_supply, k) for k in range(1, ACF_LAGS + 1)})




# 6. ② 교차상관: k주 전 국제가 변화와 이번 주 공급가 변화의 상관입니다.
# 누적 반영률: 국제가가 1원 오른 뒤 k주 안에 공급가가 몇 원 올랐는지(차분 회귀 계수의 누적합)입니다.
# 평균 반영 시차 = Σ k·계수 / Σ 계수. 클수록 국제가가 늦게 반영됩니다.
def ccf_series(changes):
    return pd.Series({k: lag_corr(changes["d_supply"], changes["d_intl"], k) for k in CCF_LAGS})


def ccf_checks(weekly, fuel, changes):
    ccf = ccf_series(changes)
    coef = diff_regression(*diff_data(weekly, fuel, lags=PASS_LAGS)).params.to_numpy()[:PASS_LAGS]
    cumulative = coef.cumsum()
    return {
        ("② 교차상관", "최고 상관 시차(주)"): ccf.idxmax(),
        ("② 교차상관", "최고 상관값"): ccf.max(),
        **{("② 교차상관", f"CCF {k}주 전"): ccf[k] for k in (0, 1, 2, 3)},
        ("② 교차상관", "역방향(국내가 먼저) 최대 |상관|"): ccf[ccf.index < 0].abs().max(),
        **{("② 교차상관", f"누적 반영률 {k}주"): cumulative[k] for k in (0, 1, 2, PASS_LAGS - 1)},
        ("② 교차상관", "평균 반영 시차(주)"): (np.arange(PASS_LAGS) * coef).sum() / coef.sum(),
    }




# 7. ③ 변동률의 표준편차: 주간 변화율(%)의 표준편차가 클수록 가격이 크게 흔들립니다.
# 첨도가 클수록 평소에는 조용하다가 가끔 크게 튀는 불규칙한 움직임입니다. (정규분포 = 0)
# 마진 변화: 공급가 − 국제가 가중평균의 주간 변화. 모형이 국제가로 설명하지 못하는 부분의 흔들림입니다.
def volatility_checks(weekly, fuel, changes):
    X, y = training_data(weekly, fuel)
    margin = (y - X.to_numpy() @ fit(X, y)["weights"]).reindex(weekly.index)
    pct_supply, pct_intl = changes["pct_supply"].dropna(), changes["pct_intl"].dropna()
    return {
        ("③ 변동률", "공급가 주간 변화율 표준편차(%)"): pct_supply.std(),
        ("③ 변동률", "국제가 주간 변화율 표준편차(%)"): pct_intl.std(),
        ("③ 변동률", "공급가 / 국제가 변동률 비율"): pct_supply.std() / pct_intl.std(),
        ("③ 변동률", "공급가 주간 변화 표준편차(원)"): changes["d_supply"].std(),
        ("③ 변동률", "공급가 변화율 첨도"): kurtosis(pct_supply),
        ("③ 변동률", "마진 주간 변화 표준편차(원)"): margin.diff().std(),
        ("③ 변동률", "마진 13주 변화 표준편차(원)"): margin.diff(13).std(),
    }




# 8. ④ 잔차 분석: 표본 밖 오차가 특정 시기에 몰리는지 봅니다.
# 같은 주가 여러 기점의 예측 구간에 들어가면 그 주의 오차는 평균해 한 주에 하나의 오차로 만듭니다.
# ARCH 검정: 차분 회귀 잔차의 크기가 몰려 다니는지(큰 오차 뒤에 큰 오차) 봅니다. p가 작으면 시기 집중이 있습니다.
def residual_checks(weekly, fuel, frame):
    by_week = frame.groupby("target")["error"].mean().abs()
    by_year = frame.groupby(frame["target"].dt.year)["error"].apply(lambda e: e.abs().mean())
    top = by_week.nlargest(int(np.ceil(len(by_week) * TOP_SHARE)))
    abs_error = frame["error"].abs()
    dX, dy = diff_data(weekly, fuel)
    residual = diff_regression(dX, dy).resid
    return {
        ("④ 잔차", "표본 밖 MAE (전체)"): abs_error.mean(),
        ("④ 잔차", "오차가 가장 큰 해"): by_year.idxmax(),
        ("④ 잔차", "그 해 MAE"): by_year.max(),
        ("④ 잔차", "그 해 MAE / 전체 MAE"): by_year.max() / abs_error.mean(),
        ("④ 잔차", f"상위 {TOP_SHARE:.0%} 주의 절대오차 비중"): top.sum() / by_week.sum(),
        ("④ 잔차", "2022년 주의 절대오차 비중"): abs_error[frame["target"].dt.year == 2022].sum() / abs_error.sum(),
        ("④ 잔차", "2022년 제외 MAE"): abs_error[frame["target"].dt.year != 2022].mean(),
        ("④ 잔차", f"ARCH-LM p (차분 회귀 잔차, {ARCH_LAGS}주)"): arch_p(residual),
    }, by_year


# ARCH-LM 검정의 p값입니다. statsmodels 0.15의 반환 형식 변경 안내는 결과와 무관해 숨깁니다.
def arch_p(residual):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return het_arch(residual, nlags=ARCH_LAGS)[1]




# 9. ⑤ 변수 중요도: 국제가 시차들이 공급가 변화를 얼마나 설명하는지 봅니다.
# 수준 R²는 단위근 때문에 모두 0.99 안팎이라 비교할 수 없으므로, 차분 회귀 R²로 비교합니다.
# 제거 시 R² 감소: 그 시차 하나를 빼고 다시 추정했을 때 줄어드는 설명력입니다.
# 순열 중요도: 표본 밖 예측에서 그 시차 값을 예측 구간 안에서 무작위로 섞었을 때 MAE가 얼마나 커지는지 봅니다.
def importance_checks(weekly, fuel, X, y):
    dX, dy = diff_data(weekly, fuel)
    full = centered_r2(diff_regression(dX, dy), dy)
    drop = {k: full - centered_r2(diff_regression(dX.drop(columns=f"lag{k}"), dy), dy) for k in range(3)}
    rng = np.random.default_rng(0)
    base, permuted = [], {k: [] for k in range(3)}
    for c in cutoffs(X):
        params = fit(X.iloc[:c], y.iloc[:c])
        test_X, actual = X.iloc[c:c + HORIZON].to_numpy(), y.iloc[c:c + HORIZON].to_numpy()
        base.append(np.abs(params["const"] + test_X @ params["weights"] - actual).mean())
        for k in range(3):
            for _ in range(PERMUTATIONS):
                shuffled = test_X.copy()
                shuffled[:, k] = rng.permutation(shuffled[:, k])
                permuted[k].append(np.abs(params["const"] + shuffled @ params["weights"] - actual).mean())
    return {
        ("⑤ 변수 중요도", "차분 회귀 R²"): full,
        **{("⑤ 변수 중요도", f"제거 시 R² 감소: {LAG_NAMES[k]}"): drop[k] for k in range(3)},
        **{("⑤ 변수 중요도", f"순열 시 MAE 증가(원): {LAG_NAMES[k]}"): np.mean(permuted[k]) - np.mean(base)
           for k in range(3)},
    }, drop, full




# 10. ⑥ Naive 비교: 국제가를 보지 않고 기점 직전 가격을 그대로 쓰는 예측과 비교합니다.
# 기술 점수 = 1 − 모형 MAE / naive MAE. 1에 가까울수록 국제가 정보가 쓸모 있고, 0이면 naive와 같습니다.
# naive 오차가 작은 유종은 원래 가격이 잘 안 움직여 맞히기 쉬운 유종입니다.
def naive_checks(changes, frame):
    model, naive = frame["error"].abs(), frame["naive_error"].abs()
    first = frame["h"] == 1
    return {
        ("⑥ Naive 비교", "1주 naive MAE = 주간 변화 평균 절대값"): changes["d_supply"].abs().mean(),
        ("⑥ Naive 비교", "32주 naive MAE"): naive.mean(),
        ("⑥ Naive 비교", "32주 모형 MAE"): model.mean(),
        ("⑥ Naive 비교", "32주 기술 점수"): 1 - model.mean() / naive.mean(),
        ("⑥ Naive 비교", "1주째 naive MAE"): naive[first].mean(),
        ("⑥ Naive 비교", "1주째 모형 MAE"): model[first].mean(),
        ("⑥ Naive 비교", "1주째 기술 점수"): 1 - model[first].mean() / naive[first].mean(),
        ("⑥ Naive 비교", "모형 MAPE(%)"): (model / frame["actual"]).mean() * 100,
        ("⑥ Naive 비교", "naive MAPE(%)"): (naive / frame["actual"]).mean() * 100,
    }




# 11. 여섯 기준을 한 그림(2행 3열)에 담습니다. 유종 색은 counterfactual.png와 같습니다.
# title을 주면 그림 위에 분석 구간 등을 적습니다. (train_test 폴더에서 학습 구간 진단에 사용)
def plot_diagnostics(results, path, title=None):
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    if title:
        fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fuels = list(FUELS)
    width = 0.26
    offsets = {fuel: (i - 1) * width for i, fuel in enumerate(fuels)}

    ax = axes[0, 0]
    for fuel in fuels:
        acf = results[fuel]["acf"]
        ax.bar(acf.index + offsets[fuel], acf.values, width, color=COLORS[fuel], edgecolor="#fcfcfb",
               linewidth=1, label=FUELS[fuel]["ko"])
    band = 1.96 / np.sqrt(results[fuels[0]]["n_changes"])
    for level in (band, -band):
        ax.axhline(level, color=MUTED, linestyle="--", linewidth=1)
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.set_title("① 자기상관: 공급가 주간 변화의 ACF", loc="left")
    ax.set_xlabel("시차 (주)   점선 = 95% 유의 경계")
    ax.legend(loc="upper right")

    ax = axes[0, 1]
    for fuel in fuels:
        ccf = results[fuel]["ccf"]
        ax.plot(ccf.index, ccf.values, color=COLORS[fuel], marker="o", markersize=4, label=FUELS[fuel]["ko"])
    ax.axhline(0, color=AXIS, linewidth=1)
    ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_title("② 교차상관: k주 전 국제가 변화 → 공급가 변화", loc="left")
    ax.set_xlabel("k (주)   음수 = 국내 공급가가 먼저")
    ax.legend(loc="upper right")

    ax = axes[0, 2]
    positions = np.arange(len(fuels))
    supply_vol = [results[f]["checks"][("③ 변동률", "공급가 주간 변화율 표준편차(%)")] for f in fuels]
    intl_vol = [results[f]["checks"][("③ 변동률", "국제가 주간 변화율 표준편차(%)")] for f in fuels]
    colors = [COLORS[f] for f in fuels]
    ax.bar(positions - 0.18, supply_vol, 0.34, color=colors, edgecolor="#fcfcfb", linewidth=1)
    ax.bar(positions + 0.18, intl_vol, 0.34, color=colors, alpha=0.35, edgecolor="#fcfcfb", linewidth=1)
    for x, value in zip(positions - 0.18, supply_vol):
        ax.annotate(f"국내 {value:.1f}", (x, value), textcoords="offset points", xytext=(0, 3), ha="center",
                    color=INK, fontsize=8)
    for x, value in zip(positions + 0.18, intl_vol):
        ax.annotate(f"국제 {value:.1f}", (x, value), textcoords="offset points", xytext=(0, 3), ha="center",
                    color=INK_2, fontsize=8)
    ax.set_xticks(positions, [FUELS[f]["ko"] for f in fuels])
    ax.set_title("③ 주간 변화율의 표준편차 (%)  진한 = 국내 공급가, 연한 = 국제가", loc="left")

    ax = axes[1, 0]
    for fuel in fuels:
        by_year = results[fuel]["by_year"]
        ax.plot(by_year.index, by_year.values, color=COLORS[fuel], marker="o", markersize=4, label=FUELS[fuel]["ko"])
    ax.set_xticks(sorted({year for fuel in fuels for year in results[fuel]["by_year"].index}))
    ax.set_title("④ 잔차: 예측 대상 연도별 표본 밖 MAE (원/리터)", loc="left")
    ax.legend(loc="upper left")

    ax = axes[1, 1]
    groups = ["차분 회귀 R²"] + [f"{name} 제거 시\nR² 감소" for name in LAG_NAMES]
    positions = np.arange(len(groups))
    for fuel in fuels:
        values = [results[fuel]["r2"]] + [results[fuel]["drop"][k] for k in range(3)]
        ax.bar(positions + offsets[fuel], values, width, color=COLORS[fuel], edgecolor="#fcfcfb", linewidth=1,
               label=FUELS[fuel]["ko"])
        ax.annotate(f"{values[0]:.2f}", (positions[0] + offsets[fuel], values[0]), textcoords="offset points",
                    xytext=(0, 3), ha="center", color=INK, fontsize=8)
    ax.set_xticks(positions, groups)
    ax.set_title("⑤ 변수 중요도: 국제가 시차의 설명력", loc="left")
    ax.legend(loc="upper right")

    ax = axes[1, 2]
    for fuel in fuels:
        frame = results[fuel]["frame"]
        by_h = frame.groupby("h")[["error", "naive_error"]].apply(lambda e: e.abs().mean())
        ax.plot(by_h.index, by_h["error"], color=COLORS[fuel], label=f"{FUELS[fuel]['ko']} 모형")
        ax.plot(by_h.index, by_h["naive_error"], color=COLORS[fuel], linestyle="--", label=f"{FUELS[fuel]['ko']} naive")
    ax.set_title("⑥ Naive 비교: 기점 뒤 h주째 MAE (원/리터)", loc="left")
    ax.set_xlabel("h (주)   실선 = 모형, 점선 = 직전 가격 유지")
    ax.legend(loc="upper left", ncol=2, fontsize=8)

    fig.savefig(path, dpi=150)
    plt.close(fig)




# 12. 유종마다 ①~⑥을 계산합니다. weekly에 정책 이전 자료 중 어느 구간을 넣느냐에 따라 분석 구간이 정해집니다.
# (train_test 폴더는 학습 구간까지만 자른 표를 넣어 같은 진단을 학습 구간으로만 합니다)
def diagnose(weekly):
    results = {}
    for fuel in FUELS:
        X, y = training_data(weekly, fuel)
        changes = pre_policy_changes(weekly, fuel)
        frame = backtest_frame(X, y)
        residual, by_year = residual_checks(weekly, fuel, frame)
        importance, drop, r2 = importance_checks(weekly, fuel, X, y)
        checks = {**acf_checks(weekly, fuel, changes), **ccf_checks(weekly, fuel, changes),
                  **volatility_checks(weekly, fuel, changes), **residual, **importance,
                  **naive_checks(changes, frame)}
        results[fuel] = {"checks": checks, "acf": acf_series(changes), "ccf": ccf_series(changes),
                         "n_changes": changes["d_supply"].notna().sum(), "by_year": by_year,
                         "drop": drop, "r2": r2, "frame": frame, "changes": changes}
    return results


# 진단 결과를 기준별 표와 연도별 표(표본 밖 MAE, 공급가 주간 변화율 표준편차)로 정리합니다.
def summary_tables(results):
    table = wide_table({fuel: result["checks"] for fuel, result in results.items()})
    yearly = pd.DataFrame({
        **{f"{FUELS[f]['ko']}_표본밖MAE": results[f]["by_year"] for f in FUELS},
        **{f"{FUELS[f]['ko']}_공급가변화율표준편차": results[f]["changes"]["pct_supply"].groupby(
            results[f]["changes"]["pct_supply"].index.year).std() for f in FUELS},
    })
    yearly.index.name = "연도"
    return table, yearly


# 정책 이전 자료 전체로 진단하고 results/regression/full_sample 폴더에 표와 그림을 저장합니다.
def main():
    try:
        results = diagnose(load_weekly())
        table, yearly = summary_tables(results)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_DIR / "fuel_diagnostics.csv", encoding="utf-8-sig", float_format="%.4f")
        yearly.to_csv(OUT_DIR / "fuel_diagnostics_by_year.csv", encoding="utf-8-sig", float_format="%.3f")
        plot_diagnostics(results, OUT_DIR / "fuel_diagnostics.png")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"유종별 진단 실패: {error}") from None
    pd.set_option("display.width", 200)
    print("[유종별 예측 난이도 진단 ①~⑥]")
    print(table.round(3).to_string())
    print("\n[연도별] 표본 밖 MAE(원)와 공급가 주간 변화율 표준편차(%)")
    print(yearly.round(2).to_string())
    print(f"\n저장 완료: {OUT_DIR}")




# 13. 직접 실행할 때만 진단을 시작합니다.
if __name__ == "__main__":
    main()
