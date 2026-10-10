# 1. 팀원 회귀분석(ols 111)과 우리 회귀분석(regression/full_sample, train_test)을 같은 조건에서 비교합니다.
# ① 그대로 실행: ols 111/regression_before_tax.py의 모형·분할 결과를 그대로 가져옵니다.
#    그 평가 구간 122주에는 정책 기간 29주가 들어 있으므로 정책 이전 93주와 정책 기간 29주로 나눠 봅니다.
# ② 같은 조건 비교: 다섯 모형을 같은 학습 행으로 학습하고 세 방법으로 평가합니다.
#    모형은 팀원 모형에서 우리 모형까지 한 가지씩 바꿔 가며 어느 차이가 성적을 가르는지 봅니다.
#    평가: (가) 팀원 분할의 정책 이전 평가 93주 (나) 우리 분할의 평가 116주 (다) 기점 이동 검증 31회 × 32주
# ③ 정책 기간 적용: 모형마다 "없었다면" 세후 공급가를 만들어 실제 세후 공급가와의 격차를 비교합니다.
# ④ 환율 계수: 국제가가 이미 원화인데도 환율 계수가 양(+)으로 나오는 이유를 마진 추세와 비교합니다.
# 팀원 코드는 고치지 않고 함수만 가져다 씁니다. 결과는 results/regression/comparison 폴더에 저장합니다.
# 실행: python "regression/comparison/compare_ols111.py"
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "regression" / "full_sample"))
sys.path.insert(0, str(ROOT / "ols 111"))
from counterfactual import (
    FUELS, INPUT_PATH, VAT, backtest, build_counterfactual, lagged_intl, load_weekly, predict, recent_margin,
    summarize, week_end,
)
from counterfactual_check import AXIS, INK, INK_2, MUTED, apply_plot_style
from regression_before_tax import load_clean_csv, run_single_model, split_indices, LAG_WEEKS as OLS_LAG




# 2. 비교할 모형과 출력 위치입니다.
# 다섯 모형은 한 단계씩만 다릅니다: 환율 제외 → 시차 0·1·2주 → 최근 마진.
# spec: 설명변수 (t1_fx = 1주 전 국제가 + 이번 주 환율, t1 = 1주 전 국제가, lags = 같은 주·1주 전·2주 전 국제가)
# mode: fixed = OLS 절편과 계수를 그대로 사용, margin = 국제가 계수를 합 1로 나누고 상수는 직전 52주 평균 마진
OUT_DIR = ROOT / "results" / "regression" / "comparison"
FX_COLUMN = "원달러환율_주간평균_원/달러 환율"
OUR_SPLIT_END = pd.Timestamp("2023-12-16")    # 우리 분할(train_test)의 학습 마지막 주
VARIANTS = {
    "ols111": ("t1_fx", "fixed", "① 팀원 모형: 국제가(1주 전) + 환율"),
    "no_fx": ("t1", "fixed", "② ①에서 환율 제외"),
    "three_lags": ("lags", "fixed", "③ ②를 시차 0·1·2주로"),
    "t1_margin": ("t1", "margin", "④ ②에 최근 마진"),
    "ours": ("lags", "margin", "⑤ 우리 모형: 시차 0·1·2주 + 최근 마진"),
}
SHORT = {"ols111": "팀원 모형", "no_fx": "환율 제외", "three_lags": "시차 3개",
         "t1_margin": "1주 시차\n+최근 마진", "ours": "우리 모형"}
PRODUCT = {spec["ko"]: fuel for fuel, spec in FUELS.items()}   # 팀원 코드의 제품 이름 → 우리 유종 키
TEAM_COLOR, OUR_COLOR, OTHER_COLOR, POLICY_FILL = "#eb6834", "#2a78d6", "#c3c2b7", "#f0efec"




# 3. 팀원 코드를 그대로 실행해 제품별 계수와 주별 예측값을 받습니다.
# 팀원 코드는 행 단위로 1주 시차를 붙이므로 결측으로 빠진 2017-10-07 주 바로 뒤 한 행만 2주 전 값이 됩니다.
def run_team_models():
    _, rows = load_clean_csv()
    train_index, test_index = split_indices(len(rows) - OLS_LAG)
    results = {}
    for product in PRODUCT:
        result = run_single_model(product, rows, train_index, test_index)
        frame = pd.DataFrame(result["예측"])
        frame.index = pd.DatetimeIndex([week_end(label) for label in frame["주차"]])
        results[PRODUCT[product]] = {"coef": result["계수"], "train_end": week_end(result["학습기간"][1]),
                                     "train": result["학습성능"], "test": result["평가성능"], "pred": frame}
    return results




# 4. 같은 조건 비교에 쓸 설명변수와 학습 함수입니다.
# 환율은 load_weekly가 읽지 않으므로 통합 자료에서 따로 읽어 같은 주별 표에 맞춥니다.
def load_fx(weekly):
    raw = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
    fx = pd.Series(raw[FX_COLUMN].to_numpy(), index=pd.DatetimeIndex([week_end(label) for label in raw["주차"]]))
    return fx.reindex(weekly.index)


def features(weekly, fx, fuel, spec):
    intl = weekly[f"{fuel}_intl_krw_l"]
    if spec == "t1_fx":
        return pd.DataFrame({"lag1": intl.shift(1), "fx": fx})
    if spec == "t1":
        return pd.DataFrame({"lag1": intl.shift(1)})
    return lagged_intl(weekly, fuel)


def fit_variant(X, y, mode):
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
    if mode == "fixed":
        return {"const": coef[0], "weights": coef[1:]}
    weights = coef[1:] / coef[1:].sum()
    return {"const": recent_margin(X, y, weights), "weights": weights}


# 다섯 모형이 모두 같은 주로 학습하도록, 모든 설명변수와 세전 공급가가 있는 정책 이전 주만 남깁니다.
def common_rows(weekly, fx, fuel):
    y = weekly[f"{fuel}_supply_pretax"]
    allX = pd.concat([features(weekly, fx, fuel, "t1_fx"), lagged_intl(weekly, fuel)], axis=1)
    return ~weekly["in_policy"] & allX.notna().all(axis=1) & y.notna()


def r2(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    return 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()




# 5. ② 한 번 나눈 평가 구간에서의 성적입니다. 오차 = 예측 − 실제이므로 편향 < 0 이면 낮게 예측한 것입니다.
# 주간 변화 R²: 예측 공급가의 주간 변화가 실제 주간 변화를 얼마나 설명하는지 (평가 구간 안의 이웃한 주끼리)
def holdout(X, y, ok, train_end, mode):
    train, test = ok & (X.index <= train_end), ok & (X.index > train_end)
    params = fit_variant(X[train], y[train], mode)
    pred = pd.Series(predict(params, X[test]), index=X.index[test])
    actual = y[test]
    error = pred - actual
    return {"MAE": error.abs().mean(), "편향": error.mean(), "RMSE": np.sqrt((error ** 2).mean()),
            "수준 R²": r2(actual, pred), "주간 변화 R²": r2(actual.diff().dropna(), pred.diff().dropna()),
            "주": int(test.sum())}, params


# (다) 기점 이동 검증: counterfactual.backtest와 같은 31개 기점, 32주 앞
def rolling(X, y, ok, mode):
    errors = backtest(X[ok], y[ok], lambda Xt, yt: fit_variant(Xt, yt, mode))
    score = summarize(errors)
    return {"MAE": score["MAE"], "편향": score["bias"], "RMSE": score["RMSE"],
            "1~4주 MAE": score["MAE_1_4w"], "29~32주 MAE": score["MAE_last_4w"], "기점": score["n_cutoffs"]}




# 6. ③ 정책 기간 격차: "없었다면" 세후 = 세전 예측 × 1.1 + 실제 유류세, 격차 = 그 값 − 실제 세후 공급가
def policy_gap(weekly, X, pretax_params=None, pretax=None, fuel=None):
    policy = weekly["in_policy"]
    if pretax is None:
        pretax = pd.Series(predict(pretax_params, X[policy]), index=X.index[policy])
    pretax = pretax.reindex(weekly.index[policy])
    return pretax * (1 + VAT) + weekly.loc[policy, f"{fuel}_tax"] - weekly.loc[policy, f"{fuel}_supply_aftertax"]




# 7. ④ 환율 계수와 마진 추세: 우리 모형의 마진(세전 공급가 − 국제가 가중평균)이 해마다 어떻게 변했는지와
# 팀원 모형의 환율 계수 × 환율 변화가 얼마인지를 같은 기간(2015년 → 2023년, 팀원 학습 구간 안)으로 비교합니다.
def fx_vs_margin(weekly, fx, fuel, ok, team_coef):
    X = lagged_intl(weekly, fuel)
    y = weekly[f"{fuel}_supply_pretax"]
    weights = fit_variant(X[ok], y[ok], "margin")["weights"]
    margin = (y - X.to_numpy() @ weights)[ok]
    by_year = margin.groupby(margin.index.year).mean()
    fx_year = fx[ok].groupby(fx[ok].index.year).mean()
    return {"마진 2015": by_year[2015], "마진 2023": by_year[2023], "마진 변화": by_year[2023] - by_year[2015],
            "환율 2015": fx_year[2015], "환율 2023": fx_year[2023],
            "환율 계수 × 환율 변화": team_coef[2] * (fx_year[2023] - fx_year[2015]),
            "마진과 환율의 상관": margin.corr(fx[ok])}




# 8. 그림: 위 = 팀원 평가 구간(정책 기간 포함)의 실제·팀원 모형·우리 모형, 아래 = 기점 이동 검증 MAE (다섯 모형)
def plot_comparison(weekly, team, ours_pred, scores, path):
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), constrained_layout=True)
    policy_start = weekly.index[weekly["in_policy"]][0]
    for col, (fuel, spec) in enumerate(FUELS.items()):
        ax = axes[0, col]
        test = team[fuel]["pred"]
        test = test[test["구분"] == "test"]
        ax.axvspan(policy_start, test.index[-1], color=POLICY_FILL, linewidth=0, label="정책 기간 (상한)")
        ax.plot(test.index, test["실제_세전가격"], color=INK_2, label="실제")
        ax.plot(test.index, test["예측_세전가격"], color=TEAM_COLOR, label="팀원 모형")
        ax.plot(ours_pred[fuel].index, ours_pred[fuel], color=OUR_COLOR, label="우리 모형")
        ax.set_title(f"{spec['ko']}: 팀원 평가 구간 세전 공급가 (원/리터)", loc="left")
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y.%m"))
        ax = axes[1, col]
        names = list(VARIANTS)
        values = [scores[(fuel, name)]["MAE"] for name in names]
        colors = [TEAM_COLOR if n == "ols111" else OUR_COLOR if n == "ours" else OTHER_COLOR for n in names]
        bars = ax.bar(range(len(names)), values, color=colors, edgecolor="#fcfcfb", linewidth=1)
        for bar, value in zip(bars, values):
            ax.annotate(f"{value:.1f}", (bar.get_x() + bar.get_width() / 2, value), textcoords="offset points",
                        xytext=(0, 3), ha="center", color=INK, fontsize=8)
        ax.set_xticks(range(len(names)), [SHORT[n] for n in names], fontsize=8)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{spec['ko']}: 기점 이동 검증 MAE (31회 × 32주, 원/리터)", loc="left")
    fig.savefig(path, dpi=150)
    plt.close(fig)




# 9. ①~④를 실행하고 표와 그림을 저장합니다.
def main():
    try:
        weekly = load_weekly()
        fx = load_fx(weekly)
        team = run_team_models()
        full_cf, _ = build_counterfactual(weekly)
        team_rows, score_rows, policy, fx_rows, scores, ours_pred = [], [], {}, {}, {}, {}
        for fuel, spec in FUELS.items():
            ko = spec["ko"]
            pred = team[fuel]["pred"]
            test = pred[pred["구분"] == "test"]
            in_policy = weekly["in_policy"].reindex(test.index).fillna(False).astype(bool)
            for part, rows in {"평가 122주 전체": test, "정책 이전 93주": test[~in_policy],
                               "정책 기간 29주": test[in_policy]}.items():
                error = rows["예측_세전가격"] - rows["실제_세전가격"]
                team_rows.append({"유종": ko, "구간": part, "주": len(rows), "R²": r2(rows["실제_세전가격"], rows["예측_세전가격"]),
                                  "MAE": error.abs().mean(), "편향(예측−실제)": error.mean()})
            y = weekly[f"{fuel}_supply_pretax"]
            ok = common_rows(weekly, fx, fuel)
            for name, (feature_spec, mode, label) in VARIANTS.items():
                X = features(weekly, fx, fuel, feature_spec)
                for evaluation, train_end in {"(가) 팀원 분할 정책 이전 93주": team[fuel]["train_end"],
                                              "(나) 우리 분할 116주": OUR_SPLIT_END}.items():
                    score, params = holdout(X, y, ok, train_end, mode)
                    score_rows.append({"평가": evaluation, "유종": ko, "모형": label, **score})
                    if name == "ours" and evaluation.startswith("(가)"):
                        # 그림용: 팀원 학습 구간으로 학습한 우리 모형을 팀원 평가 구간(정책 기간 포함)에 적용
                        shown = (ok & (X.index > train_end)) | weekly["in_policy"]
                        ours_pred[fuel] = pd.Series(predict(params, X[shown]), index=X.index[shown])
                scores[(fuel, name)] = rolling(X, y, ok, mode)
                score_rows.append({"평가": "(다) 기점 이동 검증 31회 × 32주", "유종": ko, "모형": label,
                                   **scores[(fuel, name)]})
            X_team = features(weekly, fx, fuel, "t1_fx")
            team_pretax = team[fuel]["pred"]["예측_세전가격"]
            policy[(ko, "팀원 모형 (팀원 학습 구간 ~2024-05)")] = policy_gap(weekly, X_team, pretax=team_pretax, fuel=fuel)
            policy[(ko, "팀원 모형 구성 (정책 이전 전체로 학습)")] = policy_gap(
                weekly, X_team, fit_variant(X_team[ok], y[ok], "fixed"), fuel=fuel)
            policy[(ko, "우리 모형 (full_sample)")] = full_cf.loc[weekly["in_policy"], f"{fuel}_gap"]
            fx_rows[ko] = fx_vs_margin(weekly, fx, fuel, ok, team[fuel]["coef"])
        team_table = pd.DataFrame(team_rows)
        score_table = pd.DataFrame(score_rows)
        policy_table = pd.DataFrame(policy)
        policy_table.insert(0, "주차", weekly.loc[policy_table.index, "주차"])
        fx_table = pd.DataFrame(fx_rows)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        team_table.to_csv(OUT_DIR / "team_model_as_is.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        score_table.to_csv(OUT_DIR / "same_conditions.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        policy_table.to_csv(OUT_DIR / "policy_gap.csv", encoding="utf-8-sig", float_format="%.2f")
        fx_table.to_csv(OUT_DIR / "fx_vs_margin.csv", encoding="utf-8-sig", float_format="%.4f")
        plot_comparison(weekly, team, ours_pred, scores, OUT_DIR / "compare_ols111.png")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"비교 실패: {error}") from None
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print("[① 팀원 모형 그대로] 평가 구간을 정책 이전과 정책 기간으로 나눔 (세전 원/리터)")
    print(team_table.round(3).to_string(index=False))
    for evaluation in score_table["평가"].unique():
        part = score_table[score_table["평가"] == evaluation].dropna(axis=1, how="all")
        print(f"\n[② {evaluation}]")
        print(part.drop(columns="평가").round(3).to_string(index=False))
    print("\n[③ 정책 기간 격차] \"없었다면\" 세후 − 실제 세후 공급가 (원/리터)")
    show = policy_table.loc[policy_table.index.isin(pd.to_datetime(["2026-03-28", "2026-06-27", "2026-09-12", "2026-09-26"]))]
    print(show.round(0).to_string())
    print("정책 기간 평균:")
    print(policy_table.drop(columns="주차", level=0).mean().round(1).to_string())
    print("\n[④ 환율 계수와 마진 추세] (2015년 → 2023년)")
    print(fx_table.round(3).to_string())
    print(f"\n저장 완료: {OUT_DIR}")




# 10. 직접 실행할 때만 비교를 시작합니다.
if __name__ == "__main__":
    main()
