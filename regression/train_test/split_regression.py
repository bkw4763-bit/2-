# 1. 학습·평가 분할 회귀분석입니다. 정책 이전 578주를 시간 순서대로 앞 80%(학습)와 뒤 20%(평가)로 나눕니다.
# ① 분할: 가격은 시간에 따라 이어지므로 섞지 않습니다. 섞으면 평가 주의 바로 앞뒤 주가 학습에 들어가 성적이 부풀려집니다.
# ② 학습: 학습 구간만으로 OLS → 시차 계수를 합이 1인 비중으로 → 마진은 학습 구간 마지막 52주 평균
# ③ 학습 구간 안 검증: 기점 이동 검증(2018년부터 13주마다 32주 앞)으로 모형을 비교하고 90% 구간의 오차 분위수를 구합니다.
# ④ 평가: 학습한 모형을 고치지 않고 평가 구간 116주를 실제 국제가만 넣어 맞힙니다. 평가 구간 자료는 학습에 쓰지 않습니다.
# ⑤ 적용: 같은 모형으로 정책 기간 29주의 "없었다면" 가격을 만들고 전체 기간 모형(full_sample)과 비교합니다.
# 모형·자료 처리 함수는 full_sample 폴더의 counterfactual.py를 그대로 씁니다. 방법론은 같고 학습 자료만 다릅니다.
# 실행: python "regression/train_test/split_regression.py"
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "full_sample"))
from counterfactual import (
    CHOSEN, FUELS, HORIZON, LAGS, MODELS, PROJECT_DIR, VAT,
    backtest, build_counterfactual, fit, lagged_intl, load_weekly, predict, summarize, training_data,
)
from counterfactual_check import AXIS, COLORS, INK, INK_2, MUTED, apply_plot_style




# 2. 분할 비율과 출력 위치입니다.
OUT_DIR = PROJECT_DIR / "results" / "regression" / "train_test"
TRAIN_SHARE = 0.8
FIG_START = pd.Timestamp("2023-07-01")      # 그림에서 평가 구간 앞에 학습 구간 끝 반년을 함께 보여 줍니다
NAIVE = {"model": "naive", "name": "직전 가격 유지"}
# 학습 구간이 9년이므로 비교 모형 이름을 "11년 전체 회귀" 대신 "학습 구간 전체 회귀"로 적습니다.
NAMES = {**MODELS, "baseline": "학습 구간 전체 회귀"}




# 3. 학습 행(정책 이전이면서 시차 국제가와 세전 공급가가 모두 있는 주)을 시간순으로 세어 앞 80%의 마지막 주를 구합니다.
# 세 유종은 같은 주가 빠져 있어 학습 행이 같습니다. 다르면 같은 경계로 나눌 수 없으므로 중단합니다.
def split_date(weekly):
    indexes = [training_data(weekly, fuel)[0].index for fuel in FUELS]
    if any(not index.equals(indexes[0]) for index in indexes):
        raise ValueError("유종마다 학습 주가 달라 같은 경계로 나눌 수 없습니다.")
    return indexes[0][round(len(indexes[0]) * TRAIN_SHARE) - 1]


# 학습 구간까지만 남긴 주별 표입니다. full_sample의 점검·진단 함수에 넣으면 학습 구간만으로 계산됩니다.
def train_panel(weekly, train_end):
    return weekly[weekly.index <= train_end]


def split_data(weekly, fuel, train_end, lags=LAGS):
    X, y = training_data(weekly, fuel, lags)
    train = X.index <= train_end
    return X[train], y[train], X[~train], y[~train]




# 4. 평가 구간 성적입니다. 오차 = 예측 − 실제이므로 bias < 0 이면 낮게 예측한 것입니다.
# 1~32주: 학습이 끝난 뒤 32주. 정책 기간(29주)과 같은 조건이라 정책 분석의 성적에 해당합니다.
# 33주 이후: 마진을 다시 잡지 않고 오래 쓰면 성적이 얼마나 나빠지는지 봅니다.
# 적중률: 실제 공급가가 90% 구간 안에 든 주의 비율입니다. 구간은 학습 구간 오차로만 만들었습니다.
def test_scores(pred, actual, low=None, high=None):
    error = pred - actual
    first = np.arange(len(error)) < HORIZON
    scores = {"test_MAE": error.abs().mean(), "test_bias": error.mean(), "test_RMSE": np.sqrt((error ** 2).mean()),
              "test_MAPE": (error.abs() / actual).mean() * 100,
              "test_MAE_1_32w": error[first].abs().mean(), "test_bias_1_32w": error[first].mean(),
              "test_MAE_after_32w": error[~first].abs().mean(), "test_bias_after_32w": error[~first].mean()}
    if low is not None:
        hit = (actual >= low) & (actual <= high)
        scores.update({"test_cover_1_32w": hit[first].mean(), "test_cover_all": hit.mean()})
    return scores




# 5. 한 유종의 ②~⑤를 실행합니다.
# 반환: 모형 비교표 행, 평가 구간 주별 예측, 정책 기간 "없었다면" 가격과 격차
# naive = 학습 구간 마지막 주의 공급가를 평가 구간 내내 그대로 쓰는 예측입니다. 기술 점수 = 1 − 모형 MAE / naive MAE
def evaluate(weekly, fuel, train_end):
    X_train, y_train, X_test, y_test = split_data(weekly, fuel, train_end)
    preds = pd.DataFrame({"actual": y_test, "h": np.arange(1, len(y_test) + 1), "naive": y_train.iloc[-1]})
    naive = test_scores(preds["naive"], y_test)
    rows = []
    for model in MODELS:
        params = fit(X_train, y_train, model)
        inner = summarize(backtest(X_train, y_train, lambda Xt, yt, m=model: fit(Xt, yt, m)))
        pred = pd.Series(predict(params, X_test), index=X_test.index)
        low, high = pred - inner["err_q95"], pred - inner["err_q05"]
        preds[model] = pred
        if model == CHOSEN:
            preds["low"], preds["high"] = low, high
            chosen = (params, inner)
        score = test_scores(pred, y_test, low, high)
        rows.append({"fuel": fuel, "model": model, "name": NAMES[model], "train_weeks": len(X_train),
                     "train_start": X_train.index[0].date(), "train_end": X_train.index[-1].date(),
                     "test_weeks": len(X_test), "test_start": X_test.index[0].date(), "test_end": X_test.index[-1].date(),
                     "const": params["const"], **{f"w_lag{k}": params["weights"][k] for k in range(LAGS)},
                     **{f"train_{key}": value for key, value in inner.items()}, **score,
                     "test_skill_vs_naive": 1 - score["test_MAE"] / naive["test_MAE"]})
    rows.append({"fuel": fuel, **NAIVE, "train_weeks": len(X_train), "test_weeks": len(X_test), **naive,
                 "test_skill_vs_naive": 0.0})
    return rows, preds, apply_policy(weekly, fuel, *chosen)


# "없었다면" 세후 = 세전 예측 × 1.1 + 실제 유류세. 격차 = "없었다면" 세후 − 실제 세후 공급가
def apply_policy(weekly, fuel, params, inner):
    policy = weekly[weekly["in_policy"]]
    pretax = pd.Series(predict(params, lagged_intl(weekly, fuel)[weekly["in_policy"]]), index=policy.index)
    tax, supply = policy[f"{fuel}_tax"], policy[f"{fuel}_supply_aftertax"]
    return pd.DataFrame({
        "cf_aftertax": pretax * (1 + VAT) + tax,
        "gap": pretax * (1 + VAT) + tax - supply,
        "gap_low": (pretax - inner["err_q95"]) * (1 + VAT) + tax - supply,
        "gap_high": (pretax - inner["err_q05"]) * (1 + VAT) + tax - supply,
    })




# 6. 그림: 2행 3열. 위 = 평가 구간의 실제·예측·90% 구간·naive, 아래 = 정책 기간 격차 (학습 80% 모형 대 전체 모형)
def plot_split(weekly, preds, policy, train_end, path):
    apply_plot_style()
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for col, (fuel, spec) in enumerate(FUELS.items()):
        color = COLORS[fuel]
        ax = axes[0, col]
        actual = weekly.loc[FIG_START:preds[fuel].index[-1], f"{fuel}_supply_pretax"]
        test = preds[fuel]
        ax.fill_between(test.index, test["low"], test["high"], color=color, alpha=0.15, linewidth=0, label="90% 구간")
        ax.plot(actual.index, actual, color=INK_2, label="실제")
        ax.plot(test.index, test[CHOSEN], color=color, label="예측 (학습 80% 모형)")
        ax.plot(test.index, test["naive"], color=MUTED, linestyle="--", linewidth=1.2, label="naive (직전 가격 유지)")
        ax.axvline(train_end, color=MUTED, linestyle=":", linewidth=1)
        ax.set_title(f"{spec['ko']}: 평가 구간 세전 공급가 (원/리터)", loc="left")
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * 0.3)        # 범례가 선을 가리지 않도록 위쪽에 여백을 둡니다
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax = axes[1, col]
        ax.fill_between(policy.index, policy[f"{fuel}_split_gap_low"], policy[f"{fuel}_split_gap_high"], color=color,
                        alpha=0.15, linewidth=0, label="90% 구간")
        ax.plot(policy.index, policy[f"{fuel}_split_gap"], color=color, marker="o", markersize=3, label="학습 80% 모형")
        ax.plot(policy.index, policy[f"{fuel}_full_gap"], color=INK_2, label="전체 기간 모형")
        ax.axhline(0, color=AXIS, linewidth=1)
        ax.set_title(f"{spec['ko']}: 정책 기간 격차 (원/리터)", loc="left")
        ax.legend(loc="upper right", fontsize=8)
        for ax in axes[:, col]:
            ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%y.%m"))
    fig.savefig(path, dpi=150)
    plt.close(fig)




# 7. 분할 → 유종별 학습·평가·적용 → 저장 → 요약 출력을 실행합니다.
def main():
    try:
        weekly = load_weekly()
        train_end = split_date(weekly)
        full_cf, _ = build_counterfactual(weekly)
        rows, preds = [], {}
        policy = weekly.loc[weekly["in_policy"], ["주차"]].copy()
        for fuel in FUELS:
            fuel_rows, preds[fuel], applied = evaluate(weekly, fuel, train_end)
            rows.extend(fuel_rows)
            for column in applied:
                policy[f"{fuel}_split_{column}"] = applied[column]
            policy[f"{fuel}_full_gap"] = full_cf.loc[policy.index, f"{fuel}_gap"]
            policy[f"{fuel}_gap_diff"] = policy[f"{fuel}_split_gap"] - policy[f"{fuel}_full_gap"]
        models = pd.DataFrame(rows)
        test_table = pd.concat({fuel: frame for fuel, frame in preds.items()}, axis=1)
        test_table.columns = [f"{fuel}_{name}" for fuel, name in test_table.columns]
        test_table.insert(0, "주차", weekly.loc[test_table.index, "주차"])
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        models.to_csv(OUT_DIR / "split_models.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        test_table.to_csv(OUT_DIR / "split_test_predictions.csv", encoding="utf-8-sig", float_format="%.2f")
        policy.to_csv(OUT_DIR / "split_counterfactual.csv", encoding="utf-8-sig", float_format="%.2f")
        plot_split(weekly, preds, policy, train_end, OUT_DIR / "split_test.png")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"분할 회귀분석 실패: {error}") from None
    first = models.iloc[0]
    print(f"학습: {first['train_start']} ~ {first['train_end']} ({first['train_weeks']}주)"
          f" / 평가: {first['test_start']} ~ {first['test_end']} ({first['test_weeks']}주)"
          f" / 정책 기간 {len(policy)}주는 적용에만 사용")
    pd.set_option("display.width", 220)
    fitted = models[models["model"] != NAIVE["model"]]
    print("\n[② 학습 결과와 ③ 학습 구간 안 기점 이동 검증] 세전 원/리터")
    print(fitted[["fuel", "name", "const", "w_lag0", "w_lag1", "w_lag2", "train_MAE", "train_bias",
                  "train_err_q05", "train_err_q95", "train_n_cutoffs"]].round(3).to_string(index=False))
    print("\n[④ 평가 구간] 세전 원/리터, 기술 점수 = 1 − MAE / naive MAE")
    print(models[["fuel", "name", "test_MAE", "test_bias", "test_MAPE", "test_MAE_1_32w", "test_bias_1_32w",
                  "test_MAE_after_32w", "test_cover_1_32w", "test_cover_all", "test_skill_vs_naive"]]
          .round(3).to_string(index=False))
    print("\n[⑤ 정책 기간 격차] 학습 80% 모형 / 전체 기간 모형 (원/리터)")
    for week in policy.index[[1, 4, 13, 22, 26, 28]]:
        row = policy.loc[week]
        print(f"  {row['주차']}: " + " / ".join(
            f"{FUELS[f]['ko']} {row[f'{f}_split_gap']:.0f} vs {row[f'{f}_full_gap']:.0f}" for f in FUELS))
    print("  정책 기간 평균 차이(학습 80% − 전체): " + " / ".join(
        f"{FUELS[f]['ko']} {policy[f'{f}_gap_diff'].mean():+.1f}" for f in FUELS))
    print(f"\n저장 완료: {OUT_DIR}")




# 8. 직접 실행할 때만 분할 회귀분석을 시작합니다.
if __name__ == "__main__":
    main()
