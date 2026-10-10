# 1. 학습·평가 분할 분석의 유종별 6가지 기준 진단과 R²입니다. 학습 구간과 평가 구간에서 같은 항목을 나란히 계산합니다.
# ①~③ 자기상관·교차상관·변동률: 자료 자체의 성질이므로 각 구간의 자료로 계산합니다.
# ④~⑥ 잔차·변수 중요도·Naive 비교: 학습 구간은 학습 구간 안 기점 이동 검증(22회 × 32주)으로,
#      평가 구간은 학습 구간에서 정한 계수·마진을 고치지 않고 평가 구간에 적용해 계산합니다.
# R²: 수준 R²는 단위근 때문에 높게 나오므로 참고만 하고, 주간 변화(차분) R²로 설명력을 봅니다.
#      평가 구간의 "학습 계수" R²가 표본 밖 R²이고, "그 구간에서 다시 추정" R²는 평가 구간 자료로 다시 맞춘 참고값입니다.
#      주간 변화 외에 4주·13주 변화의 R²도 봅니다. "없었다면" 가격은 몇 주에 걸친 수준이라 긴 변화의 설명력이 더 중요합니다.
# 평가 구간 끝 10주(2026년 1~3월 급등기)는 주간 변화가 매우 커 상관·R²를 좌우하므로, 이 10주를 뺀 평가 구간도 함께 계산합니다.
# 실행: python "regression/train_test/split_diagnostics.py"
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "full_sample"))
from counterfactual import FUELS, HORIZON, fit, lagged_intl, load_weekly, predict
from counterfactual_check import centered_r2, diff_data, diff_regression, wide_table
from fuel_diagnostics import (
    ARCH_LAGS, LAG_NAMES, PASS_LAGS, PERMUTATIONS, TOP_SHARE,
    acf_checks, acf_series, arch_p, ccf_checks, ccf_series, diagnose, lag_corr, naive_checks, plot_diagnostics,
    pre_policy_changes, summary_tables, volatility_checks,
)
from split_regression import OUT_DIR, split_data, split_date, train_panel




# 2. 평가 구간 진단 매개변수입니다.
TEST_STEP = 4         # 평가 구간 기점 간격(주). 116주 안에 학습 구간과 같은 22개 기점을 두려고 4주마다 둡니다
MIN_YEAR_WEEKS = 8    # 연도별 오차에서 이보다 적은 주만 든 해는 뺍니다 (평가 구간의 2023년은 12월 2주뿐)
CALM_END = pd.Timestamp("2025-12-31")   # 급등기를 뺀 평가 구간의 끝
R2_HORIZONS = (4, 13)                   # 주간 변화 외에 R²를 볼 변화 기간(주)
PERIODS = {"train": "학습", "test": "평가", "calm": "평가(2026년 제외)"}




# 3. 평가 구간 주와, 시차(최대 7주 전)·차분 계산에 필요한 직전 주를 남긴 표입니다.
# full_sample의 진단 함수는 in_policy가 참인 주를 계산에서 빼므로, 직전 주(학습 구간)에 참을 표시해
# 시차 값으로만 쓰고 진단 대상에서는 빠지게 합니다. 정책 기간은 표에 넣지 않습니다.
# end를 주면 평가 구간을 그 날까지로 줄입니다.
def test_panel(weekly, train_end, end=None):
    end = weekly.index[~weekly["in_policy"]][-1] if end is None else end
    panel = weekly.loc[train_end - pd.Timedelta(weeks=PASS_LAGS + 1):end].copy()
    panel["in_policy"] = panel.index <= train_end
    return panel


# 학습 구간에서 정한 모형(채택 모형의 계수·마진, 차분 회귀 계수)과 두 구간의 수준·차분 자료를 모읍니다.
# 평가 구간 자료는 panel의 진단 대상 주로 줄입니다. level은 k주 변화 R²에 쓸 공급가와 가중 국제가(전체 기간)입니다.
def split_model(weekly, fuel, train_end, train, panel):
    X_train, y_train, X_test, y_test = split_data(weekly, fuel, train_end)
    keep = X_test.index.isin(panel.index[~panel["in_policy"]])
    X_test, y_test = X_test[keep], y_test[keep]
    dX_train, dy_train = diff_data(train, fuel)
    dX_test, dy_test = diff_data(panel, fuel)
    params = fit(X_train, y_train)
    weighted = pd.Series(lagged_intl(weekly, fuel).to_numpy() @ params["weights"], index=weekly.index)
    return {"params": params, "b": diff_regression(dX_train, dy_train).params.to_numpy(),
            "train": (X_train, y_train, dX_train, dy_train), "test": (X_test, y_test, dX_test, dy_test),
            "level": (weekly[f"{fuel}_supply_pretax"], weighted)}


def r2(actual, pred):
    actual, pred = np.asarray(actual, dtype=float), np.asarray(pred, dtype=float)
    return 1 - ((actual - pred) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()




# 4. R²: 같은 계수로 학습 구간(표본 안)과 평가 구간(표본 밖)의 설명력을 비교합니다.
# 수준: 채택 모형의 공급가 예측. 단위근이 있는 수준끼리라 높게 나오므로 참고만 합니다.
# 변화(채택 모형 비중): 마진은 주마다 같으므로 예측 공급가의 주간 변화 = 비중 · 국제가 주간 변화입니다.
# 변화(학습 구간 차분 회귀 계수): 비중으로 정규화하기 전의 차분 회귀 계수를 그대로 씁니다.
# 변화(그 구간에서 다시 추정): 그 구간 자료로 차분 회귀를 다시 맞춘 값. 평가 구간에서는 상한선 같은 참고값입니다.
# k주 변화(채택 모형 비중): k주 동안의 공급가 변화를 같은 기간 가중 국제가 변화로 설명한 R². 구간 안의 주에서 끝나는 변화만 씁니다.
def r2_checks(model, period):
    X, y, dX, dy = model[period]
    supply, weighted = model["level"]
    longer = {}
    for k in R2_HORIZONS:
        actual, pred = supply.diff(k).reindex(X.index), weighted.diff(k).reindex(X.index)
        ok = actual.notna() & pred.notna()
        longer[("R²", f"{k}주 변화: 채택 모형 비중")] = r2(actual[ok], pred[ok])
    return {
        ("R²", "수준: 채택 모형 (참고)"): r2(y, predict(model["params"], X)),
        ("R²", "변화: 채택 모형 비중"): r2(dy, dX.to_numpy() @ model["params"]["weights"]),
        ("R²", "변화: 학습 구간 차분 회귀 계수"): r2(dy, dX.to_numpy() @ model["b"]),
        ("R²", "변화: 그 구간에서 다시 추정"): centered_r2(diff_regression(dX, dy), dy),
        **longer,
    }




# 5. 학습 구간 ④에 오차의 자기상관 두 줄을 더합니다. 평가 구간과 같은 항목으로 비교하기 위해서입니다.
# 표본 밖 오차 1주 자기상관: 같은 기점의 32주 예측 안에서 이웃한 두 주 오차의 상관
# 차분 회귀 잔차 1주 자기상관: 국제가 변화로 설명하고 남은 공급가 변화의 1주 자기상관
def train_extra(result, model, train):
    errors = result["frame"].pivot(index="cutoff", columns="h", values="error").to_numpy()
    _, _, dX, dy = model["train"]
    residual = (dy - dX.to_numpy() @ model["b"]).reindex(train.index)
    return {("④ 잔차", "표본 밖 오차 1주 자기상관"): np.corrcoef(errors[:, :-1].ravel(), errors[:, 1:].ravel())[0, 1],
            ("④ 잔차", "차분 회귀 잔차 1주 자기상관"): lag_corr(residual, residual, 1)}




# 6. ⑥ 평가 구간 기점 이동 표: 평가 구간 처음부터 4주마다 기점을 두고 32주씩 봅니다.
# 모형은 학습 구간 계수·마진 그대로이고, naive는 기점 직전 주의 실제 공급가를 씁니다. (naive는 최신 가격을 쓰므로 모형에 불리한 비교)
def test_frame(model):
    X_test, y_test, _, _ = model["test"]
    pred = predict(model["params"], X_test)
    last_train = model["train"][1].iloc[-1]
    frames = []
    for c in range(0, len(y_test) - HORIZON + 1, TEST_STEP):
        window = slice(c, c + HORIZON)
        frames.append(pd.DataFrame({
            "cutoff": y_test.index[c], "h": np.arange(1, HORIZON + 1), "target": y_test.index[window],
            "actual": y_test.iloc[window].to_numpy(), "pred": pred[window],
            "naive": last_train if c == 0 else y_test.iloc[c - 1],
        }))
    frame = pd.concat(frames, ignore_index=True)
    frame["error"] = frame["pred"] - frame["actual"]
    frame["naive_error"] = frame["naive"] - frame["actual"]
    return frame




# 7. ④ 평가 구간 잔차: 학습 구간 모형을 평가 구간 116주에 한 번 적용한 오차입니다.
# 차분 회귀 잔차는 학습 구간 차분 회귀 계수로 평가 구간 공급가 변화를 설명하고 남은 값입니다.
def test_residual_checks(model, panel):
    X_test, y_test, dX, dy = model["test"]
    error = pd.Series(predict(model["params"], X_test), index=y_test.index) - y_test
    abs_error = error.abs()
    counts = abs_error.groupby(abs_error.index.year).size()
    by_year = abs_error.groupby(abs_error.index.year).mean()[counts >= MIN_YEAR_WEEKS]
    top = abs_error.nlargest(int(np.ceil(len(abs_error) * TOP_SHARE)))
    residual = (dy - dX.to_numpy() @ model["b"]).reindex(panel.index)
    return {
        ("④ 잔차", "표본 밖 MAE (전체)"): abs_error.mean(),
        ("④ 잔차", "오차가 가장 큰 해"): by_year.idxmax(),
        ("④ 잔차", "그 해 MAE"): by_year.max(),
        ("④ 잔차", "그 해 MAE / 전체 MAE"): by_year.max() / abs_error.mean(),
        ("④ 잔차", f"상위 {TOP_SHARE:.0%} 주의 절대오차 비중"): top.sum() / abs_error.sum(),
        ("④ 잔차", f"ARCH-LM p (차분 회귀 잔차, {ARCH_LAGS}주)"): arch_p(residual.dropna()),
        ("④ 잔차", "표본 밖 오차 1주 자기상관"): lag_corr(error, error, 1),
        ("④ 잔차", "차분 회귀 잔차 1주 자기상관"): lag_corr(residual, residual, 1),
    }, by_year




# 8. ⑤ 평가 구간 변수 중요도: 모두 학습 구간에서 정한 계수로 평가 구간을 설명합니다.
# 제거 시 R² 감소: 학습 구간에서 그 시차를 빼고 다시 추정한 계수로 평가 구간 R²를 구해, 전체 시차일 때보다 얼마나 줄어드는지 봅니다.
# 순열 중요도: ⑥과 같은 32주 창마다 그 시차 값을 무작위로 섞었을 때 MAE가 얼마나 커지는지 봅니다.
def test_importance_checks(model):
    _, _, dX_train, dy_train = model["train"]
    X_test, y_test, dX_test, dy_test = model["test"]
    full = r2(dy_test, dX_test.to_numpy() @ model["b"])
    drop = {}
    for k in range(3):
        column = f"lag{k}"
        coef = diff_regression(dX_train.drop(columns=column), dy_train).params.to_numpy()
        drop[k] = full - r2(dy_test, dX_test.drop(columns=column).to_numpy() @ coef)
    rng = np.random.default_rng(0)
    params, X, y = model["params"], X_test.to_numpy(), y_test.to_numpy()
    base, permuted = [], {k: [] for k in range(3)}
    for c in range(0, len(y) - HORIZON + 1, TEST_STEP):
        window_X, actual = X[c:c + HORIZON], y[c:c + HORIZON]
        base.append(np.abs(params["const"] + window_X @ params["weights"] - actual).mean())
        for k in range(3):
            for _ in range(PERMUTATIONS):
                shuffled = window_X.copy()
                shuffled[:, k] = rng.permutation(shuffled[:, k])
                permuted[k].append(np.abs(params["const"] + shuffled @ params["weights"] - actual).mean())
    return {
        ("⑤ 변수 중요도", "차분 회귀 R²"): full,
        **{("⑤ 변수 중요도", f"제거 시 R² 감소: {LAG_NAMES[k]}"): drop[k] for k in range(3)},
        **{("⑤ 변수 중요도", f"순열 시 MAE 증가(원): {LAG_NAMES[k]}"): np.mean(permuted[k]) - np.mean(base)
           for k in range(3)},
    }, drop, full




# 9. 평가 구간 ①~⑥을 계산합니다. 반환 형식은 fuel_diagnostics.diagnose와 같아 같은 그림 함수에 넣을 수 있습니다.
# ③의 마진은 평가 구간 자료로 다시 맞춘 비중으로 계산합니다. 비중이 세 시차에 고르게 나뉘어 있어 마진 크기에 주는 영향은 작습니다.
def diagnose_test(panel, fuel, model):
    changes = pre_policy_changes(panel, fuel)
    frame = test_frame(model)
    residual, by_year = test_residual_checks(model, panel)
    importance, drop, full = test_importance_checks(model)
    checks = {**acf_checks(panel, fuel, changes), **ccf_checks(panel, fuel, changes),
              **volatility_checks(panel, fuel, changes), **residual, **importance, **naive_checks(changes, frame)}
    return {"checks": checks, "acf": acf_series(changes), "ccf": ccf_series(changes),
            "n_changes": changes["d_supply"].notna().sum(), "by_year": by_year,
            "drop": drop, "r2": full, "frame": frame, "changes": changes}


# 기준(①~⑥) 순서를 지키면서 추가 항목을 같은 기준의 끝에 붙이고, 새 기준(R²)은 맨 뒤에 둡니다.
def merged(checks, extra):
    out = {}
    groups = list(dict.fromkeys(key[0] for key in checks))
    for group in groups:
        out.update({key: value for key, value in checks.items() if key[0] == group})
        out.update({key: value for key, value in extra.items() if key[0] == group})
    out.update({key: value for key, value in extra.items() if key[0] not in groups})
    return out




# 10. 학습·평가·평가(2026년 제외) 세 구간을 진단하고 results/regression/train_test 폴더에 표와 그림을 저장합니다.
# 표의 열은 (구간, 유종)입니다. 평가 구간에 없는 항목(2022년 비중 등)은 빈칸입니다.
def main():
    try:
        weekly = load_weekly()
        train_end = split_date(weekly)
        train = train_panel(weekly, train_end)
        panels = {"test": test_panel(weekly, train_end), "calm": test_panel(weekly, train_end, CALM_END)}
        train_results = diagnose(train)
        train_yearly = summary_tables(train_results)[1]
        rows = {period: {} for period in PERIODS}
        test_results, test_yearly = {}, {}
        for fuel in FUELS:
            models = {key: split_model(weekly, fuel, train_end, train, panel) for key, panel in panels.items()}
            rows["train"][fuel] = merged(train_results[fuel]["checks"], {
                **train_extra(train_results[fuel], models["test"], train), **r2_checks(models["test"], "train")})
            for key, panel in panels.items():
                result = diagnose_test(panel, fuel, models[key])
                rows[key][fuel] = merged(result["checks"], r2_checks(models[key], "test"))
                if key == "test":
                    test_results[fuel] = result
            pct = test_results[fuel]["changes"]["pct_supply"].dropna()
            counts = pct.groupby(pct.index.year).size()
            test_yearly[f"{FUELS[fuel]['ko']}_평가MAE"] = test_results[fuel]["by_year"]
            test_yearly[f"{FUELS[fuel]['ko']}_공급가변화율표준편차"] = pct.groupby(pct.index.year).std()[counts >= MIN_YEAR_WEEKS]
        tables = {PERIODS[period]: wide_table(rows[period]) for period in PERIODS}
        table = pd.concat(tables, axis=1).reindex(tables["학습"].index)
        yearly = pd.concat([train_yearly, pd.DataFrame(test_yearly)])
        yearly.index.name = "연도"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        flat = table.copy()
        flat.columns = [f"{period}_{fuel}" for period, fuel in flat.columns]
        flat.to_csv(OUT_DIR / "split_diagnostics.csv", encoding="utf-8-sig", float_format="%.4f")
        yearly.to_csv(OUT_DIR / "split_diagnostics_by_year.csv", encoding="utf-8-sig", float_format="%.3f")
        plot_diagnostics(train_results, OUT_DIR / "split_diagnostics.png",
                         title=f"학습 구간(~{train_end.date()})만으로 계산한 유종별 진단")
        test_start = test_results[next(iter(FUELS))]["frame"]["cutoff"].min()
        plot_diagnostics(test_results, OUT_DIR / "split_test_diagnostics.png",
                         title=f"평가 구간({test_start.date()} ~) 진단: ④~⑥은 학습 구간 모형을 고치지 않고 적용")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"분할 유종별 진단 실패: {error}") from None
    pd.set_option("display.width", 250)
    print(f"[유종별 진단과 R²] 학습 = ~{train_end.date()}, 평가 = {test_start.date()} ~, "
          f"평가(2026년 제외) = {test_start.date()} ~ {CALM_END.date()}")
    print(table.round(3).to_string())
    print("\n[연도별] 학습 구간 표본 밖 MAE, 평가 구간 MAE(원)와 공급가 주간 변화율 표준편차(%)")
    print(yearly.round(2).to_string())
    print(f"\n저장 완료: {OUT_DIR}")




# 11. 직접 실행할 때만 진단을 시작합니다.
if __name__ == "__main__":
    main()
