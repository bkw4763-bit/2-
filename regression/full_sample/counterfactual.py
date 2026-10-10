# 1. 1단계 회귀분석: "상한제가 없었다면" 정유사가 공급했을 가격을 추정합니다.
# ① merged_weekly_data_clean.csv 불러오기: 주차 표기를 그 주의 마지막 날(토요일)로 바꿉니다.
# ② 주별 표 만들기: 결측 행 제거로 빠진 주는 빈칸으로 되살려 시차가 실제 1·2주 전을 가리키게 합니다.
# ③ 유류세 역산: 유류세 = 세후 공급가 − 1.1 × 세전 공급가
# ④ 모형: 국제가 3개 시차 + 같은 주 환율로 OLS → 국제가 계수만 합 1 → 마지막 52주 잔차로 상수 보정
# ⑤ 검증: 2018년부터 13주마다 기점을 옮겨, 기점 이전 자료로 학습하고 다음 32주를 맞혀 봅니다.
# ⑥ 결과 저장: results/regression/full_sample 폴더에 주별 "없었다면" 가격과 모형 비교표를 저장합니다.
# 방법론: regression_method.md / 실행: python "regression/full_sample/counterfactual.py"
# 정책 이전 자료 전체로 학습합니다. 학습·평가를 80:20으로 나눈 분석은 regression/train_test 폴더에 있습니다.
# 필요 라이브러리: pip install numpy pandas statsmodels matplotlib
import re
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd




# 2. 입력·출력 위치와 모형 매개변수입니다. 값을 바꿀 때는 여기만 수정합니다.
# 상한은 세후 공급가에 걸리므로 "없었다면" 가격도 세후로 바꿔 실제 세후 공급가와 비교합니다.
PROJECT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_DIR / "csv" / "merged_weekly_data_clean.csv"
OUT_DIR = PROJECT_DIR / "results" / "regression" / "full_sample_exchange_rate"
EXCHANGE_COLUMN = "원달러환율_주간평균_원/달러 환율"
POLICY_START = pd.Timestamp("2026-03-13")   # 최고가격제 시행일. 이 날이 든 주부터 학습에서 뺍니다
VAT = 0.1
LAGS = 3                                     # 같은 주, 1주 전, 2주 전 국제가
MARGIN_WEEKS = 52                            # 마진을 잡는 정책 직전 기간
BACKTEST_START = pd.Timestamp("2018-01-01")  # 첫 검증 기점. 그 전 3년은 최소 학습 기간
BACKTEST_STEP = 13                           # 기점 간격(주). 분기마다 한 번
HORIZON = 32                                 # 기점마다 맞혀 보는 주 수. 정책 기간(29주)보다 길게
MODELS = {"baseline": "11년 전체 회귀", "recent_margin": "최근 마진"}
CHOSEN = "recent_margin"

# 유종별로 국내 공급가와 국제 제품가를 짝지어 회귀합니다.
# 경유는 차량용(자동차용경유)만 분석하므로 국제가도 경유(0.001%)만 씁니다. 경유(0.05%)는 선박·산업용이라 쓰지 않습니다.
# 휘발유(95RON)도 고급휘발유용이라 쓰지 않습니다. 아래에 적힌 열만 읽으므로 두 열은 회귀에 들어가지 않습니다.
FUELS = {
    "gasoline": {
        "ko": "휘발유",
        "intl": "국제_석유제품가격(원)_휘발유(92RON)",
        "pretax": "정유사_주간공급가격_제품별(세전)_보통휘발유",
        "aftertax": "정유사_주간공급가격_제품별(세후)_보통휘발유",
    },
    "diesel": {
        "ko": "경유",
        "intl": "국제_석유제품가격(원)_경유(0.001%)",
        "pretax": "정유사_주간공급가격_제품별(세전)_자동차용경유",
        "aftertax": "정유사_주간공급가격_제품별(세후)_자동차용경유",
    },
    "kerosene": {
        "ko": "등유",
        "intl": "국제_석유제품가격(원)_등유",
        "pretax": "정유사_주간공급가격_제품별(세전)_실내등유",
        "aftertax": "정유사_주간공급가격_제품별(세후)_실내등유",
    },
}




# 3. 오피넷 주차 표기를 날짜로 바꿉니다.
# "M월 N주"는 그 달의 N번째 수요일이 든 일~토 주간이므로, 그 수요일 + 3일(토요일)을 주의 끝으로 둡니다.
# 예: 2026년03월2주 → 3월 11일(두 번째 수요일) → 2026-03-14
def week_end(label):
    match = re.fullmatch(r"(\d{4})년(\d{2})월([1-5])주", str(label).strip())
    if not match:
        raise ValueError(f"주차 형식이 잘못되었습니다: {label}")
    year, month, number = map(int, match.groups())
    first = date(year, month, 1)
    wednesday = first + timedelta(days=(2 - first.weekday()) % 7 + 7 * (number - 1))
    if wednesday.month != month:
        raise ValueError(f"존재하지 않는 주차입니다: {label}")
    return pd.Timestamp(wednesday + timedelta(days=3))




# 4. 결측 행을 뺀 통합 자료를 읽어 회귀용 주별 표로 만듭니다.
# 첫 주부터 마지막 주까지 모든 토요일을 인덱스로 두고, 결측 행 제거로 빠진 주는 빈칸으로 남깁니다.
# 빈칸을 채우지 않으므로, 빠진 주가 시차(1·2주 전)에 걸리는 행은 학습에서 자동으로 빠집니다.
# 열 이름: {fuel}_intl_krw_l(국제가, 원/리터), {fuel}_supply_pretax·aftertax, {fuel}_tax, in_policy
def load_weekly(path=INPUT_PATH):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.index = pd.DatetimeIndex([week_end(label) for label in raw["주차"]], name="week_end")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("주차가 중복되었거나 시간순이 아닙니다.")
    columns = [spec[key] for spec in FUELS.values() for key in ("intl", "pretax", "aftertax")] + [EXCHANGE_COLUMN]
    if raw[columns].isna().any().any():
        raise ValueError("결측이 남아 있습니다. 결측 행을 제거한 자료를 사용하세요.")
    full = pd.date_range(raw.index[0], raw.index[-1], freq="W-SAT", name="week_end")
    if not raw.index.isin(full).all():
        raise ValueError("주의 끝 날짜가 토요일이 아닌 주차가 있습니다.")
    weekly = pd.DataFrame({"주차": raw["주차"]}).reindex(full)
    weekly["exchange_rate"] = raw[EXCHANGE_COLUMN].reindex(full)
    for fuel, spec in FUELS.items():
        weekly[f"{fuel}_intl_krw_l"] = raw[spec["intl"]].reindex(full)
        weekly[f"{fuel}_supply_pretax"] = raw[spec["pretax"]].reindex(full)
        weekly[f"{fuel}_supply_aftertax"] = raw[spec["aftertax"]].reindex(full)
        weekly[f"{fuel}_tax"] = weekly[f"{fuel}_supply_aftertax"] - (1 + VAT) * weekly[f"{fuel}_supply_pretax"]
    weekly["in_policy"] = weekly.index >= POLICY_START
    return weekly




# 5. 국제가(원/리터)의 같은 주·1주 전·2주 전과 같은 주 환율을 사용합니다.
# 주별 표가 달력상 연속이므로 shift(k)는 정확히 k주 전 값을 가리킵니다.
def lagged_intl(weekly, fuel, lags=LAGS, include_exchange=True):
    series = weekly[f"{fuel}_intl_krw_l"]
    features = pd.DataFrame({f"lag{k}": series.shift(k) for k in range(lags)})
    if include_exchange:
        features["exchange_rate"] = weekly["exchange_rate"]
    return features


def normalize_weights(coef, intl_count):
    """국제가격 계수만 합 1로 맞춘다. 환율 계수는 원/L per 원/USD 단위 그대로 유지."""
    weights = np.asarray(coef, dtype=float).copy()
    total = weights[:intl_count].sum()
    if not np.isfinite(weights).all() or abs(total) < 1e-12:
        raise ValueError("국제가격 계수 합이 0이거나 유효하지 않습니다.")
    weights[:intl_count] /= total
    return weights


def select_lags(X, lags):
    columns = [f"lag{k}" for k in range(lags)]
    if "exchange_rate" in X.columns:
        columns.append("exchange_rate")
    return X[columns]




# 6. 정책 이전 주 중 시차 국제가와 세전 공급가가 모두 있는 주만 학습 행 (X, y)로 돌려줍니다.
# 52주보다 적으면 마진을 잡을 수 없으므로 중단합니다.
def training_data(weekly, fuel, lags=LAGS):
    X = lagged_intl(weekly, fuel, lags)
    y = weekly[f"{fuel}_supply_pretax"]
    ok = ~weekly["in_policy"] & X.notna().all(axis=1) & y.notna()
    if ok.sum() < MARGIN_WEEKS:
        raise ValueError(f"{fuel}: 학습할 주가 {ok.sum()}개뿐입니다.")
    return X[ok], y[ok]




# 7. 모형을 학습해 {"const": 상수, "weights": 시차별 계수}를 돌려줍니다. 예측식은 두 모형이 같습니다.
# 예측 = const + 국제가 비중 · 국제가 3개 시차 + 환율 계수 × 같은 주 환율
# baseline: OLS의 상수와 계수를 그대로 씁니다.
# recent_margin: 국제가 계수만 합 1로 맞추고 환율 계수는 유지, 상수는 마지막 52주 평균 잔차로 보정합니다.
# 정유사 마진이 2015년 약 60원에서 최근 약 100원으로 커져, 11년 상수 하나로는 최근 가격을 낮게 예측합니다.
def fit(X, y, model=CHOSEN, margin_weeks=MARGIN_WEEKS, feature_scale=None):
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]
    if model == "baseline":
        return {"const": coef[0], "weights": coef[1:]}
    intl_count = len(X.columns) - int("exchange_rate" in X.columns)
    scale = np.ones(len(X.columns)) if feature_scale is None else np.asarray(feature_scale, dtype=float)
    if scale.shape != (len(X.columns),) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError("표준화 척도는 설명변수마다 유한한 양수여야 합니다.")
    # 계수 합 제약은 원/리터 단위에서 적용한 뒤 표준화 좌표로 되돌린다.
    weights = normalize_weights(coef[1:] / scale, intl_count) * scale
    return {"const": recent_margin(X, y, weights, margin_weeks), "weights": weights}




# 8. 최근 margin_weeks주 동안 "세전 공급가 − 국제가 가중평균 − 환율 효과"의 평균으로 상수를 보정합니다.
def recent_margin(X, y, weights, margin_weeks=MARGIN_WEEKS):
    recent_X, recent_y = X.to_numpy()[-margin_weeks:], y.to_numpy()[-margin_weeks:]
    return float(np.mean(recent_y - recent_X @ weights))


def predict(params, X):
    return params["const"] + X.to_numpy() @ params["weights"]




# 9. 기점 이동 표본 밖 검증입니다.
# 기점 = 2018-01-01 이후 학습 행 위치 중 뒤에 32주가 남는 곳을 13주 간격으로 고른 것입니다.
# 기점 이전 자료로 학습하고 다음 32주의 실제 국제가·환율로 공급가를 맞히는 조건부 예측입니다.
# 반환: 오차(예측 − 실제) 행렬. 행 = 기점, 열 = 기점 뒤 몇 주째. bias < 0 이면 낮게 예측한 것입니다.
def cutoffs(X):
    return [i for i in range(len(X))
            if X.index[i] >= BACKTEST_START and i + HORIZON <= len(X)][::BACKTEST_STEP]


def backtest(X, y, fit_fn):
    errors = []
    for c in cutoffs(X):
        params = fit_fn(X.iloc[:c], y.iloc[:c])
        errors.append(predict(params, X.iloc[c:c + HORIZON]) - y.iloc[c:c + HORIZON].to_numpy())
    return np.array(errors)


def summarize(errors):
    e = errors.ravel()
    return {"MAE": np.abs(e).mean(), "bias": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAE_1_4w": np.abs(errors[:, :4]).mean(), "MAE_last_4w": np.abs(errors[:, -4:]).mean(),
            "err_q05": np.quantile(e, 0.05), "err_q95": np.quantile(e, 0.95), "n_cutoffs": len(errors)}




# 10. 유종·모형별로 학습과 검증을 하고 주별 "없었다면" 가격표와 모형 비교표를 만듭니다.
# 세후 변환: "없었다면" 세후 = 세전 예측 × 1.1 + 실제 유류세 (상한과 같은 세금 기준으로 비교)
# 90% 구간: 오차 = 예측 − 실제 → 실제 = 예측 − 오차이므로 하한에는 q95, 상한에는 q05를 뺍니다.
# 격차: "없었다면" 세후 − 실제 세후 공급가. 상한이 걸린 주에는 실제 공급가가 상한과 거의 같습니다.
def build_counterfactual(weekly):
    out = weekly[["주차", "in_policy"]].copy()
    rows = []
    for fuel in FUELS:
        X, y = training_data(weekly, fuel)
        X_all = lagged_intl(weekly, fuel)
        tax = weekly[f"{fuel}_tax"]
        for model in MODELS:
            params = fit(X, y, model)
            score = summarize(backtest(X, y, lambda Xt, yt, m=model: fit(Xt, yt, m)))
            rows.append({"fuel": fuel, "model": model, "name": MODELS[model], "train_weeks": len(X),
                         "train_start": X.index[0].date(), "train_end": X.index[-1].date(),
                         "const": params["const"], **{f"w_lag{k}": params["weights"][k] for k in range(LAGS)},
                         "w_exchange_rate": params["weights"][LAGS],
                         "w_sum": params["weights"][:LAGS].sum(), **score})
            pretax = pd.Series(predict(params, X_all), index=weekly.index)
            if model == "baseline":
                out[f"{fuel}_baseline_aftertax"] = pretax * (1 + VAT) + tax
                continue
            out[f"{fuel}_cf_pretax"] = pretax
            out[f"{fuel}_cf_aftertax"] = pretax * (1 + VAT) + tax
            out[f"{fuel}_cf_low"] = (pretax - score["err_q95"]) * (1 + VAT) + tax
            out[f"{fuel}_cf_high"] = (pretax - score["err_q05"]) * (1 + VAT) + tax
        out[f"{fuel}_supply_aftertax"] = weekly[f"{fuel}_supply_aftertax"]
        out[f"{fuel}_gap"] = out[f"{fuel}_cf_aftertax"] - out[f"{fuel}_supply_aftertax"]
        out[f"{fuel}_gap_low"] = out[f"{fuel}_cf_low"] - out[f"{fuel}_supply_aftertax"]
        out[f"{fuel}_gap_high"] = out[f"{fuel}_cf_high"] - out[f"{fuel}_supply_aftertax"]
    return out, pd.DataFrame(rows)




# 11. 정책 기간의 격차를 유종별로 보여 줍니다. 괄호 안은 90% 구간입니다.
def gap_table(cf):
    policy = cf[cf["in_policy"]]
    table = pd.DataFrame({"주차": policy["주차"]})
    for fuel, spec in FUELS.items():
        table[spec["ko"]] = [f"{gap:.0f} ({low:.0f}~{high:.0f})" for gap, low, high in
                             zip(policy[f"{fuel}_gap"], policy[f"{fuel}_gap_low"], policy[f"{fuel}_gap_high"])]
    return table




# 12. 불러오기 → 학습·검증 → 저장 → 요약 출력을 실행합니다.
def main():
    try:
        weekly = load_weekly()
        cf, models = build_counterfactual(weekly)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cf.to_csv(OUT_DIR / "counterfactual.csv", encoding="utf-8-sig", float_format="%.2f")
        models.to_csv(OUT_DIR / "counterfactual_models.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"회귀분석 실패: {error}") from None
    gaps = weekly.index[weekly["주차"].isna()]
    print(f"주별 표: {weekly.index[0].date()} ~ {weekly.index[-1].date()}, {len(weekly)}주"
          f" (결측 행 제거로 빠진 주 {len(gaps)}개: {', '.join(str(d.date()) for d in gaps)})")
    print(f"학습: {models['train_start'].iloc[0]} ~ {models['train_end'].iloc[0]}, {models['train_weeks'].iloc[0]}주"
          f" / 정책 기간: {int(weekly['in_policy'].sum())}주")
    pd.set_option("display.width", 200)
    print("\n[모형 비교] 기점 이동 검증, 세전 원/리터, 오차 = 예측 − 실제")
    show = models[["fuel", "name", "const", "w_lag0", "w_lag1", "w_lag2", "w_sum",
                   "MAE", "bias", "RMSE", "MAE_1_4w", "MAE_last_4w", "err_q05", "err_q95", "n_cutoffs"]]
    print(show.round(3).to_string(index=False))
    print(f"\n[격차] \"없었다면\" 세후 공급가 − 실제 세후 공급가, 원/리터 (모형: {MODELS[CHOSEN]})")
    print(gap_table(cf).to_string())
    print(f"\n저장 완료: {OUT_DIR}")




# 13. 직접 실행할 때만 회귀분석을 시작합니다.
if __name__ == "__main__":
    main()
