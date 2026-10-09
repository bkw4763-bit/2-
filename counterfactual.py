"""1단계: 유종별 "상한제가 없었다면" 정유사 공급가.

질문: 상한이 없었다면 정유사는 이번 주에 얼마에 팔았을까?
상한제 기간에는 국제가와 국내 공급가의 연결이 정책으로 끊겨 있으므로,
상한제 이전(in_policy == False) 자료로만 '국제가 → 세전 공급가' 관계를 배우고,
그 관계에 상한제 기간의 실제 국제가를 넣어 "없었다면" 가격을 만든다.

모형 두 가지를 같은 방법으로 검증해 비교한다.
  - 기본 회귀 (비교용): 세전 공급가 = a + b0·국제가(같은 주) + b1·(1주 전) + b2·(2주 전)
      상한제 이전 11년 전체로 한 번 추정한다. 처음 합의한 방법.
  - 최근 마진 (채택): 위 회귀의 b0·b1·b2 를 합이 1이 되도록 나눠 '시차별 비중'으로만 쓰고,
      상수(정유사 마진)는 상한제 직전 52주의 평균으로 다시 잡는다.
      세전 공급가 = 국제가 가중평균 + 최근 마진

왜 바꿨나
  정유사 마진(세전 공급가 − 국제가)이 2015년 약 60원에서 2025년 약 100원으로 꾸준히 커졌다.
  11년 전체에 상수 하나를 맞추면 최근 가격을 일관되게 낮게 예측한다.
  기점을 옮겨 가며 검증하면 기본 회귀의 편향이 −13~−19원, 최근 마진 모형은 −2~−4원이다. (python counterfactual.py 로 확인)

예측구간
  2018년부터 13주마다 기점을 옮겨, 그때까지의 자료로 학습하고 다음 32주를 맞혀 본 오차를 모은다.
  그 오차의 5%·95% 분위수를 "없었다면" 가격에 더해 90% 구간을 만든다. (분포 가정 없이 실제 오차로 만든 구간)

결과: data/processed/counterfactual.csv (주별), counterfactual_models.csv (모형 비교표)
모형이 왜 타당한지(단위근, 시차 선택, 구간 적중률, 민감도)는 counterfactual_check.py 에서 따로 점검한다.
"""
import numpy as np
import pandas as pd

import config as cfg

LAGS = 3               # 같은 주, 1주 전, 2주 전. 국제가는 1주 뒤 공급가에 가장 크게 반영된다
MARGIN_WEEKS = 52      # 마진을 잡는 최근 기간. 26주와 52주의 검증 성적이 비슷해 계절 영향이 고르게 섞이는 52주를 쓴다
BACKTEST_START = "2018-01-01"   # 첫 검증 기점. 그 전 3년은 최소 학습 기간
BACKTEST_STEP = 13              # 기점 간격(주). 분기마다 한 번
HORIZON = 32                    # 기점마다 맞혀 보는 주 수. 상한제 기간(약 31주)보다 길게 잡는다
MODELS = {"baseline": "기본 회귀", "recent_margin": "최근 마진"}
CHOSEN = "recent_margin"


# ── 자료 준비 ───────────────────────────────────────────────────────────
def load_weekly():
    """전처리가 만든 주별 표를 읽는다. (run_pipeline.py 를 먼저 실행해야 한다)"""
    path = cfg.OUT_DIR / "weekly_panel.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} 없음. python run_pipeline.py 를 먼저 실행할 것")
    weekly = pd.read_csv(path, index_col="week_end", parse_dates=["week_end", "week_start"])
    weekly["in_policy"] = weekly["in_policy"].astype(str) == "True"   # CSV 에서 글자로 읽힌 참/거짓을 되돌린다
    return weekly


def lagged_intl(weekly, fuel):
    """국제가(원/리터)의 같은 주·1주 전·2주 전 값을 열로 늘어놓는다. 열 이름은 lag0, lag1, lag2."""
    s = weekly[f"{fuel}_intl_krw_l"]
    return pd.DataFrame({f"lag{k}": s.shift(k) for k in range(LAGS)})


def training_data(weekly, fuel):
    """상한제 이전 주 중 국제가와 세전 공급가가 모두 있는 주만 골라 (X, y) 로 돌려준다."""
    X = lagged_intl(weekly, fuel)
    y = weekly[f"{fuel}_supply_pretax"]
    ok = (~weekly["in_policy"]) & X.notna().all(axis=1) & y.notna()
    if ok.sum() < MARGIN_WEEKS:
        # 환율을 하나도 받지 못하면 국제가(원/리터)가 비어 학습할 수 없다
        raise RuntimeError(f"{fuel}: 학습할 주가 {ok.sum()}개뿐. 환율(ECOS·FRED)이 들어왔는지 확인할 것")
    return X[ok], y[ok]


# ── 모형 ────────────────────────────────────────────────────────────────
def fit(X, y, model, margin_weeks=MARGIN_WEEKS):
    """모형을 학습해 {'const': 상수, 'weights': 시차별 계수} 를 돌려준다. 두 모형 모두 예측식은 같다.

    예측 = const + weights · (같은 주, 1주 전, 2주 전 국제가)
    margin_weeks 는 최근 마진을 잡는 기간. 민감도 점검(counterfactual_check.py)에서만 바꾼다.
    """
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef = np.linalg.lstsq(A, y.to_numpy(), rcond=None)[0]          # 최소제곱 회귀 (OLS)
    if model == "baseline":
        return {"const": coef[0], "weights": coef[1:]}
    # 최근 마진: 계수를 비중으로 바꾸고(합 = 1), 마진은 최근 margin_weeks 주 평균으로
    weights = coef[1:] / coef[1:].sum()
    return {"const": recent_margin(X, y, weights, margin_weeks), "weights": weights}


def recent_margin(X, y, weights, margin_weeks=MARGIN_WEEKS):
    """최근 margin_weeks 주 동안 '세전 공급가 − 국제가 가중평균' 의 평균. 정유사 마진에 해당한다."""
    recent_X, recent_y = X.to_numpy()[-margin_weeks:], y.to_numpy()[-margin_weeks:]
    return float(np.mean(recent_y - recent_X @ weights))


def predict(params, X):
    """학습한 식에 국제가를 넣어 세전 공급가를 계산한다."""
    return params["const"] + X.to_numpy() @ params["weights"]


# ── 검증 ────────────────────────────────────────────────────────────────
def cutoffs(X):
    """검증 기점의 위치 목록. 2018년부터 13주마다, 뒤에 HORIZON 주가 남아 있는 곳까지."""
    return [i for i in range(len(X))
            if X.index[i] >= pd.Timestamp(BACKTEST_START) and i + HORIZON <= len(X)][::BACKTEST_STEP]


def backtest(X, y, fit_fn):
    """기점을 옮겨 가며 '그때까지 자료로 학습 → 다음 HORIZON 주 예측' 을 반복하고 오차를 모은다.

    상한제 기간과 같은 상황(학습 뒤 몇 달을 국제가만 보고 맞힌다)을 과거에 재현하는 검증이다.
    fit_fn(X, y) 는 {'const', 'weights'} 를 돌려주는 학습 함수. 모형마다 같은 기점으로 비교하려고 인자로 받는다.
    반환: 오차(예측 − 실제) 행렬. 행 = 기점, 열 = 기점 뒤 몇 주째
    """
    errors = []
    for c in cutoffs(X):
        params = fit_fn(X.iloc[:c], y.iloc[:c])
        errors.append(predict(params, X.iloc[c:c + HORIZON]) - y.iloc[c:c + HORIZON].to_numpy())
    return np.array(errors)


def summarize(errors):
    """오차 행렬을 성적표 한 줄로 줄인다. 오차는 '예측 − 실제' 이므로 bias < 0 이면 낮게 예측한 것이다."""
    e = errors.ravel()
    return {"MAE": np.abs(e).mean(), "bias": e.mean(), "RMSE": np.sqrt((e ** 2).mean()),
            "MAE_1_4w": np.abs(errors[:, :4]).mean(), "MAE_last_4w": np.abs(errors[:, -4:]).mean(),
            "err_q05": np.quantile(e, 0.05), "err_q95": np.quantile(e, 0.95), "n_cutoffs": len(errors)}


# ── 전체 실행 ───────────────────────────────────────────────────────────
def build_counterfactual(weekly):
    """유종·모형별로 학습과 검증을 하고, 주별 "없었다면" 가격표와 모형 비교표를 만든다.

    주별 표의 열 (fuel = gasoline, diesel, kerosene)
      {fuel}_cf_pretax      채택 모형의 "없었다면" 세전 공급가
      {fuel}_cf_aftertax    위 값 × 1.1 + 유류세. 상한과 바로 비교할 수 있는 값
      {fuel}_cf_low/high    세후 기준 90% 예측구간
      {fuel}_baseline_aftertax  비교용 기본 회귀의 세후 값
      {fuel}_supply_aftertax, {fuel}_cap  실제 세후 공급가와 상한 (대조용)
      provisional           공급가가 아직 공개되지 않은 상한제 기간의 주. 국제가로만 추정한 잠정값이다
    """
    out = pd.DataFrame(index=weekly.index)
    out["in_policy"] = weekly["in_policy"]
    supply_cols = [f"{fuel}_supply_aftertax" for fuel in cfg.FUELS]
    out["provisional"] = weekly["in_policy"] & weekly[supply_cols].isna().all(axis=1)
    rows = []
    for fuel in cfg.FUELS:
        X, y = training_data(weekly, fuel)
        X_all = lagged_intl(weekly, fuel)
        tax = weekly[f"{fuel}_tax"]
        for model in MODELS:
            params = fit(X, y, model)
            score = summarize(backtest(X, y, lambda Xt, yt, m=model: fit(Xt, yt, m)))
            rows.append({"fuel": fuel, "model": model, "name": MODELS[model], "train_weeks": len(X),
                         "train_end": X.index[-1].date(), "const": params["const"],
                         **{f"w_lag{k}": params["weights"][k] for k in range(LAGS)}, **score})
            pretax = pd.Series(predict(params, X_all), index=weekly.index)   # 국제가가 빈 주는 NaN
            if model == "baseline":
                out[f"{fuel}_baseline_aftertax"] = pretax * (1 + cfg.VAT) + tax
                continue
            out[f"{fuel}_cf_pretax"] = pretax
            out[f"{fuel}_cf_aftertax"] = pretax * (1 + cfg.VAT) + tax
            # 오차 = 예측 − 실제 → 실제 = 예측 − 오차. 그래서 하한에는 q95, 상한에는 q05 를 뺀다
            out[f"{fuel}_cf_low"] = (pretax - score["err_q95"]) * (1 + cfg.VAT) + tax
            out[f"{fuel}_cf_high"] = (pretax - score["err_q05"]) * (1 + cfg.VAT) + tax
        out[f"{fuel}_supply_aftertax"] = weekly[f"{fuel}_supply_aftertax"]
        out[f"{fuel}_cap"] = weekly[f"{fuel}_cap"]
    return out, pd.DataFrame(rows)


def gap_table(cf, start=cfg.POLICY_START):
    """상한제 기간의 '"없었다면" 세후 공급가 − 상한' 을 유종별로 보여 준다. (정부 추정과 대조용)

    '잠정' 열에 * 가 붙은 주는 공급가가 아직 공개되지 않아 실제값과 대조할 수 없는 주다.
    """
    policy = cf[cf.index >= pd.Timestamp(start)]
    table = pd.DataFrame({cfg.FUELS[f]["ko"]: (policy[f"{f}_cf_aftertax"] - policy[f"{f}_cap"]).round(0)
                          for f in cfg.FUELS})
    table["잠정"] = policy["provisional"].map({True: "*", False: ""})
    return table


def run():
    """학습 → 검증 → 저장 → 요약 출력."""
    weekly = load_weekly()
    cf, models = build_counterfactual(weekly)
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cf.to_csv(cfg.OUT_DIR / "counterfactual.csv", encoding="utf-8-sig")
    models.to_csv(cfg.OUT_DIR / "counterfactual_models.csv", index=False, encoding="utf-8-sig")
    return cf, models


def main():
    pd.set_option("display.width", 200)
    cf, models = run()
    print("[모형 비교] 기점 이동 검증, 단위 원/리터 (세전 공급가 기준, 오차 = 예측 − 실제)")
    show = models[["fuel", "name", "MAE", "bias", "RMSE", "MAE_1_4w", "MAE_last_4w", "err_q05", "err_q95",
                   "const", "w_lag0", "w_lag1", "w_lag2"]]
    print(show.round(2).to_string(index=False))
    print(f"\n[격차] 채택 모형('{MODELS[CHOSEN]}')의 \"없었다면\" 세후 공급가 − 상한, 원/리터")
    print(gap_table(cf).to_string())


if __name__ == "__main__":
    main()
