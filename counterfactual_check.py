"""1단계 "없었다면" 모형의 근거 점검. 실행: python counterfactual_check.py

counterfactual.py 의 모형이 왜 타당한지, 심사에서 나올 질문 순서대로 확인한다.
  1. 정상성·공적분  공급가·국제가 수준에는 단위근이 있다 → 수준 회귀의 R² 는 근거가 되지 못한다
  2. 차분 회귀      단위근 문제가 없는 '주간 변화' 회귀로 시차 비중을 다시 구해 채택 모형과 비교한다
  3. 변수·시차 선택 시차 1~5개, 원유(브렌트) 대 제품가를 같은 기점 이동 검증으로 비교한다
  4. 구간 적중률    2018~2021 기점의 오차로 만든 90% 구간이 2022년 이후 기점에서 몇 % 맞았나
  5. 민감도         시차 비중 방식(수준/차분)과 마진 기간(26/52/104주)을 바꿔도 격차가 유지되나
  6. 그림           위 = 실제 세후 공급가, 상한, "없었다면" 가격과 90% 구간 / 아래 = 상한제 기간 유종별 격차
결과: data/processed/counterfactual_checks.csv, counterfactual_sensitivity.csv, counterfactual.png
"""
import warnings

import matplotlib
matplotlib.use("Agg")                     # 창을 띄우지 않고 파일로만 저장한다
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller, coint

import config as cfg
import counterfactual as cf
import prepare

GOV_CHECK_WEEK = "2026-09-12"     # 정부 설명(9/18) 직전에 공급가가 확정된 주. 정부 수치와 대조한다
SPLIT_DATE = "2022-01-01"         # 구간 적중률: 이 날 이전 기점으로 구간을 만들고, 이후 기점에서 맞았는지 본다
MARGIN_OPTIONS = (26, 52, 104)    # 민감도에서 바꿔 보는 마진 기간(주)
PLOT_START = "2025-07-01"         # 그림은 상한제 전 약 8개월부터 그린다
SECTIONS = {
    "1": "[1] 정상성·공적분 — p > 0.05 면 '단위근 있음'·'공적분 없음'. 수준 회귀의 R²·일반 표준오차는 근거가 아니다",
    "2": "[2] 차분 회귀 — 주간 변화끼리 회귀해 단위근 문제를 피한다. 비중이 채택 모형과 비슷하고 '합 = 1' 이 기각되지 않으면 된다",
    "3": "[3] 변수·시차 선택 — 기점 이동 검증의 평균 오차(원/리터). 작을수록 좋다",
    "4": "[4] 예측구간 적중률 — 목표 90%",
}
# 그림 색: 기준 팔레트(dataviz 스킬)의 1·2번 범주색과 글자·눈금색
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, MUTED, GRID, BASE, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


# ── 도구 ────────────────────────────────────────────────────────────────
def lagged(series, lags):
    """한 시계열의 같은 주·1주 전 … (lags−1)주 전 값을 열로 늘어놓는다."""
    return pd.DataFrame({f"lag{k}": series.shift(k) for k in range(lags)})


def pre_policy(weekly, fuel, X):
    """상한제 이전이면서 X 와 세전 공급가가 모두 있는 주만 남긴다. (학습 자료와 같은 규칙)"""
    y = weekly[f"{fuel}_supply_pretax"]
    ok = ~weekly["in_policy"] & X.notna().all(axis=1) & y.notna()
    return X[ok], y[ok]


def fit_chosen(X, y, margin_weeks=cf.MARGIN_WEEKS):
    """채택 모형(수준 회귀 비중 + 최근 마진)으로 학습한다."""
    return cf.fit(X, y, cf.CHOSEN, margin_weeks)


def fit_diff(X, y, margin_weeks=cf.MARGIN_WEEKS):
    """시차 비중을 주간 변화(차분) 회귀로 구하고, 마진은 채택 모형과 같게 최근 평균으로 잡는다."""
    b = np.linalg.lstsq(np.diff(X.to_numpy(), axis=0), np.diff(y.to_numpy()), rcond=None)[0]
    weights = b / b.sum()
    return {"const": cf.recent_margin(X, y, weights, margin_weeks), "weights": weights}


def adf_p(series):
    """ADF 단위근 검정의 p값. p > 0.05 면 '단위근이 있다(정상 시계열이 아니다)' 를 기각하지 못한다.

    statsmodels 0.15 가 반환 형식 변경 예고 경고를 띄우는데, 결과와 무관해 숨긴다.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return adfuller(series.dropna())[1]


def mae(X, y, fit_fn=fit_chosen):
    """기점 이동 검증의 평균 절대 오차(원/리터)."""
    return float(np.abs(cf.backtest(X, y, fit_fn)).mean())


# ── 점검 1~4 ────────────────────────────────────────────────────────────
def check_fuel(weekly, fuel):
    """한 유종의 점검 1~4 결과를 {항목: 값} 으로 돌려준다. 항목 이름 앞의 숫자가 점검 번호다."""
    X, y = cf.training_data(weekly, fuel)
    chosen = fit_chosen(X, y)
    index = pd.Series(X.to_numpy() @ chosen["weights"], index=X.index)     # 국제가 가중평균
    r = {}

    # 1. 정상성(ADF)과 공적분(Engle-Granger). 둘 다 p값
    r["1 공급가 수준 ADF p"] = adf_p(y)
    r["1 공급가 차분 ADF p"] = adf_p(y.diff())
    r["1 국제가 수준 ADF p"] = adf_p(X["lag0"])
    r["1 국제가 차분 ADF p"] = adf_p(X["lag0"].diff())
    r["1 공적분 p"] = coint(y, index)[1]
    levels = sm.OLS(y, sm.add_constant(X)).fit()
    r["1 수준 회귀 R²"] = levels.rsquared
    r["1 수준 회귀 DW (2 에 가까워야 정상)"] = durbin_watson(levels.resid)

    # 2. 차분 회귀. 잔차 자기상관을 고려한 Newey-West 표준오차로 '계수 합 = 1' 을 검정한다
    dX, dy = X.diff().dropna(), y.diff().dropna()
    diff = sm.OLS(dy, dX).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
    for k in range(cf.LAGS):
        r[f"2 비중 {k}주 전 (채택: 수준 회귀)"] = chosen["weights"][k]
        r[f"2 비중 {k}주 전 (차분 회귀)"] = diff.params.iloc[k] / diff.params.sum()
    r["2 차분 회귀 계수 합"] = diff.params.sum()
    r["2 '계수 합 = 1' 검정 p"] = float(diff.t_test(" + ".join(dX.columns) + " = 1").pvalue)

    # 3. 시차 수와 설명변수 선택. 모두 같은 검증, 같은 최근 마진 방식
    intl = weekly[f"{fuel}_intl_krw_l"]
    for lags in range(1, 6):
        r[f"3 시차 {lags}개"] = mae(*pre_policy(weekly, fuel, lagged(intl, lags)))
    r["3 원유(브렌트) 시차 3개"] = mae(*pre_policy(weekly, fuel, lagged(weekly["brent_krw_l"], cf.LAGS)))

    # 4. 앞 시기 오차로 만든 구간이 뒤 시기에 맞는지 (구간을 만든 자료로 적중률을 재면 안 되므로 나눈다)
    errors = cf.backtest(X, y, fit_chosen)
    dates = X.index[cf.cutoffs(X)]
    early, late = errors[dates < SPLIT_DATE], errors[dates >= SPLIT_DATE]
    low, high = np.quantile(early, [0.05, 0.95])
    r["4 2018~21 기점 오차 5% 분위"] = low
    r["4 2018~21 기점 오차 95% 분위"] = high
    r["4 2022년 이후 기점 적중률(%)"] = 100 * np.mean((late >= low) & (late <= high))
    return r


# ── 점검 5: 민감도 ──────────────────────────────────────────────────────
def sensitivity(weekly):
    """시차 비중 방식(수준/차분) × 마진 기간(26/52/104주) 6가지로 상한제 기간의 격차를 다시 계산한다.

    반환: (주별 격차 표: 유종마다 채택값·최솟값·최댓값, 사양별 검증 오차 표)
    """
    policy = weekly["in_policy"]
    gaps, scores = pd.DataFrame(index=weekly.index[policy]), []
    for fuel in cfg.FUELS:
        X, y = cf.training_data(weekly, fuel)
        X_policy = cf.lagged_intl(weekly, fuel)[policy]
        tax, cap = weekly.loc[policy, f"{fuel}_tax"], weekly.loc[policy, f"{fuel}_cap"]
        by_spec = {}
        for method, base_fit in (("수준", fit_chosen), ("차분", fit_diff)):
            for weeks in MARGIN_OPTIONS:
                fit_fn = lambda a, b, f=base_fit, m=weeks: f(a, b, m)
                spec = f"{method} 비중·마진 {weeks}주"
                pretax = cf.predict(fit_fn(X, y), X_policy)
                by_spec[spec] = pretax * (1 + cfg.VAT) + tax - cap
                scores.append({"fuel": cfg.FUELS[fuel]["ko"], "spec": spec, "MAE": mae(X, y, fit_fn)})
        by_spec = pd.DataFrame(by_spec)
        gaps[f"{fuel}_gap"] = by_spec[f"수준 비중·마진 {cf.MARGIN_WEEKS}주"]       # 채택 모형
        gaps[f"{fuel}_gap_min"] = by_spec.min(axis=1)
        gaps[f"{fuel}_gap_max"] = by_spec.max(axis=1)
    scores = pd.DataFrame(scores).pivot(index="spec", columns="fuel", values="MAE")
    return gaps, scores[[info["ko"] for info in cfg.FUELS.values()]]     # 휘발유·경유·등유 순서로


# ── 점검 6: 그림 ────────────────────────────────────────────────────────
def style_axis(ax, title, date_format):
    """모든 칸을 같은 모양으로 꾸민다: 칸 제목, 옅은 가로 눈금선, 위·오른쪽 테두리 없음, 천 단위 쉼표."""
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=13, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=MUTED, labelsize=9, labelleft=True)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))


def draw_levels(ax, result, fuel):
    """위 칸: 실제 세후 공급가, 상한, "없었다면" 세후 공급가와 90% 구간 (상한제 전 약 8개월부터)."""
    view = result[result.index >= PLOT_START]
    ax.fill_between(view.index, view[f"{fuel}_cf_low"], view[f"{fuel}_cf_high"],
                    color=ORANGE, alpha=0.15, linewidth=0, label="90% 구간")
    ax.plot(view.index, view[f"{fuel}_cf_aftertax"], color=ORANGE, lw=1.6, label='"없었다면" 세후 공급가')
    ax.plot(view.index, view[f"{fuel}_supply_aftertax"], color=BLUE, lw=1.6, label="실제 세후 공급가")
    ax.plot(view.index, view[f"{fuel}_cap"], color=INK_2, lw=1.2, ls="--", label="상한")
    ax.axvline(pd.Timestamp(cfg.POLICY_START), color=MUTED, lw=0.8)
    ax.text(pd.Timestamp(cfg.POLICY_START), 0.98, " 상한제 시행", transform=ax.get_xaxis_transform(),
            color=MUTED, fontsize=9, va="top")
    style_axis(ax, cfg.FUELS[fuel]["ko"], "%y.%m")


def draw_gap(ax, result, fuel, caps):
    """아래 칸: 상한제 기간의 격차("없었다면" 세후 공급가 − 상한)와 90% 구간.

    점선 부분은 공급가가 아직 공개되지 않은 잠정 주이고, 회색 세로선은 상한을 바꾼 날이다.
    """
    policy = result[result["in_policy"]]
    firm_end = policy.index[~policy["provisional"]][-1]                    # 공급가가 공개된 마지막 주
    gap = policy[f"{fuel}_cf_aftertax"] - policy[f"{fuel}_cap"]
    ax.fill_between(policy.index, policy[f"{fuel}_cf_low"] - policy[f"{fuel}_cap"],
                    policy[f"{fuel}_cf_high"] - policy[f"{fuel}_cap"], color=ORANGE, alpha=0.15, linewidth=0)
    ax.plot(gap[:firm_end].index, gap[:firm_end], color=ORANGE, lw=1.8)
    ax.plot(gap[firm_end:].index, gap[firm_end:], color=ORANGE, lw=1.8, ls=":")            # 잠정 구간
    ax.axhline(0, color=BASE, lw=1)
    # 상한을 바꾼 차수 (1차 시행과, 금액이 그대로인 동결 차수는 표시하지 않는다)
    moved = caps.assign(change=caps[fuel].diff()).query("change != 0").dropna(subset=["change"])
    for _, row in moved.iterrows():
        ax.axvline(row["start_date"], color=MUTED, lw=0.8, ls="--")
        ax.text(row["start_date"], 0.98, f" {row['round']}차 {row['change']:+.0f}원",
                transform=ax.get_xaxis_transform(), color=MUTED, fontsize=9, va="top")
    # 숫자는 마지막 확정 주와 최근 잠정 주에만 단다
    ax.annotate(f"{gap[firm_end]:,.0f}", (firm_end, gap[firm_end]), textcoords="offset points",
                xytext=(-6, 8), ha="right", fontsize=10, color=INK)
    ax.annotate(f"{gap.iloc[-1]:,.0f}\n(잠정)", (gap.index[-1], gap.iloc[-1]), textcoords="offset points",
                xytext=(4, -4), ha="left", va="top", fontsize=9, color=INK_2)
    style_axis(ax, f"{cfg.FUELS[fuel]['ko']} 격차", "%m월")


def plot(result, path):
    """유종별 2행 그림을 한 장으로 저장한다. 위 = 가격 수준, 아래 = 격차.

    격차 칸 세 개는 세로축 눈금을 같게 두어 유종끼리 크기를 바로 비교할 수 있게 한다.
    """
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False})   # 윈도 한글 글꼴
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), facecolor=SURFACE)
    for ax in axes[1, 1:]:
        ax.sharey(axes[1, 0])
    caps = prepare.load_caps()
    for col, fuel in enumerate(cfg.FUELS):
        draw_levels(axes[0, col], result, fuel)
        draw_gap(axes[1, col], result, fuel, caps)
    # 맨 위 줄 = 제목, 둘째 줄 = 범례(왼쪽)와 단위·잠정 설명(오른쪽)
    # 제목의 빼기는 일반 하이픈으로 쓴다 (맑은 고딕에 수학 기호 '−' 가 없어 네모로 깨진다)
    fig.suptitle('상한제가 없었다면: 정유사 세후 공급가(위)와 격차 = "없었다면" - 상한(아래)',
                 x=0.01, y=0.995, ha="left", va="top", fontsize=14, color=INK)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.004, 0.968), ncol=4,
               frameon=False, fontsize=10, labelcolor=INK_2)
    fig.text(0.99, 0.958, "원/리터, 주별 · 아래 점선 = 잠정(공급가 미공개 주)", ha="right", va="top",
             fontsize=9.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=2.5)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ── 실행 ────────────────────────────────────────────────────────────────
def main():
    pd.set_option("display.width", 200)
    weekly = cf.load_weekly()
    result, _ = cf.build_counterfactual(weekly)

    checks = pd.DataFrame({cfg.FUELS[f]["ko"]: check_fuel(weekly, f) for f in cfg.FUELS})
    for number, title in SECTIONS.items():
        print(f"\n{title}")
        print(checks[checks.index.str.startswith(number)].round(3).to_string())

    gaps, scores = sensitivity(weekly)
    print("\n[5] 민감도 — 사양 6가지의 검증 평균 오차(원/리터)와 격차(\"없었다면\" 세후 공급가 − 상한, 원/리터)")
    print(scores.round(1).to_string())
    confirmed = gaps.index[~result.loc[gaps.index, "provisional"]]
    weeks = {f"정부 대조 주 {GOV_CHECK_WEEK}": pd.Timestamp(GOV_CHECK_WEEK),
             f"마지막 확정 주 {confirmed[-1].date()}": confirmed[-1],
             f"최근 주 {gaps.index[-1].date()} (잠정)": gaps.index[-1]}
    table = pd.DataFrame({cfg.FUELS[f]["ko"]: {name: "{:.0f} ({:.0f}~{:.0f})".format(
        *gaps.loc[week, [f"{f}_gap", f"{f}_gap_min", f"{f}_gap_max"]]) for name, week in weeks.items()}
        for f in cfg.FUELS})
    print("  격차: 채택값 (6가지 사양의 최소~최대)")
    print(table.to_string())

    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    checks.to_csv(cfg.OUT_DIR / "counterfactual_checks.csv", encoding="utf-8-sig")
    gaps.to_csv(cfg.OUT_DIR / "counterfactual_sensitivity.csv", encoding="utf-8-sig")
    plot(result, cfg.OUT_DIR / "counterfactual.png")
    print(f"\n[6] 그림 저장: {cfg.OUT_DIR / 'counterfactual.png'}")


if __name__ == "__main__":
    main()
