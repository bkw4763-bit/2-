"""석유 최고가격제의 판매가격 인하 효과 추정 (일간 회귀). 결과를 KDI <표 1>과 비교한다.

방법
  1) 최고가격제 시행 이전 일간 자료로 유종별 회귀식을 추정한다.
       주유소 판매가격_t = a + b0*국제가격_t + b1*국제가격_(t-7) + b2*국제가격_(t-14) + c*유류세_t + e_t
  2) 국제가격 반영 시차를 달리한 모형 3개를 추정한다 (당일만 / 7일 전까지 / 14일 전까지).
  3) 시행 이후 날짜에 실제 국제가격을 넣어 "최고가격제가 없었을 경우"의 가상 가격을 예측한다.
     이때 시행 직전 7일 동안 회귀식이 실제 가격과 벌어진 만큼(잔차)을 예측값에 더한다.
  4) 모형별 가상 가격의 중윗값을 대표값으로, 실제 - 가상을 정책 효과로 본다.

설정은 KDI 표에 맞추지 않고, 시행 이전 자료만으로 예측이 얼마나 맞는지 검증해 정했다. (--validate)

자료 (모두 오피넷 일간. build_dataset.py 가 input 의 원자료를 합쳐 만든다)
  data/opinet_daily.csv
    date            날짜
    *_retail        주유소 평균 판매가격(원/리터): 보통휘발유, 자동차용 경유, 실내등유
    *_intl          국제 제품가격(원/리터): 휘발유(92RON), 경유(0.001%), 등유. 주말·휴일은 직전 영업일 값
    *_tax           그날 적용된 유류세 합계(원/리터, 부가세 제외)

사용법
  python kdi_price_cap_replication.py              효과 추정과 KDI 표 1 비교
  python kdi_price_cap_replication.py --validate   가상 시행일 검증 표 추가
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장한다
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

import build_dataset

# ----------------------------------------------------------------------------
# 설정
# ----------------------------------------------------------------------------
SAMPLE_START = "2021-11-12"  # 추정 표본 시작일. 유류세 인하가 시작된 날
POLICY_START = "2026-03-13"  # 최고가격제 시행일. 추정 표본은 그 전날까지다
# KDI 표 1의 '3월 4주 차'에 해당하는 날짜. 오피넷 주간 판매가격은 일~목 5일 평균이다.
TARGET_FIRST, TARGET_LAST = "2026-03-22", "2026-03-26"
ANCHOR_DAYS = 7  # 잔차 보정에 쓰는 일수 (시행 직전 며칠의 잔차를 평균할지)

# 국제가격 반영 시차를 달리한 모형: 이름 → 회귀식에 넣는 시차(일)
LAG_MODELS = {"당일": [0], "7일 전까지": [0, 7], "14일 전까지": [0, 7, 14]}
# 검증에서 비교하는 대안: 시차를 28일 전까지 늘린 모형 5개
LONGER_LAG_MODELS = {**LAG_MODELS, "21일 전까지": [0, 7, 14, 21], "28일 전까지": [0, 7, 14, 21, 28]}

# 유종: (판매가격 열, 국제가격 열, 유류세 열 또는 None)
PRODUCTS = {
    "휘발유": ("gasoline_retail", "gasoline_intl", "gasoline_tax"),
    "자동차용 경유": ("diesel_retail", "diesel_intl", "diesel_tax"),
    "실내 등유": ("kerosene_retail", "kerosene_intl", None),  # 등유 세율은 표본 기간에 변동 없음
}

# KDI <표 1> 3월 4주 차 (원/리터). 추정 결과와 나란히 놓고 비교한다.
KDI_TABLE = pd.DataFrame(
    {"휘발유": [1819, 2279, -460], "자동차용 경유": [1816, 2732, -916], "실내 등유": [1509, 2061, -552]},
    index=["실제", "가상", "효과"],
)

# 가상 시행일 검증에 쓰는 날짜. 상한제가 없던 날을 시행일인 것처럼 놓고 예측이 맞는지 본다.
NORMAL_DATES = pd.date_range("2023-01-06", "2026-01-30", freq="28D")  # 평상시: 4주 간격 41개
SHOCK_DATES = pd.to_datetime([  # 2022년 급등·급락기 10개
    "2022-03-04", "2022-03-08", "2022-03-11", "2022-03-15", "2022-03-22",
    "2022-05-06", "2022-06-03", "2022-06-17", "2022-07-08", "2022-07-22",
])
SHOCK_SAMPLE_START = "2019-01-01"  # 2022년 검증은 그 전 자료가 충분해야 하므로 표본을 더 앞에서 시작한다


# ----------------------------------------------------------------------------
# 자료 준비
# ----------------------------------------------------------------------------
def load_data(path):
    """분석용 일간 자료를 읽어 날짜를 인덱스로 돌려준다.

    회귀에 필요한 열이 빠져 있으면 엉뚱한 결과가 나오기 전에 바로 알려 준다.
    """
    df = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["date"]).set_index("date").sort_index()
    needed = [column for columns in PRODUCTS.values() for column in columns if column]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise KeyError(f"입력 파일에 다음 열이 없습니다: {missing}")
    return df


# ----------------------------------------------------------------------------
# 추정
# ----------------------------------------------------------------------------
def design(df, product, lags):
    """한 유종·한 모형의 설명변수 표를 만든다. (상수항, 시차별 국제가격, 유류세)

    lags 가 [0, 7, 14] 이면 당일·7일 전·14일 전 국제가격이 들어간다.
    주유소가 국제가격을 며칠에 걸쳐 반영하는지를 모형마다 다르게 가정하기 위해서다.
    """
    _, intl, tax = PRODUCTS[product]
    X = pd.DataFrame({f"국제가격(t-{lag})": df[intl].shift(lag) for lag in lags})
    if tax:
        # 표본 기간에 세율이 여러 번 바뀌었으므로, 그 영향이 국제가격 계수에 섞이지 않게 따로 넣는다.
        X["유류세"] = df[tax]
    return sm.add_constant(X, has_constant="add")


def fit(df, product, lags, start, end):
    """표본 구간(start~end)으로 회귀식을 추정하고, 전 기간 예측값을 함께 돌려준다.

    추정에는 시행 이전 날짜만 쓴다. 그래야 예측값이 '제도가 없을 때의 관계'를 나타낸다.
    """
    y = df[PRODUCTS[product][0]]
    X = design(df, product, lags)
    in_sample = (df.index >= start) & (df.index <= end) & X.notna().all(axis=1) & y.notna()
    result = sm.OLS(y[in_sample], X[in_sample]).fit()
    return result, result.predict(X)


def predict_without_policy(df, product, lags, start, end, anchor=True):
    """한 모형의 가상 가격(제도가 없었을 경우의 예측)을 전 기간에 대해 돌려준다.

    회귀식의 예측값은 실제 가격과 한동안 같은 방향으로 벌어져 있는 경우가 많다.
    그래서 표본 마지막 7일 동안 벌어진 평균(잔차)을 예측값에 더해, 시행 직전의 실제 가격 수준에서 출발하게 한다.
    검증에서 이 보정을 넣으면 예측 오차가 절반 가까이 줄었다. anchor=False 는 보정 없는 값이다.
    돌려주는 값은 (추정 결과, 가상 가격, 더한 잔차)이다.
    """
    result, fitted = fit(df, product, lags, start, end)
    if not anchor:
        return result, fitted, 0.0
    residual = df[PRODUCTS[product][0]] - fitted
    last_days = residual.loc[end - pd.Timedelta(days=ANCHOR_DAYS - 1): end]
    correction = float(last_days.mean())
    return result, fitted + correction, correction


def counterfactual(df, start, policy_start, lag_models, anchor=True):
    """유종별로 시행일 이후의 실제 가격, 모형별 가상 가격, 중윗값, 효과를 표로 만든다.

    추정 표본은 start 부터 시행 전날까지다. 돌려주는 값은 (유종별 표 묶음, 모형별 추정 요약표)이다.
    한 모형에 기대지 않으려고 여러 시차 모형의 중윗값을 대표 가상 가격으로 쓴다.
    효과 = 실제 - 가상(중윗값)이며, 음수이면 그만큼 가격이 낮았다는 뜻이다.
    """
    start, policy_start = pd.Timestamp(start), pd.Timestamp(policy_start)
    end = policy_start - pd.Timedelta(days=1)
    after = df.index >= policy_start
    results, stats = {}, []
    for product, (retail, _, _) in PRODUCTS.items():
        table = pd.DataFrame({"실제": df.loc[after, retail]})
        for name, lags in lag_models.items():
            result, virtual, correction = predict_without_policy(df, product, lags, start, end, anchor)
            table[f"가상|{name}"] = virtual[after]
            stats.append({
                "유종": product, "모형": name, "표본 수": int(result.nobs),
                # 계수 합 = 국제가격이 1원 오를 때 판매가격이 결국 몇 원 오르는가
                "국제가격 계수 합": result.params.filter(like="국제가격").sum(),
                "유류세 계수": result.params.get("유류세", np.nan),
                "조정 R2": result.rsquared_adj,
                "잔차 보정(원)": correction,
            })
        model_columns = [column for column in table.columns if column.startswith("가상|")]
        table["가상(중윗값)"] = table[model_columns].median(axis=1)
        table["효과"] = table["실제"] - table["가상(중윗값)"]
        results[product] = table
    return results, pd.DataFrame(stats)


def period_mean(results, first, last):
    """first~last 날짜의 평균 실제·가상·효과를 KDI 표 1과 같은 모양(행: 구분, 열: 유종)으로 정리한다."""
    rows = {}
    for product, table in results.items():
        days = table.loc[first:last]
        rows[product] = [days["실제"].mean(), days["가상(중윗값)"].mean(), days["효과"].mean()]
    return pd.DataFrame(rows, index=["실제", "가상", "효과"])


def weekly_effects(results, weeks=5):
    """시행 이후 주별 평균 효과를 만든다. 오피넷 주간 평균과 같게 일요일~목요일 5일을 평균한다."""
    first_sunday = pd.Timestamp(POLICY_START) + pd.offsets.Week(weekday=6)
    rows = []
    for n in range(weeks):
        sunday = first_sunday + pd.Timedelta(weeks=n)
        thursday = sunday + pd.Timedelta(days=4)
        effect = period_mean(results, sunday, thursday).loc["효과"]
        rows.append({"기간(일~목)": f"{sunday:%m/%d}~{thursday:%m/%d}", **effect.round(0).to_dict()})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 검증: 상한제가 없던 날을 시행일로 가정했을 때 예측이 실제와 얼마나 벌어지는가
# ----------------------------------------------------------------------------
def placebo_errors(df, dates, start, lag_models, anchor):
    """가상 시행일마다 '가상 가격 - 실제 가격'을 유종별로 구한다.

    상한제가 없던 때이므로 이 값은 0 이어야 한다. 0 에서 벗어난 만큼이 이 방법의 예측 오차다.
    비교 구간은 본 추정과 같게 시행일 9~13일 뒤의 5일 평균으로 잡는다.
    """
    first_gap = (pd.Timestamp(TARGET_FIRST) - pd.Timestamp(POLICY_START)).days
    last_gap = (pd.Timestamp(TARGET_LAST) - pd.Timestamp(POLICY_START)).days
    errors = []
    for date in dates:
        results, _ = counterfactual(df, start, date, lag_models, anchor)
        window = period_mean(results, date + pd.Timedelta(days=first_gap), date + pd.Timedelta(days=last_gap))
        errors.append(window.loc["가상"] - window.loc["실제"])
    return pd.DataFrame(errors)


def validate(df):
    """평상시와 2022년 급등·급락기의 가상 시행일 예측 오차를 요약한다.

    RMSE 는 오차의 전형적인 크기, 평균 오차는 한쪽으로 치우친 정도다. (음수 = 실제보다 낮게 예측)
    본 설정과 함께 두 가지 대안(잔차 보정을 뺀 경우, 시차를 28일 전까지 늘린 경우)을 같은 방식으로 돌려,
    지금 설정을 쓰는 근거로 삼는다.
    """
    periods = [("평상시 (2023-01~2026-01)", NORMAL_DATES, SAMPLE_START),
               ("2022년 급등·급락기", SHOCK_DATES, SHOCK_SAMPLE_START)]
    settings = [("본 설정 (14일 전까지 3개, 잔차 보정)", LAG_MODELS, True),
                ("잔차 보정 없음", LAG_MODELS, False),
                ("시차 28일 전까지 5개", LONGER_LAG_MODELS, True)]
    rows = []
    for period, dates, start in periods:
        for setting, lag_models, anchor in settings:
            errors = placebo_errors(df, dates, start, lag_models, anchor)
            for product in PRODUCTS:
                rows.append({
                    "구간": period, "설정": setting, "유종": product, "시점 수": len(dates),
                    "RMSE(원)": float(np.sqrt((errors[product] ** 2).mean())),
                    "평균 오차(원)": float(errors[product].mean()),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------
def plot(df, results, path, first="2025-12-01", last="2026-05-31"):
    """유종별로 실제 가격과 가상 가격을 한 장에 그려 저장한다.

    굵은 주황선이 가상 가격(중윗값), 가는 주황선이 모형별 가상 가격이다.
    KDI 표 1의 기준 기간(3/22~3/26)에는 평균값을 점과 숫자로 적어 표와 바로 대조할 수 있게 한다.
    한글 글꼴이 없는 컴퓨터에서는 글자가 깨지지 않도록 영어 표기로 바꾼다.
    """
    installed = {font.name for font in fm.fontManager.ttflist}
    korean = next((name for name in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR"]
                   if name in installed), None)
    if korean:
        plt.rcParams["font.family"] = korean
    plt.rcParams["axes.unicode_minus"] = False
    label = (lambda ko, en: ko) if korean else (lambda ko, en: en)
    english = {"휘발유": "Gasoline", "자동차용 경유": "Diesel", "실내 등유": "Kerosene"}
    blue, orange, ink, muted = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"
    policy_start = pd.Timestamp(POLICY_START)
    target_middle = pd.Timestamp(TARGET_FIRST) + (pd.Timestamp(TARGET_LAST) - pd.Timestamp(TARGET_FIRST)) / 2
    target = period_mean(results, TARGET_FIRST, TARGET_LAST)

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.3), facecolor="#fcfcfb")
    for ax, (product, table) in zip(np.atleast_1d(axes), results.items()):
        shown = table.loc[:last]
        ax.set_facecolor("#fcfcfb")
        for column in [c for c in shown.columns if c.startswith("가상|")]:
            ax.plot(shown.index, shown[column], color=orange, lw=0.8, alpha=0.35)
        actual = df.loc[first:last, PRODUCTS[product][0]]
        ax.plot(actual.index, actual, color=blue, lw=2, label=label("실제", "Actual"))
        ax.plot(shown.index, shown["가상(중윗값)"], color=orange, lw=2,
                label=label("가상(모형 중윗값)", "Counterfactual (median)"))

        # 기준 기간의 평균 가상 가격(위)과 평균 실제 가격(아래)에 점과 숫자를 붙인다.
        marks = [(target.loc["가상", product], orange, "bottom", 8), (target.loc["실제", product], blue, "top", -8)]
        for value, color, align, offset in marks:
            ax.plot([target_middle], [value], "o", ms=8, color=color, mec="#fcfcfb", mew=2)
            ax.annotate(f"{value:,.0f}", (target_middle, value), xytext=(0, offset), textcoords="offset points",
                        ha="center", va=align, fontsize=10, color=ink)

        ax.axvline(policy_start, color=muted, lw=0.8, ls=":")
        ax.annotate(label("최고가격제 시행 (3/13) ", "Price cap starts (Mar 13) "), (policy_start, 0.98),
                    xycoords=("data", "axes fraction"), ha="right", va="top", fontsize=9, color=muted)
        ax.set_title(label(product, english[product]), color=ink, loc="left", fontsize=12)
        ax.set_ylabel(label("원/리터", "KRW per liter"), color=muted)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y.%m"))
        ax.grid(axis="y", color="#e4e3df", lw=0.6)
        ax.tick_params(colors=muted, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.legend(frameon=False, loc="center left", labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ----------------------------------------------------------------------------
# 실행
# ----------------------------------------------------------------------------
def main():
    """자료를 읽어 회귀를 추정하고, 효과 추정 결과를 화면에 보여 준 뒤 output 폴더에 저장한다."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="석유 최고가격제 효과 추정 (KDI 표 1과 비교)")
    parser.add_argument("--data", default=str(build_dataset.OUT_FILE), help="분석용 일간 자료")
    parser.add_argument("--start", default=SAMPLE_START, help="추정 표본 시작일")
    parser.add_argument("--outdir", default=str(here / "output"), help="결과 저장 폴더")
    parser.add_argument("--validate", action="store_true", help="가상 시행일 검증 표 출력")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)

    # 분석용 자료는 저장소에 올리지 않으므로, 처음 실행할 때 원자료에서 만든다.
    if Path(args.data) == build_dataset.OUT_FILE and not build_dataset.OUT_FILE.exists():
        build_dataset.main()
    df = load_data(args.data)
    results, stats = counterfactual(df, args.start, POLICY_START, LAG_MODELS)
    sample_end = pd.Timestamp(POLICY_START) - pd.Timedelta(days=1)

    print(f"\n추정 표본: {args.start} ~ {sample_end:%Y-%m-%d} / 비교 기간: {TARGET_FIRST} ~ {TARGET_LAST}")

    print("\n[1] 추정 결과 요약")
    print(stats.round(3).to_string(index=False))

    print(f"\n[2] {TARGET_FIRST}~{TARGET_LAST} 모형별 가상 가격 (원/리터)")
    by_model = pd.DataFrame({product: table.loc[TARGET_FIRST:TARGET_LAST].filter(like="가상").mean()
                             for product, table in results.items()})
    print(by_model.round(0).to_string())

    mine = period_mean(results, TARGET_FIRST, TARGET_LAST)
    print("\n[3] 추정 결과 (원/리터)")
    print(mine.round(0).to_string())
    print("\n    KDI 표 1")
    print(KDI_TABLE.to_string())
    print("\n    차이 (추정 - KDI)")
    diff = mine - KDI_TABLE
    diff.loc["가상 차이(%)"] = diff.loc["가상"] / KDI_TABLE.loc["가상"] * 100
    print(diff.round(1).to_string())

    print("\n[4] 주별 효과 (실제 - 가상 중윗값, 원/리터)")
    print(weekly_effects(results).to_string(index=False))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.concat(results, names=["유종"]).to_csv(outdir / "counterfactual_daily.csv", encoding="utf-8-sig")
    stats.to_csv(outdir / "model_stats.csv", index=False, encoding="utf-8-sig")
    pd.concat({"추정": mine, "KDI": KDI_TABLE.astype(float)}).to_csv(outdir / "table1_comparison.csv", encoding="utf-8-sig")
    plot(df, results, outdir / "counterfactual.png")

    if args.validate:
        table = validate(df)
        table.to_csv(outdir / "validation.csv", index=False, encoding="utf-8-sig")
        print("\n[5] 가상 시행일 검증: 상한제가 없던 날을 시행일로 놓았을 때의 예측 오차 (가상 - 실제)")
        print(table.round(1).to_string(index=False))

    print(f"\n결과 저장: {outdir}")


if __name__ == "__main__":
    main()
