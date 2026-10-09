"""KDI 「석유 최고가격제와 유류세 인하의 효과」(2026.4.22) <표 1> 재현.

방법 (단순 시차 회귀. 표본 기간과 시차 설정은 표 1에 가장 가까운 것을 골랐다)
  1) 최고가격제 시행 이전 주간 자료로 유종별 회귀식을 추정한다.
       주유소 판매가격_t = a + sum_{k=0..K} b_k * 국제 제품가격_{t-k} + c * 유류세_t + e_t
  2) 국제유가 반영 시차를 달리한 모형 3개를 추정한다 (K = 0, 1, 2주).
  3) 시행 이후 주에 실제 국제가격을 넣어 "최고가격제가 없었을 경우"의 가상 가격을 예측한다.
  4) 모형별 추정치의 중윗값을 가상 가격으로, 실제 - 가상을 정책 효과로 본다.

자료 (모두 오피넷 주간. build_dataset.py 가 input 의 원자료를 합쳐 만든다)
  data/opinet_weekly.csv
    week            오피넷 주차 (예: 2026-03-4 = 2026년 3월 4주)
    *_retail        주유소 평균 판매가격(원/리터): 보통휘발유, 자동차용 경유, 실내등유
    *_intl          국제 제품가격(원/리터): 휘발유(92RON), 경유(0.001%), 등유
    *_intl_usd      국제 제품가격(달러/배럴). 참고용이며 회귀에는 쓰지 않는다
    *_tax           유류세 합계(원/리터, 부가세 제외). 주 7일 평균

사용법
  python kdi_price_cap_replication.py                 기본 설정으로 표 1 재현
  python kdi_price_cap_replication.py --sensitivity   표본 시작 시점·시차 모형 수에 따른 민감도
"""
import argparse
import datetime as dt
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
SAMPLE_START = "2021-11-3"  # 유류세 인하 시행(2021-11-12) 후 첫 온전한 주
SAMPLE_END = "2026-03-1"    # 최고가격제 시행(3/13, 3월 2주) 직전 주
POLICY_START = "2026-03-2"  # 최고가격제 시행 주
TARGET_WEEK = "2026-03-4"   # 표 1 기준 주 (1차 최고가격 적용 마지막 주)

# 국제유가 반영 시차를 달리한 모형: 이름 → 회귀식에 넣는 시차(주)
LAG_MODELS = {"시차 없음": [0], "시차 1주": [0, 1], "시차 2주": [0, 1, 2]}

# 유종: (판매가격 열, 국제가격 열, 유류세 열 또는 None)
PRODUCTS = {
    "휘발유": ("gasoline_retail", "gasoline_intl", "gasoline_tax"),
    "자동차용 경유": ("diesel_retail", "diesel_intl", "diesel_tax"),
    "실내 등유": ("kerosene_retail", "kerosene_intl", None),  # 등유 세율은 표본 기간에 변동 없음
}

# KDI <표 1> 3월 4주 차 (원/리터). 재현 결과와 비교할 목표값이다.
KDI_TABLE = pd.DataFrame(
    {"휘발유": [1819, 2279, -460], "자동차용 경유": [1816, 2732, -916], "실내 등유": [1509, 2061, -552]},
    index=["실제", "가상", "효과"],
)


# ----------------------------------------------------------------------------
# 자료 준비
# ----------------------------------------------------------------------------
def week_start(label):
    """오피넷 주차('연-월-주')의 시작일(일요일)을 돌려준다.

    오피넷의 'M월 N주'는 그 달의 N번째 수요일이 든 일~토 주간이다.
    주차 표기만으로는 앞뒤 순서를 비교할 수 없어, 표본 구간을 자를 때 이 날짜를 쓴다.
    """
    year, month, nth = map(int, label.split("-"))
    day = dt.date(year, month, 1)
    while day.weekday() != 2:  # 2 = 수요일
        day += dt.timedelta(days=1)
    return day + dt.timedelta(weeks=nth - 1, days=-3)


def load_data(path):
    """분석용 주간 자료를 읽어 주차를 인덱스로, 시간 순서로 정렬해 돌려준다.

    회귀에 필요한 열이 빠져 있으면 엉뚱한 결과가 나오기 전에 바로 알려 준다.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    needed = ["week"] + [column for columns in PRODUCTS.values() for column in columns if column]
    missing = [column for column in needed if column not in df.columns]
    if missing:
        raise KeyError(f"입력 파일에 다음 열이 없습니다: {missing}")
    df["week_start"] = [week_start(week) for week in df["week"]]
    return df.sort_values("week_start").set_index("week")


# ----------------------------------------------------------------------------
# 추정
# ----------------------------------------------------------------------------
def design(df, product, lags):
    """한 유종·한 모형의 설명변수 표를 만든다. (상수항, 시차별 국제가격, 유류세)

    lags 가 [0, 1, 2] 이면 이번 주·1주 전·2주 전 국제가격이 들어간다.
    주유소가 국제가격을 몇 주에 걸쳐 반영하는지를 모형마다 다르게 가정하기 위해서다.
    """
    _, intl, tax = PRODUCTS[product]
    X = pd.DataFrame({f"국제가격(t-{k})": df[intl].shift(k) for k in lags})
    if tax:
        # 표본 기간에 세율이 여러 번 바뀌었으므로, 그 영향이 국제가격 계수에 섞이지 않게 따로 넣는다.
        X["유류세"] = df[tax]
    return sm.add_constant(X, has_constant="add")


def fit(df, product, lags, start, end):
    """표본 구간(start~end)으로 회귀식을 추정하고, 전 기간 예측값을 함께 돌려준다.

    추정에는 최고가격제 시행 이전 주만 쓴다. 그래야 예측값이 '제도가 없을 때의 관계'를 나타낸다.
    예측은 전 기간에 대해 내므로, 시행 이후 주의 예측값이 곧 가상 가격이다.
    """
    y = df[PRODUCTS[product][0]]
    X = design(df, product, lags)
    in_sample = (df["week_start"] >= df.loc[start, "week_start"]) & (df["week_start"] <= df.loc[end, "week_start"])
    in_sample &= X.notna().all(axis=1)  # 시차 때문에 값이 빈 맨 앞 주는 뺀다
    result = sm.OLS(y[in_sample], X[in_sample]).fit()
    return result, result.predict(X)


def counterfactual(df, start, end, lag_models):
    """유종별로 시행 이후 주의 실제 가격, 모형별 가상 가격, 중윗값, 효과를 표로 만든다.

    돌려주는 값은 (유종별 표 묶음, 모형별 추정 요약표)이다.
    한 모형에 기대지 않으려고 여러 시차 모형의 중윗값을 대표 가상 가격으로 쓴다.
    효과 = 실제 - 가상(중윗값)이며, 음수이면 그만큼 가격이 낮았다는 뜻이다.
    """
    policy = df["week_start"] >= df.loc[POLICY_START, "week_start"]
    results, stats = {}, []
    for product, (retail, _, _) in PRODUCTS.items():
        table = pd.DataFrame({"주 시작": df.loc[policy, "week_start"], "실제": df.loc[policy, retail]})
        for name, lags in lag_models.items():
            result, predicted = fit(df, product, lags, start, end)
            table[f"가상|{name}"] = predicted[policy]
            stats.append({
                "유종": product, "모형": name, "표본 수": int(result.nobs),
                # 계수 합 = 국제가격이 1원 오를 때 판매가격이 결국 몇 원 오르는가
                "국제가격 계수 합": result.params.filter(like="국제가격").sum(),
                "유류세 계수": result.params.get("유류세", np.nan),
                "조정 R2": result.rsquared_adj,
            })
        model_columns = [column for column in table.columns if column.startswith("가상|")]
        table["가상(중윗값)"] = table[model_columns].median(axis=1)
        table["효과"] = table["실제"] - table["가상(중윗값)"]
        results[product] = table
    return results, pd.DataFrame(stats)


def table1(results, week=TARGET_WEEK):
    """기준 주의 실제·가상·효과를 KDI 표 1과 같은 모양(행: 구분, 열: 유종)으로 정리한다."""
    return pd.DataFrame(
        {product: [t.loc[week, "실제"], t.loc[week, "가상(중윗값)"], t.loc[week, "효과"]] for product, t in results.items()},
        index=["실제", "가상", "효과"],
    )


def sensitivity(df):
    """표본 시작·끝 시점과 시차 모형 수를 바꿨을 때 기준 주의 가상 가격이 얼마나 달라지는지 표로 만든다.

    기본 설정은 표 1에 가깝도록 고른 것이어서, 설정을 조금 바꿔도 결과가 유지되는지 보여 주기 위해서다.
    '최대 오차(%)'는 세 유종 가운데 KDI 가상 가격과 가장 많이 벌어진 정도다.
    """
    rows = []
    for start in ["2015-01-3", "2019-01-3", "2021-01-1", "2021-11-3", "2022-01-1", "2022-07-1", "2023-01-1"]:
        for end in ["2026-02-4", "2026-03-1"]:
            for max_lag in [1, 2, 3, 4]:
                models = {f"L{k}": list(range(k + 1)) for k in range(max_lag + 1)}
                virtual = table1(counterfactual(df, start, end, models)[0]).loc["가상"]
                gap = (virtual - KDI_TABLE.loc["가상"]) / KDI_TABLE.loc["가상"] * 100
                rows.append({
                    "표본 시작": start, "표본 끝": end, "시차 모형": f"0~{max_lag}주",
                    **{product: round(virtual[product]) for product in PRODUCTS}, "최대 오차(%)": gap.abs().max(),
                })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# 그림
# ----------------------------------------------------------------------------
def plot(df, results, path, weeks_before=14, weeks_after=10):
    """유종별로 실제 가격과 가상 가격을 한 장에 그려 저장한다.

    시행 전 14주와 시행 후 10주만 보여 준다. 굵은 주황선이 가상 가격(중윗값), 가는 주황선이 모형별 가상 가격이다.
    기준 주에는 두 값을 숫자로 적어 표 1과 바로 대조할 수 있게 한다.
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

    weeks = list(df.index)
    first_policy = weeks.index(POLICY_START)
    window = weeks[max(0, first_policy - weeks_before): first_policy + weeks_after]

    def midweek(labels):
        """주차 표기를 그 주의 수요일 날짜로 바꾼다. (가로축 위치)"""
        return [pd.Timestamp(df.loc[week, "week_start"]) + pd.Timedelta(days=3) for week in labels]

    fig, axes = plt.subplots(1, len(results), figsize=(5.2 * len(results), 4.3), facecolor="#fcfcfb")
    for ax, (product, table) in zip(np.atleast_1d(axes), results.items()):
        policy_weeks = [week for week in window if week in table.index]
        ax.set_facecolor("#fcfcfb")
        for column in [c for c in table.columns if c.startswith("가상|")]:
            ax.plot(midweek(policy_weeks), table.loc[policy_weeks, column], color=orange, lw=0.8, alpha=0.35)
        ax.plot(midweek(window), df.loc[window, PRODUCTS[product][0]], color=blue, lw=2, label=label("실제", "Actual"))
        ax.plot(midweek(policy_weeks), table.loc[policy_weeks, "가상(중윗값)"], color=orange, lw=2,
                label=label("가상(모형 중윗값)", "Counterfactual (median)"))

        # 기준 주의 가상 가격(위)과 실제 가격(아래)에 점과 숫자를 붙인다.
        target_x = midweek([TARGET_WEEK])[0]
        marks = [(table.loc[TARGET_WEEK, "가상(중윗값)"], orange, "bottom", 8), (table.loc[TARGET_WEEK, "실제"], blue, "top", -8)]
        for value, color, align, offset in marks:
            ax.plot([target_x], [value], "o", ms=8, color=color, mec="#fcfcfb", mew=2)
            ax.annotate(f"{value:,.0f}", (target_x, value), xytext=(0, offset), textcoords="offset points",
                        ha="center", va=align, fontsize=10, color=ink)

        ax.axvline(pd.Timestamp("2026-03-13"), color=muted, lw=0.8, ls=":")
        ax.annotate(label("최고가격제 시행 (3/13) ", "Price cap starts (Mar 13) "), (pd.Timestamp("2026-03-13"), 0.98),
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
    """자료를 읽어 회귀를 추정하고, 표 1 재현 결과를 화면에 보여 준 뒤 output 폴더에 저장한다."""
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="KDI 표 1(3월 4주 차 최고가격제 효과) 재현")
    parser.add_argument("--data", default=str(build_dataset.OUT_FILE), help="분석용 주간 자료")
    parser.add_argument("--start", default=SAMPLE_START, help="추정 표본 시작 주")
    parser.add_argument("--end", default=SAMPLE_END, help="추정 표본 마지막 주")
    parser.add_argument("--outdir", default=str(here / "output"), help="결과 저장 폴더")
    parser.add_argument("--sensitivity", action="store_true", help="표본·시차 설정별 민감도 표 출력")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.unicode.east_asian_width", True)

    # 분석용 자료는 저장소에 올리지 않으므로, 처음 실행할 때 원자료에서 만든다.
    if Path(args.data) == build_dataset.OUT_FILE and not build_dataset.OUT_FILE.exists():
        build_dataset.main()
    df = load_data(args.data)
    results, stats = counterfactual(df, args.start, args.end, LAG_MODELS)

    print(f"\n추정 표본: {args.start} ~ {args.end} / 기준 주: {TARGET_WEEK} "
          f"({df.loc[TARGET_WEEK, 'week_start']} 시작 주)")

    print("\n[1] 추정 결과 요약")
    print(stats.round(3).to_string(index=False))

    print(f"\n[2] {TARGET_WEEK} 모형별 가상 가격 (원/리터)")
    by_model = pd.DataFrame({product: t.loc[TARGET_WEEK].filter(like="가상") for product, t in results.items()}).astype(float)
    print(by_model.round(0).to_string())

    mine = table1(results)
    print("\n[3] 표 1 재현 결과 (원/리터)")
    print(mine.round(0).to_string())
    print("\n    KDI 표 1")
    print(KDI_TABLE.to_string())
    print("\n    차이 (재현 - KDI)")
    diff = mine - KDI_TABLE
    diff.loc["가상 오차(%)"] = diff.loc["가상"] / KDI_TABLE.loc["가상"] * 100
    print(diff.round(1).to_string())

    print("\n[4] 주별 효과 (실제 - 가상 중윗값, 원/리터)")
    weekly = pd.DataFrame({product: t["효과"] for product, t in results.items()})
    weekly.insert(0, "주 시작", results["휘발유"]["주 시작"])
    print(weekly.head(6).round(0).to_string())

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.concat(results, names=["유종"]).to_csv(outdir / "counterfactual_weekly.csv", encoding="utf-8-sig")
    stats.to_csv(outdir / "model_stats.csv", index=False, encoding="utf-8-sig")
    pd.concat({"재현": mine, "KDI": KDI_TABLE.astype(float)}).to_csv(outdir / "table1_comparison.csv", encoding="utf-8-sig")
    plot(df, results, outdir / "counterfactual.png")

    if args.sensitivity:
        table = sensitivity(df)
        table.to_csv(outdir / "sensitivity.csv", index=False, encoding="utf-8-sig")
        print("\n[5] 민감도: 3월 4주 가상 가격 (KDI: 2,279 / 2,732 / 2,061)")
        print(table.round(2).to_string(index=False))

    print(f"\n결과 저장: {outdir}")


if __name__ == "__main__":
    main()
