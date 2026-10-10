# 1. 2단계: 격차와 재정 부담입니다. "상한제가 없었다면" 가격과 상한의 차이를 날마다 계산하고 판매량을 곱합니다.
# 결과는 휘발유·경유·등유를 합한 정부 전체 재정 부담으로 냅니다. 상한이 유종마다 달라 격차는 유종별로 구한 뒤 금액에서 합칩니다.
# ① 상한: csv/price_caps.csv의 차수별 상한(세후 공급가 기준)을 시행일부터 다음 차수 시행일 전날까지 날마다 펼칩니다.
# ② "없었다면" 가격: 1단계 모형(regression/full_sample)으로 주별 세후 가격과 오차 경로를 만듭니다.
#    공급가가 아직 공개되지 않은 최근 주(잠정)는 merged_weekly_data.csv의 국제가로 추정하고, 유류세는 마지막 값을 씁니다.
# ③ 격차 = "없었다면" 세후 공급가 − 상한. "없었다면" 가격은 주 안에서 같다고 보고, 상한은 날마다 그날 값을 씁니다.
#    상한이 주 중간에 바뀐 주(3/27 인상, 6/27 인하)와 정책이 금요일에 시작한 첫 주를 정확히 나누기 위해서입니다.
# ④ 하루 재정 부담 = 세 유종의 max(격차, 0) × 하루 판매량의 합. 격차가 음수인 날은 시장가가 상한보다 낮아 보전할 것이 없으므로 0입니다.
# ⑤ 불확실성: 기점 이동 검증의 오차 경로 31개를 세 유종에 함께 적용해 합계 경로를 만들고, 그 5%·95% 분위수를 90% 구간으로 씁니다.
#    같은 기점의 오차를 세 유종에 같이 쓰는 이유: 세 유종의 오차가 같은 국제가 충격에서 나와 함께 움직이기 때문입니다.
#    주별 오차도 서로 상관이 커서(0.55~0.68), 주별 구간을 독립으로 더하면 합계 구간이 실제보다 좁아집니다.
# 실행: python "fiscal/fiscal.py"  (먼저 python "preprocessing/preprocess_fuel_demand.py"로 판매량 파일을 만듭니다)
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "regression" / "full_sample"))
from counterfactual import (
    FUELS, HORIZON, POLICY_START, VAT, backtest, cutoffs, fit, lagged_intl, load_weekly, predict, training_data,
    week_end,
)
from counterfactual_check import AXIS, INK, INK_2, MUTED, apply_plot_style




# 2. 입력·출력 위치와 단위입니다.
# 판매량 파일 형식: 월(YYYY-MM), 휘발유, 경유, 등유 열. 단위는 천 배럴/월 (석유공사 페트로넷 제품별 소비 통계의 단위)
CAPS_PATH = ROOT / "csv" / "price_caps.csv"
MERGED_PATH = ROOT / "csv" / "merged_weekly_data.csv"
VOLUME_PATH = ROOT / "csv" / "fuel_sales_monthly.csv"
OUT_DIR = ROOT / "results" / "fiscal"
LITERS_PER_BARREL = 158.987
EOK = 1e8                     # 1억 원
JO = 1e12                     # 1조 원
TOTAL_COLOR = "#2a78d6"       # 그림에 계열이 하나(세 유종 합계)뿐이라 기본 순서의 첫 색을 씁니다.

# 상한이 걸리는 판매량의 비중입니다. 출하량 통계의 경유는 수송·산업·선박용을 합친 값이라 차량용 비중만 씁니다.
# 차량용 경유 80%는 팀 가정값입니다(출처 확인 전). 휘발유·등유는 출하량 전체를 씁니다.
COVERED_SHARE = {"gasoline": 1.0, "diesel": 0.8, "kerosene": 1.0}
# 아직 발표되지 않은 달 = 작년 같은 달 × (최근 YOY_MONTHS개월 출하량 / 1년 전 같은 개월 출하량)
# 2026년은 정책 기간 내내 작년보다 출하량이 적어서, 작년 값을 그대로 쓰면 부담이 커지기 때문입니다.
YOY_MONTHS = 6

# 산업통상자원부 추산(2026-10-04 보도, 조지연 의원실 제출 자료, 에너지경제연구원 시뮬레이션 기반)과 대조합니다.
# 보도는 "격차 100원당 1.5조 원"의 기간을 밝히지 않았습니다. 2·3분기(9월까지) 손실 추산 2.76조 원에
# 유가 150달러 대비분 1.5조 원(= 격차 100원분)을 더했다는 설명에 맞춰, 정책 시행일 ~ 9월 30일 기간으로 가정합니다.
# 산업부 보전은 정유사 원가 기준 정산이고 우리 격차는 국제 시장가 기준이라, 급등기에는 우리 값이 더 클 수 있습니다.
MINISTRY = {"per_100won": 1.5e12, "base_estimate": 2.76e12, "reserve": 4.2e12,
            "period_end": pd.Timestamp("2026-09-30")}




# 3. ① 상한을 읽어 날마다 펼칩니다. 각 날짜에는 그날 이전에 시작한 가장 최근 차수의 상한이 적용됩니다.
def load_caps(path=CAPS_PATH):
    caps = pd.read_csv(path, encoding="utf-8-sig", parse_dates=["start_date"]).sort_values("start_date")
    missing = {"round", "start_date", *FUELS} - set(caps.columns)
    if missing:
        raise ValueError(f"상한 파일에 열이 없습니다: {sorted(missing)}")
    if caps["start_date"].min() != POLICY_START:
        raise ValueError(f"1차 시행일({caps['start_date'].min().date()})이 정책 시행일({POLICY_START.date()})과 다릅니다.")
    return caps.reset_index(drop=True)


def daily_frame(caps, end):
    days = pd.date_range(POLICY_START, end, freq="D")
    position = caps["start_date"].searchsorted(days, side="right") - 1
    frame = pd.DataFrame({"round": caps["round"].to_numpy()[position]}, index=days)
    for fuel in FUELS:
        frame[f"{fuel}_cap"] = caps[fuel].to_numpy()[position]
    frame["week_end"] = days + pd.to_timedelta((5 - days.weekday) % 7, unit="D")   # 그 날이 든 주의 토요일
    frame.index.name = "date"
    return frame




# 4. ② 공급가가 아직 공개되지 않은 최근 주를 주별 표 뒤에 붙입니다. (결측 행 제거본에서는 빠진 주)
# 국제가는 통합 원본(merged_weekly_data.csv)에서 가져오고, 유류세는 마지막으로 계산된 값을 이어 씁니다.
def extend_provisional(weekly, path=MERGED_PATH):
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw.index = pd.DatetimeIndex([week_end(label) for label in raw["주차"]])
    later = raw[raw.index > weekly.index[-1]]
    end = later.index.max() if len(later) else weekly.index[-1]
    extended = weekly.reindex(pd.date_range(weekly.index[0], end, freq="W-SAT"))
    extended.loc[later.index, "주차"] = later["주차"]
    for fuel, spec in FUELS.items():
        if later[spec["intl"]].isna().any():
            raise ValueError(f"잠정 주의 국제가가 비어 있습니다: {spec['intl']}")
        extended.loc[later.index, f"{fuel}_intl_krw_l"] = later[spec["intl"]]
        extended.loc[later.index, f"{fuel}_tax"] = weekly[f"{fuel}_tax"].dropna().iloc[-1]
    extended["in_policy"] = extended.index >= POLICY_START
    extended["provisional"] = extended.index.isin(later.index)
    return extended




# 5. ② 정책 기간 주별 "없었다면" 세후 가격과 오차 경로입니다. 1단계와 같은 학습·검증을 다시 실행합니다.
# 오차 경로: 기점 이동 검증의 기점마다 "기점 뒤 h주째 오차"를 정책 기간 h주째에 그대로 적용한 세후 가격 (31개 × 정책 주 수)
# 정책 기간이 검증 길이(32주)보다 길어지면 마지막 주의 오차를 이어 씁니다. 오차는 몇 달씩 지속되므로 가장 가까운 근사입니다.
# 세 유종의 경로를 같은 번호끼리 더하므로, 유종마다 검증 기점 날짜가 같은지 확인합니다.
def counterfactual_paths(weekly, extended):
    policy = extended["in_policy"]
    weeks = extended.index[policy]
    table = pd.DataFrame({"주차": extended.loc[policy, "주차"], "provisional": extended.loc[policy, "provisional"]},
                         index=weeks)
    paths, origins = {}, {}
    for fuel in FUELS:
        X, y = training_data(weekly, fuel)
        params = fit(X, y)
        errors = backtest(X, y, fit)
        origins[fuel] = tuple(X.index[cutoffs(X)])
        if len(weeks) > HORIZON:
            errors = np.hstack([errors, np.repeat(errors[:, -1:], len(weeks) - HORIZON, axis=1)])
        pretax = predict(params, lagged_intl(extended, fuel)[policy])
        tax = extended.loc[policy, f"{fuel}_tax"].to_numpy()
        table[f"{fuel}_cf"] = pretax * (1 + VAT) + tax
        table[f"{fuel}_supply_aftertax"] = extended.loc[policy, f"{fuel}_supply_aftertax"]
        paths[fuel] = (pretax[None, :] - errors[:, :len(weeks)]) * (1 + VAT) + tax[None, :]
    if len(set(origins.values())) > 1:
        raise ValueError("유종마다 검증 기점 날짜가 달라 오차 경로를 합칠 수 없습니다.")
    return table, paths




# 6. 판매량을 날마다의 리터로 바꾸고, 상한이 걸리는 비중(COVERED_SHARE)을 곱합니다.
# 아직 발표되지 않은 달은 작년 같은 달 값(계절 반영)에 최근 출하량의 전년 대비 비율(올해 흐름 반영)을 곱해 채웁니다.
def yoy_ratio(column):
    recent = pd.period_range(end=column.index.max(), periods=YOY_MONTHS, freq="M")
    if not set(recent).union(recent - 12) <= set(column.index):
        raise ValueError(f"전년 대비 비율을 구할 {YOY_MONTHS}개월과 1년 전 자료가 모두 있어야 합니다.")
    return column[recent].sum() / column[recent - 12].sum()


def load_volume(days, path=VOLUME_PATH):
    if not path.exists():
        raise ValueError(f"판매량 파일이 없습니다: {path.name}. preprocessing/preprocess_fuel_demand.py를 먼저 실행하세요.")
    table = pd.read_csv(path, encoding="utf-8-sig", dtype={"월": str})
    table.index = pd.PeriodIndex(table["월"], freq="M")
    months = days.to_period("M")
    volume, filled = pd.DataFrame(index=days), []
    for fuel, spec in FUELS.items():
        column = table[spec["ko"]].dropna()
        ratio = yoy_ratio(column)
        per_month = {}
        for month in months.unique():
            if month in column.index:
                value = column[month]
            elif month - 12 in column.index:
                value = column[month - 12] * ratio
                filled.append(f"{month} {spec['ko']} = {month - 12} × {ratio:.3f}")
            else:
                raise ValueError(f"{month} {spec['ko']} 판매량이 없고 작년 같은 달 값도 없습니다.")
            per_month[month] = value * 1000 * LITERS_PER_BARREL / month.days_in_month * COVERED_SHARE[fuel]
        volume[fuel] = [per_month[month] for month in months]
    return volume, filled




# 7. ③④⑤ 날마다 유종별 격차와 부담을 구하고 세 유종을 합칩니다.
# 합계 열: 판매량, 부담, 1리터당 보전액(= 부담 ÷ 판매량, 판매량 가중 평균), 누적 부담과 각각의 90% 구간
# 유종별 열은 합계의 계산 근거로 남깁니다: 상한, 격차, 하루 판매량, 부담
# 오차 경로마다 같은 계산을 해서 세 유종을 더한 합계 경로(31개 × 날짜)를 만듭니다.
def daily_burden(daily, table, paths, volume):
    position = table.index.get_indexer(daily["week_end"])
    if (position < 0).any():
        raise ValueError("\"없었다면\" 가격이 없는 주가 있습니다.")
    detail = pd.DataFrame(index=daily.index)
    total_paths = 0
    for fuel in FUELS:
        cap = daily[f"{fuel}_cap"].to_numpy(dtype=float)
        liters = volume[fuel].to_numpy()
        detail[f"{fuel}_cap"] = cap
        detail[f"{fuel}_gap"] = table[f"{fuel}_cf"].to_numpy()[position] - cap
        detail[f"{fuel}_volume_l"] = liters
        detail[f"{fuel}_burden"] = detail[f"{fuel}_gap"].clip(lower=0) * liters
        total_paths = total_paths + np.clip(paths[fuel][:, position] - cap[None, :], 0, None) * liters[None, :]
    result = daily[["round", "week_end"]].copy()
    result["provisional"] = table["provisional"].to_numpy(dtype=bool)[position]
    result["volume_l"] = sum(detail[f"{fuel}_volume_l"] for fuel in FUELS)
    result["burden"] = sum(detail[f"{fuel}_burden"] for fuel in FUELS)
    result["burden_low"], result["burden_high"] = np.quantile(total_paths, [0.05, 0.95], axis=0)
    result["per_liter"] = result["burden"] / result["volume_l"]
    result["cumulative"] = result["burden"].cumsum()
    result["cumulative_low"], result["cumulative_high"] = np.quantile(total_paths.cumsum(axis=1), [0.05, 0.95], axis=0)
    return result.join(detail), total_paths




# 8. 주별·차수별로 묶습니다. 구간은 합계 경로를 같은 기간끼리 더한 뒤의 5%·95% 분위수입니다.
# 누적 부담은 그 주·차수의 마지막 날까지의 합계입니다.
def group_burden(result, total_paths, key):
    dates = pd.Series(result.index, index=result.index).groupby(key)
    sums = result[["volume_l", "burden"]].groupby(key).sum()
    band = pd.DataFrame(total_paths.T, index=result.index).groupby(key).sum()
    last = dates.max()
    return pd.DataFrame({
        "start": dates.min().dt.date, "end": last.dt.date, "days": dates.size(),
        "volume_l": sums["volume_l"], "per_liter": sums["burden"] / sums["volume_l"],
        "burden": sums["burden"], "burden_low": band.quantile(0.05, axis=1), "burden_high": band.quantile(0.95, axis=1),
        **{column: result.loc[last, column].to_numpy() for column in ("cumulative", "cumulative_low", "cumulative_high")},
    })


def summarize(result, table, total_paths):
    weekly = group_burden(result, total_paths, result["week_end"])
    weekly.insert(0, "주차", table["주차"])
    weekly.insert(1, "provisional", table["provisional"].astype(bool))
    rounds = group_burden(result, total_paths, result["round"])
    sums = total_paths.sum(axis=1)
    totals = pd.Series({
        "기간": f"{result.index[0].date()} ~ {result.index[-1].date()}", "일수": len(result),
        "합계 (원)": result["burden"].sum(), "합계 90% 하한 (원)": np.quantile(sums, 0.05),
        "합계 90% 상한 (원)": np.quantile(sums, 0.95), "하루 평균 (원)": result["burden"].mean(),
        "하루 판매량 (L)": result["volume_l"].mean(),
        "평균 1리터당 보전액 (원)": result["burden"].sum() / result["volume_l"].sum(),
    })
    return weekly, rounds, totals.map(lambda value: round(value, 2) if isinstance(value, float) else value)




# 9. 누적 부담이 기준 금액을 처음 넘는 날입니다. 90% 구간의 상한 곡선이 먼저, 하한 곡선이 나중에 넘습니다.
# 돌려주는 값: (추정치가 넘는 날, 가장 이른 날, 가장 늦은 날). 기간 안에 넘지 않으면 None입니다.
def crossing(result, amount):
    def first(column):
        hit = result.index[result[column] >= amount]
        return hit[0] if len(hit) else None
    return first("cumulative"), first("cumulative_high"), first("cumulative_low")


def crossing_text(dates):
    point, early, late = (date.strftime("%m/%d") if date is not None else "기간 안에 넘지 않음" for date in dates)
    return f"{point} (90% 구간 {early} ~ {late})"




# 10. 산업부 추산과 대조합니다. 기간은 MINISTRY 설명대로 정책 시행일 ~ 9월 30일로 가정합니다.
# 100원당 보전액 = 100원 × 그 기간 판매량이므로, 산업부 수치에서 산업부가 쓴 판매량 규모를 거꾸로 구할 수 있습니다.
# 우리 평균 격차를 산업부 단가(100원당 1.5조)로 환산한 값과 우리 합계의 차이는 판매량 규모의 차이입니다.
def compare_ministry(result, total_paths):
    window = result.index <= MINISTRY["period_end"]
    days = int(window.sum())
    liters = result.loc[window, "volume_l"].sum()
    burden = result.loc[window, "burden"].sum()
    paths = total_paths[:, window].sum(axis=1)
    gap = burden / liters
    return pd.Series({
        "기간 (가정)": f"{result.index[window][0].date()} ~ {result.index[window][-1].date()} ({days}일)",
        "산업부: 격차 100원당 보전액 (조 원)": MINISTRY["per_100won"] / JO,
        "산업부 수치가 뜻하는 하루 판매량 (백만 L)": MINISTRY["per_100won"] / 100 / days / 1e6,
        "우리: 하루 판매량 (백만 L)": liters / days / 1e6,
        "우리: 격차 100원당 보전액 (조 원)": 100 * liters / JO,
        "우리: 판매량 가중 평균 격차 (원/리터)": gap,
        "우리: 재정 부담 합계 (조 원)": burden / JO,
        "우리: 합계 90% 구간 하한 (조 원)": np.quantile(paths, 0.05) / JO,
        "우리: 합계 90% 구간 상한 (조 원)": np.quantile(paths, 0.95) / JO,
        "우리 평균 격차 × 산업부 단가 (조 원)": gap / 100 * MINISTRY["per_100won"] / JO,
        "산업부: 2·3분기 원가 기준 손실 추산 (조 원)": MINISTRY["base_estimate"] / JO,
        "산업부: 추산이 뜻하는 평균 격차 (원/리터)": MINISTRY["base_estimate"] / MINISTRY["per_100won"] * 100,
        "산업부: 목적예비비 (조 원)": MINISTRY["reserve"] / JO,
        "우리 누적 부담이 산업부 추산(2.76조)을 넘는 날": crossing_text(crossing(result, MINISTRY["base_estimate"])),
        "우리 누적 부담이 목적예비비(4.2조)를 넘는 날": crossing_text(crossing(result, MINISTRY["reserve"])),
    })




# 11. 그림: 세 유종을 합한 정부 재정 부담입니다. 위 = 하루 부담(억 원), 아래 = 누적 부담(조 원)과 산업부 수치
# 점선 = 상한 차수가 바뀐 날, 회색 = 공급가가 공개되지 않아 국제가로만 추정한 잠정 기간
def plot_fiscal(result, caps, path):
    apply_plot_style()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True, constrained_layout=True)
    provisional = result.index[result["provisional"]]
    for ax in axes:
        for start in caps["start_date"].iloc[1:]:
            ax.axvline(start, color=AXIS, linestyle=":", linewidth=0.8)
        if len(provisional):
            ax.axvspan(provisional[0], result.index[-1] + pd.Timedelta(days=1), color="#f0efec", linewidth=0,
                       zorder=0, label="잠정 (공급가 미공개)")
    ax = axes[0]
    ax.fill_between(result.index, result["burden_low"] / EOK, result["burden_high"] / EOK, step="post",
                    color=TOTAL_COLOR, alpha=0.15, linewidth=0, label="90% 구간")
    ax.step(result.index, result["burden"] / EOK, where="post", color=TOTAL_COLOR, linewidth=2, label="하루 부담")
    for _, row in caps.iterrows():
        ax.text(row["start_date"] + pd.Timedelta(days=1), 0.97, f"{row['round']}차", transform=ax.get_xaxis_transform(),
                va="top", color=MUTED, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title("하루 재정 부담 (세 유종 합계, 억 원/일)", loc="left")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=8)

    ax = axes[1]
    reserve, estimate, end = MINISTRY["reserve"] / JO, MINISTRY["base_estimate"] / JO, MINISTRY["period_end"]
    ax.fill_between(result.index, result["cumulative_low"] / JO, result["cumulative_high"] / JO, color=TOTAL_COLOR,
                    alpha=0.15, linewidth=0, label="90% 구간 (오차 경로)")
    ax.plot(result.index, result["cumulative"] / JO, color=TOTAL_COLOR, linewidth=2, label="누적 재정 부담")
    ax.axhline(reserve, color=INK_2, linestyle="--", linewidth=1)
    ax.text(result.index[0], reserve, f" 목적예비비 {reserve:.1f}조", va="bottom", color=INK_2, fontsize=8)
    ax.hlines(estimate, result.index[0], end, color=INK_2, linestyle=":", linewidth=1)
    ax.plot(end, estimate, marker="D", markersize=7, linestyle="none", color=INK_2,
            markeredgecolor="#fcfcfb", markeredgewidth=2, label="산업부 추산 (9월까지, 원가 기준)")
    ax.annotate(f"산업부 {estimate:.2f}조", (end, estimate), textcoords="offset points", xytext=(8, -4),
                va="top", color=INK_2, fontsize=8)
    ours = result.loc[end, "cumulative"] / JO
    ax.plot(end, ours, marker="o", markersize=8, linestyle="none", color=TOTAL_COLOR, markeredgecolor="#fcfcfb",
            markeredgewidth=2)
    ax.annotate(f"9/30 우리 {ours:.2f}조", (end, ours), textcoords="offset points", xytext=(-8, 8), ha="right",
                color=INK, fontsize=8)
    # 누적 부담이 산업부 추산(9월까지)과 예비비를 처음 넘는 날을 곡선 위에 표시합니다.
    for amount, label in ((estimate, "산업부 추산 도달"), (reserve, "예비비 초과")):
        passed = crossing(result, amount * JO)[0]
        if passed is None:
            continue
        ax.plot(passed, amount, marker="o", markersize=8, linestyle="none", color=TOTAL_COLOR,
                markeredgecolor="#fcfcfb", markeredgewidth=2)
        ax.annotate(f"{label} {passed.strftime('%m/%d')}", (passed, amount), textcoords="offset points",
                    xytext=(-8, 8), ha="right", color=INK, fontsize=8)
    ax.annotate(f"{result['cumulative'].iloc[-1] / JO:.2f}조", (result.index[-1], result["cumulative"].iloc[-1] / JO),
                textcoords="offset points", xytext=(-4, 8), ha="right", color=INK, fontsize=8)
    ax.set_ylim(bottom=0)
    ax.set_title("누적 재정 부담 (세 유종 합계, 조 원)", loc="left")
    ax.legend(loc="upper left", fontsize=8)
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m/%d"))
    fig.savefig(path, dpi=150)
    plt.close(fig)




# 12. 상한과 실제 세후 공급가를 대조해 상한 자료가 맞는지 점검합니다. 상한이 유종마다 달라 유종별로 봅니다.
# 상한이 걸린 주에는 실제 세후 공급가가 상한 바로 아래(수 원)에 있어야 합니다. 크게 벗어난 주를 이유와 함께 보여 줍니다.
def check_caps(result, table):
    lines = []
    for fuel, spec in FUELS.items():
        cap = result.groupby("week_end")[f"{fuel}_cap"].mean()
        diff = (table[f"{fuel}_supply_aftertax"] - cap).dropna()
        near = diff.between(-10, 1)
        lines.append(f"  {spec['ko']}: {near.sum()}/{len(diff)}주가 상한 −10 ~ +1원 안. 벗어난 주: "
                     + ", ".join(f"{d.strftime('%m/%d')}({v:+.0f})" for d, v in diff[~near].items()))
    return lines




# 13. ①~⑤를 실행하고 results/fiscal 폴더에 저장한 뒤 산업부 추산과 대조합니다.
def main():
    try:
        weekly_panel = load_weekly()
        caps = load_caps()
        extended = extend_provisional(weekly_panel)
        table, paths = counterfactual_paths(weekly_panel, extended)
        daily = daily_frame(caps, table.index[-1])
        volume, filled = load_volume(daily.index)
        result, total_paths = daily_burden(daily, table, paths, volume)
        weekly, rounds, totals = summarize(result, table, total_paths)
        ministry = compare_ministry(result, total_paths)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        result.to_csv(OUT_DIR / "fiscal_daily.csv", encoding="utf-8-sig", float_format="%.2f")
        weekly.to_csv(OUT_DIR / "fiscal_weekly.csv", encoding="utf-8-sig", float_format="%.2f")
        rounds.to_csv(OUT_DIR / "fiscal_rounds.csv", encoding="utf-8-sig", float_format="%.2f")
        totals.to_csv(OUT_DIR / "fiscal_totals.csv", header=["값"], encoding="utf-8-sig", float_format="%.2f")
        ministry.to_csv(OUT_DIR / "fiscal_ministry_check.csv", header=["값"], encoding="utf-8-sig")
        plot_fiscal(result, caps, OUT_DIR / "fiscal.png")
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"2단계 계산 실패: {error}") from None
    provisional = table.index[table["provisional"].to_numpy(dtype=bool)]
    print(f"정책 기간: {result.index[0].date()} ~ {result.index[-1].date()} ({len(result)}일, {len(table)}주, "
          f"잠정 {len(provisional)}주: {', '.join(table.loc[provisional, '주차'])})")
    print("\n[상한 자료 점검] 실제 세후 공급가 − 주 평균 상한")
    print("\n".join(check_caps(result, table)))
    print(f"\n[판매량] {VOLUME_PATH.name}, 상한 적용 비중: "
          + ", ".join(f"{FUELS[f]['ko']} {share:.0%}" for f, share in COVERED_SHARE.items()))
    if filled:
        print("  발표 전이라 추정한 달 (작년 같은 달 × 최근 전년 대비 비율): " + ", ".join(filled))
    print("\n[주별 재정 부담] 세 유종 합계(억 원), 괄호 = 90% 구간 / 1리터당 보전액 / 누적(조 원), * = 잠정")
    for week, row in weekly.iloc[-6:].iterrows():
        mark = "*" if row["provisional"] else " "
        print(f"  {row['주차']}{mark} ({row['days']}일): {row['burden'] / EOK:,.0f} ({row['burden_low'] / EOK:,.0f}~"
              f"{row['burden_high'] / EOK:,.0f}) / {row['per_liter']:.0f}원 / 누적 {row['cumulative'] / JO:.2f}조")
    print("\n[차수별 재정 부담] 세 유종 합계(억 원), 괄호 = 90% 구간 / 1리터당 보전액 / 누적(조 원)")
    for number, row in rounds.iterrows():
        print(f"  {number}차 {row['start']}~{row['end']} ({row['days']}일): {row['burden'] / EOK:,.0f} "
              f"({row['burden_low'] / EOK:,.0f}~{row['burden_high'] / EOK:,.0f}) / {row['per_liter']:.0f}원 / "
              f"누적 {row['cumulative'] / JO:.2f}조 ({row['cumulative_low'] / JO:.2f}~{row['cumulative_high'] / JO:.2f})")
    print(f"\n[전체] {totals['기간']} ({totals['일수']}일)")
    print(f"  재정 부담 합계: {totals['합계 (원)'] / EOK:,.0f}억 원 ({totals['합계 90% 하한 (원)'] / EOK:,.0f}~"
          f"{totals['합계 90% 상한 (원)'] / EOK:,.0f}) / 하루 평균 {totals['하루 평균 (원)'] / EOK:,.1f}억 원")
    print(f"  상한 적용 판매량 하루 {totals['하루 판매량 (L)'] / 1e6:,.1f}백만 L, "
          f"평균 1리터당 보전액 {totals['평균 1리터당 보전액 (원)']:.0f}원")
    print("\n[산업부 추산과 대조] 기간은 보도에 없어 정책 시행일 ~ 9/30으로 가정")
    for name, value in ministry.items():
        print(f"  {name}: {value:,.2f}" if isinstance(value, float) else f"  {name}: {value}")
    print(f"\n저장 완료: {OUT_DIR}")




# 14. 직접 실행할 때만 계산을 시작합니다.
if __name__ == "__main__":
    main()
