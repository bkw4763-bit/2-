"""일간 공변량을 기존 KNN으로 보정한 뒤 주간 공급가격 실험용 표를 만든다.

실행: python 딥러닝/prepare_weekly.py
네트워크 호출 없이 읽는다. 공변량은 과거 실측 이웃의 거리 가중 KNN으로 보정한다.
"""
import re

import numpy as np
import pandas as pd

import settings as cfg


def read_supply():
    """공급가 주차를 토요일로 변환한다. 실제 타겟은 보간하지 않는다."""
    frames = []
    for path in sorted((cfg.ROOT / "input").glob("supply_weekly_pretax*.csv")):
        try:
            table = pd.read_csv(path, encoding="cp949")
        except UnicodeDecodeError:
            table = pd.read_csv(path, encoding="utf-8-sig")
        def week_end(value):
            year, month, nth = map(int, re.findall(r"\d+", str(value))[:3])
            first = pd.Timestamp(year=year + 2000 if year < 100 else year, month=month, day=1)
            return first + pd.Timedelta(days=(2-first.dayofweek) % 7 + 7*(nth-1) + 3)
        table.index = pd.DatetimeIndex(table.iloc[:, 0].map(week_end), name="date")
        frames.append(table.iloc[:, 1:].apply(pd.to_numeric, errors="coerce"))
    if not frames:
        raise FileNotFoundError("원자료 없음: supply_weekly_pretax*.csv")
    table = pd.concat(frames).sort_index()
    return table[~table.index.duplicated(keep="last")]


def parse_date(value):
    """오피넷의 두 자리·네 자리 연도 표기를 실제 날짜로 변환한다."""
    year, month, day = map(int, re.findall(r"\d+", str(value))[:3])
    return pd.Timestamp(year=year + 2000 if year < 100 else year, month=month, day=day)


def read_prices(pattern):
    """분할 CSV를 읽고 중복 날짜를 제거하며 가격을 숫자로 변환한다."""
    frames = []
    for path in sorted((cfg.ROOT / "input").glob(pattern)):
        try:
            frame = pd.read_csv(path, encoding="cp949")
        except UnicodeDecodeError:
            frame = pd.read_csv(path, encoding="utf-8-sig")
        frame.index = pd.DatetimeIndex(frame.iloc[:, 0].map(parse_date), name="date")
        frames.append(frame.iloc[:, 1:].apply(pd.to_numeric, errors="coerce"))
    if not frames:
        raise FileNotFoundError(f"원자료 없음: {pattern}")
    result = pd.concat(frames).sort_index()
    return result[~result.index.duplicated(keep="last")]


def knn_impute_covariates(raw, columns):
    """최근 과거 실측 최대 252행에서 표준화 거리의 5개 이웃을 선택한다.

    세 유종 판매가·국제가격·환율과 날짜를 거리 특성으로 사용한다.
    보정값을 다른 날짜의 이웃으로 재사용하지 않는다. 최초 과거 이웃이 없는
    시작 부분만 최초 완전 관측일로 초기화하고 미래 출처를 명시한다.
    """
    values = raw.to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError("원자료에 무한대가 있습니다.")
    complete = np.flatnonzero(np.isfinite(values).all(axis=1))
    if not len(complete):
        raise ValueError("KNN 보정에 사용할 완전 실측 행이 없습니다.")
    filled = raw.copy()
    details = []
    features = np.column_stack([values, (raw.index-raw.index[0]).days.to_numpy()])
    for i in range(len(raw)):
        missing = [col for col in columns if pd.isna(raw.iloc[i][col])]
        if not missing:
            continue
        candidates = complete[complete < i][-cfg.KNN_HISTORY_ROWS:]
        method = "knn"
        if not len(candidates):
            candidates = complete[:1]
            method = "initial_knn"
        available = np.isfinite(features[i])
        donor_features = features[candidates][:, available]
        scale = donor_features.std(axis=0)
        scale[scale < 1e-12] = 1.0
        distance = np.sqrt(np.mean(((donor_features-features[i, available])/scale)**2, axis=1))
        order = np.argsort(distance, kind="stable")[:cfg.KNN_NEIGHBORS]
        neighbors = candidates[order]
        distance = distance[order]
        exact = distance < 1e-12
        weights = exact.astype(float) if exact.any() else 1.0/distance
        weights /= weights.sum()
        source_dates = raw.index[neighbors]
        for col in missing:
            result = float(np.dot(weights, values[neighbors, raw.columns.get_loc(col)]))
            filled.loc[raw.index[i], col] = result
            details.append({"date": raw.index[i], "variable": col, "filled_value": result,
                            "source_date": source_dates.max(), "method": method,
                            "neighbors": len(neighbors),
                            "donor_dates": "|".join(day.strftime("%Y-%m-%d") for day in source_dates),
                            "donor_weights": "|".join(f"{weight:.10g}" for weight in weights)})
    return filled, pd.DataFrame(details, columns=["date", "variable", "filled_value", "source_date", "method",
                                                 "neighbors", "donor_dates", "donor_weights"])


def build_daily():
    """하루·유종당 한 행을 생성하고 정책 이전 학습 가능 여부를 표시한다."""
    retail = read_prices("retail_daily*.csv")
    intl = read_prices("intl_daily*.csv")
    fx_table = pd.read_csv(cfg.ROOT / "data" / "raw" / "ecos_fx_krw_usd.csv", parse_dates=["date"])
    fx = fx_table.drop_duplicates("date", keep="last").set_index("date")["value"]
    fx = pd.to_numeric(fx, errors="raise")
    days = pd.date_range(cfg.DATA_START, retail.index.max(), freq=cfg.DAILY_FREQ, name="date")
    wide = pd.DataFrame(index=days)
    for fuel, (retail_col, intl_col) in cfg.FUELS.items():
        wide[f"retail_{fuel}"] = retail[retail_col].reindex(days)
        wide[f"intl_{fuel}"] = intl[intl_col].reindex(days)
    wide["fx"] = fx.reindex(days)
    filled, audit = knn_impute_covariates(wide, [f"intl_{fuel}" for fuel in cfg.FUELS] + ["fx"])
    frames = []
    for fuel, (retail_col, intl_col) in cfg.FUELS.items():
        frame = pd.DataFrame(index=days)
        frame["retail"] = retail[retail_col].reindex(days)  # 정답은 보간하지 않는다.
        for variable, col in (("intl_usd", f"intl_{fuel}"), ("fx", "fx")):
            frame[variable] = filled[col]
            frame[variable + "_imputed"] = wide[col].isna()
            frame[variable + "_observed_date"] = days
            frame[variable + "_fill_method"] = "observed"
            frame[variable + "_neighbors"] = 0
            frame[variable + "_donor_dates"] = ""
            frame[variable + "_donor_weights"] = ""
            for row in audit.loc[audit.variable == col].to_dict("records"):
                frame.loc[row["date"], [variable + "_observed_date", variable + "_fill_method",
                    variable + "_neighbors", variable + "_donor_dates", variable + "_donor_weights"]] = [
                    row["source_date"], row["method"], row["neighbors"], row["donor_dates"], row["donor_weights"]]
            frame[variable + "_age_days"] = (pd.Series(days, index=days)-frame[variable + "_observed_date"]).dt.days
        frame["intl_krw_l"] = frame["intl_usd"] * frame["fx"] / 158.987
        frame["fuel"] = fuel
        frame["time_idx"] = (days - days.min()).days
        frame["weekday"] = days.dayofweek
        frame["month"] = days.month
        frame["in_policy"] = days >= pd.Timestamp(cfg.POLICY_START)
        required = ["retail", "intl_usd", "fx"]
        frame["usable"] = frame[required].notna().all(axis=1)
        frame["available_on_date"] = ((frame.intl_usd_observed_date <= frame.index) &
                                      (frame.fx_observed_date <= frame.index))
        frame["train_eligible"] = frame["usable"] & frame["available_on_date"] & ~frame["in_policy"]
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["fuel", "date"])


def build_weekly():
    """일간 KNN 보정 후 일~토 주간 평균을 계산해 실제 공급가와 결합한다."""
    daily = build_daily()
    supply = read_supply()
    first = pd.Timestamp(cfg.DATA_START)
    if first.dayofweek != 6:
        first += pd.Timedelta(days=(6-first.dayofweek) % 7)
    # 첫 불완전 주는 제외하고, 공변량과 타겟을 모두 확보한 마지막 완전 주까지만 쓴다.
    last = min(supply.index.max(), daily.date.max())
    dates = pd.date_range(first + pd.Timedelta(days=6), last, freq=cfg.TIME_FREQ, name="date")
    frames = []
    for fuel, (supply_col, _) in cfg.FUELS.items():
        one = daily.loc[daily.fuel == fuel].set_index("date").sort_index()
        frame = one[["intl_usd", "fx", "intl_krw_l"]].resample(cfg.TIME_FREQ).mean().reindex(dates)
        frame["supply_pretax"] = supply[supply_col].reindex(dates)
        frame["week_start"] = dates - pd.Timedelta(days=6)
        frame["supply_available_date"] = dates + pd.Timedelta(days=6)
        frame["daily_observations"] = one.intl_krw_l.resample(cfg.TIME_FREQ).count().reindex(dates)
        for variable in ("intl_usd", "fx"):
            frame[variable + "_imputed"] = one[variable + "_imputed"].resample(cfg.TIME_FREQ).sum().reindex(dates)
            frame[variable + "_observed_date"] = one[variable + "_observed_date"].resample(cfg.TIME_FREQ).max().reindex(dates)
        frame["fuel"] = fuel
        frame["time_idx"] = np.arange(len(dates))
        # 시행일을 포함한 3/8~14 혼합 주도 정책 구간으로 분류한다.
        frame["in_policy"] = dates >= pd.Timestamp(cfg.POLICY_START)
        frame["usable"] = frame[["supply_pretax", "intl_krw_l"]].notna().all(axis=1) & frame.daily_observations.eq(5)
        frame["available_on_date"] = ((frame.intl_usd_observed_date <= frame.index) &
                                      (frame.fx_observed_date <= frame.index))
        frame["train_eligible"] = frame.usable & frame.available_on_date & ~frame.in_policy
        frames.append(frame.reset_index())
    return pd.concat(frames, ignore_index=True).sort_values(["fuel", "date"]), daily


def main():
    """주간 실험 자료와 기존 일간 KNN 보정 근거를 새로 저장한다."""
    data, daily = build_weekly()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(cfg.OUT_DIR / "weekly_panel.csv", index=False, encoding="utf-8-sig")
    data.loc[~data.in_policy].to_csv(cfg.OUT_DIR / "pre_policy.csv", index=False, encoding="utf-8-sig")
    summary = data.groupby("fuel").agg(rows=("date", "size"), start=("date", "min"),
                                      end=("date", "max"), usable=("usable", "sum"),
                                      train_eligible=("train_eligible", "sum"))
    summary.to_csv(cfg.OUT_DIR / "quality_report.csv", encoding="utf-8-sig")
    reports = []
    for variable in ("intl_usd", "fx"):
        marked = daily.loc[daily[variable + "_imputed"],
                           ["fuel", "date", variable, variable + "_observed_date",
                            variable + "_age_days", variable + "_fill_method"]].copy()
        marked.columns = ["fuel", "date", "filled_value", "source_date", "age_days", "method"]
        marked["variable"] = variable
        for field in ("neighbors", "donor_dates", "donor_weights"):
            marked[field] = daily.loc[marked.index, variable + "_" + field]
        reports.append(marked)
    pd.concat(reports, ignore_index=True).to_csv(cfg.OUT_DIR / "imputation_report.csv", index=False, encoding="utf-8-sig")
    counts = []
    for fuel, table in data.groupby("fuel"):
        for variable in ("supply_pretax", "intl_usd", "fx", "intl_krw_l"):
            counts.append({"fuel": fuel, "variable": variable,
                           "missing_daily_before_fill": int(table[variable + "_imputed"].sum()) if variable in ("intl_usd", "fx") else None,
                           "missing_weekly_after_fill": int(table[variable].isna().sum()), "weeks": len(table)})
    pd.DataFrame(counts).to_csv(cfg.OUT_DIR / "missing_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string())


if __name__ == "__main__":
    main()


