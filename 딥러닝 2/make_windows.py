"""최종 테스트를 보류하고 개발 기간 안의 expanding rolling-origin 계획을 만든다."""
import hashlib
import json

import numpy as np
import pandas as pd

import settings as cfg

SPLIT_DIR = cfg.OUT_DIR / "splits"
DATE_COLUMNS = ["origin", "context_start", "target_start", "target_end"]


def feature_columns(fuel):
    return ["supply_pretax", "intl_krw_l"]


def split_data(data):
    policy, validation, test = map(pd.Timestamp, [cfg.POLICY_START, cfg.VALIDATION_START, cfg.TEST_START])
    if not pd.Timestamp(cfg.DATA_START) < validation < test < policy:
        raise ValueError("날짜는 자료 시작 < 검증 시작 < 테스트 시작 < 정책 시작 순서여야 합니다.")
    return {"history": data.loc[data.date < validation].copy(),
            "validation": data.loc[(data.date >= validation) & (data.date < test)].copy(),
            "test": data.loc[(data.date >= test) & (data.date < policy)].copy(),
            "policy": data.loc[data.date >= policy].copy()}


def base_origins():
    rows = []
    periods = {"validation": (cfg.VALIDATION_START, pd.Timestamp(cfg.TEST_START)-pd.Timedelta(days=1)),
               "test": (cfg.TEST_START, pd.Timestamp(cfg.POLICY_START)-pd.Timedelta(days=1))}
    for split, (start, end) in periods.items():
        for first in pd.date_range(start, end, freq=cfg.TIME_FREQ)[::cfg.ORIGIN_STEP_WEEKS]:
            future = pd.date_range(first, periods=max(cfg.HORIZONS), freq=cfg.TIME_FREQ)
            if future[-1] <= pd.Timestamp(end):
                rows.append({"split": split, "origin": first-pd.tseries.frequencies.to_offset(cfg.TIME_FREQ),
                             "target_start": first, "target_end": future[-1]})
    return pd.DataFrame(rows)


def plan_windows(data, origins, years):
    if data.duplicated(["fuel", "date"]).any():
        raise ValueError("날짜·유종 중복이 있습니다.")
    groups = {f: g.set_index("date").sort_index() for f, g in data.groupby("fuel")}
    if set(groups) != set(cfg.FUELS):
        raise ValueError("세 유종 모두 필요합니다.")
    rows, status = [], []
    for year in years:
        for base in origins.to_dict("records"):
            origin = pd.Timestamp(base["origin"])
            start = origin+pd.Timedelta(days=1)-pd.DateOffset(years=year)
            days = pd.date_range(start, origin, freq=cfg.TIME_FREQ)
            future = pd.date_range(base["target_start"], base["target_end"], freq=cfg.TIME_FREQ)
            reasons = []
            for fuel, table in groups.items():
                if not len(days) or days[0] < table.index.min():
                    reasons.append(f"{fuel}: insufficient_history")
                    continue
                context = table.reindex(days)
                if not np.isfinite(context[feature_columns(fuel)].to_numpy(dtype=float)).all():
                    reasons.append(f"{fuel}: missing_context")
                for col in ("intl_usd_observed_date", "fx_observed_date"):
                    if col in context and (pd.to_datetime(context[col]) > origin).any():
                        reasons.append(f"{fuel}: future_imputation_source")
                if len(future) != max(cfg.HORIZONS) or not np.isfinite(table.reindex(future).supply_pretax.to_numpy(dtype=float)).all():
                    reasons.append(f"{fuel}: missing_target")
            record = {**base, "config_id": f"years_{year:02d}", "context_years": year,
                      "context_start": days[0] if len(days) else start, "context_weeks": len(days)}
            status.append({**record, "status": "skipped" if reasons else "eligible", "reason": "; ".join(reasons)})
            if not reasons:
                rows.extend({**record, "fuel": fuel, "horizon_weeks": len(future)} for fuel in cfg.FUELS)
    columns = ["split", "config_id", "context_years", "fuel", *DATE_COLUMNS, "context_weeks", "horizon_weeks"]
    return pd.DataFrame(rows, columns=columns), pd.DataFrame(status)


def build_windows(data):
    origins = base_origins()
    windows, status = plan_windows(data, origins, cfg.CONTEXT_YEARS)
    expected = int(origins.loc[origins.split == "validation", "origin"].nunique())
    eligible = status[(status.split == "validation") & (status.status == "eligible")].groupby("context_years").size()
    accepted = eligible[eligible == expected].index.tolist()
    if not accepted:
        raise ValueError("전체 공통 검증 기간을 지원하는 입력 길이가 없습니다.")
    status["included_candidate"] = status.context_years.isin(accepted)
    status.loc[(status.status == "eligible") & ~status.included_candidate, "reason"] = "incomplete_validation_coverage"
    windows = windows.loc[windows.context_years.isin(accepted)].copy()
    summary = []
    for year, group in status.groupby("context_years"):
        v = group[group.split == "validation"]
        summary.append({"context_years": year, "config_id": f"years_{year:02d}",
                        "included_candidate": year in accepted, "required_validation_origins": expected,
                        "eligible_validation_origins": int(v.status.eq("eligible").sum()),
                        "reason": "" if year in accepted else "insufficient_full_validation_history_or_missing_inputs"})
    return windows, status, pd.DataFrame(summary)


def get_window(data, window):
    series = data.loc[data.fuel == window["fuel"]].set_index("date").sort_index()
    dates = pd.date_range(window["context_start"], window["origin"], freq=cfg.TIME_FREQ)
    history = series.reindex(dates)
    for col in ("intl_usd_observed_date", "fx_observed_date"):
        if col in history and (pd.to_datetime(history[col]) > pd.Timestamp(window["origin"])).any():
            raise ValueError("보정에 예측 기준일 이후 자료가 사용되었습니다.")
    context = history[feature_columns(window["fuel"])].copy()
    future_dates = pd.date_range(window["target_start"], window["target_end"], freq=cfg.TIME_FREQ)
    known_future = pd.DataFrame({"weekday": future_dates.dayofweek, "month": future_dates.month}, index=future_dates)
    return context, known_future, series.reindex(future_dates).supply_pretax.copy()


def experiment_spec():
    source_names = ["settings.py", "prepare_weekly.py", "make_windows.py", "rolling_origin.py",
                    "evaluation.py", "ai_protocol.py", "predict_timesfm.py"]
    return {"schema_version": 3, "data_start": cfg.DATA_START, "validation_start": cfg.VALIDATION_START,
            "test_start": cfg.TEST_START, "policy_start": cfg.POLICY_START, "time_frequency": cfg.TIME_FREQ,
            "horizons_weeks": cfg.HORIZONS, "requested_context_years": cfg.CONTEXT_YEARS,
            "step_weeks": cfg.ORIGIN_STEP_WEEKS, "knn_neighbors": cfg.KNN_NEIGHBORS,
            "knn_history_rows": cfg.KNN_HISTORY_ROWS,
            "data_sha256": hashlib.sha256((cfg.OUT_DIR/"weekly_panel.csv").read_bytes()).hexdigest(),
            "code_sha256": hashlib.sha256(b"".join((cfg.HERE/name).read_bytes() for name in source_names)).hexdigest()}


def load_experiment():
    protocol = json.loads((SPLIT_DIR/"protocol.json").read_text(encoding="utf-8"))
    if protocol.get("spec") != experiment_spec():
        raise ValueError("자료 또는 코드·설정이 변경되었습니다. prepare_weekly.py와 make_windows.py를 다시 실행하세요.")
    data = pd.read_csv(cfg.OUT_DIR/"weekly_panel.csv", parse_dates=["date"])
    windows = pd.read_csv(SPLIT_DIR/"windows.csv", parse_dates=DATE_COLUMNS)
    if protocol.get("windows_sha256") != hashlib.sha256((SPLIT_DIR/"windows.csv").read_bytes()).hexdigest():
        raise ValueError("예측 구간 파일이 변경되었습니다. make_windows.py를 다시 실행하세요.")
    return data, windows, protocol


def result_dir(protocol):
    return cfg.RESULT_DIR/protocol["experiment_id"]


def main():
    data = pd.read_csv(cfg.OUT_DIR/"weekly_panel.csv", parse_dates=["date"])
    splits = split_data(data)
    windows, status, candidates = build_windows(data)
    if set(windows.split) != {"validation", "test"}:
        raise ValueError("검증·테스트 구간이 모두 필요합니다.")
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    summary = []
    for name, table in splits.items():
        table.to_csv(SPLIT_DIR/f"{name}.csv", index=False, encoding="utf-8-sig")
        summary.append({"split": name, "start": table.date.min(), "end": table.date.max(), "rows": len(table)})
    windows.to_csv(SPLIT_DIR/"windows.csv", index=False, encoding="utf-8-sig")
    from rolling_origin import make_folds
    make_folds(windows).to_csv(SPLIT_DIR/"cv_folds.csv", index=False, encoding="utf-8-sig")
    status.to_csv(SPLIT_DIR/"availability.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(SPLIT_DIR/"candidates.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(summary).to_csv(SPLIT_DIR/"split_summary.csv", index=False, encoding="utf-8-sig")
    spec = experiment_spec()
    protocol = {"spec": spec, "experiment_id": hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16],
                "windows_sha256": hashlib.sha256((SPLIT_DIR/"windows.csv").read_bytes()).hexdigest(),
                "accepted_context_years": candidates.loc[candidates.included_candidate,"context_years"].tolist(),
                "validation_origins": int(windows.loc[windows.split=="validation","origin"].nunique()),
                "test_origins": int(windows.loc[windows.split=="test","origin"].nunique()),
                "selection": "공통 기준일의 세 유종 동일 가중 4주 경로 MAE",
                "test_rule": "검증으로 고정한 설정만 사용; 학습형 모델은 테스트 직전에 최종 refit 후 고정",
                "timesfm": "사전학습 가중치 고정, 기준일마다 과거 문맥만 갱신",
                "imputation": "과거 실측 KNN; 첫날만 최초 관측 초기화, 입력의 출처는 기준일 이하여야 함",
                "information_assumption": "주간 실적 기준 사후 검증; 공급가의 다음 주 공개 지연과 사전학습 중복 미반영",
                "policy_rule": "정책 이후는 모델 선택과 최종 테스트에서 제외"}
    (SPLIT_DIR/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))
    print(f"검증 {protocol['validation_origins']}개 / 테스트 {protocol['test_origins']}개 기준일; 후보 {protocol['accepted_context_years']}년")


if __name__ == "__main__":
    main()
