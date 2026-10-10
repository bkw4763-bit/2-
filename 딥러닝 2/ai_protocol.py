"""XGBoost·TimesFM-3의 폴드 내부 분할과 모델 선택·설정 고정 절차.

실행: python 딥러닝/ai_protocol.py
현재는 분석 설계와 데이터 분할을 생성하며 모델 학습·추론은 실행하지 않는다.
"""
import json
from pathlib import Path

import pandas as pd

import settings as cfg
from make_windows import feature_columns, load_experiment, result_dir
from evaluation import rank_candidates
from rolling_origin import get_training_samples, make_folds


def split_xgb_training(data, fuel, train_cutoff, horizon):
    """과거 학습 표본을 학습·조기 종료·재학습으로 나누고 정답 중첩을 제거한다."""
    samples = get_training_samples(data, fuel, train_cutoff, horizon)
    stop_start = pd.date_range(end=pd.Timestamp(train_cutoff), periods=cfg.XGB_EARLY_STOP_WEEKS, freq=cfg.TIME_FREQ)[0]
    # 조기 종료 구간은 정답 날짜로 정의한다. 이 구간 입력은 그보다 horizon일 이전이다.
    stopping = samples.loc[samples.target_date >= stop_start].copy()
    if stopping.empty:
        raise ValueError("조기 종료 표본이 없습니다.")
    # 학습 정답은 조기 종료 예측의 첫 기준일에 관측 가능해야 한다.
    fitting = samples.loc[samples.target_date <= stopping.index.min()].copy()
    if fitting.empty:
        raise ValueError("학습 표본이 없습니다.")
    if not fitting.target_date.max() < stopping.target_date.min():
        raise ValueError("학습·조기 종료 정답 구간이 중첩됩니다.")
    return fitting, stopping, samples


def refit_xgb(model_factory, params, fitting, stopping, all_samples, fuel):
    """내부 구간에서 트리 수를 고른 뒤 외부 정답 없이 과거 전체로 재학습한다.

    model_factory는 XGBRegressor와 같은 생성자다. 모델 패키지는 실행 시 제공한다.
    외부 CV 또는 최종 테스트 정답을 eval_set에 전달하지 않는다.
    """
    columns = feature_columns(fuel)
    estimator = model_factory(**params, n_estimators=cfg.XGB_MAX_TREES,
                              early_stopping_rounds=cfg.XGB_EARLY_STOP_ROUNDS,
                              objective="reg:squarederror", eval_metric="mae", random_state=42)
    estimator.fit(fitting[columns], fitting.target,
                  eval_set=[(stopping[columns], stopping.target)], verbose=False)
    trees = int(estimator.best_iteration) + 1
    # 트리 수 확정 후 조기 종료 구간도 포함한다. 외부 미래 정답은 포함하지 않는다.
    final = model_factory(**params, n_estimators=trees, objective="reg:squarederror",
                          eval_metric="mae", random_state=42)
    final.fit(all_samples[columns], all_samples.target)
    return final, trees


def lock_settings(predictions, destination, protocol, windows):
    """실제 폴드·정답·후보 일치를 검증한 뒤 TimesFM 설정을 한 번만 고정한다."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError("고정 파일을 덮어쓸 수 없습니다.")
    if predictions.empty or not predictions.split.eq("validation").all() or not predictions.model.eq("timesfm3").all():
        raise ValueError("TimesFM 전체 검증 예측만 설정 선택에 사용할 수 있습니다.")
    expected = windows[windows.split == "validation"]
    keys = ["config_id", "fuel", "origin"]
    actual = predictions[keys].drop_duplicates().copy()
    actual["origin"] = pd.to_datetime(actual.origin)
    if set(map(tuple, actual.to_numpy())) != set(map(tuple, expected[keys].to_numpy())):
        raise ValueError("모든 후보·유종·공통 검증 기준일을 평가해야 합니다.")
    # 평균 폴드 수만으로는 서로 다른 날짜를 평가한 결과를 검출할 수 없다.
    for _, group in predictions.groupby(keys):
        group = group.sort_values("lead")
        origin = pd.Timestamp(group.origin.iloc[0])
        expected_dates = pd.date_range(origin+pd.tseries.frequencies.to_offset(cfg.TIME_FREQ), periods=max(cfg.HORIZONS), freq=cfg.TIME_FREQ)
        if not pd.DatetimeIndex(pd.to_datetime(group.date)).equals(expected_dates):
            raise ValueError("예측 날짜가 공통 검증 계획과 다릅니다.")
    ranking = rank_candidates(predictions)
    choice = ranking.iloc[0]
    record = {"schema_version": 3, "experiment_id": protocol["experiment_id"], "model": "timesfm3",
              "config_id": choice.config_id, "context_years": int(choice.context_years),
              "validation_mae": float(choice.mean_fuel_mae), "validation_origins": int(choice.origins),
              "test_start": cfg.TEST_START, "policy_start": cfg.POLICY_START,
              "mode": "zero_shot_no_weight_updates", "selection_metric": "equal_fuel_mean_4week_path_mae"}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)
    return record


def load_locked_settings(path, protocol):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("experiment_id") != protocol["experiment_id"] or record.get("model") != "timesfm3":
        raise ValueError("현재 데이터·설계로 확정한 TimesFM 설정이 아닙니다.")
    if record.get("context_years") not in protocol["accepted_context_years"]:
        raise ValueError("확정 입력 연수가 현재 후보에 없습니다.")
    return record


def main():
    """모델별 CV와 최종 재학습 계획을 저장한다. 최종 테스트 성능은 계산하지 않는다."""
    data, windows, protocol = load_experiment()
    folds = make_folds(windows)
    rows = []
    # 각 예측 주를 직접 예측하는 1~4주 모형의 분할을 모두 기록한다.
    for fold in folds.loc[folds.split == "validation"].to_dict("records"):
        for fuel in cfg.FUELS:
            for horizon in range(1, max(cfg.HORIZONS) + 1):
                fitting, stopping, refit = split_xgb_training(data, fuel, fold["train_cutoff"], horizon)
                rows.append({"stage": "validation_cv", "fold_id": fold["fold_id"], "origin": fold["origin"],
                             "fuel": fuel, "horizon": horizon, "train_cutoff": fold["train_cutoff"],
                             "fit_rows": len(fitting), "fit_target_end": fitting.target_date.max(),
                             "stop_input_start": stopping.index.min(), "stop_target_start": stopping.target_date.min(),
                             "stop_target_end": stopping.target_date.max(), "stop_rows": len(stopping),
                             "refit_rows": len(refit)})
    cutoff = pd.Timestamp(cfg.TEST_START) - pd.Timedelta(days=1)
    for fuel in cfg.FUELS:
        for horizon in range(1, max(cfg.HORIZONS) + 1):
            fitting, stopping, refit = split_xgb_training(data, fuel, cutoff, horizon)
            rows.append({"stage": "final_refit", "fold_id": 0, "origin": cutoff, "fuel": fuel,
                         "horizon": horizon, "train_cutoff": cutoff, "fit_rows": len(fitting),
                         "fit_target_end": fitting.target_date.max(), "stop_input_start": stopping.index.min(),
                         "stop_target_start": stopping.target_date.min(), "stop_target_end": stopping.target_date.max(),
                         "stop_rows": len(stopping), "refit_rows": len(refit)})
    out = result_dir(protocol) / "plans"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "xgb_training_plan.csv", index=False, encoding="utf-8-sig")
    fm = windows.copy()
    fm["mode"] = "zero_shot"
    fm["weight_updates"] = False
    fm["settings_lock_required"] = fm.split.eq("test")
    fm.to_csv(out / "timesfm_evaluation_plan.csv", index=False, encoding="utf-8-sig")
    print("XGBoost 학습·조기 종료·재학습 계획과 TimesFM zero-shot 평가 계획 저장 완료")


if __name__ == "__main__":
    main()
