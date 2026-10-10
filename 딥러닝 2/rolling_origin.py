"""공통 rolling-origin 평가와 기준일별 학습 정답 누출 방지 도구."""
import numpy as np
import pandas as pd

import settings as cfg
from make_windows import feature_columns, get_window


def make_folds(windows):
    """기준일별 폴드와 학습 종료일을 명시한다. 최종 테스트 학습은 고정한다."""
    folds = windows[["split", "origin", "target_start", "target_end"]].drop_duplicates()
    folds = folds.sort_values(["split", "origin"]).copy()
    folds["fold_id"] = folds.groupby("split").cumcount() + 1
    folds["train_cutoff"] = folds["origin"]
    test = folds["split"] == "test"
    # 테스트 정답으로 가중치나 설정을 갱신하지 않는다. 문맥은 기준일까지 갱신 가능하다.
    folds.loc[test, "train_cutoff"] = pd.tseries.frequencies.to_offset(cfg.TIME_FREQ).rollback(
        pd.Timestamp(cfg.TEST_START) - pd.Timedelta(days=1))
    if not (folds.train_cutoff <= folds.origin).all():
        raise ValueError("학습 종료일이 예측 기준일보다 늦습니다.")
    return folds.reset_index(drop=True)


def get_training_samples(data, fuel, train_cutoff, horizon):
    """비교 회귀·XGBoost용 입력과 미래 정답을 정답 관측 가능 범위 안에서 만든다."""
    if horizon < 1:
        raise ValueError("예측 길이는 1주 이상이어야 합니다.")
    cutoff = pd.Timestamp(train_cutoff)
    if cutoff >= pd.Timestamp(cfg.POLICY_START):
        raise ValueError("정책 이후 자료는 학습에 사용할 수 없습니다.")
    series = data.loc[(data.fuel == fuel) & (data.date <= cutoff)].set_index("date").sort_index()
    if series.index.duplicated().any():
        raise ValueError("학습 자료 날짜가 중복됩니다.")
    features = series[feature_columns(fuel)]
    target_dates = features.index + pd.tseries.frequencies.to_offset(cfg.TIME_FREQ) * horizon
    targets = series.supply_pretax.reindex(target_dates).to_numpy()
    valid = (target_dates <= cutoff) & features.notna().all(axis=1).to_numpy() & np.isfinite(targets)
    for column in ("intl_usd_observed_date", "fx_observed_date"):
        if column in series:
            # 첫날 초기화 출처가 다음 날이면 해당 행을 직접 학습 입력으로 사용하지 않는다.
            valid &= (pd.to_datetime(series[column]).to_numpy() <= features.index.to_numpy())
    samples = features.loc[valid].copy()
    samples["target_date"] = target_dates[valid]
    samples["target"] = targets[valid]
    return samples


def run_rolling_origin(data, windows, predictor, model, split="validation"):
    """과거·미래 달력만 예측기에 전달한 후 숨겨 둔 정답을 붙여 평가 표를 만든다.

    predictor(context, known_future, metadata)는 4주 예측 배열을 반환한다.
    TimesFM은 폴드별 가중치를 갱신하지 않는다. 학습형 모형은 train_cutoff로
    get_training_samples를 호출하고 전처리·조기 종료에도 평가 정답을 사용하지 않는다.
    """
    selected = windows.loc[windows.split == split]
    if selected.empty:
        raise ValueError("평가할 폴드가 없습니다.")
    folds = make_folds(selected).set_index("origin")
    rows = []
    for window in selected.to_dict("records"):
        context, known_future, truth = get_window(data, window)
        fold = folds.loc[window["origin"]]
        if len(context) != window["context_weeks"] or context.isna().any().any():
            raise ValueError("과거 입력 구간의 길이 또는 결측이 잘못됐습니다.")
        if truth.isna().any() or context.index.max() >= truth.index.min():
            raise ValueError("정답 결측 또는 과거·미래 구간 중첩이 있습니다.")
        metadata = {**window, "fold_id": int(fold.fold_id), "train_cutoff": fold.train_cutoff}
        predicted = np.asarray(predictor(context.copy(), known_future.copy(), metadata), dtype=float)
        if predicted.shape != (len(truth),) or not np.isfinite(predicted).all():
            raise ValueError("예측 결과는 유한한 4주 1차원 배열이어야 합니다.")
        for lead, (day, actual, prediction) in enumerate(zip(truth.index, truth, predicted), start=1):
            rows.append({"model": model, "split": split, "fold_id": metadata["fold_id"],
                         "config_id": window["config_id"], "context_years": window["context_years"],
                         "fuel": window["fuel"], "origin": window["origin"],
                         "train_cutoff": metadata["train_cutoff"], "context_weeks": window["context_weeks"],
                         "date": day, "lead": lead, "actual": actual, "prediction": prediction})
    return pd.DataFrame(rows)
