"""모든 예측 모형에 공통으로 적용할 원/리터 단위 평가 지표."""
import numpy as np
import pandas as pd

import settings as cfg


def score_predictions(predictions):
    """1·2·4주 지점 및 누적 구간의 MAE·RMSE를 유종별로 계산한다."""
    keys = ["model", "split", "fuel", "config_id", "context_years"]
    if predictions.duplicated(keys + ["origin", "lead"]).any():
        raise ValueError("예측 기준일·예측일 중복이 있습니다.")
    if not np.isfinite(predictions[["actual", "prediction"]].to_numpy(dtype=float)).all():
        raise ValueError("평가값에 결측 또는 무한대가 있습니다.")
    for _, group in predictions.groupby(keys + ["origin"]):
        if set(group.lead) != set(range(1, max(cfg.HORIZONS)+1)):
            raise ValueError("각 기준일의 1~4주 예측이 모두 있어야 합니다.")
    rows = []
    for key, group in predictions.groupby(keys):
        for horizon in cfg.HORIZONS:
            for measure, selected in (("endpoint", group.loc[group.lead == horizon]),
                                      ("path", group.loc[group.lead.between(1, horizon)])):
                if selected.empty:
                    continue
                error = selected.prediction - selected.actual
                rows.append(dict(zip(keys, key), horizon=horizon, measure=measure,
                                 min_context_weeks=int(group.context_weeks.min()), max_context_weeks=int(group.context_weeks.max()),
                                 mae=float(error.abs().mean()), rmse=float(np.sqrt((error ** 2).mean())),
                                 observations=len(selected), origins=selected.origin.nunique()))
    return pd.DataFrame(rows)


def rank_candidates(predictions):
    metrics = score_predictions(predictions)
    selected = metrics[(metrics.horizon == max(cfg.HORIZONS)) & (metrics.measure == "path")]
    ranked = selected.groupby(["model", "config_id", "context_years"], as_index=False).agg(
        mean_fuel_mae=("mae", "mean"), mean_fuel_rmse=("rmse", "mean"),
        fuels=("fuel", "nunique"), origins=("origins", "min"))
    ranked = ranked.sort_values(["mean_fuel_mae", "context_years", "config_id"])
    ranked.insert(0, "rank", range(1, len(ranked)+1))
    return ranked


def score_folds(predictions):
    """각 기준일의 지표를 별도 저장해 평균 성능 뒤의 시점별 변동을 확인한다."""
    results = []
    for (split, fold_id, origin), group in predictions.groupby(["split", "fold_id", "origin"]):
        scores = score_predictions(group)
        scores["fold_id"] = fold_id
        scores["origin"] = origin
        results.append(scores)
    return pd.concat(results, ignore_index=True)
