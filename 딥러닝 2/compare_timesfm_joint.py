"""세 유종 공동 예측과 개별 예측·회귀를 동일한 80:20 CV로 비교한다.

실행: .venv-timesfm/Scripts/python.exe -X utf8 딥러닝/compare_timesfm_joint.py
2015년부터의 과거를 최대 15년 사용하며 입력 길이를 선택하지 않는다.
"""
import hashlib
import json

import numpy as np
import pandas as pd

import run_regression_timesfm as base

FUELS = tuple(base.cfg.FUELS)
OUT = base.OUT.parent / "compare_joint_regression_data"


def joint_inputs(weekly, cutoff, horizon, context_weeks):
    """같은 날짜의 타겟 3개·국제가격 3개를 정렬하고 미래 타겟을 제외한다."""
    aligned = [base.inputs(weekly, fuel, cutoff, horizon, context_weeks) for fuel in FUELS]
    dates = aligned[0][2]
    lengths = {len(item[0]) for item in aligned}
    if len(lengths) != 1 or any(not item[2].equals(dates) for item in aligned):
        raise ValueError("공동 예측의 유종별 입력 날짜가 다릅니다.")
    return np.stack([item[0] for item in aligned]), np.concatenate([item[1] for item in aligned]), dates


def joint_forecast(model, weekly, cutoff, horizon, context_weeks):
    targets, intl, dates = joint_inputs(weekly, cutoff, horizon, context_weeks)
    output = next(iter(model.predict_batch(contexts=[targets], horizon=horizon,
        past_future_covariates=[intl], return_quantiles=True, use_symmetric_averaging=False)))
    values = np.asarray(output.forecast, dtype=float)
    quantiles = np.asarray(output.quantiles, dtype=float)
    if (values.shape != (len(FUELS), horizon)
            or quantiles.shape != (len(FUELS), horizon, len(model.config.quantiles))
            or not np.isfinite(values).all() or not np.isfinite(quantiles).all()):
        raise ValueError("TimesFM 공동 예측 출력 형태 또는 값이 잘못되었습니다.")
    return dates, values, quantiles


def evaluate_joint(model, weekly, cutoffs, split):
    rows = []
    for cutoff in cutoffs:
        weeks = min(780, base.available_context_weeks(weekly, cutoff))
        dates, predicted, _ = joint_forecast(model, weekly, cutoff, base.regression.HORIZON, weeks)
        for i, fuel in enumerate(FUELS):
            actual = weekly.loc[dates, f"{fuel}_supply_pretax"].to_numpy()
            rows.extend({"split": split, "model": "timesfm3_joint", "context": "780", "fuel": fuel,
                "cutoff": cutoff, "date": date, "lead": lead, "actual": a, "prediction": p,
                "actual_context_weeks": weeks}
                for lead, (date, a, p) in enumerate(zip(dates, actual, predicted[i]), 1))
    return pd.DataFrame(rows)


def save_joint_policy(model, weekly, context_weeks, out_dir):
    # 세 유종을 하나의 표본으로 한 번만 추론한 뒤 기존 정책 출력 형식을 재사용한다.
    cached = None
    def select_fuel(model, expanded, fuel, cutoff, horizon, weeks):
        nonlocal cached
        if cached is None:
            cached = joint_forecast(model, expanded, cutoff, horizon, weeks)
        dates, values, quantiles = cached
        i = FUELS.index(fuel)
        return dates, values[i], quantiles[i]
    base.save_policy(model, weekly, context_weeks, out_dir, forecast_fn=select_fuel)


def main():
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    torch.set_num_threads(4)
    weekly = base.regression.load_weekly()
    validation, testing, _, _, split_spec = base.chronological_cv(weekly, (None,))
    OUT.mkdir(parents=True, exist_ok=True)
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(base.ROOT / "모델" / "timesfm-3.0-pytorch"),
        per_core_batch_size=1, device="cpu", local_files_only=True))
    tables = []
    with torch.inference_mode():
        for split, cutoffs in (("validation", validation), ("test", testing)):
            separate = base.evaluate(model, weekly, cutoffs, (780,), split, allow_short_history=True)
            joint = evaluate_joint(model, weekly, cutoffs, split)
            combined = pd.concat([separate, joint], ignore_index=True)
            combined.to_csv(OUT / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
            tables.append(base.metrics(combined))
            print(f"{split}: 개별·공동·회귀 비교 완료", flush=True)
        cutoff = weekly.index[weekly.in_policy][0]
        weeks = min(780, base.available_context_weeks(weekly, cutoff))
        save_joint_policy(model, weekly, weeks, OUT)
    scores = pd.concat(tables, ignore_index=True)
    scores.to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    paired = scores[scores.model.eq("timesfm3_joint")].merge(
        scores[scores.model.eq("timesfm3")], on=["split", "fuel", "context"], suffixes=("_joint", "_separate"))
    paired["mae_change_joint_minus_separate"] = paired.mae_joint - paired.mae_separate
    paired["rmse_change_joint_minus_separate"] = paired.rmse_joint - paired.rmse_separate
    paired.to_csv(OUT / "joint_vs_separate.csv", index=False, encoding="utf-8-sig")
    spec = {**split_spec, "source_sha256": hashlib.sha256((base.cfg.OUT_DIR / "weekly_panel.csv").read_bytes()).hexdigest(),
        "target": "supply_pretax", "fuels_in_order": FUELS, "covariates": "all three intl_krw_l series",
        "fixed_context_years": 15, "context_selection": "none; user-fixed",
        "short_history_rule": "use all available up to 780 weeks", "policy_actual_context_weeks": weeks,
        "horizon_weeks": base.regression.HORIZON, "policy_end": str(base.POLICY_END.date()),
        "future_international_prices": "observed ex-post; unknown policy future held constant",
        "post_policy_actual_supply_as_input": False, "weight_updates": False,
        "comparison_scope": "joint adds other fuels' past targets and international covariates; architecture and information both differ"}
    (OUT / "locked_settings.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(scores[scores.split.eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
