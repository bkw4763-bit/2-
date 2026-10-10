"""회귀와 동일한 주간 자료·실제 국제가격 경로로 TimesFM을 비교한다.

실행: .venv-timesfm/Scripts/python.exe 딥러닝/run_regression_timesfm.py
미래 국제가격을 안다고 가정하는 사후 조건부 추정이며 사전 예측과 구분한다.
"""
import hashlib
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config as cfg
import counterfactual as regression

OUT = ROOT / "딥러닝" / "output" / "expanding_window"
POLICY_END = regression.POLICY_END


def available_context_weeks(weekly, cutoff):
    """세 유종 모두 공급가·국제가를 확보한 과거 길이의 최솟값을 반환한다."""
    return min(len(weekly.loc[weekly.index < cutoff,
        [f"{fuel}_supply_pretax", f"{fuel}_intl_krw_l"]].dropna()) for fuel in cfg.FUELS)


def policy_input(weekly):
    """연말을 포함하는 마지막 주까지 확장하고 미관측 국제가격·세금·상한을 유지한다."""
    final_week = pd.offsets.Week(weekday=5).rollforward(POLICY_END)
    dates = pd.date_range(weekly.index.min(), final_week, freq="W-SAT")
    expanded = weekly.reindex(dates).copy()
    for fuel in cfg.FUELS:
        for field in ("intl_krw_l", "tax", "cap"):
            column = f"{fuel}_{field}"
            expanded[column] = expanded[column].ffill()
    expanded["in_policy"] = expanded.index >= pd.Timestamp(cfg.POLICY_START)
    return expanded


def save_policy(model, weekly, context_weeks, out_dir, forecast_fn=None):
    """시행일부터 연말까지의 주간 미시행 가격을 출력하고 가정 구간을 표시한다."""
    forecast_fn = forecast if forecast_fn is None else forecast_fn
    expanded = policy_input(weekly)
    cutoff = expanded.index[expanded.in_policy][0]
    horizon = len(expanded.loc[expanded.index >= cutoff])
    rows = []
    levels = list(model.config.quantiles)
    for fuel in cfg.FUELS:
        dates, predicted, quantiles = forecast_fn(model, expanded, fuel, cutoff, horizon, context_weeks)
        for i, date in enumerate(dates):
            tax = expanded.loc[date, f"{fuel}_tax"]
            aftertax = predicted[i] * (1+cfg.VAT) + tax
            rows.append({"fuel": fuel, "date": min(date, POLICY_END),
                "week_start": max(date-pd.Timedelta(days=6), pd.Timestamp(cfg.POLICY_START)),
                "week_end": min(date, POLICY_END), "full_week_end": date,
                "international_price_assumed": date > weekly.index.max(),
                "tax_and_cap_assumed": date > weekly.index.max(),
                "cf_pretax": predicted[i], "cf_aftertax": aftertax,
                "actual_pretax": expanded.loc[date, f"{fuel}_supply_pretax"],
                "cap": expanded.loc[date, f"{fuel}_cap"],
                "gap_to_cap": aftertax-expanded.loc[date, f"{fuel}_cap"],
                **{f"q{round(q*100):02d}_pretax": quantiles[i, j] for j, q in enumerate(levels)}})
    pd.DataFrame(rows).to_csv(out_dir / "policy_counterfactual.csv", index=False, encoding="utf-8-sig")
    spec = {"start": cfg.POLICY_START, "end": str(POLICY_END.date()), "horizon_weeks": horizon,
            "last_source_week": str(weekly.index.max().date()),
            "unobserved_international_price": "last available weekly price held constant",
            "unobserved_tax_and_cap": "last available values held constant; scenario, not announced policy",
            "boundary_weeks": "display interval clipped to policy start and December 31",
            "post_policy_actual_supply_as_input": False}
    (out_dir / "policy_scenario.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")


def inputs(weekly, fuel, cutoff, horizon, context_weeks):
    """예측 직전까지의 실제 공급가와 과거·미래 국제가만 만든다. 미래 타겟은 읽지 않는다."""
    history = weekly.loc[weekly.index < cutoff]
    history = history.loc[history[f"{fuel}_supply_pretax"].notna() & history[f"{fuel}_intl_krw_l"].notna()]
    if context_weeks is not None:
        if len(history) < context_weeks:
            raise ValueError(f"요청 문맥 {context_weeks}주에 과거 자료 {len(history)}주가 부족합니다.")
        history = history.tail(context_weeks)
    if history.empty:
        raise ValueError("과거 공급가격이 없습니다.")
    future = weekly.loc[weekly.index >= cutoff].iloc[:horizon]
    dates = history.index.append(future.index)
    if len(future) != horizon or not dates.equals(pd.date_range(dates.min(), periods=len(dates), freq="W-SAT")):
        raise ValueError("주간 입력이 불연속이거나 예측 구간이 부족합니다.")
    target = history[f"{fuel}_supply_pretax"].to_numpy(dtype=np.float32)
    intl = weekly.loc[dates, f"{fuel}_intl_krw_l"].to_numpy(dtype=np.float32)[None, :]
    if not np.isfinite(target).all() or not np.isfinite(intl).all():
        raise ValueError("회귀 입력에 남은 결측이 있습니다. 별도 보정 없이 같은 자료를 사용합니다.")
    return target, intl, future.index


def forecast(model, weekly, fuel, cutoff, horizon, context_weeks):
    """국제 제품가격을 알려진 공변량으로 주고 공급가 경로와 분위수를 반환한다."""
    target, intl, dates = inputs(weekly, fuel, cutoff, horizon, context_weeks)
    outputs = list(model.predict_batch(contexts=[target], horizon=horizon,
        past_future_covariates=[intl], return_quantiles=True, use_symmetric_averaging=False))
    output = outputs[0]
    values = np.asarray(output.forecast, dtype=float)
    quantiles = np.asarray(output.quantiles, dtype=float)
    if values.shape != (horizon,) or not np.isfinite(values).all() or not np.isfinite(quantiles).all():
        raise ValueError("TimesFM 출력이 잘못되었습니다.")
    return dates, values, quantiles


def regression_forecast(weekly, fuel, cutoff, model, horizon=regression.EVALUATION_HORIZON):
    """같은 기점의 이전 자료만으로 회귀를 학습한다."""
    X, y = regression.training_data(weekly, fuel)
    before = X.index < cutoff
    params = regression.fit(X.loc[before], y.loc[before], model)
    future = X.loc[X.index >= cutoff].iloc[:horizon]
    return regression.predict(params, future)


def metrics(predictions):
    """Score fixed leads separately; each target week occurs once per lead."""
    table = predictions.assign(error=predictions.prediction-predictions.actual)
    return table.groupby(["split", "model", "context", "fuel", "lead"], dropna=False).apply(
        lambda g: pd.Series({"mae": g.error.abs().mean(), "bias": g.error.mean(),
            "rmse": np.sqrt((g.error**2).mean()), "origins": g.cutoff.nunique(),
            "weeks": g.date.nunique(), "prediction_pairs": len(g)}),
        include_groups=False).reset_index()


def evaluate(model, weekly, cutoffs, candidates, split, allow_short_history=False, horizon=regression.EVALUATION_HORIZON, score_lead_only=False):
    """공급가 정답을 예측 후에만 붙이고, 같은 자료의 두 회귀와 비교한다."""
    rows = []
    for fuel in cfg.FUELS:
        for cutoff in cutoffs:
            for context_weeks in candidates:
                available = weekly.loc[weekly.index < cutoff, [f"{fuel}_supply_pretax", f"{fuel}_intl_krw_l"]].dropna()
                effective = min(context_weeks, len(available)) if allow_short_history else context_weeks
                dates, predicted, _ = forecast(model, weekly, fuel, cutoff, horizon, effective)
                actual = weekly.loc[dates, f"{fuel}_supply_pretax"].to_numpy()
                label = "all" if context_weeks is None else str(context_weeks)
                rows.extend({"split": split, "model": "timesfm3", "context": label, "fuel": fuel,
                    "cutoff": cutoff, "date": date, "lead": lead, "actual": a, "prediction": p,
                    "actual_context_weeks": len(available) if effective is None else effective}
                    for lead, (date, a, p) in enumerate(zip(dates, actual, predicted), 1)
                    if not score_lead_only or lead == horizon)
            for name in regression.MODELS:
                predicted = regression_forecast(weekly, fuel, cutoff, name, horizon)
                rows.extend({"split": split, "model": name, "context": "all", "fuel": fuel,
                    "cutoff": cutoff, "date": date, "lead": lead, "actual": a, "prediction": p}
                    for lead, (date, a, p) in enumerate(zip(dates, actual, predicted), 1)
                    if not score_lead_only or lead == horizon)
            if score_lead_only:
                X, y = regression.training_data(weekly, fuel)
                last = y.loc[y.index < cutoff].iloc[-1]
                rows.append({"split": split, "model": "no_change", "context": "all", "fuel": fuel,
                    "cutoff": cutoff, "date": dates[-1], "lead": horizon,
                    "actual": actual[-1], "prediction": last})
        print(f"{split}: {fuel}, {len(cutoffs)}개 기점 완료", flush=True)
    return pd.DataFrame(rows)


def main(out_dir=OUT):
    """Weekly expanding-history evaluation, separately for each lead."""
    import torch
    from timesfm3 import ModelConfig, TimesFM3Evaluator
    torch.set_num_threads(4)
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--policy-only", action="store_true")
    actions.add_argument("--evaluation-only", action="store_true")
    args = parser.parse_args()
    weekly = regression.load_weekly()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256((cfg.OUT_DIR / "weekly_panel.csv").read_bytes()).hexdigest()
    if args.policy_only:
        spec = json.loads((out_dir / "locked_settings.json").read_text(encoding="utf-8"))
        if digest != spec["source_sha256"] or spec["chosen_context"] != "all":
            raise ValueError("Re-run evaluation with the current expanding-window settings")
    model = TimesFM3Evaluator(ModelConfig(checkpoint_path=str(ROOT / "\uBAA8\uB378" / "timesfm-3.0-pytorch"),
        per_core_batch_size=1, device="cpu", local_files_only=True))
    with torch.inference_mode():
        if args.policy_only:
            save_policy(model, weekly, None, out_dir)
            return
        train, test, plan, spec = regression.evaluation_plan(weekly)
        train.to_csv(cfg.OUT_DIR / "regression_train.csv", encoding="utf-8-sig")
        test.to_csv(cfg.OUT_DIR / "regression_test.csv", encoding="utf-8-sig")
        spec.update({"source_sha256": digest, "chosen_context": "all",
            "context_selection": "none; expanding history", "fixed_context_years": None,
            "future_international_prices": "actual observed; ex-post conditional estimates",
            "weight_updates": False, "public_release_delay_accounted": False,
            "policy_actual_context_weeks": available_context_weeks(weekly, weekly.index[weekly.in_policy][0])})
        scores = []
        for split, leads in plan.items():
            predictions = pd.concat([evaluate(model, weekly, cutoffs, (None,), split,
                horizon=lead, score_lead_only=True) for lead, cutoffs in leads.items()], ignore_index=True)
            predictions.to_csv(out_dir / f"{split}_predictions.csv", index=False, encoding="utf-8-sig")
            scores.append(metrics(predictions))
        all_scores = pd.concat(scores, ignore_index=True)
        all_scores.to_csv(out_dir / "model_comparison.csv", index=False, encoding="utf-8-sig")
        (out_dir / "locked_settings.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
        if not args.evaluation_only:
            save_policy(model, weekly, None, out_dir)
    print(all_scores[all_scores.split.eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
