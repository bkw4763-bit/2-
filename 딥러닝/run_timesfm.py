# 1. regression 폴더 회귀분석과 같은 자료·같은 설명변수로 TimesFM 3.0을 돌려 세전 공급가격을 제품별로 예측합니다.
# 자료와 변수는 load_training_set.py(= regression의 주별 표·국제가 3개 시차·정책 이전 80/20 분할)에서 가져옵니다.
#   - 사전학습 가중치를 고정하고 학습(파인튜닝) 없이 추론만 합니다.
#   - 학습셋의 세전 공급가격 전체를 과거 문맥(context)으로 넣습니다.
#   - 설명변수(국제가 같은 주·1주 전·2주 전)는 과거·미래 모두 알려진 공변량(past_future_covariates)으로 넣습니다.
#     회귀분석도 평가셋의 실제 국제가를 쓰므로 조건은 같습니다. 즉 국제가가 주어진 조건부 추정입니다.
#   - 이 파일은 학습셋 → 평가셋 한 번만 예측합니다. 추론 설정을 고르는 학습셋 안 롤링 오리진 교차검증은
#     cv_timesfm.py에 있습니다. 비교용 회귀(최근 마진·전체 회귀)는 같은 학습셋으로 적합해 함께 채점합니다.
# 실행: python 딥러닝/run_timesfm.py [--device cpu|cuda] [--threads 4]
import argparse
import csv
from pathlib import Path

import numpy as np

from load_model import DEFAULT_CHECKPOINT, load_model
from load_training_set import (
    INPUT_PATH, KEY_COLUMN, LAGS, MODELS, POLICY_START, PROJECT_DIR, REGRESSION_MODELS,
    evaluate, fit_regression, load_training_set, load_weekly_table, predict_regression,
)




# 2. 결과 저장 위치입니다.
PREDICTION_DIR = PROJECT_DIR / "csv"
REPORT_PATH = PROJECT_DIR / "results" / "timesfm_before_tax_report.md"
MODEL_NAME = "timesfm3"
REGRESSION_LABELS = {"recent_margin": "회귀(최근 마진)", "baseline": "회귀(학습 구간 전체)"}




# 3. 제품 하나의 입력을 만듭니다.
# 문맥은 학습셋 y 전체, 공변량은 학습셋+평가셋 X를 이어 붙인 뒤 (변수 수, 전체 길이)로 돌립니다.
# 평가셋 y는 예측이 끝난 뒤 채점에만 쓰고 모형에는 넣지 않습니다.
def model_inputs(train, test):
    if train["설명변수"] != test["설명변수"]:
        raise ValueError(f"{train['제품']}: 학습셋과 평가셋의 설명변수가 다릅니다")
    context = np.asarray(train["y"], dtype=np.float32)
    covariates = np.concatenate([train["X"], test["X"]], axis=0).T.astype(np.float32)
    horizon = len(test["y"])
    if covariates.shape != (len(train["설명변수"]), len(context) + horizon):
        raise ValueError(f"{train['제품']}: 공변량 배열 크기가 잘못됐습니다 {covariates.shape}")
    if not np.isfinite(context).all() or not np.isfinite(covariates).all():
        raise ValueError(f"{train['제품']}: 입력에 결측 또는 무한대가 있습니다")
    return context, covariates, horizon




# 4. TimesFM을 한 번 호출해 평가셋 길이만큼의 중앙 예측과 분위수를 받습니다.
def forecast(model, context, covariates, horizon):
    outputs = list(model.predict_batch(
        contexts=[context], horizon=horizon,
        past_future_covariates=[covariates],
        return_quantiles=True, use_symmetric_averaging=False,
    ))
    if len(outputs) != 1:
        raise ValueError("모형 출력 개수가 잘못됐습니다")
    output = outputs[0]
    predicted = np.asarray(output.forecast, dtype=float)
    quantiles = np.asarray(output.quantiles, dtype=float)
    levels = list(model.config.quantiles)
    if predicted.shape != (horizon,) or quantiles.shape != (horizon, len(levels)):
        raise ValueError(f"모형 출력 크기가 잘못됐습니다: {predicted.shape}, {quantiles.shape}")
    if not np.isfinite(predicted).all() or not np.isfinite(quantiles).all():
        raise ValueError("모형 출력에 결측 또는 무한대가 있습니다")
    return predicted, quantiles, levels




# 5. 제품 하나에 대해 입력 구성 → 예측 → 평가를 수행합니다.
# 비교 기준으로 regression의 두 회귀모형(같은 학습셋으로 적합)과 '학습셋 마지막 값 유지' 무변화 예측도 함께 채점합니다.
def run_single_model(product, model, weekly):
    train = load_training_set(product, weekly, split="train")
    test = load_training_set(product, weekly, split="test")
    context, covariates, horizon = model_inputs(train, test)
    predicted, quantiles, levels = forecast(model, context, covariates, horizon)
    actual = [float(value) for value in test["y"]]
    no_change = [float(context[-1])] * horizon
    regression = {}
    for name in REGRESSION_MODELS:
        params = fit_regression(train["X"], train["y"], name)
        regression[name] = {"예측": predict_regression(params, test["X"]), "계수": params}
    quantile_names = [f"q{round(level * 100):02d}" for level in levels]
    predictions = [
        {
            KEY_COLUMN: label,
            "날짜": test["날짜"][i],
            "구분": "test",
            "실제_세전가격": actual[i],
            "예측_세전가격": float(predicted[i]),
            "잔차": actual[i] - float(predicted[i]),
            **{f"{REGRESSION_LABELS[name]}_예측": regression[name]["예측"][i] for name in REGRESSION_MODELS},
            **{name: float(quantiles[i, j]) for j, name in enumerate(quantile_names)},
        }
        for i, label in enumerate(test["주차"])
    ]
    return {
        "제품": product,
        "타겟": train["타겟"],
        "설명변수": train["설명변수"],
        "문맥길이": len(context),
        "예측길이": horizon,
        "학습기간": train["기간"],
        "평가기간": test["기간"],
        "학습경계": train["학습경계"],
        "평가성능": evaluate(actual, [float(value) for value in predicted]),
        "회귀성능": {name: evaluate(actual, regression[name]["예측"]) for name in REGRESSION_MODELS},
        "회귀계수": {name: regression[name]["계수"] for name in REGRESSION_MODELS},
        "무변화성능": evaluate(actual, no_change),
        "분위수": quantile_names,
        "예측": predictions,
    }




# 6. 제품별 예측값을 CSV로 저장합니다. TimesFM 예측·분위수와 비교 회귀 예측을 같은 행에 둡니다.
def save_predictions(result, directory=PREDICTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{MODEL_NAME}_predictions_{result['제품']}.csv"
    fieldnames = [KEY_COLUMN, "날짜", "구분", "실제_세전가격", "예측_세전가격", "잔차",
                  *(f"{REGRESSION_LABELS[name]}_예측" for name in REGRESSION_MODELS), *result["분위수"]]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["예측"]:
            writer.writerow({name: (f"{row[name]:.2f}" if isinstance(row[name], float) else row[name]) for name in fieldnames})
    return path




# 7. 세 제품의 결과를 하나의 마크다운 보고서로 정리합니다.
def save_report(results, checkpoint, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = results[0]
    lines = [
        "# 세전 공급가격 TimesFM 3.0 예측 보고서",
        "",
        "## 1. 분석 설계",
        "",
        f"- 자료: `{INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}` → regression 폴더와 같은 달력 연속 주별 표 (`regression/full_sample/counterfactual.py`)",
        "- 타겟변수: 정유사 주간공급가격 **세전** (제품별 3개 모형)",
        f"- 설명변수: regression 폴더와 동일. 해당 제품 국제가(원/리터) 시차 {LAGS}개 (같은 주·1주 전·2주 전), 환율 없음. 과거·미래 공변량으로 입력",
        f"- 학습·평가 대상: 정책({POLICY_START.date()}) 이전 주만. 정책 이전 학습 행의 앞 80%가 학습, 뒤 20%가 평가 (`regression/train_test/split_regression.py`와 같은 경계)",
        f"- 모형: TimesFM 3.0 사전학습 가중치 고정, 학습 없이 추론 (`{checkpoint}`)",
        f"- 문맥: 학습셋 {first['문맥길이']}주 ({first['학습기간'][0]} ~ {first['학습기간'][1]}, 경계 {first['학습경계']})",
        f"- 예측: 평가셋 {first['예측길이']}주 ({first['평가기간'][0]} ~ {first['평가기간'][1]}) 한 번에 예측. 기점을 옮기는 학습셋 안 교차검증은 `results/timesfm_cv/cv_report.md`",
        "- 비교 회귀: regression의 최근 마진 모형(비중 합 1, 상수 = 학습 끝 52주 마진)과 학습 구간 전체 회귀. 같은 학습셋으로 적합해 평가셋을 예측",
        "- 평가셋의 실제 국제가를 모든 모형에 주므로 국제가가 주어진 조건부 추정입니다.",
        "",
        "### 제품별 설명변수와 회귀 계수",
        "",
        "| 모형 | 타겟 | 설명변수 | 최근 마진 비중 (같은 주 / 1주 전 / 2주 전) | 마진 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for result in results:
        params = result["회귀계수"]["recent_margin"]
        w = params["weights"]
        lines.append(f"| {result['제품']} | `{result['타겟']}` | " + ", ".join(f"`{n}`" for n in result["설명변수"])
                     + f" | {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f} | {params['const']:.1f}원 |")
    lines += [
        "",
        "## 2. 평가셋 성능 비교",
        "",
        "| 제품 | 모형 | R² | RMSE | MAE | 편향 | MAPE | 최대오차 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        rows = [("TimesFM 3.0", result["평가성능"])]
        rows += [(REGRESSION_LABELS[name], result["회귀성능"][name]) for name in REGRESSION_MODELS]
        rows.append(("무변화(마지막 학습값 유지)", result["무변화성능"]))
        for name, m in rows:
            lines.append(
                f"| {result['제품']} | {name} | {m['R2']:.4f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | {m['편향']:+.2f} | "
                f"{m['MAPE']:.2f}% | {m['최대오차']:.2f} |"
            )
    lines += [
        "",
        "단위는 원/리터입니다. 편향 = 예측 − 실제의 평균. 회귀 행은 regression 폴더의 `split_report.md` 평가 구간과 같은 주·같은 모형입니다.",
        "",
        "## 3. 해석 시 유의점",
        "",
        "- TimesFM은 학습셋에 맞춰 가중치를 고치지 않습니다. 학습셋은 과거 문맥으로만 쓰이므로 학습셋 성능(R² 등)은 없습니다.",
        f"- {first['예측길이']}주를 한 번에 예측하므로 뒤쪽 주는 먼 미래 예측입니다. 최근 마진 회귀도 마진을 다시 잡지 않고 고정했습니다.",
        "- 정책 기간(상한제)은 세전 공급가가 상한에 묶여 국제가만으로 맞힐 수 없으므로 평가에서 제외했습니다. 로더의 `split=\"policy\"`로 따로 받을 수 있습니다.",
        "- 분위수 열(q10~q90)은 모형이 내는 예측 구간입니다. 예측 CSV에 함께 저장했습니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 8. 자료 불러오기 → 모형 로딩 → 제품별 예측·평가 → 저장 순서로 진행합니다.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        weekly = load_weekly_table()
        model = load_model(args.checkpoint, device=args.device, threads=args.threads)
        import torch
        with torch.inference_mode():
            results = [run_single_model(product, model, weekly) for product in MODELS]
        prediction_paths = [save_predictions(result) for result in results]
        report_path = save_report(results, args.checkpoint.resolve())
    except (OSError, UnicodeError, ValueError, KeyError, ImportError, csv.Error) as error:
        raise SystemExit(f"TimesFM 예측 실패: {error}") from None

    first = results[0]
    print(f"TimesFM 3.0 ({args.device}) · 문맥 {first['문맥길이']}주 → 예측 {first['예측길이']}주 · regression 방식 자료·변수·분할")
    print(f"학습(문맥) 기간: {first['학습기간'][0]} ~ {first['학습기간'][1]} (경계 {first['학습경계']})")
    print(f"평가 기간:       {first['평가기간'][0]} ~ {first['평가기간'][1]} (정책 이전만)")
    print()
    for result in results:
        print(f"[{result['제품']}] 타겟: {result['타겟']}")
        print(f"  설명변수 {len(result['설명변수'])}개: " + ", ".join(result["설명변수"]))
        rows = [("TimesFM", result["평가성능"])]
        rows += [(REGRESSION_LABELS[name], result["회귀성능"][name]) for name in REGRESSION_MODELS]
        rows.append(("무변화", result["무변화성능"]))
        for name, m in rows:
            print(f"  {name:<12} R²={m['R2']:.4f} RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f} 편향={m['편향']:+.2f} MAPE={m['MAPE']:.2f}%")
        print()
    print(f"보고서 저장: {report_path}")
    for path in prediction_paths:
        print(f"예측값 저장: {path}")




# 9. 직접 실행할 때만 예측을 수행합니다.
if __name__ == "__main__":
    main()
