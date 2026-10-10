# 1. 회귀분석과 같은 자료·같은 설명변수로 TimesFM 3.0을 돌려 세전 공급가격을 제품별로 예측합니다.
# 자료와 변수는 load_training_set.py(= 회귀분석의 클린파일·변수·시차·80/20 분할)에서 가져오고,
# 모형 실행은 기존 딥러닝 방식("딥러닝 2/run_regression_timesfm.py")을 그대로 따릅니다.
#   - 사전학습 가중치를 고정하고 학습(파인튜닝) 없이 추론만 합니다.
#   - 학습셋의 세전 공급가격 전체를 과거 문맥(context)으로 넣습니다.
#   - 설명변수(국제가격 t-1, 환율 t)는 과거·미래 모두 알려진 공변량(past_future_covariates)으로 넣습니다.
#     회귀분석도 평가셋의 실제 설명변수를 쓰므로 조건은 같습니다. 즉 국제가격이 주어진 조건부 추정입니다.
#   - 교차검증은 하지 않고 학습셋 → 평가셋 한 번만 예측합니다. 평가 지표는 회귀분석의 evaluate를 그대로 씁니다.
# 실행: python 딥러닝/run_timesfm.py [--device cpu|cuda] [--threads 4]
import argparse
import csv
from pathlib import Path

import numpy as np

from load_model import DEFAULT_CHECKPOINT, load_model
from load_training_set import MODELS, PROJECT_DIR, KEY_COLUMN, load_training_set, load_clean_rows, regression




# 2. 결과 저장 위치입니다. 회귀분석과 같은 폴더에 같은 열 이름으로 저장해 바로 비교할 수 있게 합니다.
PREDICTION_DIR = PROJECT_DIR / "csv"
REPORT_PATH = PROJECT_DIR / "results" / "timesfm_before_tax_report.md"
MODEL_NAME = "timesfm3"




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
# 호출 방식(return_quantiles=True, use_symmetric_averaging=False)은 기존 딥러닝 코드와 같습니다.
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
# 비교 기준으로 '학습셋 마지막 값을 그대로 유지'하는 무변화 예측도 함께 채점합니다(기존 딥러닝 코드의 no_change).
def run_single_model(product, model, rows):
    train = load_training_set(product, rows, split="train")
    test = load_training_set(product, rows, split="test")
    context, covariates, horizon = model_inputs(train, test)
    predicted, quantiles, levels = forecast(model, context, covariates, horizon)
    actual = [float(value) for value in test["y"]]
    no_change = [float(context[-1])] * horizon
    quantile_names = [f"q{round(level * 100):02d}" for level in levels]
    predictions = [
        {
            KEY_COLUMN: label,
            "구분": "test",
            "실제_세전가격": actual[i],
            "예측_세전가격": float(predicted[i]),
            "잔차": actual[i] - float(predicted[i]),
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
        "평가성능": regression.evaluate(actual, [float(value) for value in predicted]),
        "무변화성능": regression.evaluate(actual, no_change),
        "분위수": quantile_names,
        "예측": predictions,
    }




# 6. 회귀분석이 저장한 예측 CSV가 있으면 같은 평가셋에서의 회귀 성능을 읽어 비교표에 넣습니다.
def regression_test_metrics(product, directory=PREDICTION_DIR):
    path = directory / f"regression_predictions_{product}.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = [row for row in csv.DictReader(file) if row.get("구분") == "test"]
    if not rows:
        return None
    actual = [float(row["실제_세전가격"]) for row in rows]
    predicted = [float(row["예측_세전가격"]) for row in rows]
    return regression.evaluate(actual, predicted)




# 7. 제품별 예측값을 CSV로 저장합니다. 열 구성은 회귀분석 예측 파일과 같고 분위수 열만 추가됩니다.
def save_predictions(result, directory=PREDICTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{MODEL_NAME}_predictions_{result['제품']}.csv"
    fieldnames = [KEY_COLUMN, "구분", "실제_세전가격", "예측_세전가격", "잔차", *result["분위수"]]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["예측"]:
            writer.writerow({name: (f"{row[name]:.2f}" if isinstance(row[name], float) else row[name]) for name in fieldnames})
    return path




# 8. 세 모형의 결과를 하나의 마크다운 보고서로 정리합니다.
def save_report(results, checkpoint, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    first = results[0]
    lines = [
        "# 세전 공급가격 TimesFM 3.0 예측 보고서",
        "",
        "## 1. 분석 설계",
        "",
        f"- 자료: `{regression.INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}` (회귀분석과 동일한 클린파일)",
        "- 타겟변수: 정유사 주간공급가격 **세전** (제품별 3개 모형)",
        f"- 설명변수: 회귀분석과 동일 (해당 제품 국제가격 t-{regression.LAG_WEEKS} + 원달러환율 t), 과거·미래 공변량으로 입력",
        f"- 모형: TimesFM 3.0 사전학습 가중치 고정, 학습 없이 추론 (`{checkpoint}`)",
        f"- 문맥: 학습셋 {first['문맥길이']}주 ({first['학습기간'][0]} ~ {first['학습기간'][1]})",
        f"- 예측: 평가셋 {first['예측길이']}주 ({first['평가기간'][0]} ~ {first['평가기간'][1]}) 한 번에 예측, 교차검증 없음",
        "- 평가셋의 실제 국제가격·환율을 공변량으로 주므로 국제가격이 주어진 조건부 추정입니다. 회귀분석과 같은 조건입니다.",
        "",
        "### 제품별 설명변수",
        "",
        "| 모형 | 타겟 | 설명변수 |",
        "| --- | --- | --- |",
    ]
    for result in results:
        lines.append(f"| {result['제품']} | `{result['타겟']}` | " + ", ".join(f"`{n}`" for n in result["설명변수"]) + " |")
    lines += [
        "",
        "## 2. 평가셋 성능 비교",
        "",
        "| 제품 | 모형 | R² | RMSE | MAE | MAPE | 최대오차 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        rows = [("TimesFM 3.0", result["평가성능"]), ("무변화(마지막 학습값 유지)", result["무변화성능"])]
        reg = regression_test_metrics(result["제품"])
        if reg:
            rows.insert(1, ("회귀분석(OLS)", reg))
        for name, m in rows:
            lines.append(
                f"| {result['제품']} | {name} | {m['R2']:.4f} | {m['RMSE']:.2f} | {m['MAE']:.2f} | "
                f"{m['MAPE']:.2f}% | {m['최대오차']:.2f} |"
            )
    lines += [
        "",
        "단위는 원/리터입니다. 회귀분석 행은 `csv/regression_predictions_*.csv`의 평가셋 예측값으로 다시 계산했습니다.",
        "",
        "## 3. 해석 시 유의점",
        "",
        "- TimesFM은 학습셋에 맞춰 가중치를 고치지 않습니다. 학습셋은 과거 문맥으로만 쓰이므로 학습셋 성능(R² 등)은 없습니다.",
        f"- {first['예측길이']}주를 한 번에 예측하므로 뒤쪽 주는 먼 미래 예측입니다. 매주 기점을 옮기는 평가(교차검증)는 다음 단계입니다.",
        "- 분위수 열(q10~q90)은 모형이 내는 예측 구간입니다. 예측 CSV에 함께 저장했습니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 9. 자료 불러오기 → 모형 로딩 → 제품별 예측·평가 → 저장 순서로 진행합니다.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    try:
        rows = load_clean_rows()
        model = load_model(args.checkpoint, device=args.device, threads=args.threads)
        import torch
        with torch.inference_mode():
            results = [run_single_model(product, model, rows) for product in MODELS]
        prediction_paths = [save_predictions(result) for result in results]
        report_path = save_report(results, args.checkpoint.resolve())
    except (OSError, UnicodeError, ValueError, ImportError, csv.Error) as error:
        raise SystemExit(f"TimesFM 예측 실패: {error}") from None

    first = results[0]
    print(f"TimesFM 3.0 ({args.device}) · 문맥 {first['문맥길이']}주 → 예측 {first['예측길이']}주 · 교차검증 없음")
    print(f"학습(문맥) 기간: {first['학습기간'][0]} ~ {first['학습기간'][1]}")
    print(f"평가 기간:       {first['평가기간'][0]} ~ {first['평가기간'][1]}")
    print()
    for result in results:
        m, nc = result["평가성능"], result["무변화성능"]
        reg = regression_test_metrics(result["제품"])
        print(f"[{result['제품']}] 타겟: {result['타겟']}")
        print(f"  설명변수 {len(result['설명변수'])}개: " + ", ".join(result["설명변수"]))
        print(f"  TimesFM  R²={m['R2']:.4f} RMSE={m['RMSE']:.2f} MAE={m['MAE']:.2f} MAPE={m['MAPE']:.2f}%")
        if reg:
            print(f"  회귀(OLS) R²={reg['R2']:.4f} RMSE={reg['RMSE']:.2f} MAE={reg['MAE']:.2f} MAPE={reg['MAPE']:.2f}%")
        print(f"  무변화    R²={nc['R2']:.4f} RMSE={nc['RMSE']:.2f} MAE={nc['MAE']:.2f} MAPE={nc['MAPE']:.2f}%")
        print()
    print(f"보고서 저장: {report_path}")
    for path in prediction_paths:
        print(f"예측값 저장: {path}")




# 10. 직접 실행할 때만 예측을 수행합니다.
if __name__ == "__main__":
    main()
