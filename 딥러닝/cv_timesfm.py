# 1. 학습셋 안에서만 TimesFM 3.0과 regression 회귀를 같은 기점에서 교차검증해 예측오차를 비교합니다.
# 평가셋(정책 이전 뒤 20%)과 정책 기간은 쓰지 않습니다. 두 가지 프로토콜을 고를 수 있습니다.
#   --protocol regression (기본): regression 폴더의 기점 이동 백테스트와 '같은 격자'입니다.
#       첫 기점 2018-01-01, 간격 13주, 예측 32주 → counterfactual.cutoffs와 같은 기점 22개.
#       regression 보고서(split_report.md 2절)의 MAE와 숫자를 직접 맞댈 수 있습니다.
#       창이 겹치므로(간격 13 < 예측 32) 기점별 오차가 독립이 아닙니다. 그래서 모형 간 차이는
#       Diebold-Mariano 검정(Newey-West HAC + Harvey-Leybourne-Newbold 소표본 보정)으로 봅니다.
#   --protocol nonoverlap: 비중첩 26주 창(GIFT-Eval·fev-bench 방식). 창이 독립이라 쌍체 t를 그대로 씁니다.
# 공통 규칙: 기점마다 그 이전 자료만 씁니다. TimesFM은 가중치를 고정하고 문맥만 확장하며, 회귀는 기점마다
# 비중과 마진을 다시 적합합니다. 예측 구간의 실제 국제가는 모든 모형에 같이 넣습니다(조건부 추정).
# 실행: python 딥러닝/cv_timesfm.py [--protocol regression|nonoverlap] [--device cpu|cuda] [--threads 4]
import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from load_model import DEFAULT_CHECKPOINT, load_model
from load_training_set import (
    BACKTEST_HORIZON, BACKTEST_START, BACKTEST_STEP, INPUT_PATH, KEY_COLUMN, LAGS, MODELS,
    POLICY_START, PROJECT_DIR, REGRESSION_MODELS, counterfactual, evaluate, fit_regression,
    load_training_set, load_weekly_table, predict_regression, split_date,
)
from run_timesfm import forecast




# 2. 프로토콜 정의와 결과 저장 위치입니다.
#   horizon             기점마다 맞혀 보는 주 수
#   step                기점 간격(주). horizon과 같으면 창이 겹치지 않습니다.
#   start               첫 기점 날짜. None이면 min_context_ratio로 첫 창을 정합니다.
#   min_context_ratio   첫 창의 문맥이 학습셋의 최소 몇 %여야 하는지
PROTOCOLS = {
    "regression": {
        "horizon": BACKTEST_HORIZON, "step": BACKTEST_STEP, "start": BACKTEST_START.date().isoformat(),
        "min_context_ratio": None, "dir": "timesfm_cv_aligned",
        "설명": "regression 폴더 기점 이동 백테스트와 동일 격자 (창 겹침, Diebold-Mariano 검정)",
    },
    "nonoverlap": {
        "horizon": 26, "step": 26, "start": None, "min_context_ratio": 0.5, "dir": "timesfm_cv",
        "설명": "비중첩 롤링 오리진 (GIFT-Eval·fev-bench 방식, 창 독립)",
    },
}
MODEL_LABELS = {
    "timesfm_cov": "TimesFM 3.0 (공변량)",
    "timesfm_uni": "TimesFM 3.0 (공변량 없음)",
    "recent_margin": "회귀 최근 마진 (기점별 재적합)",
    "baseline": "회귀 전체 (기점별 재적합)",
    "no_change": "무변화 (기준주 가격 유지)",
}
REFERENCE = "recent_margin"   # 차이를 잴 기준 모형 (regression 채택 모형)
LEADS = (1, 4, 13, 26)




# 3. 기점(문맥 길이)을 정합니다.
# start가 있으면 regression의 cutoffs와 같은 방식입니다. 조건을 만족하는 행을 모아 step 간격으로 고릅니다.
# start가 없으면 학습셋 끝에서 step씩 거슬러 올라가며 최소 문맥 이상인 창만 남깁니다.
def fold_origins(train_size, horizon, step=None, start=None, dates=None, min_context_ratio=None):
    step = step or horizon
    if horizon < 1 or horizon >= train_size:
        raise ValueError(f"예측 길이 {horizon}는 1 이상, 학습셋 길이 {train_size} 미만이어야 합니다")
    if step < 1:
        raise ValueError(f"기점 간격은 1 이상이어야 합니다: {step}")
    if start is not None:
        if dates is None or len(dates) != train_size:
            raise ValueError("start를 쓰려면 학습셋과 길이가 같은 날짜 목록이 필요합니다")
        eligible = [i for i in range(train_size) if i + horizon <= train_size and dates[i] >= start]
        origins = eligible[::step]
    else:
        if min_context_ratio is None:
            raise ValueError("start가 없으면 min_context_ratio가 필요합니다")
        min_context = int(math.ceil(train_size * min_context_ratio))
        origins, context = [], train_size - horizon
        while context >= min_context:
            origins.append(context)
            context -= step
        origins = sorted(origins)
    if not origins:
        raise ValueError("조건을 만족하는 기점이 없습니다. horizon이나 start를 줄이세요")
    return origins


# 겹치는 창의 개수입니다. 간격 13주·예측 32주면 3이므로 이웃 2개까지 겹칩니다(HAC 시차 = 2).
def overlap_length(horizon, step):
    return max(0, int(math.ceil(horizon / step)) - 1)




# 4. 창 하나의 입력을 만듭니다. 문맥은 y[:origin], 공변량은 X[:origin+H]를 (변수 수, 길이)로 돌립니다.
# 정답 y[origin:origin+H]는 채점에만 쓰고 모형에는 넣지 않습니다.
def fold_inputs(train, origin, horizon):
    y, X = train["y"], train["X"]
    context = np.asarray(y[:origin], dtype=np.float32)
    covariates = np.asarray(X[: origin + horizon], dtype=np.float32).T
    actual = [float(value) for value in y[origin : origin + horizon]]
    if covariates.shape != (len(train["설명변수"]), origin + horizon) or len(actual) != horizon:
        raise ValueError(f"{train['제품']}: 창 입력 크기가 잘못됐습니다")
    return context, covariates, actual




# 5. 창 하나에 다섯 모형을 적용해 예측값을 모형별 사전으로 돌려줍니다.
def predict_fold(model, train, origin, horizon):
    context, covariates, actual = fold_inputs(train, origin, horizon)
    predicted_cov, _, _ = forecast(model, context, covariates, horizon)
    outputs = list(model.predict_batch(
        contexts=[context], horizon=horizon, return_quantiles=True, use_symmetric_averaging=False,
    ))
    predicted_uni = np.asarray(outputs[0].forecast, dtype=float)
    if predicted_uni.shape != (horizon,) or not np.isfinite(predicted_uni).all():
        raise ValueError("공변량 없는 TimesFM 출력이 잘못됐습니다")
    predictions = {
        "timesfm_cov": [float(v) for v in predicted_cov],
        "timesfm_uni": [float(v) for v in predicted_uni],
    }
    for name in REGRESSION_MODELS:
        params = fit_regression(train["X"][:origin], train["y"][:origin], name)
        predictions[name] = predict_regression(params, train["X"][origin : origin + horizon])
    predictions["no_change"] = [float(context[-1])] * horizon
    return actual, {name: predictions[name] for name in MODEL_LABELS}




# 6. 제품 하나에 대해 모든 기점을 돌려 긴 형식의 예측 행 목록을 만듭니다.
def run_product(product, model, weekly, protocol):
    train = load_training_set(product, weekly, split="train")
    origins = fold_origins(len(train["y"]), protocol["horizon"], protocol["step"],
                           protocol["start"], train["날짜"], protocol["min_context_ratio"])
    horizon = protocol["horizon"]
    records = []
    for fold, origin in enumerate(origins, start=1):
        actual, predictions = predict_fold(model, train, origin, horizon)
        for lead in range(horizon):
            week = train["주차"][origin + lead]
            for name, values in predictions.items():
                records.append({
                    "제품": product, "폴드": fold, "문맥길이": origin,
                    "기준주차": train["주차"][origin - 1], "기준날짜": train["날짜"][origin - 1],
                    KEY_COLUMN: week, "날짜": train["날짜"][origin + lead], "리드": lead + 1,
                    "모형": name, "실제_세전가격": actual[lead], "예측_세전가격": values[lead],
                    "잔차": actual[lead] - values[lead],
                })
    return {"제품": product, "타겟": train["타겟"], "설명변수": train["설명변수"],
            "학습기간": train["기간"], "학습길이": len(train["y"]), "기준점": origins, "예측": records}




# 7. 기점별 지표와 오차 행렬(행 = 기점, 열 = 리드)을 만듭니다. 오차는 예측 − 실제이므로 음수면 낮게 예측한 것입니다.
def fold_metrics(records):
    groups = {}
    for row in records:
        groups.setdefault((row["제품"], row["폴드"], row["모형"]), []).append(row)
    order = list(MODEL_LABELS)
    metrics = []
    for key in sorted(groups, key=lambda k: (list(MODELS).index(k[0]), k[1], order.index(k[2]))):
        rows = sorted(groups[key], key=lambda r: r["리드"])
        metrics.append({"제품": key[0], "폴드": key[1], "문맥길이": rows[0]["문맥길이"],
                        "기준주차": rows[0]["기준주차"], "모형": key[2],
                        **evaluate([r["실제_세전가격"] for r in rows], [r["예측_세전가격"] for r in rows])})
    return metrics


def error_matrices(records, horizon):
    matrices = {}
    for row in records:
        matrices.setdefault((row["제품"], row["모형"]), {}).setdefault(row["폴드"], {})[row["리드"]] = -row["잔차"]
    result = {}
    for key, folds in matrices.items():
        result[key] = np.array([[folds[f][lead] for lead in range(1, horizon + 1)] for f in sorted(folds)])
    return result




# 8. regression 보고서와 같은 집계입니다(counterfactual.summarize와 같은 항목).
# 모든 기점의 오차를 한 덩어리로 모으므로, 창이 겹치는 프로토콜에서는 중간 시기의 주가 여러 번 들어갑니다.
# 숫자를 regression 보고서와 맞대려면 그 보고서와 같은 방식이어야 하므로 그대로 따릅니다.
def pooled_summary(errors):
    flat = errors.ravel()
    return {"기점수": errors.shape[0], "예측길이": errors.shape[1], "채점점수": flat.size,
            "MAE": float(np.abs(flat).mean()), "편향": float(flat.mean()),
            "RMSE": float(math.sqrt((flat ** 2).mean())),
            "MAE_1_4주": float(np.abs(errors[:, :4]).mean()),
            "MAE_마지막4주": float(np.abs(errors[:, -4:]).mean()),
            "오차_q05": float(np.quantile(flat, 0.05)), "오차_q95": float(np.quantile(flat, 0.95))}




# 9. 겹치는 기점에서도 쓸 수 있는 유의성 검정입니다.
# Newey-West HAC 분산(Bartlett 가중)으로 기점 간 상관을 보정하고, Diebold-Mariano 통계량에
# Harvey-Leybourne-Newbold 소표본 보정을 곱한 뒤 t분포로 p값을 구합니다.
# 손실은 기점별 경로 MAE이고, 차이 = 모형 − 기준 모형이므로 음수면 모형이 더 좋습니다.
def hac_variance(values, lag):
    series = np.asarray(values, dtype=float)
    count = len(series)
    centered = series - series.mean()
    variance = float(centered @ centered) / count
    for j in range(1, min(lag, count - 1) + 1):
        weight = 1.0 - j / (lag + 1)
        variance += 2.0 * weight * float(centered[j:] @ centered[:-j]) / count
    return variance


def diebold_mariano(diff, lag):
    diff = np.asarray(diff, dtype=float)
    count = len(diff)
    if count < 3 or np.allclose(diff, 0.0):
        return {"평균차": float(diff.mean()) if count else float("nan"), "DM_t": float("nan"), "DM_p": float("nan"), "HAC시차": lag}
    variance = hac_variance(diff, lag)
    if variance <= 0:
        variance = float(np.var(diff, ddof=1))
    statistic = float(diff.mean()) / math.sqrt(variance / count)
    horizon = lag + 1
    factor = (count + 1 - 2 * horizon + horizon * (horizon - 1) / count) / count
    corrected = statistic * math.sqrt(factor) if factor > 0 else statistic
    return {"평균차": float(diff.mean()), "DM_t": corrected,
            "DM_p": float(2 * stats.t.sf(abs(corrected), count - 1)), "HAC시차": lag}


def paired_t(diff):
    diff = np.asarray(diff, dtype=float)
    count = len(diff)
    spread = float(np.std(diff, ddof=1)) if count > 1 else 0.0
    if count < 2 or spread == 0:
        return {"부분집합_n": count, "부분집합_t": float("nan"), "부분집합_p": float("nan")}
    statistic = float(diff.mean()) / (spread / math.sqrt(count))
    return {"부분집합_n": count, "부분집합_t": statistic, "부분집합_p": float(2 * stats.t.sf(abs(statistic), count - 1))}




# 10. 제품·모형별 종합표입니다. 기점별 MAE 차이에 DM 검정과, 겹치지 않는 기점만 고른 부분집합의 쌍체 t를 함께 냅니다.
def summarize(metrics, matrices, horizon, step):
    lag = overlap_length(horizon, step)
    stride = lag + 1
    summary = []
    for product in sorted({m["제품"] for m in metrics}, key=list(MODELS).index):
        by_model = {name: [m for m in metrics if m["제품"] == product and m["모형"] == name] for name in MODEL_LABELS}
        naive = {m["폴드"]: m["MAE"] for m in by_model["no_change"]}
        reference = {m["폴드"]: m["MAE"] for m in by_model[REFERENCE]}
        for name, rows in by_model.items():
            maes = np.array([m["MAE"] for m in rows])
            ratios = np.log(np.maximum(maes, 1e-9) / np.array([max(naive[m["폴드"]], 1e-9) for m in rows]))
            diff = np.array([m["MAE"] - reference[m["폴드"]] for m in rows])
            summary.append({
                "제품": product, "모형": name, **pooled_summary(matrices[(product, name)]),
                "MAE평균": float(maes.mean()), "MAE표준편차": float(maes.std(ddof=1)) if len(maes) > 1 else float("nan"),
                "skill_vs_무변화": float(1.0 - math.exp(ratios.mean())),
                "무변화승수": int(sum(1 for m in rows if m["MAE"] < naive[m["폴드"]])),
                "기준보다_좋은_기점": int((diff < 0).sum()),
                **diebold_mariano(diff, lag), **paired_t(diff[::stride]),
            })
    return summary


def lead_metrics(records, horizon, leads=LEADS):
    leads = sorted({lead for lead in (*leads, horizon) if lead <= horizon})
    table = []
    for product in sorted({r["제품"] for r in records}, key=list(MODELS).index):
        for name in MODEL_LABELS:
            row = {"제품": product, "모형": name}
            for lead in leads:
                errors = [abs(r["잔차"]) for r in records if r["제품"] == product and r["모형"] == name and r["리드"] == lead]
                row[f"MAE_리드{lead}"] = float(np.mean(errors))
            table.append(row)
    return leads, table




# 11. 프로토콜이 정말 regression과 같은지 확인합니다.
# counterfactual.backtest(기점·예측길이·적합 모두 regression 코드)로 다시 계산해 기점 목록과 MAE를 맞춰 봅니다.
def verify_protocol(weekly, results, matrices, tolerance=0.01):
    train_end = split_date(weekly)
    checks = []
    for result in results:
        fuel = MODELS[result["제품"]]["fuel"]
        X, y = counterfactual.training_data(weekly, fuel)
        inside = X.index <= train_end
        X, y = X[inside], y[inside]
        cutoffs = counterfactual.cutoffs(X)
        if cutoffs != result["기준점"]:
            raise ValueError(f"{result['제품']}: 기점이 regression과 다릅니다 ({len(cutoffs)}개 vs {len(result['기준점'])}개)")
        for name in REGRESSION_MODELS:
            errors = counterfactual.backtest(X, y, lambda a, b, n=name: counterfactual.fit(a, b, n))
            theirs = counterfactual.summarize(errors)
            mine = pooled_summary(matrices[(result["제품"], name)])
            gap = abs(theirs["MAE"] - mine["MAE"])
            if gap > tolerance or np.abs(errors - matrices[(result["제품"], name)]).max() > tolerance:
                raise ValueError(f"{result['제품']} {name}: regression 백테스트와 오차가 다릅니다 (MAE 차 {gap:.4f}원)")
            checks.append({"제품": result["제품"], "모형": name, "기점수": len(cutoffs),
                           "regression_MAE": theirs["MAE"], "딥러닝_MAE": mine["MAE"], "차이": gap})
    return checks




# 12. CSV와 마크다운 보고서를 저장합니다.
def save_csv(rows, path, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in row.items() if k in fieldnames})
    return path


def save_report(results, metrics, summary, leads, lead_table, checks, protocol, name, checkpoint, path):
    first = results[0]
    horizon, step = protocol["horizon"], protocol["step"]
    lag = overlap_length(horizon, step)
    lines = [
        f"# 세전 공급가격 교차검증 보고서 — {protocol['설명']}",
        "",
        "## 1. 검증 설계",
        "",
        f"- 프로토콜: `--protocol {name}`. 예측 길이 {horizon}주, 기점 간격 {step}주, 기점 {len(first['기준점'])}개.",
        f"- 자료·변수·분할: regression 폴더와 동일 (`regression/full_sample/counterfactual.py`). 해당 제품 국제가 시차 {LAGS}개(같은 주·1주 전·2주 전), 환율 없음.",
        f"- 구간: 학습셋만 사용 ({first['학습기간'][0]} ~ {first['학습기간'][1]}, {first['학습길이']}주). 평가셋과 정책 기간({POLICY_START.date()} 이후)은 쓰지 않음.",
        f"- 첫 기점 문맥 {first['기준점'][0]}주, 마지막 기점 문맥 {first['기준점'][-1]}주.",
        f"- TimesFM: 사전학습 가중치 고정, 기점마다 문맥만 확장 (`{checkpoint}`). 회귀 두 모형: 기점마다 비중과 마진을 다시 적합.",
        "- 예측 구간의 **실제** 국제가를 모든 모형에 같이 넣습니다. 즉 국제가가 주어진 조건부 예측이고, 모형 간 입력 정보는 같습니다.",
        f"- 창 겹침: 간격 {step}주 < 예측 {horizon}주 → 한 주가 최대 {lag + 1}개 창에 등장." if lag else "- 창이 겹치지 않으므로 기점별 지표를 독립 표본으로 봅니다.",
        "",
    ]
    if lag:
        lines += [
            "### 겹침을 감안한 검정",
            "",
            f"- 기점별 경로 MAE의 차이(모형 − {MODEL_LABELS[REFERENCE]})에 Diebold-Mariano 검정을 씁니다.",
            f"- 분산은 Newey-West HAC(Bartlett 가중, 시차 {lag})로 보정하고 Harvey-Leybourne-Newbold 소표본 보정을 곱했습니다.",
            f"- 추가로 겹치지 않는 기점만 {lag + 1}개마다 골라낸 부분집합에서 쌍체 t도 함께 냈습니다.",
            "- 두 검정 모두 음수면 그 모형이 기준보다 오차가 작다는 뜻이고, p가 작을수록 차이가 뚜렷합니다.",
            "",
        ]
    if checks:
        lines += [
            "## 2. 프로토콜 일치 검증",
            "",
            "regression 코드(`counterfactual.backtest`)로 같은 기점을 다시 돌려 오차가 같은지 확인했습니다.",
            "기점 목록이 같고 주별 오차가 0.01원 안에서 같아야 통과합니다.",
            "",
            "| 제품 | 모형 | 기점수 | regression MAE | 딥러닝 CV MAE | 차이 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for c in checks:
            lines.append(f"| {c['제품']} | {MODEL_LABELS[c['모형']]} | {c['기점수']} | {c['regression_MAE']:.2f} | "
                         f"{c['딥러닝_MAE']:.2f} | {c['차이']:.4f} |")
        lines += ["", "따라서 아래 TimesFM 수치는 `results/regression/train_test/split_report.md` 2절의 기점 이동 검증 MAE와 같은 평면에 있습니다.", ""]
    lines += [
        f"## {3 if checks else 2}. 모형별 종합 (세전 원/리터, regression 보고서와 같은 집계)",
        "",
        "| 제품 | 모형 | MAE | 편향 | RMSE | 1~4주 MAE | 마지막 4주 MAE | 오차 90% 구간 | skill vs 무변화 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        lines.append(f"| {s['제품']} | {MODEL_LABELS[s['모형']]} | {s['MAE']:.1f} | {s['편향']:+.1f} | {s['RMSE']:.1f} | "
                     f"{s['MAE_1_4주']:.1f} | {s['MAE_마지막4주']:.1f} | {s['오차_q05']:+.0f} ~ {s['오차_q95']:+.0f} | {s['skill_vs_무변화']:+.3f} |")
    lines += [
        "",
        f"모든 기점의 주별 오차를 한 덩어리로 모아 계산했습니다(기점 {summary[0]['기점수']}개 × {summary[0]['예측길이']}주 = {summary[0]['채점점수']}개). 오차 = 예측 − 실제.",
        "",
        f"## {4 if checks else 3}. 기준 모형 대비 유의성",
        "",
        f"기준은 {MODEL_LABELS[REFERENCE]}입니다. 차이는 기점별 경로 MAE의 (모형 − 기준) 평균이고 음수면 모형이 더 좋습니다.",
        "",
        "| 제품 | 모형 | 기점별 MAE 평균±표준편차 | 기준 대비 차 | 기준보다 좋은 기점 | DM t | DM p | 비중첩 t | 비중첩 p |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for s in summary:
        lines.append(f"| {s['제품']} | {MODEL_LABELS[s['모형']]} | {s['MAE평균']:.1f} ± {s['MAE표준편차']:.1f} | {s['평균차']:+.1f} | "
                     f"{s['기준보다_좋은_기점']}/{s['기점수']} | {s['DM_t']:+.2f} | {s['DM_p']:.3f} | {s['부분집합_t']:+.2f} | {s['부분집합_p']:.3f} |")
    lines += [
        "",
        f"비중첩 열은 기점을 {lag + 1}개마다 골라낸 {summary[0]['부분집합_n']}개 기점의 쌍체 t입니다." if lag else "",
        "",
        f"## {5 if checks else 4}. 예측 거리별 MAE (모든 기점 평균)",
        "",
        "| 제품 | 모형 | " + " | ".join(f"{lead}주 앞" for lead in leads) + " |",
        "| --- | --- | " + " | ".join("---" for _ in leads) + " |",
    ]
    for row in lead_table:
        lines.append(f"| {row['제품']} | {MODEL_LABELS[row['모형']]} | "
                     + " | ".join(f"{row[f'MAE_리드{lead}']:.1f}" for lead in leads) + " |")
    lines += [
        "",
        "거리를 섞지 않고 같은 거리끼리만 비교한 표입니다. 거리별로 모형의 우열이 바뀌는지 볼 때 씁니다.",
        "",
        f"## {6 if checks else 5}. 기점별 MAE",
        "",
        "| 제품 | 폴드 | 기준주차 | 문맥 | " + " | ".join(MODEL_LABELS[n] for n in MODEL_LABELS) + " |",
        "| --- | --- | --- | --- | " + " | ".join("---" for _ in MODEL_LABELS) + " |",
    ]
    by_fold = {}
    for m in metrics:
        by_fold.setdefault((m["제품"], m["폴드"], m["기준주차"], m["문맥길이"]), {})[m["모형"]] = m["MAE"]
    for (product, fold, origin_week, context), row in by_fold.items():
        lines.append(f"| {product} | {fold} | {origin_week} | {context}주 | "
                     + " | ".join(f"{row[n]:.1f}" for n in MODEL_LABELS) + " |")
    lines += [
        "",
        f"## {7 if checks else 6}. 해석 시 유의점",
        "",
        "- 두 모형은 같은 기점, 같은 주, 같은 국제가 입력으로 채점했습니다. 다른 점은 모형이 자료에서 무엇을 배우는지입니다. 회귀는 비중 3개와 마진 1개를 이 자료에서 추정하고, TimesFM은 외부 사전학습 가중치를 고정한 채 학습셋을 문맥으로만 봅니다.",
        "- TimesFM의 사전학습 자료 기간은 알 수 없습니다. '학습 자료가 기점 이전으로 한정됐다'는 말은 회귀에만 엄격히 성립합니다.",
        f"- 기점 {summary[0]['기점수']}개는 작은 표본입니다. p값은 참고로 보고, 기점별 MAE 표의 패턴과 함께 읽으세요.",
        "- 이 검증은 학습셋 안에서만 합니다. 평가셋 한 번 예측은 `run_timesfm.py`, 평가셋 모형 비교는 `results/regression/train_test/common_test_report.md`에 있습니다.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(line for line in lines if line is not None), encoding="utf-8")
    return path




# 13. 자료 불러오기 → 모형 로딩 → 제품별 기점 예측 → 지표·검정 → 저장 순서로 진행합니다.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=tuple(PROTOCOLS), default="regression")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    protocol = PROTOCOLS[args.protocol]
    out_dir = args.out_dir or PROJECT_DIR / "results" / protocol["dir"]
    try:
        weekly = load_weekly_table()
        model = load_model(args.checkpoint, device=args.device, threads=args.threads)
        import torch
        results = []
        with torch.inference_mode():
            for product in MODELS:
                result = run_product(product, model, weekly, protocol)
                results.append(result)
                print(f"[{product}] 기점 {len(result['기준점'])}개 예측 완료 (문맥 {result['기준점'][0]}~{result['기준점'][-1]}주)", flush=True)
        records = [r for result in results for r in result["예측"]]
        metrics = fold_metrics(records)
        matrices = error_matrices(records, protocol["horizon"])
        checks = verify_protocol(weekly, results, matrices) if args.protocol == "regression" else []
        summary = summarize(metrics, matrices, protocol["horizon"], protocol["step"])
        leads, lead_table = lead_metrics(records, protocol["horizon"])
        paths = [
            save_csv(records, out_dir / "cv_predictions.csv",
                     ["제품", "폴드", "문맥길이", "기준주차", "기준날짜", KEY_COLUMN, "날짜", "리드", "모형",
                      "실제_세전가격", "예측_세전가격", "잔차"]),
            save_csv(metrics, out_dir / "cv_fold_metrics.csv",
                     ["제품", "폴드", "문맥길이", "기준주차", "모형", "n", "R2", "RMSE", "MAE", "편향", "MAPE", "최대오차"]),
            save_csv(summary, out_dir / "cv_summary.csv", list(summary[0].keys())),
            save_report(results, metrics, summary, leads, lead_table, checks, protocol, args.protocol,
                        args.checkpoint.resolve(), out_dir / "cv_report.md"),
        ]
        if checks:
            save_csv(checks, out_dir / "cv_protocol_check.csv", list(checks[0].keys()))
    except (OSError, UnicodeError, ValueError, KeyError, ImportError, csv.Error) as error:
        raise SystemExit(f"TimesFM 교차검증 실패: {error}") from None

    print()
    print(f"{protocol['설명']} · 예측 {protocol['horizon']}주 · 간격 {protocol['step']}주 · 기점 {len(results[0]['기준점'])}개 · 학습셋 {results[0]['학습길이']}주만 사용")
    if checks:
        print(f"프로토콜 일치 검증 통과: regression 백테스트와 MAE 최대 차 {max(c['차이'] for c in checks):.4f}원")
    print()
    for s in summary:
        print(f"  {s['제품']:<4} {MODEL_LABELS[s['모형']]:<24} MAE {s['MAE']:7.1f}  편향 {s['편향']:+7.1f}  "
              f"기준대비 {s['평균차']:+6.1f} (DM t {s['DM_t']:+5.2f}, p {s['DM_p']:.3f})  skill {s['skill_vs_무변화']:+.3f}")
    print()
    for path in paths:
        print(f"저장: {path.relative_to(PROJECT_DIR).as_posix()}")




# 14. 직접 실행할 때만 교차검증을 수행합니다.
if __name__ == "__main__":
    main()
