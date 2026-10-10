# 1. 학습셋 안 교차검증(평면 A)과 공통 평가셋(평면 B)의 오차를 한 보고서로 묶습니다.
# 딥러닝/cv_timesfm.py가 저장한 예측 파일을 읽어, 거기 들어간 regression 회귀 예측을
# regression 폴더 코드(counterfactual.training_data → fit → predict)로 독립적으로 다시 적합해 일치를 확인합니다.
# 그 뒤 두 평면의 지표를 나란히 적어, 같은 모형이 검증 구간과 평가 구간에서 어떻게 달라지는지 봅니다.
# 실행: python regression/train_test/apply_cv_folds.py [--cv-dir results/timesfm_cv_aligned]
#       (먼저 python 딥러닝/cv_timesfm.py 와 apply_common_test.py 실행)
import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR / "regression" / "full_sample"))
from counterfactual import CHOSEN, FUELS, MODELS, fit, lagged_intl, load_weekly, predict, week_end  # noqa: E402




# 2. 입력·출력 위치와 모형 이름입니다. 결과 파일 이름에는 읽어 온 교차검증 폴더 이름을 붙입니다.
DEFAULT_CV_DIR = PROJECT_DIR / "results" / "timesfm_cv_normalized_exchange_rate"
COMMON_TEST_PATH = PROJECT_DIR / "results" / "regression" / "train_test_exchange_rate" / "common_test_models.csv"
OUT_DIR = PROJECT_DIR / "results" / "regression" / "train_test_exchange_rate"
NAMES = {"recent_margin": "regression 최근 마진 (기점별 재적합)", "baseline": "regression 전체 회귀 (기점별 재적합)",
         "timesfm_cov": "TimesFM 3.0 (공변량)", "timesfm_uni": "TimesFM 3.0 (공변량 없음)", "no_change": "무변화 (기준주 가격 유지)"}
ORDER = ["recent_margin", "baseline", "timesfm_cov", "timesfm_uni", "no_change"]
LEADS = (1, 4, 13, 26)
KO_TO_FUEL = {spec["ko"]: fuel for fuel, spec in FUELS.items()}
TOLERANCE = 0.01




# 3. 교차검증 예측 파일을 읽어 (제품, 폴드, 주차) 단위의 넓은 표로 바꿉니다. 모형별 예측이 열이 됩니다.
def load_cv_predictions(path):
    if not path.is_file():
        raise FileNotFoundError(f"교차검증 예측 파일이 없습니다. 먼저 python 딥러닝/cv_timesfm.py 를 실행하세요: {path}")
    long = pd.read_csv(path, encoding="utf-8-sig")
    missing = [m for m in ORDER if m not in set(long["모형"])]
    if missing:
        raise ValueError(f"예측 파일에 모형이 없습니다: {', '.join(missing)}. 딥러닝/cv_timesfm.py를 다시 실행하세요")
    keys = ["제품", "폴드", "문맥길이", "기준주차", "주차", "리드"]
    wide = long.pivot_table(index=keys, columns="모형", values="예측_세전가격").reset_index()
    actual = long.drop_duplicates(keys).set_index(keys)["실제_세전가격"]
    wide["actual"] = actual.loc[list(map(tuple, wide[keys].to_numpy()))].to_numpy()
    return wide.sort_values(["제품", "폴드", "리드"]).reset_index(drop=True)




# 4. 각 창에 대해 counterfactual.py 경로로 regression 모형을 다시 적합해 예측 파일의 값과 비교합니다.
# 학습 행 조건은 counterfactual.training_data와 같습니다(정책 이전, 시차 국제가·세전가 모두 있음).
def check_regression(wide, weekly, product):
    fuel = KO_TO_FUEL[product]
    X_all = lagged_intl(weekly, fuel)
    y_all = weekly[f"{fuel}_supply_pretax"]
    complete = X_all.notna().all(axis=1) & y_all.notna() & ~weekly["in_policy"]
    part = wide[wide["제품"] == product]
    worst = 0.0
    for fold, rows in part.groupby("폴드"):
        origin = week_end(rows["기준주차"].iloc[0])
        train = complete & (weekly.index <= origin)
        targets = pd.DatetimeIndex([week_end(label) for label in rows["주차"]])
        if targets.min() <= origin or X_all.loc[targets].isna().any().any():
            raise ValueError(f"{product} 폴드 {fold}: 예측 구간이 기준주차 이전이거나 국제가가 비어 있습니다")
        if not np.allclose(y_all.loc[targets].to_numpy(), rows["actual"].to_numpy(), atol=TOLERANCE):
            raise ValueError(f"{product} 폴드 {fold}: regression 자료의 실제값이 예측 파일과 다릅니다")
        for model in MODELS:
            params = fit(X_all[train], y_all[train], model)
            diff = np.abs(predict(params, X_all.loc[targets]) - rows[model].to_numpy()).max()
            worst = max(worst, float(diff))
            if diff > TOLERANCE:
                raise ValueError(f"{product} 폴드 {fold} {model}: 딥러닝 경로와 counterfactual 경로의 예측이 다릅니다 (최대 {diff:.3f}원)")
    return worst




# 5. 지표입니다. 편향 = 예측 − 실제의 평균. skill = 1 − exp(평균 log(MAE / 무변화 MAE)), fev-bench 방식.
def metrics(actual, predicted):
    error = np.asarray(predicted, dtype=float) - np.asarray(actual, dtype=float)
    sst = ((actual - actual.mean()) ** 2).sum()
    return {"n": len(error), "R2": 1 - (error ** 2).sum() / sst if sst else np.nan,
            "RMSE": math.sqrt((error ** 2).mean()), "MAE": np.abs(error).mean(), "bias": error.mean(),
            "MAPE": (np.abs(error) / np.abs(actual)).mean() * 100}


def fold_table(table):
    rows = []
    for (product, fold), part in table.groupby(["제품", "폴드"], sort=False):
        for model in ORDER:
            rows.append({"제품": product, "폴드": fold, "기준주차": part["기준주차"].iloc[0], "모형": model,
                         **metrics(part["actual"].to_numpy(), part[model].to_numpy())})
    return pd.DataFrame(rows)


def summary_table(folds, table):
    rows = []
    for product in folds["제품"].unique():
        part = folds[folds["제품"] == product].set_index(["모형", "폴드"])
        naive = part.loc["no_change", "MAE"]
        reference = part.loc[CHOSEN, "MAE"]
        pooled = table[table["제품"] == product]
        for model in ORDER:
            m = part.loc[model]
            whole = metrics(pooled["actual"].to_numpy(), pooled[model].to_numpy())
            diff = (m["MAE"] - reference).to_numpy()
            rows.append({"제품": product, "모형": model, "기점수": len(m),
                         "MAE평균": m["MAE"].mean(), "MAE표준편차": m["MAE"].std(ddof=1),
                         "RMSE평균": m["RMSE"].mean(), "MAPE평균": m["MAPE"].mean(), "편향평균": m["bias"].mean(),
                         "전체MAE": whole["MAE"], "전체RMSE": whole["RMSE"], "전체R2": whole["R2"],
                         "skill_vs_무변화": 1 - math.exp(np.log(np.maximum(m["MAE"], 1e-9) / np.maximum(naive, 1e-9)).mean()),
                         "무변화승수": int((m["MAE"] < naive).sum()),
                         "최근마진대비_MAE차_평균": diff.mean(),
                         "최근마진보다_좋은_기점": int((diff < 0).sum())})
    return pd.DataFrame(rows)


def lead_table(table, leads):
    rows = []
    for product in table["제품"].unique():
        part = table[table["제품"] == product]
        for model in ORDER:
            row = {"제품": product, "모형": model}
            for lead in leads:
                sub = part[part["리드"] == lead]
                row[f"MAE_{lead}"] = (sub[model] - sub["actual"]).abs().mean()
            rows.append(row)
    return pd.DataFrame(rows)




# 6. 공통 평가셋 결과(apply_common_test.py)를 읽어 보고서에 붙일 표를 만듭니다. 파일이 없으면 생략합니다.
def load_common_test(path=COMMON_TEST_PATH):
    if not path.is_file():
        return None
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame["유종"] = frame["fuel"].map({fuel: spec["ko"] for fuel, spec in FUELS.items()})
    return frame




# 7. 보고서를 마크다운으로 정리합니다.
def save_report(folds, summary, leads, lead_rows, common, worst, cv_dir, horizon, step, path):
    first = folds[folds["모형"] == "no_change"].drop_duplicates(["폴드"])
    lines = ["# 학습셋 교차검증과 공통 평가셋에서의 예측오차 비교", "",
             "## 1. 설계", "",
             "- 자료·변수·분할: 국제가 같은 주·1주 전·2주 전과 같은 주 환율을 함께 사용하며 정책 이전만 평가합니다.",
             f"- 평면 A (학습셋 안 교차검증): `{cv_dir.relative_to(PROJECT_DIR).as_posix()}`의 기점 {first['폴드'].nunique()}개 × 예측 {horizon}주, 기점 간격 {step}주 "
             f"({first['기준주차'].iloc[0]} 기준 ~ {first['기준주차'].iloc[-1]} 기준). 기점마다 그 이전 자료만 사용.",
             f"- 회귀 재적합 점검: 예측 파일의 회귀 값을 counterfactual 경로로 다시 적합해 비교. 최대 차이 {worst:.4f}원 (허용 {TOLERANCE}원).",
             "- 평면 B (공통 평가셋): `apply_common_test.py`의 정책 이전 뒤 20% (`split_regression.py`와 같은 경계). 학습 경계에서 한 번 고정.",
             "- 모든 모형에 예측 구간의 **실제** 국제가를 넣었으므로 국제가가 주어진 조건부 예측입니다.",
             "- 편향 = 예측 − 실제. skill = 1 − exp(평균 log(MAE / 무변화 MAE)).",
             f"- 기점별 차이의 유의성 검정(Diebold-Mariano)은 `{(cv_dir / 'cv_report.md').relative_to(PROJECT_DIR).as_posix()}`에 있습니다.",
             "", f"## 2. 평면 A: 학습셋 안 기점 {first['폴드'].nunique()}개 (세전 원/리터)", "",
             "| 제품 | 모형 | 기점별 MAE 평균 ± 표준편차 | RMSE 평균 | MAPE 평균 | 편향 평균 | 전체 MAE | skill vs 무변화 | 최근 마진 대비 MAE 차 | 최근 마진보다 좋은 기점 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for _, s in summary.iterrows():
        lines.append(f"| {s['제품']} | {NAMES[s['모형']]} | {s['MAE평균']:.1f} ± {s['MAE표준편차']:.1f} | {s['RMSE평균']:.1f} | "
                     f"{s['MAPE평균']:.2f}% | {s['편향평균']:+.1f} | {s['전체MAE']:.1f} | {s['skill_vs_무변화']:+.3f} | "
                     f"{s['최근마진대비_MAE차_평균']:+.1f} | {s['최근마진보다_좋은_기점']}/{s['기점수']} |")
    lines += ["", f"### 예측 거리별 MAE (기점 {first['폴드'].nunique()}개 평균)", "",
              "| 제품 | 모형 | " + " | ".join(f"{lead}주 앞" for lead in leads) + " |",
              "| --- | --- | " + " | ".join("---" for _ in leads) + " |"]
    for _, r in lead_rows.iterrows():
        lines.append(f"| {r['제품']} | {NAMES[r['모형']]} | " + " | ".join(f"{r[f'MAE_{lead}']:.1f}" for lead in leads) + " |")
    lines += ["", "### 기점별 MAE", "",
              "| 제품 | 폴드 | 기준주차 | 최근 마진 | 전체 회귀 | TimesFM | 무변화 | TimesFM − 최근 마진 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    pivot = folds.pivot_table(index=["제품", "폴드", "기준주차"], columns="모형", values="MAE", sort=False)
    for (product, fold, origin), r in pivot.iterrows():
        lines.append(f"| {product} | {fold} | {origin} | {r['recent_margin']:.1f} | {r['baseline']:.1f} | "
                     f"{r['timesfm_cov']:.1f} | {r['no_change']:.1f} | {r['timesfm_cov'] - r['recent_margin']:+.1f} |")
    if common is not None:
        for segment in ["정책 이전", "전체", "정책 기간"]:
            part = common[common["segment"] == segment]
            if part.empty:
                continue
            lines += ["", f"## 3. 평면 B: 공통 평가셋 — {segment} (세전 원/리터, `common_test_models.csv`)", "",
                      "| 유종 | 모형 | 주 | R² | RMSE | MAE | 편향 | MAPE |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
            for _, r in part.iterrows():
                lines.append(f"| {r['유종']} | {r['name']} | {int(r['n'])} | {r['R2']:.3f} | {r['RMSE']:.1f} | {r['MAE']:.1f} | {r['bias']:+.1f} | {r['MAPE']:.2f}% |")
    lines += ["", "## 4. 읽는 법", "",
              "- 평면 A와 B의 MAE를 직접 비교하지 마세요. A는 기점마다 재적합하고 예측 거리가 짧으며, B는 학습 경계에서 한 번 고정해 끝까지 밀고 갑니다.",
              "- 평면 A 안에서는 두 모형이 같은 기점·같은 주·같은 국제가 입력으로 채점됐으므로 그대로 비교할 수 있습니다.",
              "- 평면 B는 regression 폴더 `split_report.md`의 평가 구간과 같은 주이므로 그 보고서 수치와 직접 비교할 수 있습니다.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 8. 읽기 → 회귀 재적합 점검 → 채점 → 저장 → 요약 출력 순서로 진행합니다.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-dir", type=Path, default=DEFAULT_CV_DIR,
                        help="딥러닝/cv_timesfm.py 결과 폴더 (기본: results/timesfm_cv_aligned)")
    args = parser.parse_args()
    cv_dir = args.cv_dir if args.cv_dir.is_absolute() else PROJECT_DIR / args.cv_dir
    suffix = "" if cv_dir.name == DEFAULT_CV_DIR.name else f"_{cv_dir.name}"
    try:
        weekly = load_weekly()
        table = load_cv_predictions(cv_dir / "cv_predictions.csv")
        worst = max(check_regression(table, weekly, product) for product in table["제품"].unique())
        horizon = int(table["리드"].max())
        contexts = sorted(table["문맥길이"].unique())
        step = int(contexts[1] - contexts[0]) if len(contexts) > 1 else horizon
        leads = sorted({lead for lead in (*LEADS, horizon) if lead <= horizon})
        folds = fold_table(table)
        summary = summary_table(folds, table)
        lead_rows = lead_table(table, leads)
        common = load_common_test()
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        table.to_csv(OUT_DIR / f"cv_folds_predictions{suffix}.csv", index=False, encoding="utf-8-sig", float_format="%.2f")
        folds.to_csv(OUT_DIR / f"cv_folds_metrics{suffix}.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
        report = save_report(folds, summary, leads, lead_rows, common, worst, cv_dir, horizon, step,
                             OUT_DIR / f"cv_folds_report{suffix}.md")
    except (OSError, ValueError, KeyError, csv.Error) as error:
        raise SystemExit(f"창별 비교 실패: {error}") from None

    pd.set_option("display.width", 220)
    print(f"평면 A: 기점 {folds['폴드'].nunique()}개 × {horizon}주 (간격 {step}주), 제품 {folds['제품'].nunique()}개 · "
          f"회귀 재적합 점검 최대 차이 {worst:.4f}원. 세전 원/리터\n")
    show = summary.assign(모형=summary["모형"].map(NAMES))[
        ["제품", "모형", "MAE평균", "MAE표준편차", "편향평균", "전체MAE", "skill_vs_무변화", "최근마진대비_MAE차_평균", "최근마진보다_좋은_기점"]]
    print(show.round(2).to_string(index=False))
    print("\n예측 거리별 MAE")
    print(lead_rows.assign(모형=lead_rows["모형"].map(NAMES)).round(1).to_string(index=False))
    if common is None:
        print("\n주의: common_test_models.csv가 없어 평가셋 비교는 보고서에서 빠졌습니다.")
    print(f"\n저장: {report.relative_to(PROJECT_DIR).as_posix()} 및 같은 폴더의 CSV 2개")




# 9. 직접 실행할 때만 비교를 시작합니다.
if __name__ == "__main__":
    main()
