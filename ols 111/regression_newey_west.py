# 1. 제품별 OLS 회귀에 Newey-West(HAC) 표준오차를 적용해 p값과 신뢰구간을 보정합니다.
# 자기상관 진단(results/회귀/autocorrelation_report.md)에서 세 모형 모두 잔차에 강한 양의 자기상관이 확인됐습니다.
# 자기상관이 있어도 OLS 계수는 불편(unbiased)이지만 표준오차가 과소추정되어 p값이 실제보다 작게 나옵니다.
# Newey-West는 계수는 그대로 두고 분산-공분산 행렬만 자기상관·이분산에 강건하게 다시 계산합니다.
# 실행: python "ols 111/regression_newey_west.py"
import csv
import math

from regression_before_tax import (
    INPUT_PATH,
    LAG_WEEKS,
    MODELS,
    PROJECT_DIR,
    SPLIT_MODE,
    build_design,
    check_structure,
    evaluate,
    feature_names,
    fit_ols,
    load_clean_csv,
    predict,
    split_indices,
    take,
    two_sided_p_value,
)




# 2. 결과 저장 위치와 HAC 설정입니다.
# HAC_LAG=None이면 Newey-West(1994) 경험칙 L = floor(4·(n/100)^(2/9)) 를 씁니다(n=488이면 5).
# 잔차 ACF가 20주차까지 유의했으므로, 더 긴 시차에서도 결론이 유지되는지 SENSITIVITY_LAGS로 함께 확인합니다.
REPORT_PATH = PROJECT_DIR / "results" / "회귀" / "regression_newey_west_report.md"
COEF_TABLE_PATH = PROJECT_DIR / "csv" / "regression_newey_west_coefficients.csv"
HAC_LAG = None
SENSITIVITY_LAGS = [0, 5, 10, 20]
SMALL_SAMPLE_CORRECTION = True




# 3. Newey-West 경험칙 시차입니다.
def newey_west_bandwidth(count):
    return int(math.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))




# 4. 행렬 보조 연산입니다(설계행렬이 작아 순수 파이썬으로 충분합니다).
def outer(a, b):
    return [[x * y for y in b] for x in a]


def add_into(target, source, scale=1.0):
    for i, row in enumerate(source):
        for j, value in enumerate(row):
            target[i][j] += scale * value


def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]




# 5. Newey-West HAC 분산-공분산 행렬을 계산합니다.
# V = (X'X)^-1 · S · (X'X)^-1,
# S = Σ_t u_t² x_t x_t'  +  Σ_{l=1}^{L} w_l Σ_{t>l} u_t u_{t-l} (x_t x_{t-l}' + x_{t-l} x_t'),
# w_l = 1 − l/(L+1) (Bartlett 가중치). L=0이면 White의 이분산-강건 분산과 같습니다.
# SMALL_SAMPLE_CORRECTION이 켜져 있으면 n/(n−k)를 곱해 소표본을 보정합니다(statsmodels 기본값과 동일).
def newey_west_covariance(features, residuals, gram_inverse, lag):
    design = [[1.0, *row] for row in features]
    count, width = len(design), len(design[0])
    meat = [[0.0] * width for _ in range(width)]
    for t in range(count):
        add_into(meat, outer(design[t], design[t]), residuals[t] ** 2)
    for l in range(1, lag + 1):
        weight = 1.0 - l / (lag + 1.0)
        for t in range(l, count):
            cross = residuals[t] * residuals[t - l]
            block = outer(design[t], design[t - l])
            add_into(meat, block, weight * cross)
            add_into(meat, [list(col) for col in zip(*block, strict=True)], weight * cross)
    covariance = matmul(matmul(gram_inverse, meat), gram_inverse)
    if SMALL_SAMPLE_CORRECTION:
        scale = count / (count - width)
        covariance = [[value * scale for value in row] for row in covariance]
    return covariance




# 6. OLS 표준오차(비교용)와 HAC 표준오차를 같은 계수에 대해 계산합니다.
def coefficient_table(coefficients, names, ols_variance_diag, hac_variance_diag, degrees_of_freedom):
    labels = ["절편(상수)", *names]
    table = []
    for index, (name, value) in enumerate(zip(labels, coefficients, strict=True)):
        ols_se = math.sqrt(max(ols_variance_diag[index], 0.0))
        hac_se = math.sqrt(max(hac_variance_diag[index], 0.0))
        ols_t = value / ols_se if ols_se else float("nan")
        hac_t = value / hac_se if hac_se else float("nan")
        table.append(
            {
                "변수": name,
                "계수": value,
                "OLS_표준오차": ols_se,
                "OLS_t": ols_t,
                "OLS_p": two_sided_p_value(ols_t, degrees_of_freedom),
                "HAC_표준오차": hac_se,
                "HAC_t": hac_t,
                "HAC_p": two_sided_p_value(hac_t, degrees_of_freedom),
                "HAC_하한95": value - 1.96 * hac_se,
                "HAC_상한95": value + 1.96 * hac_se,
                "표준오차_배율": hac_se / ols_se if ols_se else float("nan"),
            }
        )
    return table




# 7. 제품 하나에 대해 OLS 적합 → HAC 분산 → 계수표 → 시차 민감도를 계산합니다.
def run_single_model(product, rows, train_index, test_index):
    names = feature_names(product)
    labels, features, targets = build_design(rows, product)
    train_features, train_targets = take(features, train_index), take(targets, train_index)
    test_features, test_targets = take(features, test_index), take(targets, test_index)

    coefficients, gram_inverse = fit_ols(train_features, train_targets)
    fitted = predict(coefficients, train_features)
    residuals = [a - p for a, p in zip(train_targets, fitted, strict=True)]
    count, width = len(train_targets), len(coefficients)
    degrees_of_freedom = count - width
    sigma_squared = sum(r * r for r in residuals) / degrees_of_freedom
    ols_diag = [sigma_squared * gram_inverse[i][i] for i in range(width)]

    lag = newey_west_bandwidth(count) if HAC_LAG is None else HAC_LAG
    hac = newey_west_covariance(train_features, residuals, gram_inverse, lag)
    hac_diag = [hac[i][i] for i in range(width)]
    table = coefficient_table(coefficients, names, ols_diag, hac_diag, degrees_of_freedom)

    sensitivity = []
    for trial_lag in SENSITIVITY_LAGS:
        trial = newey_west_covariance(train_features, residuals, gram_inverse, trial_lag)
        trial_diag = [trial[i][i] for i in range(width)]
        sensitivity.append(
            {
                "시차": trial_lag,
                "표": coefficient_table(coefficients, names, ols_diag, trial_diag, degrees_of_freedom),
            }
        )

    return {
        "제품": product,
        "타겟": MODELS[product]["타겟"],
        "설명변수": names,
        "HAC시차": lag,
        "계수표": table,
        "민감도": sensitivity,
        "자유도": degrees_of_freedom,
        "학습성능": evaluate(train_targets, fitted),
        "평가성능": evaluate(test_targets, predict(coefficients, test_features)),
        "학습기간": (labels[train_index[0]], labels[train_index[-1]]),
        "평가기간": (labels[test_index[0]], labels[test_index[-1]]),
    }




# 8. 수치 표기와 유의성 표시입니다.
def fmt(value, digits=4):
    if value != value:
        return "-"
    return f"{value:.{digits}f}"


def stars(p):
    if p != p:
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""




# 9. 세 모형의 계수표를 CSV로 저장합니다.
def save_coefficients_csv(results, path=COEF_TABLE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "모형", "변수", "계수",
        "OLS_표준오차", "OLS_t", "OLS_p",
        "HAC_표준오차", "HAC_t", "HAC_p", "HAC_하한95", "HAC_상한95",
        "표준오차_배율", "HAC_시차",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            for item in result["계수표"]:
                writer.writerow(
                    {
                        "모형": result["제품"],
                        "변수": item["변수"],
                        "계수": fmt(item["계수"]),
                        "OLS_표준오차": fmt(item["OLS_표준오차"]),
                        "OLS_t": fmt(item["OLS_t"], 3),
                        "OLS_p": fmt(item["OLS_p"]),
                        "HAC_표준오차": fmt(item["HAC_표준오차"]),
                        "HAC_t": fmt(item["HAC_t"], 3),
                        "HAC_p": fmt(item["HAC_p"]),
                        "HAC_하한95": fmt(item["HAC_하한95"]),
                        "HAC_상한95": fmt(item["HAC_상한95"]),
                        "표준오차_배율": fmt(item["표준오차_배율"], 2),
                        "HAC_시차": result["HAC시차"],
                    }
                )
    return path




# 10. 보고서를 마크다운으로 정리합니다.
def product_section(number, result):
    lines = [
        f"## {number}. {result['제품']} 모형",
        "",
        f"- 타겟변수: `{result['타겟']}`",
        "- 설명변수: " + ", ".join(f"`{n}`" for n in result["설명변수"]),
        f"- 학습 기간: {result['학습기간'][0]} ~ {result['학습기간'][1]} ({result['학습성능']['n']}주), 잔차 자유도 {result['자유도']}",
        f"- HAC 시차 L = {result['HAC시차']} (Newey-West 경험칙), Bartlett 가중치, 소표본 보정 n/(n−k) 적용",
        f"- 학습 R² {result['학습성능']['R2']:.4f} / 평가 R² {result['평가성능']['R2']:.4f} (계수가 같으므로 OLS와 동일)",
        "",
        "### 계수표: OLS 표준오차 vs Newey-West 표준오차",
        "",
        "| 변수 | 계수 | OLS SE | OLS p | HAC SE | HAC p | HAC 95% 신뢰구간 | SE 배율 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in result["계수표"]:
        lines.append(
            f"| {item['변수']} | {fmt(item['계수'])} | {fmt(item['OLS_표준오차'])} | {fmt(item['OLS_p'])}{stars(item['OLS_p'])} | "
            f"**{fmt(item['HAC_표준오차'])}** | **{fmt(item['HAC_p'])}** {stars(item['HAC_p'])} | "
            f"[{fmt(item['HAC_하한95'], 3)}, {fmt(item['HAC_상한95'], 3)}] | ×{fmt(item['표준오차_배율'], 2)} |"
        )
    lines += [
        "",
        "`*` p<0.05, `**` p<0.01, `***` p<0.001. SE 배율 = HAC SE ÷ OLS SE (1보다 크면 OLS가 표준오차를 과소추정했다는 뜻).",
        "",
        "### HAC 시차 민감도 (p값)",
        "",
        "| 변수 | " + " | ".join(f"L={s['시차']}" for s in result["민감도"]) + " |",
        "| --- |" + " --- |" * len(result["민감도"]),
    ]
    for index, item in enumerate(result["계수표"]):
        cells = [f"{fmt(s['표'][index]['HAC_p'])}{stars(s['표'][index]['HAC_p'])}" for s in result["민감도"]]
        lines.append(f"| {item['변수']} | " + " | ".join(cells) + " |")
    lines += ["", "L=0은 White 이분산-강건 표준오차(자기상관 미고려)입니다. 시차를 늘려도 유의성 결론이 유지되면 강건한 결과입니다.", ""]
    return lines


def save_report(results, total_rows, train_size, test_size, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 세전 공급가격 회귀분석 보고서 — Newey-West(HAC) 표준오차 적용",
        "",
        "## 0. 분석 설계",
        "",
        f"- 자료: `{INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}` (시차 적용 후 {total_rows}주)",
        f"- 모형·분할·계수: `ols 111/regression_before_tax.py`와 동일 (제품별 국제가격 t-{LAG_WEEKS} + 환율 t, "
        f"{SPLIT_MODE} 분할 학습 {train_size}주 / 평가 {test_size}주)",
        "- 달라진 점: 표준오차를 Newey-West HAC 분산으로 계산해 **잔차 자기상관·이분산에 강건한 p값과 신뢰구간**을 제시",
        "- 근거: `results/회귀/autocorrelation_report.md` — 세 모형 모두 잔차 Durbin-Watson 0.87~1.12, Ljung-Box p≈0",
        "- 계수 추정치, R², RMSE 등 적합 성능은 OLS와 완전히 같습니다. 바뀌는 것은 '그 계수가 얼마나 확실한가'뿐입니다.",
        "",
        "### 요약",
        "",
        "| 모형 | 변수 | 계수 | OLS p | HAC p | 판정 변화 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        for item in result["계수표"][1:]:
            before = "유의" if item["OLS_p"] < 0.05 else "비유의"
            after = "유의" if item["HAC_p"] < 0.05 else "비유의"
            change = "변화 없음" if before == after else f"**{before} → {after}**"
            lines.append(
                f"| {result['제품']} | {item['변수']} | {fmt(item['계수'], 3)} | {fmt(item['OLS_p'])}{stars(item['OLS_p'])} | "
                f"{fmt(item['HAC_p'])}{stars(item['HAC_p'])} | {change} |"
            )
    lines.append("")
    for number, result in enumerate(results, start=1):
        lines += product_section(number, result)

    lines += ["## 해석", ""]
    ratios = [item["표준오차_배율"] for result in results for item in result["계수표"][1:]]
    lines.append(
        f"- HAC 표준오차는 OLS 표준오차의 {min(ratios):.1f}~{max(ratios):.1f}배입니다. "
        "잔차 자기상관 때문에 OLS가 그만큼 불확실성을 과소평가하고 있었다는 뜻입니다."
    )
    for result in results:
        price = result["계수표"][1]
        fx = result["계수표"][2]
        lines.append(
            f"- {result['제품']}: 국제가격 계수 {fmt(price['계수'], 3)}는 HAC 기준으로도 p={fmt(price['HAC_p'])}로 매우 유의하며 "
            f"95% 신뢰구간 [{fmt(price['HAC_하한95'], 2)}, {fmt(price['HAC_상한95'], 2)}]"
            + ("이 1을 포함합니다 → 전가율 1(완전 전가)과 통계적으로 구분되지 않습니다." if price["HAC_하한95"] <= 1.0 <= price["HAC_상한95"] else "이 1을 포함하지 않습니다.")
            + f" 환율 계수는 HAC p={fmt(fx['HAC_p'])}로 "
            + ("유의합니다." if fx["HAC_p"] < 0.05 else "유의하지 않습니다.")
        )
    lines += [
        "- 발표 시 문구 예시: \"잔차에 자기상관이 확인되어(Durbin-Watson 0.87~1.12) Newey-West HAC 표준오차를 사용했다. "
        "국제가격의 국내 세전가격 전가율은 세 제품 모두 1에 가깝고 HAC 기준으로도 1% 수준에서 유의하다.\"",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 11. 불러오기 → 구조 점검 → 80/20 분할 → 제품별 OLS+HAC → 저장 순서로 진행합니다.
def main():
    try:
        columns, rows = load_clean_csv(INPUT_PATH)
        errors = check_structure(columns, rows)
        if errors:
            raise ValueError("구조 점검 실패:\n" + "\n".join(errors[:20]))
        usable_rows = len(rows) - LAG_WEEKS
        train_index, test_index = split_indices(usable_rows)
        results = [run_single_model(product, rows, train_index, test_index) for product in MODELS]
        csv_path = save_coefficients_csv(results)
        report_path = save_report(results, usable_rows, len(train_index), len(test_index))
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise SystemExit(f"Newey-West 회귀분석 실패: {error}") from None

    print(f"학습 {len(train_index)}주 / 평가 {len(test_index)}주, 제품별 모형 {len(MODELS)}개, Newey-West HAC 표준오차")
    for result in results:
        print()
        print(f"[{result['제품']}] HAC 시차 L={result['HAC시차']}  학습 R²={result['학습성능']['R2']:.4f}")
        print(f"   {'변수':<40} {'계수':>9} {'OLS SE':>8} {'OLS p':>7} {'HAC SE':>8} {'HAC p':>7} {'배율':>5}")
        for item in result["계수표"]:
            print(
                f"   {item['변수']:<40} {item['계수']:>9.4f} {item['OLS_표준오차']:>8.4f} {item['OLS_p']:>7.4f} "
                f"{item['HAC_표준오차']:>8.4f} {item['HAC_p']:>7.4f} {item['표준오차_배율']:>5.2f} {stars(item['HAC_p'])}"
            )
    print()
    print(f"보고서 저장: {report_path}")
    print(f"계수표 저장: {csv_path}")




# 12. 직접 실행할 때만 분석을 수행합니다.
if __name__ == "__main__":
    main()
