# 1. 제품별 회귀모형 각각에 대해 설명변수의 다중공선성을 VIF(분산팽창계수)로 진단합니다.
# 실행: python "ols 111/vif_analysis.py"
# 같은 폴더의 regression_before_tax.py에서 모형 정의·자료 불러오기·시차 적용·분할·OLS 함수를 그대로 가져옵니다.
# 경유 / 등유 / 휘발유 모형은 설명변수 집합이 서로 다르므로 VIF도 모형마다 따로 계산합니다.
import csv
import math

from regression_before_tax import (
    CURRENT_FEATURES,
    INPUT_PATH,
    LAG_WEEKS,
    MODELS,
    PROJECT_DIR,
    SPLIT_MODE,
    build_design,
    check_structure,
    feature_names,
    fit_ols,
    load_clean_csv,
    predict,
    split_indices,
    take,
)




# 2. 결과 저장 위치와 판단 기준을 정합니다.
# VIF 10 이상이면 심각한 공선성, 5 이상이면 주의가 필요한 수준으로 봅니다.
REPORT_PATH = PROJECT_DIR / "results" / "회귀" / "vif_report.md"
VIF_TABLE_PATH = PROJECT_DIR / "csv" / "vif_table.csv"
VIF_WARNING = 5.0
VIF_SEVERE = 10.0
# 변수를 줄일 때 전체 모형보다 학습 R²이 이 값 이상 떨어지면 정보 손실로 봅니다.
R2_TOLERANCE = 0.01




# 3. 한 변수를 나머지 변수들로 회귀해 R²을 구하고 VIF = 1 / (1 - R²) 로 계산합니다.
# 다른 변수들이 그 변수를 거의 완벽하게 설명하면 R²→1, VIF→∞ 가 됩니다.
# 설명변수가 하나뿐이면 보조회귀를 만들 수 없으므로 VIF = 1(공선성 없음)로 둡니다.
def r_squared(actual, predicted):
    mean = sum(actual) / len(actual)
    sse = sum((a - p) ** 2 for a, p in zip(actual, predicted, strict=True))
    sst = sum((a - mean) ** 2 for a in actual)
    return 1.0 - sse / sst if sst else 0.0


def vif_for_column(features, names, target_index):
    if len(names) == 1:
        return 1.0, 0.0
    others = [[row[j] for j in range(len(names)) if j != target_index] for row in features]
    column = [row[target_index] for row in features]
    try:
        coefficients, _ = fit_ols(others, column)
    except ValueError:
        return float("inf"), 1.0
    fitted = predict(coefficients, others)
    r2 = r_squared(column, fitted)
    vif = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return vif, r2


def compute_vif_table(features, names):
    table = []
    for index, name in enumerate(names):
        vif, r2 = vif_for_column(features, names, index)
        table.append({"변수": name, "VIF": vif, "R2_보조회귀": r2, "허용도": 1.0 / vif if vif else float("nan")})
    return table




# 4. 설명변수끼리의 피어슨 상관계수 행렬을 구합니다.
# VIF가 높을 때 '누구와 누가' 겹치는지는 상관행렬에서 바로 읽을 수 있습니다.
def correlation_matrix(features, names):
    width = len(names)
    columns = [[row[j] for row in features] for j in range(width)]
    means = [sum(col) / len(col) for col in columns]
    stds = [math.sqrt(sum((v - m) ** 2 for v in col) / len(col)) for col, m in zip(columns, means, strict=True)]
    matrix = []
    for a in range(width):
        row = []
        for b in range(width):
            if stds[a] == 0 or stds[b] == 0:
                row.append(float("nan"))
                continue
            cov = sum((x - means[a]) * (y - means[b]) for x, y in zip(columns[a], columns[b], strict=True)) / len(columns[a])
            row.append(cov / (stds[a] * stds[b]))
        matrix.append(row)
    return matrix




# 5. VIF가 가장 높은 변수를 하나씩 제거하면서 모든 변수가 기준 아래로 내려올 때까지 반복합니다.
# 각 단계의 VIF와, 그 변수 집합으로 해당 제품을 회귀했을 때 학습 R²을 함께 기록합니다.
def stepwise_elimination(train_features, train_targets, names, threshold=VIF_SEVERE):
    remaining = list(range(len(names)))
    steps = []
    while True:
        subset = [[row[j] for j in remaining] for row in train_features]
        subset_names = [names[j] for j in remaining]
        table = compute_vif_table(subset, subset_names)
        coefficients, _ = fit_ols(subset, train_targets)
        fit_r2 = r_squared(train_targets, predict(coefficients, subset))
        worst = max(table, key=lambda item: item["VIF"])
        steps.append({"변수": subset_names, "VIF표": table, "학습R2": fit_r2, "제거": None})
        if worst["VIF"] < threshold or len(remaining) <= 1:
            break
        steps[-1]["제거"] = worst["변수"]
        remaining.remove(names.index(worst["변수"]))
    return steps




# 6. 제품 하나의 설명변수에 대해 VIF 표 → 상관행렬 → 단계적 제거를 수행합니다.
# 회귀모형과 똑같이 학습셋만 사용해, 모형이 실제로 본 자료 기준으로 공선성을 진단합니다.
def diagnose_product(product, rows, train_index):
    names = feature_names(product)
    _, features, targets = build_design(rows, product)
    train_features, train_targets = take(features, train_index), take(targets, train_index)
    steps = stepwise_elimination(train_features, train_targets, names)
    full_r2 = steps[0]["학습R2"]
    affordable = [step for step in steps if step["학습R2"] >= full_r2 - R2_TOLERANCE]
    return {
        "제품": product,
        "타겟": MODELS[product]["타겟"],
        "설명변수": names,
        "VIF표": compute_vif_table(train_features, names),
        "상관행렬": correlation_matrix(train_features, names),
        "단계": steps,
        "권장": affordable[-1],
    }




# 7. VIF 값에 따라 판정 문구를 붙입니다.
def judge(vif):
    if vif == float("inf"):
        return "완전공선성"
    if vif >= VIF_SEVERE:
        return "심각"
    if vif >= VIF_WARNING:
        return "주의"
    return "양호"


def fmt(value, digits=2):
    if value == float("inf"):
        return "∞"
    if value != value:
        return "-"
    return f"{value:.{digits}f}"


def short_name(name):
    return name.replace("국제_석유제품가격(원)_", "국제 ").replace("원달러환율_주간평균_원/달러 환율", "환율")




# 8. 세 모형의 VIF 표를 하나의 CSV로 저장합니다(모형 열로 구분).
def save_vif_csv(diagnoses, path=VIF_TABLE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["모형", "변수", "VIF", "허용도", "R2_보조회귀", "판정"])
        writer.writeheader()
        for diagnosis in diagnoses:
            for item in diagnosis["VIF표"]:
                writer.writerow(
                    {
                        "모형": diagnosis["제품"],
                        "변수": item["변수"],
                        "VIF": fmt(item["VIF"], 4),
                        "허용도": fmt(item["허용도"], 6),
                        "R2_보조회귀": fmt(item["R2_보조회귀"], 6),
                        "판정": judge(item["VIF"]),
                    }
                )
    return path




# 9. 제품별 진단 결과를 마크다운 보고서로 정리합니다.
def product_section(number, diagnosis):
    names = diagnosis["설명변수"]
    short = [short_name(n) for n in names]
    lines = [
        f"## {number}. {diagnosis['제품']} 모형",
        "",
        f"- 타겟변수: `{diagnosis['타겟']}`",
        "- 설명변수: " + ", ".join(f"`{n}`" for n in names),
        "",
        "### VIF 표",
        "",
        "| 변수 | VIF | 허용도(1/VIF) | 보조회귀 R² | 판정 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for item in diagnosis["VIF표"]:
        lines.append(
            f"| {item['변수']} | {fmt(item['VIF'], 1)} | {fmt(item['허용도'], 4)} | "
            f"{fmt(item['R2_보조회귀'], 4)} | {judge(item['VIF'])} |"
        )

    lines += ["", "### 상관행렬", "", "| | " + " | ".join(short) + " |", "| --- |" + " --- |" * len(short)]
    for name, row in zip(short, diagnosis["상관행렬"], strict=True):
        lines.append(f"| {name} | " + " | ".join(fmt(v, 3) for v in row) + " |")

    steps = diagnosis["단계"]
    lines += ["", f"### 단계적 제거 (VIF 최대 변수부터, 모든 VIF < {VIF_SEVERE:.0f}까지)", ""]
    if len(steps) == 1:
        lines += [f"모든 변수의 VIF가 {VIF_SEVERE:.0f} 미만이므로 제거할 변수가 없습니다. 학습 R² {steps[0]['학습R2']:.4f}.", ""]
    else:
        lines += ["| 단계 | 남은 변수 | 최대 VIF | 학습 R² | 제거 |", "| --- | --- | --- | --- | --- |"]
        for index, step in enumerate(steps, start=1):
            worst = max(item["VIF"] for item in step["VIF표"])
            removed = f"`{step['제거']}`" if step["제거"] else "(종료)"
            lines.append(
                f"| {index} | " + ", ".join(short_name(n) for n in step["변수"]) + f" | {fmt(worst, 1)} | {step['학습R2']:.4f} | {removed} |"
            )
        lines.append("")

    recommended = diagnosis["권장"]
    max_vif = max(item["VIF"] for item in recommended["VIF표"])
    worst_full = max(item["VIF"] for item in diagnosis["VIF표"])
    lines += ["### 판단", ""]
    if worst_full < VIF_SEVERE:
        lines.append(f"- 공선성 문제 없음(최대 VIF {fmt(worst_full, 1)}). 계수를 그대로 해석해도 됩니다.")
    else:
        lines.append(f"- 최대 VIF {fmt(worst_full, 1)}로 심각한 공선성이 있습니다. 개별 계수의 부호·크기를 따로 해석하면 안 됩니다.")
        if len(recommended["변수"]) < len(names):
            lines.append(
                f"- R² 손실 {R2_TOLERANCE} 이내로 줄일 수 있는 가장 작은 집합: "
                + ", ".join(f"`{n}`" for n in recommended["변수"])
                + f" (학습 R² {recommended['학습R2']:.4f}, 최대 VIF {fmt(max_vif, 1)}). 계수 해석이 필요하면 이 집합을 권합니다."
            )
        else:
            lines.append(f"- 변수를 하나라도 빼면 학습 R²이 {R2_TOLERANCE} 이상 떨어지므로 전체 집합을 유지하는 편이 낫습니다.")
        lines.append("- 예측만 목적이면 공선성은 적합값에 영향을 주지 않으므로 전체 집합을 그대로 써도 됩니다.")
    lines.append("")
    return lines


def save_report(diagnoses, sample_size, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 제품별 회귀모형 다중공선성(VIF) 진단 보고서",
        "",
        "## 0. 분석 설계",
        "",
        f"- 자료: `{INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- 표본: 학습셋 {sample_size}주 ({SPLIT_MODE} 분할의 앞 80%) — 회귀계수를 추정하는 자료와 동일",
        f"- 설명변수: 해당 제품의 국제가격 ({LAG_WEEKS}주 시차, t-{LAG_WEEKS}) + 환율 (같은 주 값, 모든 모형 공통)",
        "- 세 모형은 설명변수 집합이 서로 다르므로 VIF를 모형마다 따로 계산합니다. 모형 간 비교는 하지 않습니다.",
        "- VIFⱼ = 1 / (1 − R²ⱼ), R²ⱼ는 변수 j를 같은 모형의 나머지 변수들로 회귀했을 때의 결정계수",
        f"- 판정 기준: VIF ≥ {VIF_SEVERE:.0f} 심각 · {VIF_WARNING:.0f} ≤ VIF < {VIF_SEVERE:.0f} 주의 · VIF < {VIF_WARNING:.0f} 양호",
        "",
        "| 모형 | 설명변수 수 | 최대 VIF | 판정 |",
        "| --- | --- | --- | --- |",
    ]
    for diagnosis in diagnoses:
        worst = max(item["VIF"] for item in diagnosis["VIF표"])
        lines.append(f"| {diagnosis['제품']} | {len(diagnosis['설명변수'])} | {fmt(worst, 1)} | {judge(worst)} |")
    lines.append("")
    for number, diagnosis in enumerate(diagnoses, start=1):
        lines += product_section(number, diagnosis)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 10. 불러오기 → 구조 점검 → 시차 적용 → 학습셋 추출 → 제품별 VIF 진단 → 저장 순서로 진행합니다.
def main():
    try:
        columns, rows = load_clean_csv(INPUT_PATH)
        errors = check_structure(columns, rows)
        if errors:
            raise ValueError("구조 점검 실패:\n" + "\n".join(errors[:20]))
        train_index, _ = split_indices(len(rows) - LAG_WEEKS)
        diagnoses = [diagnose_product(product, rows, train_index) for product in MODELS]
        csv_path = save_vif_csv(diagnoses)
        report_path = save_report(diagnoses, len(train_index))
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise SystemExit(f"VIF 분석 실패: {error}") from None

    print(f"학습셋 {len(train_index)}주 기준, 제품별 모형 {len(MODELS)}개 (국제가격 t-{LAG_WEEKS}, 환율 t)")
    for diagnosis in diagnoses:
        print()
        print(f"[{diagnosis['제품']}] 설명변수 {len(diagnosis['설명변수'])}개")
        for item in diagnosis["VIF표"]:
            print(f"   {item['변수']:<44} VIF={fmt(item['VIF'], 1):>8}  {judge(item['VIF'])}")
        steps = diagnosis["단계"]
        if len(steps) > 1:
            print(f"   단계적 제거: " + " → ".join(f"{len(s['변수'])}개(R²={s['학습R2']:.4f})" for s in steps))
        recommended = diagnosis["권장"]
        print(f"   권장 집합: " + ", ".join(short_name(n) for n in recommended["변수"]) + f"  (R²={recommended['학습R2']:.4f})")
    print()
    print(f"보고서 저장: {report_path}")
    print(f"VIF 표 저장: {csv_path}")




# 11. 직접 실행할 때만 분석을 수행합니다.
if __name__ == "__main__":
    main()
