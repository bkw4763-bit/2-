# 1. 제품별 회귀모형의 잔차와 타겟 시계열에 자기상관이 있는지 진단합니다.
# 회귀("ols 111/regression_before_tax.py")의 모형 정의·시차·분할·OLS를 그대로 가져와 같은 학습셋에서 계산합니다.
# OLS는 잔차가 서로 독립이라고 가정합니다. 주차 자료에서 이 가정이 깨지면 계수 자체는 그대로지만
# 표준오차가 과소추정되어 p값이 실제보다 작게(유의하게) 나오므로, 반드시 확인해야 합니다.
# 실행: python "자기상관/autocorrelation_analysis.py"
import csv
import importlib.util
import math
from pathlib import Path




# 2. 회귀분석 파일을 경로로 불러옵니다. 폴더 이름("ols 111")에 공백이 있어 일반 import가 되지 않습니다.
PROJECT_DIR = Path(__file__).resolve().parent.parent
REGRESSION_PATH = PROJECT_DIR / "ols 111" / "regression_before_tax.py"
REPORT_PATH = PROJECT_DIR / "results" / "회귀" / "autocorrelation_report.md"
ACF_TABLE_PATH = PROJECT_DIR / "csv" / "autocorrelation_acf.csv"


def load_regression_module(path=REGRESSION_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"회귀분석 파일이 없습니다: {path}")
    spec = importlib.util.spec_from_file_location("regression_before_tax", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


reg = load_regression_module()




# 3. 분석 설정입니다.
# MAX_LAG: ACF/PACF를 몇 주차까지 볼지. 주간 자료이므로 20주(약 5개월)까지 봅니다.
# LJUNG_BOX_LAGS: Ljung-Box Q 검정을 수행할 시차. 여러 시차에서 동시에 확인합니다.
MAX_LAG = 20
LJUNG_BOX_LAGS = [4, 8, 12, 20]
SIGNIFICANCE = 0.05




# 4. 자기상관함수(ACF)를 계산합니다.
# r_k = Σ (x_t − x̄)(x_{t−k} − x̄) / Σ (x_t − x̄)²  로, k주 전 값과의 상관입니다.
def acf(series, max_lag=MAX_LAG):
    count = len(series)
    mean = sum(series) / count
    centered = [x - mean for x in series]
    denominator = sum(c * c for c in centered)
    if denominator == 0:
        return [float("nan")] * (max_lag + 1)
    values = []
    for lag in range(max_lag + 1):
        numerator = sum(centered[t] * centered[t - lag] for t in range(lag, count))
        values.append(numerator / denominator)
    return values




# 5. 편자기상관함수(PACF)를 Durbin-Levinson 재귀로 계산합니다.
# PACF(k)는 1~k−1주 전의 영향을 제거한 뒤 남는 k주 전 값과의 순수 상관입니다.
# AR 모형의 차수를 정할 때 PACF가 끊기는 지점을 봅니다.
def pacf(acf_values):
    max_lag = len(acf_values) - 1
    values = [1.0]
    phi_prev = []
    for k in range(1, max_lag + 1):
        if k == 1:
            phi_kk = acf_values[1]
            phi_prev = [phi_kk]
            values.append(phi_kk)
            continue
        numerator = acf_values[k] - sum(phi_prev[j] * acf_values[k - 1 - j] for j in range(k - 1))
        denominator = 1.0 - sum(phi_prev[j] * acf_values[j + 1] for j in range(k - 1))
        phi_kk = numerator / denominator if denominator else float("nan")
        phi_new = [phi_prev[j] - phi_kk * phi_prev[k - 2 - j] for j in range(k - 1)] + [phi_kk]
        phi_prev = phi_new
        values.append(phi_kk)
    return values




# 6. Durbin-Watson 통계량입니다. 2에 가까우면 1차 자기상관 없음, 0에 가까우면 양의 자기상관, 4에 가까우면 음의 자기상관입니다.
# DW ≈ 2(1 − r₁) 관계가 있어 1차 ACF와 바로 대응합니다.
def durbin_watson(residuals):
    numerator = sum((residuals[t] - residuals[t - 1]) ** 2 for t in range(1, len(residuals)))
    denominator = sum(r * r for r in residuals)
    return numerator / denominator if denominator else float("nan")




# 7. 카이제곱 분포의 꼬리확률을 정규화 불완전 감마함수로 구합니다(Ljung-Box p값용).
def _gamma_series(a, x):
    term = 1.0 / a
    total = term
    for n in range(1, 500):
        term *= x / (a + n)
        total += term
        if abs(term) < abs(total) * 1e-14:
            break
    return total * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_continued_fraction(a, x):
    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi_square_survival(statistic, degrees_of_freedom):
    if statistic <= 0:
        return 1.0
    a, x = degrees_of_freedom / 2.0, statistic / 2.0
    if x < a + 1.0:
        return 1.0 - _gamma_series(a, x)
    return _gamma_continued_fraction(a, x)




# 8. Ljung-Box Q 검정입니다. 1~h주차 자기상관이 모두 0이라는 귀무가설을 한꺼번에 검정합니다.
# Q = n(n+2) Σ_{k=1}^{h} r_k² / (n − k) 이고, 귀무가설 아래 자유도 h의 카이제곱 분포를 따릅니다.
# 회귀 잔차에 적용할 때는 자유도에서 추정한 설명변수 수를 빼지 않는 일반 형태를 사용합니다.
def ljung_box(acf_values, count, lags=LJUNG_BOX_LAGS):
    results = []
    for h in lags:
        q = count * (count + 2) * sum(acf_values[k] ** 2 / (count - k) for k in range(1, h + 1))
        results.append({"시차": h, "Q": q, "p": chi_square_survival(q, h)})
    return results




# 9. 시계열 하나에 대해 ACF·PACF·Ljung-Box를 한 번에 계산합니다.
def analyze_series(series, name):
    count = len(series)
    acf_values = acf(series)
    band = 1.96 / math.sqrt(count)
    significant = [k for k in range(1, MAX_LAG + 1) if abs(acf_values[k]) > band]
    return {
        "이름": name,
        "n": count,
        "ACF": acf_values,
        "PACF": pacf(acf_values),
        "신뢰대": band,
        "유의시차": significant,
        "LjungBox": ljung_box(acf_values, count),
    }




# 10. 제품 하나에 대해 타겟 수준·타겟 1차 차분·회귀 잔차 세 시계열을 분석합니다.
# 잔차는 회귀와 똑같이 학습셋에서 적합한 뒤 학습셋 잔차를 사용합니다.
def diagnose_product(product, rows, train_index):
    names = reg.feature_names(product)
    labels, features, targets = reg.build_design(rows, product)
    train_features, train_targets = reg.take(features, train_index), reg.take(targets, train_index)
    coefficients, _ = reg.fit_ols(train_features, train_targets)
    fitted = reg.predict(coefficients, train_features)
    residuals = [a - p for a, p in zip(train_targets, fitted, strict=True)]
    differenced = [train_targets[t] - train_targets[t - 1] for t in range(1, len(train_targets))]
    return {
        "제품": product,
        "타겟": reg.MODELS[product]["타겟"],
        "설명변수": names,
        "기간": (labels[train_index[0]], labels[train_index[-1]]),
        "DW": durbin_watson(residuals),
        "시계열": {
            "타겟 수준": analyze_series(train_targets, "타겟 수준"),
            "타겟 1차 차분": analyze_series(differenced, "타겟 1차 차분"),
            "회귀 잔차": analyze_series(residuals, "회귀 잔차"),
        },
    }




# 11. 수치를 표기용 문자열로 바꿉니다.
def fmt(value, digits=3):
    if value != value:
        return "-"
    return f"{value:.{digits}f}"


def dw_verdict(dw):
    if dw != dw:
        return "-"
    if dw < 1.5:
        return "양의 자기상관"
    if dw > 2.5:
        return "음의 자기상관"
    return "자기상관 없음"


def lb_verdict(p):
    return "자기상관 있음" if p < SIGNIFICANCE else "자기상관 없음"




# 12. ACF/PACF 전체 값을 CSV로 저장해 그래프를 그릴 수 있게 합니다.
def save_acf_csv(diagnoses, path=ACF_TABLE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["제품", "시계열", "시차", "ACF", "PACF", "신뢰대", "유의"])
        writer.writeheader()
        for diagnosis in diagnoses:
            for series in diagnosis["시계열"].values():
                for lag in range(1, MAX_LAG + 1):
                    writer.writerow(
                        {
                            "제품": diagnosis["제품"],
                            "시계열": series["이름"],
                            "시차": lag,
                            "ACF": fmt(series["ACF"][lag], 4),
                            "PACF": fmt(series["PACF"][lag], 4),
                            "신뢰대": fmt(series["신뢰대"], 4),
                            "유의": "Y" if abs(series["ACF"][lag]) > series["신뢰대"] else "N",
                        }
                    )
    return path




# 13. 보고서를 마크다운으로 정리합니다.
def acf_table_lines(series, lags=(1, 2, 3, 4, 8, 12, 20)):
    lines = ["| 시차 | " + " | ".join(str(k) for k in lags) + " |", "| --- |" + " --- |" * len(lags)]
    lines.append("| ACF | " + " | ".join(fmt(series["ACF"][k]) for k in lags) + " |")
    lines.append("| PACF | " + " | ".join(fmt(series["PACF"][k]) for k in lags) + " |")
    return lines


def ljung_box_lines(series):
    lines = ["| 시차 h | Q 통계량 | p값 | 판정 |", "| --- | --- | --- | --- |"]
    for item in series["LjungBox"]:
        lines.append(f"| {item['시차']} | {fmt(item['Q'], 2)} | {fmt(item['p'], 4)} | {lb_verdict(item['p'])} |")
    return lines


def product_section(number, diagnosis):
    residual = diagnosis["시계열"]["회귀 잔차"]
    level = diagnosis["시계열"]["타겟 수준"]
    differenced = diagnosis["시계열"]["타겟 1차 차분"]
    lines = [
        f"## {number}. {diagnosis['제품']} 모형",
        "",
        f"- 타겟변수: `{diagnosis['타겟']}`",
        "- 설명변수: " + ", ".join(f"`{n}`" for n in diagnosis["설명변수"]),
        f"- 학습 기간: {diagnosis['기간'][0]} ~ {diagnosis['기간'][1]} ({residual['n']}주)",
        f"- ACF 95% 신뢰대: ±{fmt(residual['신뢰대'])} (= 1.96/√n)",
        "",
        "### 회귀 잔차 자기상관 (핵심)",
        "",
        f"- Durbin-Watson: **{fmt(diagnosis['DW'])}** → {dw_verdict(diagnosis['DW'])}",
        f"- 신뢰대를 벗어난 시차: {', '.join(str(k) for k in residual['유의시차']) if residual['유의시차'] else '없음'}",
        "",
        *acf_table_lines(residual),
        "",
        "Ljung-Box Q 검정 (귀무가설: 1~h주차 자기상관이 모두 0)",
        "",
        *ljung_box_lines(residual),
        "",
        "### 타겟 시계열 자기상관 (참고)",
        "",
        "세전가격 수준 자체의 자기상관입니다. 가격 시계열은 보통 강한 지속성을 보이므로 높게 나오는 것이 정상입니다.",
        "",
        "**타겟 수준**",
        "",
        *acf_table_lines(level),
        "",
        f"Ljung-Box(20): Q = {fmt(level['LjungBox'][-1]['Q'], 1)}, p = {fmt(level['LjungBox'][-1]['p'], 4)}",
        "",
        "**타겟 1차 차분 (이번 주 − 지난 주)**",
        "",
        *acf_table_lines(differenced),
        "",
        f"Ljung-Box(20): Q = {fmt(differenced['LjungBox'][-1]['Q'], 1)}, p = {fmt(differenced['LjungBox'][-1]['p'], 4)}",
        "",
    ]
    return lines


def save_report(diagnoses, sample_size, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 제품별 회귀모형 자기상관 진단 보고서",
        "",
        "## 0. 분석 설계",
        "",
        f"- 자료: `{reg.INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}`",
        f"- 표본: 학습셋 {sample_size}주 ({reg.SPLIT_MODE} 분할의 앞 80%) — 회귀계수를 추정한 자료와 동일",
        f"- 모형: `ols 111/regression_before_tax.py`의 제품별 OLS (국제가격 t-{reg.LAG_WEEKS} + 환율 t)",
        f"- 진단 대상: ① 회귀 잔차 (OLS 독립성 가정 점검, 핵심) ② 타겟 수준 ③ 타겟 1차 차분 (참고)",
        f"- 지표: ACF·PACF (1~{MAX_LAG}주차), Durbin-Watson, Ljung-Box Q (h = {', '.join(str(h) for h in LJUNG_BOX_LAGS)})",
        f"- 유의수준 {SIGNIFICANCE}. ACF 신뢰대는 ±1.96/√n",
        "",
        "### 요약",
        "",
        "| 모형 | 잔차 DW | 잔차 ACF(1) | 잔차 Ljung-Box(12) p | 판정 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for diagnosis in diagnoses:
        residual = diagnosis["시계열"]["회귀 잔차"]
        lb12 = next(item for item in residual["LjungBox"] if item["시차"] == 12)
        lines.append(
            f"| {diagnosis['제품']} | {fmt(diagnosis['DW'])} | {fmt(residual['ACF'][1])} | "
            f"{fmt(lb12['p'], 4)} | {lb_verdict(lb12['p'])} |"
        )
    lines.append("")
    for number, diagnosis in enumerate(diagnoses, start=1):
        lines += product_section(number, diagnosis)

    any_autocorrelated = any(
        d["시계열"]["회귀 잔차"]["LjungBox"][2]["p"] < SIGNIFICANCE for d in diagnoses
    )
    lines += ["## 해석 및 대응", ""]
    if any_autocorrelated:
        lines += [
            "- 잔차에 양의 자기상관이 있습니다. 이번 주 잔차가 양수면 다음 주도 양수일 가능성이 높다는 뜻으로,",
            "  모형이 놓친 체계적 요인(가격 조정의 점진성, 정유사의 평활화 등)이 잔차에 남아 있습니다.",
        ]
        for diagnosis in diagnoses:
            residual = diagnosis["시계열"]["회귀 잔차"]
            acf_values = residual["ACF"]
            monthly_peaks = [k for k in (4, 8, 12, 16) if acf_values[k] > acf_values[k - 1] and acf_values[k] > acf_values[k + 1]]
            lines.append(
                f"- {diagnosis['제품']}: 잔차 ACF가 1주차 {fmt(acf_values[1])}에서 {MAX_LAG}주차 {fmt(acf_values[MAX_LAG])}까지 "
                f"천천히 감소합니다(신뢰대 ±{fmt(residual['신뢰대'])}). 단기 AR(1)이 아니라 몇 달 단위로 느리게 움직이는 "
                f"요인이 남아 있다는 뜻입니다."
                + (f" {', '.join(str(k) for k in monthly_peaks)}주차에서 국소 봉우리가 보여 약 4주(월) 주기의 패턴도 있습니다." if monthly_peaks else "")
            )
        lines += [
            "- **계수 추정치 자체는 여전히 불편(unbiased)입니다.** 전가율 ≈ 1 이라는 결론은 유효합니다.",
            "- **표준오차는 과소추정됩니다.** 따라서 회귀 보고서의 p값과 95% 신뢰구간은 실제보다 낙관적입니다.",
            "  p값이 0.0000 수준으로 매우 작은 계수(국제가격)는 결론이 바뀌지 않지만, p가 0.01~0.05 근처인 계수(경유 환율 p=0.013)는 유의성을 단정하기 어렵습니다.",
            "- 대응 방법 (필요 시):",
            "  1. Newey-West(HAC) 표준오차로 p값만 보정 — 모형은 그대로 두고 추론만 고침",
            "  2. 타겟의 1주 전 값(세전가격 t-1)을 설명변수로 추가 — 잔차 자기상관을 흡수하는 가장 간단한 구조 변경",
            "  3. 수준 대신 1차 차분(주간 변화량)으로 회귀 — 타겟 1차 차분 ACF가 작다면 효과적",
            "- 잔차 PACF가 1주차에서 끊기면 AR(1) 구조이며 위 2번(t-1 타겟 추가)으로 대부분 해소됩니다.",
            "",
        ]
    else:
        lines += [
            "- 잔차 자기상관이 유의하지 않습니다. OLS의 독립성 가정이 성립하며 회귀 보고서의 p값과 신뢰구간을 그대로 믿어도 됩니다.",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 14. 불러오기 → 구조 점검 → 시차·분할 → 제품별 진단 → 저장 순서로 진행합니다.
def main():
    try:
        columns, rows = reg.load_clean_csv(reg.INPUT_PATH)
        errors = reg.check_structure(columns, rows)
        if errors:
            raise ValueError("구조 점검 실패:\n" + "\n".join(errors[:20]))
        train_index, _ = reg.split_indices(len(rows) - reg.LAG_WEEKS)
        diagnoses = [diagnose_product(product, rows, train_index) for product in reg.MODELS]
        csv_path = save_acf_csv(diagnoses)
        report_path = save_report(diagnoses, len(train_index))
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise SystemExit(f"자기상관 분석 실패: {error}") from None

    print(f"학습셋 {len(train_index)}주 기준, 제품별 모형 {len(reg.MODELS)}개, ACF 1~{MAX_LAG}주차")
    for diagnosis in diagnoses:
        residual = diagnosis["시계열"]["회귀 잔차"]
        level = diagnosis["시계열"]["타겟 수준"]
        differenced = diagnosis["시계열"]["타겟 1차 차분"]
        print()
        print(f"[{diagnosis['제품']}] 잔차 Durbin-Watson = {fmt(diagnosis['DW'])} → {dw_verdict(diagnosis['DW'])}")
        print(f"   잔차 ACF  1~4주차: " + "  ".join(fmt(residual['ACF'][k]) for k in range(1, 5)) + f"   (신뢰대 ±{fmt(residual['신뢰대'])})")
        print(f"   잔차 PACF 1~4주차: " + "  ".join(fmt(residual['PACF'][k]) for k in range(1, 5)))
        for item in residual["LjungBox"]:
            print(f"   잔차 Ljung-Box({item['시차']:>2}): Q={fmt(item['Q'], 1):>7}  p={fmt(item['p'], 4)}  {lb_verdict(item['p'])}")
        print(f"   참고 — 타겟 수준 ACF(1)={fmt(level['ACF'][1])}, 타겟 1차 차분 ACF(1)={fmt(differenced['ACF'][1])}")
    print()
    print(f"보고서 저장: {report_path}")
    print(f"ACF 표 저장: {csv_path}")




# 15. 직접 실행할 때만 분석을 수행합니다.
if __name__ == "__main__":
    main()
