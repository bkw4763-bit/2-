# 1. 세전 공급가격을 예측하는 다중선형회귀(OLS) 모형을 제품별로 학습·평가합니다.
# 실행: python "ols 111/regression_before_tax.py"
import csv
import math
import random
from pathlib import Path




# 2. 입력 자료와 결과 저장 위치를 정합니다.
PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_DIR / "csv" / "merged_weekly_data_clean.csv"
REPORT_PATH = PROJECT_DIR / "results" / "회귀" / "regression_before_tax_report.md"
PREDICTION_DIR = PROJECT_DIR / "csv"

KEY_COLUMN = "주차"




# 3. 제품별로 타겟변수(세전 공급가격)와 그 제품의 국내 기준 등급에 해당하는 국제가격 하나를 설명변수로 묶습니다.
# 경유: 국내 자동차용 경유는 황 함량 10ppm(0.001%) 기준이므로 경유(0.001%)를 씁니다. 경유(0.05%)는 선박·산업용 등급이라 제외.
# 휘발유: 국내 보통휘발유는 옥탄가 91 이상으로 92RON이 벤치마크입니다. 95RON은 고급휘발유 쪽이라 제외.
# 같은 제품의 두 등급을 함께 넣으면 상관 0.999로 VIF가 수백까지 올라가 계수를 해석할 수 없게 됩니다(vif_report.md 참고).
# 세후 공급가격(세전 + 세금)과 주유소 판매가격(세전가격 이후에 형성)은 누수·하위 변수라 어느 모형에도 넣지 않습니다.
MODELS = {
    "경유": {
        "타겟": "정유사_주간공급가격_제품별(세전)_자동차용경유",
        "국제가격": [
            "국제_석유제품가격(원)_경유(0.001%)",
        ],
    },
    "등유": {
        "타겟": "정유사_주간공급가격_제품별(세전)_실내등유",
        "국제가격": [
            "국제_석유제품가격(원)_등유",
        ],
    },
    "휘발유": {
        "타겟": "정유사_주간공급가격_제품별(세전)_보통휘발유",
        "국제가격": [
            "국제_석유제품가격(원)_휘발유(92RON)",
        ],
    },
}

# 원달러환율은 모든 제품 모형에 공통으로 들어가는 설명변수입니다.
CURRENT_FEATURES = [
    "원달러환율_주간평균_원/달러 환율",
]

TARGETS = {product: spec["타겟"] for product, spec in MODELS.items()}




# 4. 국제가격이 국내 공급가격에 반영되기까지의 시차(주)를 적용합니다.
# LAG_WEEKS=1이면 이번 주 세전가격을 '지난주' 국제가격으로 설명합니다(환율은 이번 주 값 사용).
# 시차만큼 앞쪽 행은 과거 값이 없어 자료에서 빠집니다. 0으로 두면 시차 없이 같은 주 값을 씁니다.
# 자료는 행 단위로 당깁니다. 결측으로 빠진 2017년10월1주 직후 한 행만 실제로는 2주 전 값이 됩니다.
LAG_WEEKS = 1


def lag_label(name, lag=LAG_WEEKS):
    return f"{name} (t-{lag})" if lag else name


# 제품 모형에 실제로 들어가는 설명변수 이름 목록(시차 표기 포함)입니다.
def feature_names(product):
    return [lag_label(name) for name in MODELS[product]["국제가격"]] + CURRENT_FEATURES




# 5. 학습셋 80% / 평가셋 20%로 나눕니다.
# 주차별 시계열이므로 기본값은 시간순 분할("chronological")입니다.
# 앞 80% 기간으로 학습하고 뒤 20% 기간을 예측하므로 미래 정보가 학습에 섞이지 않습니다.
# 무작위 분할을 보고 싶으면 SPLIT_MODE를 "random"으로 바꿉니다(RANDOM_SEED로 재현).
SPLIT_MODE = "chronological"
TRAIN_RATIO = 0.8
RANDOM_SEED = 42




# 6. CSV를 읽어 열 이름과 행 목록을 돌려줍니다.
# 엑셀이 붙이는 UTF-8 BOM을 함께 처리합니다.
def load_clean_csv(path=INPUT_PATH):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not columns:
        raise ValueError(f"열 이름을 읽지 못했습니다: {path}")
    if not rows:
        raise ValueError(f"자료 행이 없습니다: {path}")
    return columns, rows




# 7. 세 모형이 쓰는 열이 모두 있는지, 숫자로 바꿀 수 있는지 먼저 점검합니다.
# 문제가 있으면 어느 주차의 어느 열인지 알려 주고 멈춥니다.
def needed_columns():
    names = []
    for spec in MODELS.values():
        names += [spec["타겟"], *spec["국제가격"]]
    names += CURRENT_FEATURES
    return list(dict.fromkeys(names))


def check_structure(columns, rows):
    errors = []
    needed = needed_columns()
    for name in [KEY_COLUMN, *needed]:
        if name not in columns:
            errors.append(f"필요한 열이 없습니다: {name}")
    if errors:
        return errors
    for index, row in enumerate(rows, start=2):
        for name in needed:
            text = (row.get(name) or "").strip().replace(",", "")
            if not text:
                errors.append(f"{index}행({row.get(KEY_COLUMN)}) {name}: 값이 비어 있습니다")
                continue
            try:
                float(text)
            except ValueError:
                errors.append(f"{index}행({row.get(KEY_COLUMN)}) {name}: 숫자가 아닙니다 '{text}'")
    return errors




# 8. 점검을 통과한 행에서 제품 모형의 설명변수 행렬과 타겟값을 꺼냅니다.
# 국제가격은 LAG_WEEKS만큼 앞 행의 값을, 환율과 타겟은 해당 행의 값을 사용합니다.
# 시차를 만들 과거 행이 없는 맨 앞 LAG_WEEKS개 행은 제외합니다.
def to_number(row, name):
    return float((row[name] or "").strip().replace(",", ""))


def build_design(rows, product, lag=LAG_WEEKS):
    if lag < 0:
        raise ValueError(f"LAG_WEEKS는 0 이상이어야 합니다: {lag}")
    if len(rows) <= lag:
        raise ValueError(f"시차 {lag}주를 적용하기에 자료 {len(rows)}행이 부족합니다")
    spec = MODELS[product]
    labels, features, targets = [], [], []
    for index in range(lag, len(rows)):
        current, past = rows[index], rows[index - lag]
        labels.append(current[KEY_COLUMN])
        features.append(
            [to_number(past, name) for name in spec["국제가격"]]
            + [to_number(current, name) for name in CURRENT_FEATURES]
        )
        targets.append(to_number(current, spec["타겟"]))
    return labels, features, targets




# 9. 전체 자료를 학습 80% / 평가 20%로 나눕니다.
# 시간순 분할은 앞쪽을 학습으로, 무작위 분할은 고정 시드로 섞어서 나눕니다.
def split_indices(count, mode=SPLIT_MODE, ratio=TRAIN_RATIO, seed=RANDOM_SEED):
    train_size = int(round(count * ratio))
    train_size = max(1, min(count - 1, train_size))
    order = list(range(count))
    if mode == "random":
        random.Random(seed).shuffle(order)
    elif mode != "chronological":
        raise ValueError(f"SPLIT_MODE는 'chronological' 또는 'random'이어야 합니다: {mode}")
    return order[:train_size], order[train_size:]


def take(values, indices):
    return [values[i] for i in indices]




# 10. 가우스-조르당 소거법으로 역행렬을 구합니다(부분 피벗으로 수치 안정성 확보).
def invert_matrix(matrix):
    size = len(matrix)
    work = [list(row) + [1.0 if i == j else 0.0 for j in range(size)] for i, row in enumerate(matrix)]
    for col in range(size):
        pivot = max(range(col, size), key=lambda r: abs(work[r][col]))
        if abs(work[pivot][col]) < 1e-12:
            raise ValueError("설명변수가 서로 완전히 겹쳐 역행렬을 구할 수 없습니다")
        work[col], work[pivot] = work[pivot], work[col]
        scale = work[col][col]
        work[col] = [value / scale for value in work[col]]
        for row in range(size):
            if row == col:
                continue
            factor = work[row][col]
            if factor:
                work[row] = [a - factor * b for a, b in zip(work[row], work[col], strict=True)]
    return [row[size:] for row in work]




# 11. 최소제곱법으로 절편과 기울기를 추정합니다.
# 설계행렬 맨 앞에 1 열을 붙여 절편을 함께 추정합니다.
def fit_ols(features, targets):
    rows = len(features)
    width = len(features[0]) + 1
    if rows <= width:
        raise ValueError(f"표본 {rows}개로는 계수 {width}개를 추정할 수 없습니다")
    design = [[1.0, *row] for row in features]
    gram = [[sum(design[i][a] * design[i][b] for i in range(rows)) for b in range(width)] for a in range(width)]
    moment = [sum(design[i][a] * targets[i] for i in range(rows)) for a in range(width)]
    gram_inverse = invert_matrix(gram)
    coefficients = [sum(gram_inverse[a][b] * moment[b] for b in range(width)) for a in range(width)]
    return coefficients, gram_inverse




# 12. 추정한 계수로 예측값을 계산합니다.
def predict(coefficients, features):
    return [coefficients[0] + sum(c * x for c, x in zip(coefficients[1:], row, strict=True)) for row in features]




# 13. 스튜던트 t 분포의 양쪽 꼬리 확률(p-value)을 정규화 불완전 베타 함수로 구합니다.
def _beta_continued_fraction(a, b, x):
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    result = d
    for m in range(1, 201):
        m2 = 2 * m
        numerator = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        result *= d * c
        numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + numerator * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + numerator / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        step = d * c
        result *= step
        if abs(step - 1.0) < 1e-12:
            break
    return result


def regularized_beta(a, b, x):
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _beta_continued_fraction(a, b, x) / a
    return 1.0 - front * _beta_continued_fraction(b, a, 1.0 - x) / b


def two_sided_p_value(t_statistic, degrees_of_freedom):
    if degrees_of_freedom <= 0:
        return float("nan")
    denominator = degrees_of_freedom + t_statistic * t_statistic
    return regularized_beta(degrees_of_freedom / 2.0, 0.5, degrees_of_freedom / denominator)




# 14. 예측 성능 지표를 계산합니다.
# R2는 설명력, RMSE/MAE는 평균 오차 크기, MAPE는 오차의 상대 비율(%)입니다.
def evaluate(actual, predicted):
    count = len(actual)
    if count == 0:
        return {"n": 0}
    mean = sum(actual) / count
    residuals = [a - p for a, p in zip(actual, predicted, strict=True)]
    sse = sum(r * r for r in residuals)
    sst = sum((a - mean) ** 2 for a in actual)
    absolute = [abs(r) for r in residuals]
    percentage = [abs(r) / abs(a) for r, a in zip(residuals, actual, strict=True) if a]
    return {
        "n": count,
        "R2": 1.0 - sse / sst if sst else float("nan"),
        "RMSE": math.sqrt(sse / count),
        "MAE": sum(absolute) / count,
        "MAPE": 100.0 * sum(percentage) / len(percentage) if percentage else float("nan"),
        "최대오차": max(absolute),
    }




# 15. 계수별 표준오차·t값·p값과 95% 신뢰구간을 계산합니다.
# 잔차분산 sigma^2 = SSE / (n - k) 를 (X'X)^-1 의 대각원소에 곱해 표준오차를 얻습니다.
def summarize_coefficients(coefficients, gram_inverse, features, targets, names):
    fitted = predict(coefficients, features)
    residuals = [a - p for a, p in zip(targets, fitted, strict=True)]
    degrees_of_freedom = len(targets) - len(coefficients)
    variance = sum(r * r for r in residuals) / degrees_of_freedom
    labels = ["절편(상수)", *names]
    summary = []
    for index, (name, value) in enumerate(zip(labels, coefficients, strict=True)):
        error = math.sqrt(max(variance * gram_inverse[index][index], 0.0))
        t_statistic = value / error if error else float("nan")
        summary.append(
            {
                "변수": name,
                "계수": value,
                "표준오차": error,
                "t": t_statistic,
                "p": two_sided_p_value(t_statistic, degrees_of_freedom),
                "하한95": value - 1.96 * error,
                "상한95": value + 1.96 * error,
            }
        )
    return summary, degrees_of_freedom




# 16. 분산팽창계수(VIF)로 설명변수끼리 겹치는 정도를 확인합니다.
# VIF가 10을 넘으면 그 변수의 계수 부호와 크기를 단독으로 해석하기 어렵습니다.
# 설명변수가 환율 하나뿐인 모형은 보조회귀를 만들 수 없으므로 VIF를 1로 둡니다.
def compute_vif(features, names):
    width = len(names)
    if width == 1:
        return {names[0]: 1.0}
    values = []
    for target_index in range(width):
        others = [[row[j] for j in range(width) if j != target_index] for row in features]
        column = [row[target_index] for row in features]
        try:
            coefficients, _ = fit_ols(others, column)
        except ValueError:
            values.append(float("inf"))
            continue
        fitted = predict(coefficients, others)
        mean = sum(column) / len(column)
        sse = sum((a - p) ** 2 for a, p in zip(column, fitted, strict=True))
        sst = sum((a - mean) ** 2 for a in column)
        r_squared = 1.0 - sse / sst if sst else 0.0
        values.append(1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf"))
    return dict(zip(names, values, strict=True))




# 17. 제품 하나에 대해 설계행렬 구성 → 분할 → 학습 → 평가를 수행합니다.
def run_single_model(product, rows, train_index, test_index):
    names = feature_names(product)
    labels, features, targets = build_design(rows, product)
    train_features, train_targets = take(features, train_index), take(targets, train_index)
    test_features, test_targets = take(features, test_index), take(targets, test_index)

    coefficients, gram_inverse = fit_ols(train_features, train_targets)
    coefficient_summary, degrees_of_freedom = summarize_coefficients(
        coefficients, gram_inverse, train_features, train_targets, names
    )
    train_predicted = predict(coefficients, train_features)
    test_predicted = predict(coefficients, test_features)

    def prediction_rows(indices, predicted, split_name):
        return [
            {
                KEY_COLUMN: labels[i],
                "구분": split_name,
                "실제_세전가격": targets[i],
                "예측_세전가격": value,
                "잔차": targets[i] - value,
            }
            for i, value in zip(indices, predicted, strict=True)
        ]

    return {
        "제품": product,
        "타겟": MODELS[product]["타겟"],
        "설명변수": names,
        "계수": coefficients,
        "계수표": coefficient_summary,
        "자유도": degrees_of_freedom,
        "학습성능": evaluate(train_targets, train_predicted),
        "평가성능": evaluate(test_targets, test_predicted),
        "VIF": compute_vif(train_features, names),
        "학습기간": (labels[train_index[0]], labels[train_index[-1]]),
        "평가기간": (labels[test_index[0]], labels[test_index[-1]]),
        "예측": prediction_rows(train_index, train_predicted, "train")
        + prediction_rows(test_index, test_predicted, "test"),
    }




# 18. 제품별 예측값을 CSV로 저장해 그래프나 추가 분석에 쓸 수 있게 합니다.
def save_predictions(result, directory=PREDICTION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"regression_predictions_{result['제품']}.csv"
    fieldnames = [KEY_COLUMN, "구분", "실제_세전가격", "예측_세전가격", "잔차"]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["예측"]:
            writer.writerow(
                {
                    KEY_COLUMN: row[KEY_COLUMN],
                    "구분": row["구분"],
                    "실제_세전가격": f"{row['실제_세전가격']:.2f}",
                    "예측_세전가격": f"{row['예측_세전가격']:.2f}",
                    "잔차": f"{row['잔차']:.2f}",
                }
            )
    return path




# 19. 세 모형의 결과를 하나의 마크다운 보고서로 정리합니다.
def save_report(results, total_rows, train_size, test_size, path=REPORT_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 세전 공급가격 회귀분석 보고서",
        "",
        "## 1. 분석 설계",
        "",
        f"- 자료: `{INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}` (시차 적용 후 {total_rows}주)",
        "- 타겟변수: 정유사 주간공급가격 **세전** (제품별 3개 모형)",
        f"- 설명변수: 해당 제품의 국제가격 (**{LAG_WEEKS}주 전 값, t-{LAG_WEEKS}**) + 원달러환율 (같은 주 값)",
        f"- 분할: {SPLIT_MODE} / 학습 {train_size}주({train_size / total_rows:.1%}) · "
        f"평가 {test_size}주({test_size / total_rows:.1%})",
        "- 모형: 다중선형회귀(최소제곱법, 절편 포함)",
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
        "### 제외한 변수와 이유",
        "",
        "| 제외 변수 | 이유 |",
        "| --- | --- |",
        "| 다른 제품의 국제가격 | 경유 모형에 국제 휘발유가격을 넣는 식의 교차 사용은 해석이 불가능하고 공선성만 키움 |",
        "| 국제 경유(0.05%) | 선박·산업용 등급. 국내 자동차용 경유 기준(10ppm)은 0.001%이며 둘은 상관 0.999로 중복 |",
        "| 국제 휘발유(95RON) | 고급휘발유 벤치마크. 국내 보통휘발유 기준은 92RON이며 둘은 상관 0.999로 중복 |",
        "| 정유사 공급가격(세후) 4개 | 세전가격 + 세금이므로 타겟이 그대로 들어가는 누수 |",
        "| 주유소 평균판매가격 4개 | 세전가격 결정 이후에 형성되는 하위 변수 |",
        "| 세전 고급휘발유 | 같은 공시표에서 동시에 결정되어 설명변수로 부적절 |",
        "",
        "## 2. 성능 요약",
        "",
        "| 제품 | 설명변수 수 | 학습 R² | 평가 R² | 학습 RMSE | 평가 RMSE | 평가 MAE | 평가 MAPE |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        train, test = result["학습성능"], result["평가성능"]
        lines.append(
            f"| {result['제품']} | {len(result['설명변수'])} | {train['R2']:.4f} | {test['R2']:.4f} | "
            f"{train['RMSE']:.2f} | {test['RMSE']:.2f} | {test['MAE']:.2f} | {test['MAPE']:.2f}% |"
        )
    lines += ["", "단위는 원/리터입니다.", ""]

    for number, result in enumerate(results, start=1):
        lines += [
            f"## 3-{number}. {result['제품']} 모형",
            "",
            f"- 타겟변수: `{result['타겟']}`",
            "- 설명변수: " + ", ".join(f"`{n}`" for n in result["설명변수"]),
            f"- 학습 기간: {result['학습기간'][0]} ~ {result['학습기간'][1]} ({result['학습성능']['n']}주)",
            f"- 평가 기간: {result['평가기간'][0]} ~ {result['평가기간'][1]} ({result['평가성능']['n']}주)",
            f"- 잔차 자유도: {result['자유도']}",
            "",
            "### 회귀계수",
            "",
            "| 변수 | 계수 | 표준오차 | t | p | 95% 신뢰구간 | VIF |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in result["계수표"]:
            vif = result["VIF"].get(item["변수"])
            vif_text = "-" if vif is None else f"{vif:.1f}"
            lines.append(
                f"| {item['변수']} | {item['계수']:.4f} | {item['표준오차']:.4f} | {item['t']:.2f} | "
                f"{item['p']:.4f} | [{item['하한95']:.3f}, {item['상한95']:.3f}] | {vif_text} |"
            )
        train, test = result["학습성능"], result["평가성능"]
        lines += [
            "",
            "### 성능 지표",
            "",
            "| 지표 | 학습셋 | 평가셋 |",
            "| --- | --- | --- |",
            f"| 표본수 | {train['n']} | {test['n']} |",
            f"| R² | {train['R2']:.4f} | {test['R2']:.4f} |",
            f"| RMSE | {train['RMSE']:.2f} | {test['RMSE']:.2f} |",
            f"| MAE | {train['MAE']:.2f} | {test['MAE']:.2f} |",
            f"| MAPE | {train['MAPE']:.2f}% | {test['MAPE']:.2f}% |",
            f"| 최대오차 | {train['최대오차']:.2f} | {test['최대오차']:.2f} |",
            "",
        ]

    lines += [
        "## 4. 해석 시 유의점",
        "",
        "- 각 모형의 설명변수는 해당 제품 국제가격 1개 + 환율 1개뿐이라 VIF가 2 안팎으로 공선성 문제가 없습니다.",
        "  국제가격 계수는 '지난주 국제가격 1원 상승 → 이번 주 세전가격 몇 원 상승'(전가율)으로 그대로 읽으면 됩니다.",
        "- 시간순 분할이므로 평가 성능은 '과거로 학습해 이후 기간을 맞힌' 성적입니다.",
        "  무작위 분할로 바꾸면 같은 추세 구간이 학습·평가에 함께 들어가 성능이 과대평가됩니다.",
        "- 2026년 3월부터 국제 경유가격이 약 3배(800→2,450원/L)로 급등했지만 세전 공급가격은 약 1,350원에서",
        "  멈춘 뒤 7월 말부터 약 1,212원에 고정되어 있습니다. 국제가격과 무관하게 가격이 묶인 구간이라",
        "  선형모형이 따라갈 수 없고, 평가셋 R²가 음수로 떨어지는 원인은 전적으로 이 구간입니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path




# 20. 불러오기 → 구조 점검 → 80/20 분할 → 제품별 회귀 → 저장 순서로 진행합니다.
def main():
    try:
        columns, rows = load_clean_csv(INPUT_PATH)
        errors = check_structure(columns, rows)
        if errors:
            raise ValueError("구조 점검 실패:\n" + "\n".join(errors[:20]))
        usable_rows = len(rows) - LAG_WEEKS
        train_index, test_index = split_indices(usable_rows)
        results = [run_single_model(product, rows, train_index, test_index) for product in MODELS]
        prediction_paths = [save_predictions(result) for result in results]
        report_path = save_report(results, usable_rows, len(train_index), len(test_index))
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise SystemExit(f"회귀분석 실패: {error}") from None

    print(
        f"전체 {len(rows)}주 → 시차 {LAG_WEEKS}주 제외 {usable_rows}주 → "
        f"학습 {len(train_index)}주 / 평가 {len(test_index)}주 ({SPLIT_MODE} 분할)"
    )
    print(f"제품별 모형 {len(MODELS)}개: 해당 제품 국제가격(t-{LAG_WEEKS}) + 환율(t) → 세전 공급가격")
    print()
    for result in results:
        train, test = result["학습성능"], result["평가성능"]
        print(f"[{result['제품']}] 타겟: {result['타겟']}")
        print(f"  설명변수 {len(result['설명변수'])}개: " + ", ".join(result["설명변수"]))
        print(f"  학습 R²={train['R2']:.4f} RMSE={train['RMSE']:.2f} MAE={train['MAE']:.2f}")
        print(f"  평가 R²={test['R2']:.4f} RMSE={test['RMSE']:.2f} MAE={test['MAE']:.2f} MAPE={test['MAPE']:.2f}%")
        for item in result["계수표"]:
            mark = "*" if item["p"] < 0.05 else " "
            vif = result["VIF"].get(item["변수"])
            vif_text = "" if vif is None else f"  VIF={vif:.1f}"
            print(f"   {mark} {item['변수']:<40} {item['계수']:>10.4f}  (p={item['p']:.4f}){vif_text}")
        print()
    print(f"보고서 저장: {report_path}")
    for path in prediction_paths:
        print(f"예측값 저장: {path}")




# 21. 직접 실행할 때만 분석을 수행합니다.
if __name__ == "__main__":
    main()
