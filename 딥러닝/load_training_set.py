# 1. 딥러닝 모형(TimesFM 등)에 넣을 학습셋을 제품별로 불러옵니다.
# 데이터 클린파일(csv/merged_weekly_data_clean.csv)을 읽고, 변수 구성·시차·분할 규칙은
# 회귀분석("ols 111/regression_before_tax.py")에 정의된 것을 그대로 가져와 사용합니다.
# 따라서 회귀 쪽에서 변수나 시차를 바꾸면 이 학습셋도 똑같이 바뀝니다.
# 교차검증(CV)은 아직 하지 않고, 앞 80% 기간의 학습셋만 제품별(경유·등유·휘발유)로 만듭니다.
# 실행: python 딥러닝/load_training_set.py
import csv
import importlib.util
import re
from pathlib import Path

import numpy as np




# 2. 회귀분석 파일을 불러옵니다. 폴더 이름("ols 111")에 공백이 있어 일반 import가 되지 않으므로
# 파일 경로로 직접 모듈을 읽습니다. 불러오기만 하고 회귀분석을 실행하지는 않습니다.
PROJECT_DIR = Path(__file__).resolve().parent.parent
REGRESSION_PATH = PROJECT_DIR / "ols 111" / "regression_before_tax.py"
OUTPUT_DIR = PROJECT_DIR / "csv"


def load_regression_module(path=REGRESSION_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"회귀분석 파일이 없습니다: {path}")
    spec = importlib.util.spec_from_file_location("regression_before_tax", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


regression = load_regression_module()




# 3. 회귀분석과 공유하는 설정입니다. 값은 전부 회귀 파일에서 가져옵니다.
#   INPUT_PATH        데이터 클린파일 경로
#   MODELS            제품별 타겟(세전 공급가격)과 그 제품의 국제가격 목록
#   CURRENT_FEATURES  모든 제품에 공통으로 들어가는 환율
#   LAG_WEEKS         국제가격 시차(주). 1이면 지난주 국제가격으로 이번 주 세전가격을 설명
#   TRAIN_RATIO       시간순 앞쪽 학습셋 비율
INPUT_PATH = regression.INPUT_PATH
KEY_COLUMN = regression.KEY_COLUMN
MODELS = regression.MODELS
CURRENT_FEATURES = regression.CURRENT_FEATURES
LAG_WEEKS = regression.LAG_WEEKS
TRAIN_RATIO = regression.TRAIN_RATIO
feature_names = regression.feature_names




# 4. 데이터 클린파일을 읽고 점검합니다. 열 누락·비숫자 점검은 회귀분석의 check_structure를 쓰고,
# 시계열이므로 주차가 시간순으로 정렬돼 있는지만 추가로 확인합니다.
def week_key(label):
    """'2015년01월1주' 형태의 주차 표기를 (연, 월, 주) 숫자 묶음으로 바꿉니다."""
    numbers = re.findall(r"\d+", str(label))
    if len(numbers) < 3:
        raise ValueError(f"주차 표기를 읽을 수 없습니다: '{label}'")
    return tuple(int(value) for value in numbers[:3])


def check_order(rows):
    errors = []
    previous = None
    for index, row in enumerate(rows, start=2):
        try:
            current = week_key(row.get(KEY_COLUMN))
        except ValueError as error:
            errors.append(f"{index}행: {error}")
            continue
        if previous is not None and current <= previous:
            errors.append(f"{index}행({row.get(KEY_COLUMN)}): 주차가 시간순이 아닙니다")
        previous = current
    return errors


def load_clean_rows(path=INPUT_PATH):
    columns, rows = regression.load_clean_csv(path)
    errors = regression.check_structure(columns, rows) + check_order(rows)
    if errors:
        raise ValueError("구조 점검 실패:\n" + "\n".join(errors[:20]))
    return rows




# 5. 제품 하나의 학습셋을 만듭니다.
# 설계행렬은 회귀분석의 build_design(국제가격 t-1, 환율 t, 맨 앞 LAG_WEEKS행 제외)으로 만들고,
# 학습/평가 경계도 회귀분석의 split_indices(시간순 80/20)로 잡아 두 모형이 같은 기간을 보게 합니다.
# 반환값은 딥러닝 모형에 바로 넣을 수 있는 numpy 배열과 주차·변수 이름을 담은 사전입니다.
#   "y": 세전 공급가격 (표본수,)            → 예측 대상 시계열
#   "X": 설명변수 행렬 (표본수, 변수수)    → 국제가격(t-1) + 환율(t), 공변량으로 사용
#   "주차": 각 행의 주차 표기 목록
# split을 "test"나 "all"로 바꾸면 같은 경계로 평가셋 또는 전체 구간을 받을 수 있습니다.
def load_training_set(product, rows=None, split="train"):
    if product not in MODELS:
        raise ValueError(f"제품은 {', '.join(MODELS)} 중 하나여야 합니다: {product}")
    if rows is None:
        rows = load_clean_rows()
    labels, features, targets = regression.build_design(rows, product)
    train_index, test_index = regression.split_indices(len(labels), mode="chronological")
    if split == "train":
        selected = train_index
    elif split == "test":
        selected = test_index
    elif split == "all":
        selected = train_index + test_index
    else:
        raise ValueError(f"split은 'train', 'test', 'all' 중 하나여야 합니다: {split}")
    X = np.asarray(regression.take(features, selected), dtype=np.float32)
    y = np.asarray(regression.take(targets, selected), dtype=np.float32)
    if not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError(f"{product} 자료에 결측 또는 무한대가 있습니다")
    labels = regression.take(labels, selected)
    return {
        "제품": product,
        "구분": split,
        "타겟": MODELS[product]["타겟"],
        "설명변수": feature_names(product),
        "시차": LAG_WEEKS,
        "주차": labels,
        "기간": (labels[0], labels[-1]),
        "X": X,
        "y": y,
    }




# 6. 세 제품의 학습셋을 한 번에 만듭니다. 자료는 한 번만 읽고 점검합니다.
def load_all_training_sets(split="train"):
    rows = load_clean_rows()
    return {product: load_training_set(product, rows, split) for product in MODELS}




# 7. 제품별 학습셋을 CSV로 저장해 어떤 행·변수가 들어갔는지 눈으로 확인할 수 있게 합니다.
def save_training_set(result, directory=OUTPUT_DIR):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"training_set_{result['제품']}.csv"
    fieldnames = [KEY_COLUMN, *result["설명변수"], result["타겟"]]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        for label, features, target in zip(result["주차"], result["X"], result["y"], strict=True):
            writer.writerow([label, *(f"{value:.2f}" for value in features), f"{target:.2f}"])
    return path




# 8. 불러오기 → 구조 점검 → 회귀와 같은 시차·분할 → 제품별 요약 출력·저장 순서로 진행합니다.
def main():
    try:
        results = load_all_training_sets()
        paths = [save_training_set(result) for result in results.values()]
    except (OSError, UnicodeError, ValueError, csv.Error) as error:
        raise SystemExit(f"학습셋 불러오기 실패: {error}") from None

    first = next(iter(results.values()))
    print(f"자료: {INPUT_PATH.relative_to(PROJECT_DIR).as_posix()}")
    print(f"변수·시차·분할 규칙: {REGRESSION_PATH.relative_to(PROJECT_DIR).as_posix()} 에서 가져옴")
    print(f"시차 {LAG_WEEKS}주 적용 후 앞 {TRAIN_RATIO:.0%}를 학습셋으로 사용 (교차검증 없음)")
    print(f"학습 기간: {first['기간'][0]} ~ {first['기간'][1]} ({len(first['y'])}주)")
    print()
    for result in results.values():
        X, y = result["X"], result["y"]
        print(f"[{result['제품']}] 타겟: {result['타겟']}")
        print(f"  설명변수 {len(result['설명변수'])}개: " + ", ".join(result["설명변수"]))
        print(f"  X 크기 {X.shape}, y 크기 {y.shape}")
        print(f"  y 요약: 평균 {y.mean():.1f} / 최소 {y.min():.1f} / 최대 {y.max():.1f} 원/L")
        for name, column in zip(result["설명변수"], X.T, strict=True):
            print(f"   - {name:<40} 평균 {column.mean():>9.2f}  최소 {column.min():>9.2f}  최대 {column.max():>9.2f}")
        print()
    for path in paths:
        print(f"학습셋 저장: {path.relative_to(PROJECT_DIR).as_posix()}")




# 9. 직접 실행할 때만 학습셋을 만듭니다. 다른 파일에서는 load_training_set / load_all_training_sets를 import해 씁니다.
if __name__ == "__main__":
    main()
