"""학습 구간만으로 국제가격·환율을 Z-score 표준화합니다.

실행: python preprocessing/normalize_training_data.py
출력: csv/normalized/제품별 train, test, policy CSV 및 scaler JSON.

가격 수준과 선형 관계를 유지하는 Z-score를 기본으로 선택합니다.
국제가격 급등도 경제적 신호이므로 이상치로 잘라내지 않습니다.
어떤 방법이 예측에 최적인지는 학습 구간 내부 교차검증으로 비교해야 합니다.
타겟은 원/L로 유지합니다. 기존 회귀의 계수 합=1 제약은 단위에 의존하므로
이 출력은 기존 회귀 입력에 자동 적용하지 않습니다.
교차검증에서는 매 fold의 학습 데이터로 fit_scaler를 다시 호출해야 합니다.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "딥러닝"))
from load_training_set import MODELS, load_training_set, load_weekly_table  # noqa: E402


def fit_scaler(train):
    """반드시 현재 학습 구간만 전달합니다 (모집단 표준편차, ddof=0)."""
    X = np.asarray(train["X"], dtype=np.float64)
    if X.ndim != 2 or not X.size or not np.isfinite(X).all():
        raise ValueError("학습 설명변수는 유한한 값의 비어 있지 않은 2차원 배열이어야 합니다.")
    mean, std = X.mean(axis=0), X.std(axis=0)
    return {
        "method": "z_score",
        "features": list(train["설명변수"]),
        "mean": mean.tolist(),
        "scale": np.where(std == 0, 1.0, std).tolist(),
        "constant_features": np.flatnonzero(std == 0).tolist(),
        "train_rows": len(X),
        "train_start": str(train["날짜"][0]),
        "train_end": str(train["날짜"][-1]),
        "target_scaled": False,
    }


def transform(data, scaler):
    if list(data["설명변수"]) != scaler["features"]:
        raise ValueError("설명변수 이름 또는 순서가 학습 데이터와 다릅니다.")
    X = np.asarray(data["X"], dtype=np.float64)
    if X.ndim != 2 or X.shape[1] != len(scaler["features"]) or not np.isfinite(X).all():
        raise ValueError("설명변수 크기가 잘못됐거나 결측·무한대가 있습니다.")
    return {**data, "X": (X - np.asarray(scaler["mean"])) / np.asarray(scaler["scale"])}


def inverse_transform(X, scaler):
    """표준화된 설명변수를 국제가격·환율의 원래 단위로 복원합니다."""
    return np.asarray(X, dtype=np.float64) * np.asarray(scaler["scale"]) + np.asarray(scaler["mean"])


def save_csv(data, path):
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["주차", "날짜", *data["설명변수"], data["타겟"]])
        writer.writerows(
            [week, day, *features, float(target)]
            for week, day, features, target in zip(
                data["주차"], data["날짜"], data["X"], data["y"], strict=True
            )
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "csv" / "normalized")
    args = parser.parse_args()
    weekly = load_weekly_table()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for product in MODELS:
        train = load_training_set(product, weekly, "train")
        scaler = fit_scaler(train)
        for split in ("train", "test", "policy"):
            data = train if split == "train" else load_training_set(product, weekly, split)
            save_csv(transform(data, scaler), args.output_dir / f"{product}_{split}.csv")
        (args.output_dir / f"{product}_scaler.json").write_text(
            json.dumps(scaler, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Z-score 표준화 완료: {len(MODELS)}개 제품, 학습 구간만 적합 → {args.output_dir}")


if __name__ == "__main__":
    main()
