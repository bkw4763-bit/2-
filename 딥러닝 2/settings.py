"""주간 정유사 세전 공급가격 실험의 경로와 사전 고정 설정."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT_DIR = HERE / "data"
POLICY_START = "2026-03-13"
HORIZONS = [1, 2, 4]  # 주간 관측 수
TEST_START = "2024-03-13"  # 정책 시행 직전 약 2년을 최종 보류
VALIDATION_START = "2022-03-13"  # 테스트 이전 개발 기간 안의 롤링 검증
CONTEXT_YEARS = list(range(1, 16))  # 실제 달력 연수
ORIGIN_STEP_WEEKS = 1
RESULT_DIR = HERE / "output" / "rolling_cv"
DATA_START = "2015-01-01"
TIME_FREQ = "W-SAT"  # 공급실적의 일~토 주간, 토요일을 관측일로 표시한다.
DAILY_FREQ = "B"  # 기존 KNN은 일간 공변량에 적용한 후 주간 평균을 낸다.
KNN_NEIGHBORS = 5
KNN_HISTORY_ROWS = 252
MODEL_ID = "google/timesfm-3.0-pytorch"
XGB_EARLY_STOP_WEEKS = 13
XGB_EARLY_STOP_ROUNDS = 30
XGB_MAX_TREES = 1000
XGB_CANDIDATES = [
    {"max_depth": 2, "learning_rate": 0.03, "reg_lambda": 10.0},
    {"max_depth": 3, "learning_rate": 0.03, "reg_lambda": 10.0},
]
FUELS = {
    "gasoline": ("보통휘발유", "휘발유(92RON)"),
    "diesel": ("자동차용경유", "경유(0.001%)"),
    "kerosene": ("실내등유", "등유"),
}
