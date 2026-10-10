# 2~4단계(재정·물가·최적화) 공통 설정과 자료 불러오기입니다.
# 값을 바꿀 때는 이 파일만 고칩니다. 확인하지 못한 값은 None 으로 두고, 실행할 때 명령행으로 넣을 수 있습니다.
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "regression" / "full_sample"))
import counterfactual as cfm  # 1단계 회귀(주차 변환, 주별 표, 유종 정의)를 그대로 다시 씁니다

CF_DIR = PROJECT_DIR / "results" / "regression" / "full_sample"
CF_PATH = CF_DIR / "counterfactual.csv"                  # 1단계 주별 "없었다면" 가격과 격차
MODELS_PATH = CF_DIR / "counterfactual_models.csv"       # 1단계 계수(비중 w, 마진 m)
INTL_PATH = PROJECT_DIR / "csv" / "international_petroleum_prices_2015011_2026101_krw.csv"  # 최신 국제가(원/리터)
RETAIL_PATH = PROJECT_DIR / "csv" / "gas_station_weekly_prices.csv"                         # 주유소 주간 판매가
OUT_DIR = PROJECT_DIR / "results" / "policy"

FUELS = ["gasoline", "diesel", "kerosene"]
KO = {"gasoline": "휘발유", "diesel": "경유", "kerosene": "등유"}
VAT = cfm.VAT
POLICY_START = cfm.POLICY_START                          # 2026-03-13

# ── 확인된 값 (CLAUDE.md '자료에서 확인한 사실') ──────────────────────────────
CURRENT_CAP = {"gasoline": 1784.0, "diesel": 1773.0, "kerosene": 1380.0}   # 10차 상한(세후 공급가, 원/리터)
FUEL_TAX = {"gasoline": 698.5, "diesel": 436.3, "kerosene": 80.2}         # 현재 유류세(부가세 포함). 11월 말 이후는 결정 필요
# 상한 조정 후 판매가 반응(조정 전날 대비, 원). {차수: (상한 변화, {유종: (1주, 2주, 4주)})}
RESPONSE = {
    2: (210, {"gasoline": (112, 170, 187), "diesel": (107, 167, 185), "kerosene": (56, 85, 104)}),
    7: (-150, {"gasoline": (-100, -124, -135), "diesel": (-103, -130, -142), "kerosene": (-25, -35, -43)}),
}

# ── 미확인 값: 출처 확인 후 채웁니다 (명령행 --volume, --weights, --cpi-limit 로도 넣을 수 있음) ──
DAILY_VOLUME_L = None   # {"gasoline": 리터/일, "diesel": ..., "kerosene": ...}  석유공사 석유수급통계
CPI_WEIGHTS = None      # {"gasoline": 1000 기준 가중치, ...}                    국가데이터처 소비자물가 가중치
CPI_LIMIT_PP = None     # 월 물가 기여도 한도(%p). 팀 결정

# ── 최적화 설정 (합의안 + 제안값) ─────────────────────────────────────────────
ACTIONS = np.array([-60, 0, 30, 60, 90, 120], dtype=float)  # 유종별 조정 폭(원). 최대 인상폭 = 120원
ROUNDS = 6               # 11차부터 6차수
ROUND_WEEKS = 4          # 차수 간격 4주
N_PATHS = 500            # 국제가 경로 수
BLOCK = 4                # 블록 부트스트랩 묶음 길이(주)
SEED = 42                # 난수 시드(재현용)
BOOT_START = pd.Timestamp("2015-01-01")   # 경로 추출 기간 시작. 끝은 상한제 직전. 결정 필요(최근 3년이면 2023-03-01)
PASS_MODE = "path"       # "path" = 1·2·4주 반영 경로, "4w" = 4주 값만




# 1. 오피넷 주차("2026년09월4주" 또는 "26년09월4주")를 그 주 토요일로 바꿉니다. 1단계 규칙을 그대로 씁니다.
def to_week_end(label):
    label = str(label).strip()
    return cfm.week_end(label if label[:4].isdigit() and label[4] == "년" else "20" + label)


# 2. 1단계 결과(주별 "없었다면" 가격과 격차)를 읽습니다.
def load_cf():
    cf = pd.read_csv(CF_PATH, encoding="utf-8-sig", index_col="week_end", parse_dates=True)
    return cf


# 3. 1단계 '최근 마진' 모형의 계수를 {유종: (마진 m, 비중 w[3])}로 돌려줍니다.
def load_params():
    m = pd.read_csv(MODELS_PATH, encoding="utf-8-sig")
    m = m[m["model"] == cfm.CHOSEN].set_index("fuel")
    return {f: (float(m.loc[f, "const"]), m.loc[f, ["w_lag0", "w_lag1", "w_lag2"]].to_numpy(float)) for f in FUELS}


# 4. 국제가(원/리터) 주별 표. 공급가보다 최근 주까지 있어 결정 시점의 상태로 씁니다.
def load_intl():
    raw = pd.read_csv(INTL_PATH, encoding="cp949")
    cols = {"gasoline": raw.columns[2], "diesel": raw.columns[4], "kerosene": raw.columns[3]}  # 92RON, 0.001%, 등유
    out = pd.DataFrame({f: raw[c].to_numpy(float) for f, c in cols.items()},
                       index=pd.DatetimeIndex([to_week_end(x) for x in raw.iloc[:, 0]], name="week_end"))
    return out


# 5. 최신 주유소 판매가(원/리터)와 기준 주차. 물가 기여도의 분모로 씁니다.
def latest_retail():
    raw = pd.read_csv(RETAIL_PATH, encoding="cp949")
    last = raw.iloc[-1]
    return {"gasoline": float(last.iloc[2]), "diesel": float(last.iloc[3]), "kerosene": float(last.iloc[4])}, str(last.iloc[0])


# 6. "없었다면" 세후 공급가 = (m + w0·x_t + w1·x_(t−1) + w2·x_(t−2)) × 1.1 + 유류세.
# x 의 마지막 축이 시간입니다. 앞 두 주는 시차용이라 결과 길이는 x 보다 2 짧습니다. (미래 경로에도 같은 식을 씁니다)
def cf_aftertax(x, margin, w, tax):
    pretax = margin + w[0] * x[..., 2:] + w[1] * x[..., 1:-1] + w[2] * x[..., :-2]
    return pretax * (1 + VAT) + tax


# 7. 명령행 인자 "a,b,c"를 유종별 사전으로 바꿉니다. 비어 있으면 config 값을 씁니다.
def parse_triplet(text, default):
    if text is None:
        return default
    values = [float(v) for v in text.split(",")]
    if len(values) != 3:
        raise SystemExit("값 3개(휘발유,경유,등유)를 쉼표로 넣으세요.")
    return dict(zip(FUELS, values))
