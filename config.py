"""설정 파일.

API 키 이름, 통계 코드, 유종 정의, 파일 이름, 단위 환산 상수를 한곳에 모은다.
None 으로 둔 값은 직접 확인해서 채워야 하는 항목이다.
"""
from pathlib import Path
import os

import pandas as pd

# ── 폴더 ────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "input"               # 오피넷에서 내려받은 CSV 를 넣는 곳
RAW_DIR = BASE_DIR / "data" / "raw"          # 정리한 원자료(캐시 겸용)
OUT_DIR = BASE_DIR / "data" / "processed"    # 분석용 표
CAPS_FILE = BASE_DIR / "caps.csv"            # 차수별 최고가격 (직접 입력)

# ── 기간 ────────────────────────────────────────────────────────────────
START_DATE = "2015-01-01"      # 학습 자료 시작일
POLICY_START = "2026-03-13"    # 석유 최고가격제 시행일
TODAY = pd.Timestamp.today().normalize()   # 수집·전처리가 같은 '오늘'을 쓰도록 한곳에서 정한다


# ── API 키: 코드에 직접 쓰지 않고 환경변수 또는 .env 에서 읽는다 ─────────
def load_env(path=BASE_DIR / ".env"):
    """.env 파일(KEY=VALUE 형식)이 있으면 환경변수로 올린다."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


ENV_KEYS = {
    "ecos": "ECOS_API_KEY",      # 한국은행 ECOS
    "kosis": "KOSIS_API_KEY",    # 통계청 KOSIS
    "opinet": "OPINET_API_KEY",  # 오피넷 유가정보
}


def get_key(name):
    """환경변수에서 키를 읽는다. 없으면 None."""
    return os.environ.get(ENV_KEYS[name])


# ── 유종 정의 ───────────────────────────────────────────────────────────
# col      : 오피넷 국내 가격 파일의 열 이름
# intl_col : 오피넷 국제 석유제품(싱가포르) 파일의 열 이름
# opinet   : 오피넷 API 제품 코드
# keyword  : KOSIS 품목 이름에서 찾을 단어
FUELS = {
    "gasoline": {"ko": "휘발유", "col": "보통휘발유", "intl_col": "휘발유(92RON)",
                 "opinet": "B027", "keyword": "휘발유"},
    "diesel": {"ko": "경유", "col": "자동차용경유", "intl_col": "경유(0.001%)",
               "opinet": "D047", "keyword": "경유"},
    "kerosene": {"ko": "등유", "col": "실내등유", "intl_col": "등유",
                 "opinet": "C004", "keyword": "등유"},
}

# ── 오피넷에서 내려받은 파일 (input 폴더). 연도별로 나눠 받았으면 이름 뒤에 _2015 처럼 붙인다 ──
INPUT_FILES = {
    "retail": "retail_daily*.csv",                    # 주유소 평균판매가격, 일간, 원/리터
    "supply_pretax": "supply_weekly_pretax*.csv",     # 정유사 주간공급가격, 세전, 원/리터
    "supply_aftertax": "supply_weekly_aftertax*.csv", # 정유사 주간공급가격, 세후, 원/리터
    "intl": "intl_daily*.csv",                        # 국제 석유제품(싱가포르), 일간, 달러/배럴
}

# ── ECOS 통계 코드 ──────────────────────────────────────────────────────
# 환율·물가·금리 코드는 대회 제공 자료에 실린 값.
# 국제유가는 '9.1.6.3. 국제상품가격'(902Y003) 표의 항목 (2026-10-09 항목 목록에서 확인, 월별).
ECOS_SERIES = {
    "fx_krw_usd": {"stat": "731Y001", "cycle": "D", "items": ["0000001"]},
    "cpi_total": {"stat": "901Y009", "cycle": "M", "items": ["0"]},
    "base_rate": {"stat": "722Y001", "cycle": "M", "items": ["0101000"]},
    "dubai_usd_bbl": {"stat": "902Y003", "cycle": "M", "items": ["010102"]},
}

# ── KOSIS 품목별 소비자물가지수 ─────────────────────────────────────────
# DT_1J22112 = '품목별 소비자물가지수(품목성질별: 2020=100)'. 대회 제공 자료의 검색 결과에 나온 표.
# 표 안의 품목 코드는 실행 시 메타 조회로 자동 탐색한다. 탐색 결과를 한 번 눈으로 확인할 것.
KOSIS_CPI_TABLE = {"orgId": "101", "tblId": "DT_1J22112"}
# 자동 탐색이 틀리면 아래에 직접 지정한다. 예: {"itmId": "T", "objL1": "T10", "objL2": "D02+D03+D04"}
KOSIS_CPI_QUERY = None

# ── FRED 계열 (키 없이 CSV 주소로 받는다) ───────────────────────────────
FRED_SERIES = {
    "brent_usd_bbl": "DCOILBRENTEU",   # 브렌트유, 달러/배럴, 일별
    "fx_krw_usd_fred": "DEXKOUS",      # 원/달러, 일별. ECOS 환율을 못 받을 때 대신 쓴다
}

# ── 단위 환산과 세금 ────────────────────────────────────────────────────
BARREL_L = 158.987       # 1 배럴 = 158.987 리터
VAT = 0.10               # 부가가치세율
# 세후 공급가 = 세전 공급가 × (1 + VAT) + 유류세(부가세 포함)
# 이 관계가 내려받은 자료에서 그대로 성립하므로, 유류세는 '세후 − 1.1 × 세전' 으로 역산한다.

# ── 직접 채워야 하는 값 ─────────────────────────────────────────────────
# 소비자물가 가중치 (전체 1000 기준). 국가데이터처 가중치 표에서 확인.
CPI_WEIGHTS = {"gasoline": None, "diesel": None, "kerosene": None}
# 유종별 하루 판매량 (리터). 석유공사 통계에서 확인.
DAILY_VOLUME_L = {"gasoline": None, "diesel": None, "kerosene": None}
