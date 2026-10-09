"""1단계 수집.

두 가지 경로로 자료를 모아 data/raw 에 같은 모양의 CSV 로 저장한다.
- 파일: 오피넷에서 내려받은 CSV (input 폴더). API 가 과거 값을 주지 않는 자료.
- API : ECOS, KOSIS, 오피넷, FRED. 매일 바뀌는 값과 공식 통계.
모든 호출은 실패하면 직전에 저장한 파일(캐시)로 대체한다.
offline=True 이면 호출하지 않고 저장본만 읽는다. (시연 중 와이파이 장애 대비)
"""
import io
import re
import time
from datetime import date
from urllib.parse import quote

import pandas as pd
import requests

import config as cfg

TODAY = pd.Timestamp(date.today())


# ═══════════════════════════════════════════════════════════════════════
# 공통 도구
# ═══════════════════════════════════════════════════════════════════════
def http_get(url, params=None, timeout=10, retries=2):
    """GET 요청. 실패하면 1초 쉬고 한 번 더 시도한다. (장애 때 오래 멈추지 않도록 짧게 잡는다)"""
    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as err:
            last_error = err
            if attempt + 1 < retries:
                time.sleep(1)
    raise RuntimeError(f"호출 실패: {url} ({last_error})")


def save_raw(df, name):
    """원자료를 data/raw/<name>.csv 로 저장한다."""
    cfg.RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(cfg.RAW_DIR / f"{name}.csv", index=False, encoding="utf-8-sig")


def load_raw(name):
    """저장된 원자료를 읽는다. 없으면 None."""
    path = cfg.RAW_DIR / f"{name}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, encoding="utf-8-sig")
    for col in ("date", "week_start"):                     # 날짜 열은 다시 날짜형으로
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])
    return df


def with_cache(name, fetch_fn, offline, report):
    """수집 함수를 실행하고 저장한다. 실패하면 저장본으로 대체하고 그 사실을 기록한다."""
    status, message, df = "OK", "", None
    if not offline:
        try:
            df = fetch_fn()
            if df is None or df.empty:
                raise RuntimeError("응답에 자료가 없음")
            save_raw(df, name)
        except Exception as err:  # 어떤 오류든 저장본으로 넘어간다
            message = str(err)[:70]
            df = None
    if df is None:
        df = load_raw(name)
        status = "CACHE" if df is not None else "FAIL"
    last, lag = "", ""
    if df is not None and "date" in df.columns and len(df):
        last_ts = pd.to_datetime(df["date"]).max()
        last, lag = str(last_ts.date()), (TODAY - last_ts).days     # 실제 마지막 관측일과 지연 일수
    report.append({"name": name, "status": status, "rows": 0 if df is None else len(df),
                   "last_date": last, "lag_days": lag, "message": message})
    return df


# ═══════════════════════════════════════════════════════════════════════
# 한국은행 ECOS
# ═══════════════════════════════════════════════════════════════════════
def _ecos_period(ts, cycle):
    """조회 기간을 주기별 형식으로 바꾼다. D=YYYYMMDD, M=YYYYMM, Q=YYYYQn, A=YYYY"""
    ts = pd.Timestamp(ts)
    if cycle == "D":
        return ts.strftime("%Y%m%d")
    if cycle == "M":
        return ts.strftime("%Y%m")
    if cycle == "Q":
        return f"{ts.year}Q{ts.quarter}"
    return str(ts.year)


def parse_ecos_time(text, cycle):
    """ECOS 의 TIME 값을 날짜로 바꾼다. 주기마다 형식이 다르다."""
    text = str(text)
    if cycle == "D":
        return pd.to_datetime(text, format="%Y%m%d")
    if cycle == "M":
        return pd.to_datetime(text, format="%Y%m")
    if cycle == "Q":
        year, quarter = int(text[:4]), int(text[-1])
        return pd.Timestamp(year=year, month=3 * quarter - 2, day=1)
    return pd.Timestamp(year=int(text[:4]), month=1, day=1)


def fetch_ecos(stat, cycle, items, start=cfg.START_DATE, end=None, page=10000):
    """ECOS StatisticSearch 로 시계열을 받는다. 반환: date, value"""
    key = cfg.get_key("ecos")
    if not key:
        raise RuntimeError("ECOS_API_KEY 없음")
    end = end or TODAY
    rows, first = [], 1
    while True:
        parts = ["https://ecos.bok.or.kr/api/StatisticSearch", key, "json", "kr",
                 str(first), str(first + page - 1), stat, cycle,
                 _ecos_period(start, cycle), _ecos_period(end, cycle), *items]
        data = http_get("/".join(parts)).json()
        # ECOS 는 오류도 HTTP 200 으로 보내므로 RESULT 코드를 직접 확인한다
        if "StatisticSearch" not in data:
            result = data.get("RESULT", {})
            if result.get("CODE") == "INFO-200":   # 해당 자료 없음
                break
            raise RuntimeError(f"ECOS 오류: {result.get('CODE')} {result.get('MESSAGE')}")
        block = data["StatisticSearch"]
        rows.extend(block["row"])
        if first + page - 1 >= int(block["list_total_count"]):   # 마지막 쪽이면 종료
            break
        first += page
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    out = pd.DataFrame({
        "date": [parse_ecos_time(t, cycle) for t in df["TIME"]],
        "value": pd.to_numeric(df["DATA_VALUE"], errors="coerce"),   # 값이 문자열로 온다
    })
    return out.dropna().sort_values("date").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# 통계청 KOSIS
# ═══════════════════════════════════════════════════════════════════════
KOSIS_META_URL = "https://kosis.kr/openapi/statisticsData.do"
KOSIS_DATA_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


def _kosis_check(data):
    """KOSIS 오류 응답({'err': '30', 'errMsg': ...})을 예외로 바꾼다."""
    if isinstance(data, dict) and "err" in data:
        raise RuntimeError(f"KOSIS 오류 {data.get('err')}: {data.get('errMsg')}")
    return data


def kosis_url(base, params):
    """KOSIS 요청 주소를 직접 조립한다.

    여러 코드를 묶는 '+' 는 제공 자료의 예시처럼 글자 그대로 보내야 한다.
    requests 의 params 에 맡기면 '+' 가 %2B 로 바뀌므로 itmId·objL 값만 '+' 를 보존한다.
    인증키는 '+', '=' 같은 문자가 들어 있을 수 있어 전부 인코딩한다.
    """
    parts = []
    for name, value in params.items():
        keep_plus = name == "itmId" or name.startswith("objL")
        parts.append(f"{name}={quote(str(value), safe='+' if keep_plus else '')}")
    return base + "?" + "&".join(parts)


def fetch_kosis_meta(org_id, tbl_id):
    """통계표의 항목·분류 코드 목록을 받는다. (itmId, objL 값을 찾는 데 쓴다)"""
    key = cfg.get_key("kosis")
    if not key:
        raise RuntimeError("KOSIS_API_KEY 없음")
    params = {"method": "getMeta", "type": "ITM", "apiKey": key, "format": "json",
              "jsonVD": "Y", "orgId": org_id, "tblId": tbl_id}
    return pd.DataFrame(_kosis_check(http_get(kosis_url(KOSIS_META_URL, params)).json()))


def build_kosis_query(meta):
    """메타에서 휘발유·경유·등유 품목 코드를 찾아 조회 조건을 만든다.

    반환: (조회 조건 dict, 품목 코드→유종 dict, 품목이 들어 있는 분류 단계 번호)
    자동 탐색이 틀릴 수 있으므로 선택한 코드를 화면에 출력한다.
    """
    items = meta[meta["OBJ_ID"] == "ITEM"]                 # 항목(지수 등)
    classes = meta[meta["OBJ_ID"] != "ITEM"]               # 분류(품목, 지역 등)
    levels = list(dict.fromkeys(classes["OBJ_ID"]))        # 분류 단계 순서 = objL1, objL2 ...

    code_to_fuel, fuel_level = {}, None
    for fuel, info in cfg.FUELS.items():
        # 이름이 정확히 일치하는 품목을 먼저 찾고, 없으면 포함하는 품목을 찾는다
        exact = classes[classes["ITM_NM"].str.strip() == info["keyword"]]
        hit = exact if len(exact) else classes[classes["ITM_NM"].str.contains(info["keyword"], na=False)]
        if hit.empty:
            raise RuntimeError(f"KOSIS 메타에서 '{info['keyword']}' 품목을 찾지 못함")
        row = hit.iloc[0]
        code_to_fuel[str(row["ITM_ID"])] = fuel
        fuel_level = levels.index(row["OBJ_ID"]) + 1

    query = {"itmId": str(items.iloc[0]["ITM_ID"]) if len(items) else "ALL"}
    for i, obj in enumerate(levels, start=1):
        if i == fuel_level:
            query[f"objL{i}"] = "+".join(code_to_fuel)     # '+' 로 여러 품목을 한 번에
        else:
            sub = classes[classes["OBJ_ID"] == obj]
            nation = sub[sub["ITM_NM"].str.contains("전국", na=False)]
            query[f"objL{i}"] = str((nation if len(nation) else sub).iloc[0]["ITM_ID"])
    print(f"  [KOSIS] 자동 선택한 조회 조건: {query}")
    return query, code_to_fuel, fuel_level


def fetch_kosis_fuel_cpi(start=cfg.START_DATE, end=None):
    """휘발유·경유·등유 소비자물가지수(월)를 받는다. 반환: date, fuel, value"""
    key = cfg.get_key("kosis")
    if not key:
        raise RuntimeError("KOSIS_API_KEY 없음")
    table = cfg.KOSIS_CPI_TABLE
    meta = fetch_kosis_meta(table["orgId"], table["tblId"])
    query, code_to_fuel, level = build_kosis_query(meta)
    if cfg.KOSIS_CPI_QUERY:                                # 직접 지정한 조건이 있으면 그것을 쓴다
        query = dict(cfg.KOSIS_CPI_QUERY)
    params = {"method": "getList", "apiKey": key, "format": "json", "jsonVD": "Y",
              "orgId": table["orgId"], "tblId": table["tblId"], "prdSe": "M",
              "startPrdDe": pd.Timestamp(start).strftime("%Y%m"),
              "endPrdDe": pd.Timestamp(end or TODAY).strftime("%Y%m"), **query}
    df = pd.DataFrame(_kosis_check(http_get(kosis_url(KOSIS_DATA_URL, params)).json()))
    out = pd.DataFrame({
        "date": pd.to_datetime(df["PRD_DE"].astype(str), format="%Y%m"),
        "fuel": df[f"C{level}"].astype(str).map(code_to_fuel),
        "value": pd.to_numeric(df["DT"], errors="coerce"),
    })
    return out.dropna().sort_values(["fuel", "date"]).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# 오피넷
# ═══════════════════════════════════════════════════════════════════════
OPINET_URL = "https://www.opinet.co.kr/api"
OPINET_CODE_TO_FUEL = {info["opinet"]: fuel for fuel, info in cfg.FUELS.items()}


def _opinet_rows(endpoint, extra=None):
    """오피넷 API 를 호출해 RESULT.OIL 목록을 돌려준다."""
    key = cfg.get_key("opinet")
    if not key:
        raise RuntimeError("OPINET_API_KEY 없음")
    params = {"certkey": key, "out": "json", **(extra or {})}
    return http_get(f"{OPINET_URL}/{endpoint}", params=params).json()["RESULT"]["OIL"]


def _opinet_fuel(row):
    """응답 한 줄이 어느 유종인지 찾는다. 제품 코드를 먼저 보고, 없으면 제품명으로 판단한다."""
    fuel = OPINET_CODE_TO_FUEL.get(row.get("PRODCD"))
    if fuel:
        return fuel
    name = str(row.get("PRODNM", ""))
    if "고급" in name:                       # 고급휘발유는 상한 대상이 아니다
        return None
    for fuel, info in cfg.FUELS.items():
        if info["keyword"] in name:
            return fuel
    return None


def fetch_opinet_today():
    """오늘의 전국 주유소 평균 판매가. 반환: date, fuel, price"""
    rows = []
    for r in _opinet_rows("avgAllPrice.do"):
        fuel = _opinet_fuel(r)
        if fuel:
            rows.append({"date": pd.to_datetime(str(r["TRADE_DT"]), format="%Y%m%d"),
                         "fuel": fuel, "price": float(r["PRICE"])})
    return pd.DataFrame(rows)


def fetch_opinet_recent():
    """최근 7일 전국 일일 평균 판매가. 반환: date, fuel, price"""
    rows = []
    for fuel, info in cfg.FUELS.items():
        for r in _opinet_rows("avgRecentPrice.do", {"prodcd": info["opinet"]}):
            day = r.get("DATE") or r.get("TRADE_DT")       # 필드 이름이 명세마다 다를 수 있다
            rows.append({"date": pd.to_datetime(str(day), format="%Y%m%d"),
                         "fuel": fuel, "price": float(r["PRICE"])})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# FRED (키 없이 CSV 로 받는다)
# ═══════════════════════════════════════════════════════════════════════
def fetch_fred(series_id):
    """FRED 일별 시계열을 받는다. 반환: date, value"""
    resp = http_get("https://fred.stlouisfed.org/graph/fredgraph.csv", params={"id": series_id})
    raw = pd.read_csv(io.StringIO(resp.text))
    out = pd.DataFrame({
        "date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),      # 첫 열이 날짜
        "value": pd.to_numeric(raw.iloc[:, 1], errors="coerce"),      # 휴일은 '.' 로 온다
    })
    out = out.dropna()
    return out[out["date"] >= cfg.START_DATE].reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════
# 오피넷에서 내려받은 파일 (input 폴더)
# ═══════════════════════════════════════════════════════════════════════
def parse_kdate(text):
    """'2015년01월01일' 또는 '15년01월02일' 을 날짜로 바꾼다."""
    y, m, d = (int(x) for x in re.findall(r"\d+", str(text))[:3])
    return pd.Timestamp(y + 2000 if y < 100 else y, m, d)


def week_label_to_range(label):
    """'26년09월4주' 같은 주차 표기를 (일요일, 토요일) 날짜로 바꾼다.

    오피넷의 주간 자료는 일요일~토요일을 한 주로 묶고, 그 주의 수요일이 속한 달에 배정한다.
    (2015년 1월 ~ 2026년 9월 141개 달의 주 수가 이 규칙과 모두 일치함을 확인했다.)
    따라서 'M월 N주' 는 그 달의 N번째 수요일이 들어 있는 주다.
    """
    y, m, n = (int(x) for x in re.findall(r"\d+", str(label))[:3])
    first = pd.Timestamp(y + 2000 if y < 100 else y, m, 1)
    wednesday = first + pd.Timedelta(days=(2 - first.dayofweek) % 7 + 7 * (n - 1))   # 월=0 … 수=2
    return wednesday - pd.Timedelta(days=3), wednesday + pd.Timedelta(days=3)


def read_input_csv(kind):
    """input 폴더에서 해당 종류의 파일을 모두 읽어 하나로 합친다. 없으면 None."""
    files = sorted(cfg.INPUT_DIR.glob(cfg.INPUT_FILES[kind]))
    frames = []
    for path in files:
        try:
            frames.append(pd.read_csv(path, encoding="cp949"))        # 오피넷 CSV 는 cp949
        except UnicodeDecodeError:
            frames.append(pd.read_csv(path, encoding="utf-8-sig"))
    if not frames:
        return None
    raw = pd.concat(frames, ignore_index=True)
    return raw.drop_duplicates(subset=[raw.columns[0]], keep="last")   # 나눠 받은 파일의 겹친 기간 정리


def _to_long(raw, dates, column_key, value_name):
    """넓은 표(유종이 열)를 긴 표(date, fuel, 값)로 바꾼다. 고급휘발유 등 쓰지 않는 열은 버린다."""
    frames = []
    for fuel, info in cfg.FUELS.items():
        if info[column_key] not in raw.columns:
            raise RuntimeError(f"파일에 '{info[column_key]}' 열이 없음. 실제 열: {list(raw.columns)}")
        values = pd.to_numeric(raw[info[column_key]].astype(str).str.replace(",", ""), errors="coerce")
        frames.append(pd.DataFrame({"date": dates, "fuel": fuel, value_name: values.to_numpy()}))
    out = pd.concat(frames).dropna()
    out = out[out["date"] >= cfg.START_DATE]
    return out.sort_values(["fuel", "date"]).reset_index(drop=True)


def load_retail_file():
    """주유소 평균 판매가(일간, 원/리터). 반환: date, fuel, price"""
    raw = read_input_csv("retail")
    if raw is None:
        raise RuntimeError("input 폴더에 retail_daily*.csv 없음")
    dates = pd.DatetimeIndex([parse_kdate(t) for t in raw.iloc[:, 0]])
    return _to_long(raw, dates, "col", "price")


def load_intl_file():
    """국제 석유제품 가격(싱가포르, 일간, 달러/배럴). 반환: date, fuel, usd_bbl"""
    raw = read_input_csv("intl")
    if raw is None:
        raise RuntimeError("input 폴더에 intl_daily*.csv 없음")
    dates = pd.DatetimeIndex([parse_kdate(t) for t in raw.iloc[:, 0]])
    out = _to_long(raw, dates, "intl_col", "usd_bbl")
    if out["usd_bbl"].median() > 500:      # 달러/배럴이면 수십~200 안팎. 그보다 크면 원 단위로 받은 것이다
        raise RuntimeError("국제 가격이 달러/배럴이 아닌 것으로 보임. 단위를 '$' 로 두고 다시 받을 것")
    return out


def load_supply_file(kind):
    """정유사 주간 공급가(원/리터). kind = 'supply_pretax' 또는 'supply_aftertax'.

    반환: date(주의 마지막 날인 토요일), week_start(일요일), label(원래 표기), fuel, price
    """
    raw = read_input_csv(kind)
    if raw is None:
        raise RuntimeError(f"input 폴더에 {cfg.INPUT_FILES[kind]} 없음")
    ranges = [week_label_to_range(t) for t in raw.iloc[:, 0]]
    ends = pd.DatetimeIndex([r[1] for r in ranges])
    out = _to_long(raw, ends, "col", "price")
    lookup = pd.DataFrame({"date": ends, "week_start": [r[0] for r in ranges],
                           "label": raw.iloc[:, 0].astype(str).to_numpy()})
    return out.merge(lookup, on="date", how="left")[["date", "week_start", "label", "fuel", "price"]]


# ═══════════════════════════════════════════════════════════════════════
# 전체 수집
# ═══════════════════════════════════════════════════════════════════════
def merge_retail(file_df, recent_df, today_df):
    """내려받은 판매가에 API 로 받은 최근 값을 덧붙인다. 겹치는 날은 나중에 받은 값을 쓴다.

    오피넷 API 는 최근 며칠만 주므로, 파일 이후 날짜는 이렇게 이어 붙여 이력을 쌓는다.
    """
    parts = [load_raw("opinet_retail"), file_df, recent_df, today_df]
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return None
    merged = pd.concat(parts)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.drop_duplicates(["date", "fuel"], keep="last")
    return merged.sort_values(["fuel", "date"]).reset_index(drop=True)


def collect_all(offline=False):
    """모든 자료를 받아 {이름: 표} 와 수집 결과 보고를 돌려준다."""
    cfg.load_env()
    report, raw = [], {}

    # ① 내려받은 파일: 오프라인이어도 읽는다 (인터넷이 필요 없다)
    file_retail = with_cache("file_retail", load_retail_file, False, report)
    raw["opinet_intl"] = with_cache("opinet_intl", load_intl_file, False, report)
    for kind in ("supply_pretax", "supply_aftertax"):
        raw[f"opinet_{kind}"] = with_cache(f"opinet_{kind}", lambda k=kind: load_supply_file(k), False, report)

    # ② API
    for name, spec in cfg.ECOS_SERIES.items():
        raw[f"ecos_{name}"] = with_cache(
            f"ecos_{name}", lambda s=spec: fetch_ecos(s["stat"], s["cycle"], s["items"]),
            offline, report)
    raw["kosis_fuel_cpi"] = with_cache("kosis_fuel_cpi", fetch_kosis_fuel_cpi, offline, report)
    today = with_cache("opinet_today", fetch_opinet_today, offline, report)
    recent = with_cache("opinet_recent", fetch_opinet_recent, offline, report)
    for name, series_id in cfg.FRED_SERIES.items():
        raw[f"fred_{name}"] = with_cache(
            f"fred_{name}", lambda s=series_id: fetch_fred(s), offline, report)

    # ③ 판매가 = 파일 + API 최근 값
    raw["opinet_retail"] = with_cache(
        "opinet_retail", lambda: merge_retail(file_retail, recent, today), False, report)
    return raw, pd.DataFrame(report)
