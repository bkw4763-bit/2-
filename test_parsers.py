"""수집·전처리 함수 점검. 키와 인터넷 없이 실행한다: python test_parsers.py

각 기관의 응답 모양을 흉내 낸 가짜 응답과 작은 표본 파일을 넣어, 함수가 올바른 표를 만드는지 확인한다.
ECOS 응답은 실제 호출 결과(2026-10-09, 원/달러 환율)를 그대로 옮긴 것이다.
KOSIS·오피넷 응답은 대회 제공 자료의 예시와 같은 필드 이름으로 만든 것이다.
"""
import os

import pandas as pd

import collect
import config as cfg

os.environ.update({"ECOS_API_KEY": "k", "KOSIS_API_KEY": "ab+c/d=", "OPINET_API_KEY": "k"})


class FakeResponse:
    """requests 응답 대역."""
    def __init__(self, payload=None, text=""):
        self._payload, self.text = payload, text

    def json(self):
        return self._payload


def fake_http(routes, calls):
    """주소에 들어 있는 단어로 가짜 응답을 골라 주는 http_get 대역을 만든다."""
    def _get(url, params=None, **_):
        calls.append((url, params))
        for word, resp in routes.items():
            if word in url:
                return resp
        raise RuntimeError("등록되지 않은 주소: " + url)
    return _get


def test_ecos():
    row = lambda t, v: {"STAT_CODE": "731Y001", "ITEM_CODE1": "0000001", "UNIT_NAME": "원",
                        "TIME": t, "DATA_VALUE": v}
    ok = FakeResponse({"StatisticSearch": {"list_total_count": 3, "row": [
        row("20260901", "1373.6"), row("20260902", "1370.3"), row("20260903", "1369.4")]}})
    calls = []
    collect.http_get = fake_http({"StatisticSearch": ok}, calls)
    df = collect.fetch_ecos("731Y001", "D", ["0000001"], start="2026-09-01", end="2026-09-10")
    assert list(df.columns) == ["date", "value"] and len(df) == 3
    assert df["value"].dtype == float and df["value"].iloc[0] == 1373.6      # 문자열 → 숫자
    assert df["date"].iloc[0] == pd.Timestamp("2026-09-01")
    assert calls[0][0].endswith("/731Y001/D/20260901/20260910/0000001")       # 주소 조립 순서
    # 오류 응답은 HTTP 200 이어도 예외가 나야 한다
    collect.http_get = fake_http({"StatisticSearch": FakeResponse(
        {"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다."}})}, [])
    try:
        collect.fetch_ecos("731Y001", "D", ["0000001"])
        raise AssertionError("오류 응답을 잡지 못함")
    except RuntimeError as err:
        assert "INFO-100" in str(err)
    # 주기별 시점 형식
    assert collect.parse_ecos_time("202608", "M") == pd.Timestamp("2026-08-01")
    assert collect.parse_ecos_time("2026Q2", "Q") == pd.Timestamp("2026-04-01")
    assert collect._ecos_period("2026-10-09", "Q") == "2026Q4"
    print("ECOS 통과")


def test_kosis():
    meta = FakeResponse([
        {"OBJ_ID": "ITEM", "OBJ_NM": "항목", "ITM_ID": "T", "ITM_NM": "소비자물가지수"},
        {"OBJ_ID": "A", "OBJ_NM": "시도별", "ITM_ID": "T10", "ITM_NM": "전국"},
        {"OBJ_ID": "A", "OBJ_NM": "시도별", "ITM_ID": "T11", "ITM_NM": "서울특별시"},
        {"OBJ_ID": "B", "OBJ_NM": "품목별", "ITM_ID": "D01", "ITM_NM": "고급휘발유"},
        {"OBJ_ID": "B", "OBJ_NM": "품목별", "ITM_ID": "D02", "ITM_NM": "휘발유"},
        {"OBJ_ID": "B", "OBJ_NM": "품목별", "ITM_ID": "D03", "ITM_NM": "경유"},
        {"OBJ_ID": "B", "OBJ_NM": "품목별", "ITM_ID": "D04", "ITM_NM": "등유"},
    ])
    rows = [{"C1": "T10", "C2": code, "ITM_ID": "T", "PRD_DE": prd, "DT": val}
            for code, val in (("D02", "121.5"), ("D03", "140.2"), ("D04", "151.0"))
            for prd in ("202608", "202609")]
    calls = []
    collect.http_get = fake_http({"statisticsData.do": meta,
                                  "statisticsParameterData.do": FakeResponse(rows)}, calls)
    df = collect.fetch_kosis_fuel_cpi(start="2026-08-01", end="2026-09-30")
    assert set(df["fuel"]) == {"gasoline", "diesel", "kerosene"} and len(df) == 6
    assert df[df["fuel"] == "gasoline"]["value"].iloc[0] == 121.5
    data_url = calls[-1][0]
    assert "objL2=D02+D03+D04" in data_url        # '+' 가 그대로 남아야 한다 (고급휘발유 D01 은 제외)
    assert "objL1=T10" in data_url                 # 지역은 '전국'
    assert "apiKey=ab%2Bc%2Fd%3D" in data_url      # 인증키의 특수문자는 인코딩
    # 오류 응답
    collect.http_get = fake_http({"statisticsData.do": FakeResponse(
        {"err": "30", "errMsg": "데이터가 존재하지 않습니다."})}, [])
    try:
        collect.fetch_kosis_fuel_cpi()
        raise AssertionError("오류 응답을 잡지 못함")
    except RuntimeError as err:
        assert "30" in str(err)
    print("KOSIS 통과")


def test_opinet():
    today = FakeResponse({"RESULT": {"OIL": [
        {"TRADE_DT": "20261009", "PRODCD": "B034", "PRODNM": "고급휘발유", "PRICE": "2101.11", "DIFF": "+0.1"},
        {"TRADE_DT": "20261009", "PRODCD": "B027", "PRODNM": "휘발유", "PRICE": "1851.20", "DIFF": "+0.5"},
        {"TRADE_DT": "20261009", "PRODCD": "D047", "PRODNM": "자동차용경유", "PRICE": "1848.70", "DIFF": "-0.2"},
        {"TRADE_DT": "20261009", "PRODCD": "C004", "PRODNM": "실내등유", "PRICE": "1479.90", "DIFF": "0"},
        {"TRADE_DT": "20261009", "PRODCD": "K015", "PRODNM": "자동차용부탄", "PRICE": "1100.00", "DIFF": "0"}]}})
    collect.http_get = fake_http({"avgAllPrice": today}, [])
    df = collect.fetch_opinet_today()
    assert len(df) == 3 and set(df["fuel"]) == {"gasoline", "diesel", "kerosene"}
    assert df.set_index("fuel").loc["gasoline", "price"] == 1851.20            # 고급휘발유가 아닌 보통휘발유
    # 제품 코드가 없는 응답도 제품명으로 처리
    no_code = FakeResponse({"RESULT": {"OIL": [
        {"TRADE_DT": "20261009", "PRODNM": "고급휘발유", "PRICE": "2101.11"},
        {"TRADE_DT": "20261009", "PRODNM": "휘발유", "PRICE": "1851.20"},
        {"TRADE_DT": "20261009", "PRODNM": "자동차용경유", "PRICE": "1848.70"},
        {"TRADE_DT": "20261009", "PRODNM": "실내등유", "PRICE": "1479.90"}]}})
    collect.http_get = fake_http({"avgAllPrice": no_code}, [])
    df = collect.fetch_opinet_today()
    assert len(df) == 3 and df.set_index("fuel").loc["gasoline", "price"] == 1851.20
    print("오피넷 통과")


def test_fred():
    text = "observation_date,DCOILBRENTEU\n2014-12-31,55.27\n2026-10-01,104.10\n2026-10-02,.\n2026-10-05,\n2026-10-06,106.30\n"
    collect.http_get = fake_http({"fredgraph": FakeResponse(text=text)}, [])
    df = collect.fetch_fred("DCOILBRENTEU")
    assert len(df) == 2                              # 시작일 이전 1행, 빈 값 2행 제외
    assert df["value"].tolist() == [104.10, 106.30]
    print("FRED 통과")


def test_cache_fallback():
    """호출이 실패하면 저장본으로 넘어가고, 저장본도 없으면 FAIL 로 기록되는지 확인한다."""
    import tempfile
    from pathlib import Path
    cfg.RAW_DIR = Path(tempfile.mkdtemp())
    report = []
    def boom():
        raise RuntimeError("네트워크 끊김")
    assert collect.with_cache("x", boom, False, report) is None and report[-1]["status"] == "FAIL"
    good = pd.DataFrame({"date": pd.to_datetime(["2026-10-01"]), "value": [1.0]})
    assert collect.with_cache("x", lambda: good, False, report) is not None and report[-1]["status"] == "OK"
    df = collect.with_cache("x", boom, False, report)
    assert df is not None and report[-1]["status"] == "CACHE" and "끊김" in report[-1]["message"]
    print("캐시 대체 통과")


def test_week_label():
    """주차 표기 → (일요일, 토요일). 'M월 N주' 는 그 달의 N번째 수요일이 든 주다."""
    T = pd.Timestamp
    assert collect.week_label_to_range("15년01월1주") == (T("2015-01-04"), T("2015-01-10"))
    assert collect.week_label_to_range("26년03월3주") == (T("2026-03-15"), T("2026-03-21"))
    assert collect.week_label_to_range("26년09월4주") == (T("2026-09-20"), T("2026-09-26"))
    assert collect.week_label_to_range("15년04월5주") == (T("2015-04-26"), T("2015-05-02"))   # 달을 넘는 주
    assert collect.parse_kdate("2026년10월08일") == collect.parse_kdate("26년10월08일") == T("2026-10-08")
    print("주차 변환 통과")


def test_input_files():
    """오피넷 내려받기 파일과 같은 모양(cp949, 한글 머리글)의 표본을 만들어 읽어 본다."""
    import tempfile
    from pathlib import Path
    cfg.INPUT_DIR = Path(tempfile.mkdtemp())
    write = lambda name, text: (cfg.INPUT_DIR / name).write_text(text, encoding="cp949")
    write("retail_daily.csv", "구분,고급휘발유,보통휘발유,자동차용경유,실내등유\n"
          "2026년10월07일,2361.24,1857.38,1843.25,1572.61\n2026년10월08일,2366.13,1857.29,1843.20,1572.51\n")
    write("intl_daily.csv", "기간,휘발유(95RON),휘발유(92RON),등유,경유(0.001%),경유(0.05%)\n"
          "26년10월07일,153.67,147.32,173.46,168.36,160.70\n26년10월08일,155.64,150.12,184.79,177.26,169.59\n")
    head = "구분,고급휘발유,보통휘발유,자동차용경유,실내등유\n"
    write("supply_weekly_pretax.csv", head + "26년09월3주,1296.33,985.05,1213.23,1179.36\n26년09월4주,1365.65,984.80,1212.72,1178.15\n")
    write("supply_weekly_aftertax.csv", head + "26년09월3주,2164.03,1782.02,1770.90,1377.51\n26년09월4주,2240.28,1781.75,1770.34,1376.18\n")

    retail = collect.load_retail_file()
    assert len(retail) == 6 and retail.set_index(["fuel", "date"]).loc[("gasoline", "2026-10-08"), "price"] == 1857.29
    intl = collect.load_intl_file()
    pick = intl.set_index(["fuel", "date"])["usd_bbl"]
    assert pick[("gasoline", "2026-10-08")] == 150.12       # 95RON 이 아니라 92RON
    assert pick[("diesel", "2026-10-08")] == 177.26         # 0.05% 가 아니라 0.001%
    assert pick[("kerosene", "2026-10-08")] == 184.79
    aft = collect.load_supply_file("supply_aftertax")
    row = aft[(aft["fuel"] == "gasoline") & (aft["label"] == "26년09월4주")].iloc[0]
    assert row["price"] == 1781.75 and row["date"] == pd.Timestamp("2026-09-26") and row["week_start"] == pd.Timestamp("2026-09-20")

    # 나눠 받은 파일은 합쳐서 읽는다
    write("retail_daily_2.csv", "구분,고급휘발유,보통휘발유,자동차용경유,실내등유\n2026년10월09일,2366.00,1857.00,1843.00,1572.00\n")
    assert len(collect.load_retail_file()) == 9
    # 원 단위로 받은 국제 가격은 걸러 낸다
    write("intl_daily.csv", "기간,휘발유(95RON),휘발유(92RON),등유,경유(0.001%),경유(0.05%)\n26년10월08일,1311.03,1264.53,1556.57,1493.14,1428.53\n")
    try:
        collect.load_intl_file()
        raise AssertionError("원 단위 파일을 걸러 내지 못함")
    except RuntimeError as err:
        assert "달러" in str(err)
    print("내려받은 파일 읽기 통과")


def test_prepare():
    """환산과 유류세 역산을 확인한다. 10월 8일 환율 1339.2원이면 휘발유 150.12달러/배럴 = 1264.5원/리터
    (오피넷 화면이 같은 날 원 단위로 보여 준 1264.53원과 일치)."""
    import tempfile
    from pathlib import Path
    import prepare
    days = pd.date_range("2026-09-01", "2026-10-08", freq="D")
    long = lambda value, col: pd.concat([pd.DataFrame({"date": days, "fuel": f, col: value}) for f in cfg.FUELS])
    week = lambda price: pd.concat([pd.DataFrame({"date": pd.to_datetime(["2026-09-19", "2026-09-26"]),
                                                   "fuel": f, "price": price}) for f in cfg.FUELS])
    raw = {"ecos_fx_krw_usd": pd.DataFrame({"date": days, "value": 1339.2}),
           "opinet_intl": long(150.12, "usd_bbl"), "opinet_retail": long(1857.29, "price"),
           "opinet_supply_pretax": week(984.80), "opinet_supply_aftertax": week(1781.75)}
    cfg.OUT_DIR = Path(tempfile.mkdtemp())
    t = prepare.prepare_all(raw)
    d, w, latest = t["daily"], t["weekly"], t["latest"].set_index("fuel")
    assert abs(d.loc["2026-10-08", "gasoline_intl_krw_l"] - 1264.5) < 0.1
    assert abs(d.loc["2026-09-22", "gasoline_tax"] - (1781.75 - 1.1 * 984.80)) < 0.01       # 698.47
    assert d.loc["2026-10-08", "gasoline_tax"] == d.loc["2026-09-22", "gasoline_tax"]        # 마지막 주 이후 유지
    assert d.loc["2026-10-08", "gasoline_cap"] == 1784 and pd.isna(d.loc["2026-03-12", "gasoline_cap"])
    assert w.loc["2026-09-26", "gasoline_supply_aftertax"] == 1781.75 and w.loc["2026-09-26", "days"] == 7
    assert pd.isna(w.loc["2026-10-03", "gasoline_supply_pretax"]) and not pd.isna(w.loc["2026-10-03", "gasoline_tax"])
    assert latest.loc["gasoline", "retail_minus_cap"] == 73.3 and latest.loc["gasoline", "intl_krw_l"] == 1264.5
    assert str(latest.loc["gasoline", "supply_week_end"]) == "2026-09-26"
    assert len(t["rounds"]) == 10 and t["rounds"]["gasoline_cap_change"].tolist()[1] == 210
    print("전처리 통과")


def test_counterfactual():
    """정답을 아는 가짜 자료로 "없었다면" 모형을 확인한다.

    세전 공급가 = 0.3·국제가(같은 주) + 0.5·(1주 전) + 0.2·(2주 전) + 마진,
    마진은 앞쪽 200주 60원, 뒤쪽 200주 100원으로 커지게 만든다.
    최근 마진 모형은 비중 (0.3, 0.5, 0.2) 와 마진 100원을 거의 되찾아야 하고,
    상한제 기간의 세후 값은 '× 1.1 + 유류세' 를 몇 원 안에서 맞혀야 한다.
    (마진이 뛰면 수준 회귀의 비중이 조금 흔들리므로 정확히 같지는 않다)
    """
    import numpy as np
    import counterfactual as cf_mod
    weeks = pd.date_range("2015-01-03", periods=420, freq="W-SAT")
    intl = pd.Series(np.random.default_rng(0).normal(0, 30, len(weeks)).cumsum() + 600, index=weeks)
    margin = pd.Series(np.where(np.arange(len(weeks)) < 200, 60.0, 100.0), index=weeks)
    pretax = 0.3 * intl + 0.5 * intl.shift(1) + 0.2 * intl.shift(2) + margin
    policy = weeks >= weeks[400]
    weekly = pd.DataFrame({"in_policy": policy}, index=weeks)
    for fuel in cfg.FUELS:
        weekly[f"{fuel}_intl_krw_l"] = intl
        weekly[f"{fuel}_supply_pretax"] = pretax.where(~policy)        # 상한제 기간은 관측하지 않은 것으로
        weekly[f"{fuel}_supply_aftertax"] = None
        weekly[f"{fuel}_tax"] = 700.0
        weekly[f"{fuel}_cap"] = 1500.0
    X, y = cf_mod.training_data(weekly, "gasoline")
    params = cf_mod.fit(X, y, "recent_margin")
    assert np.isclose(params["weights"].sum(), 1.0)
    assert np.allclose(params["weights"], [0.3, 0.5, 0.2], atol=0.05) and abs(params["const"] - 100) < 1
    out, models = cf_mod.build_counterfactual(weekly)
    week = weeks[410]
    assert abs(out.loc[week, "gasoline_cf_aftertax"] - (pretax[week] * 1.1 + 700)) < 5
    assert out.loc[week, "gasoline_cf_low"] <= out.loc[week, "gasoline_cf_aftertax"] <= out.loc[week, "gasoline_cf_high"]
    # 마진이 커진 자료에서는 11년 상수 하나인 기본 회귀가 낮게 예측해야 한다 (bias < 0)
    score = models[models["fuel"] == "gasoline"].set_index("model")
    assert score.loc["baseline", "bias"] < score.loc["recent_margin", "bias"] <= 1e-6
    # 공급가가 비어 있는 상한제 기간의 주는 모두 잠정으로 표시된다
    assert out["provisional"].sum() == policy.sum() and not out.loc[~policy, "provisional"].any()

    # 근거 점검 (counterfactual_check.py)
    import counterfactual_check as chk
    # 차분 회귀는 마진이 한 번 뛰어도 흔들리지 않으므로 비중을 수준 회귀보다 더 정확히 되찾는다
    diff = chk.fit_diff(X, y)
    assert np.allclose(diff["weights"], [0.3, 0.5, 0.2], atol=0.01) and abs(diff["const"] - 100) < 1
    # 민감도 표의 '채택값' 은 counterfactual.py 의 격차와 같아야 한다
    gaps, _ = chk.sensitivity(weekly)
    expected = (out["gasoline_cf_aftertax"] - out["gasoline_cap"])[policy]
    assert np.allclose(gaps["gasoline_gap"], expected)
    assert (gaps["gasoline_gap_min"] <= gaps["gasoline_gap"]).all() and (gaps["gasoline_gap"] <= gaps["gasoline_gap_max"]).all()
    print("\"없었다면\" 모형 통과")


if __name__ == "__main__":
    test_week_label()
    test_input_files()
    test_prepare()
    test_counterfactual()
    test_ecos()
    test_kosis()
    test_opinet()
    test_fred()
    test_cache_fallback()
    print("모든 점검 통과")
