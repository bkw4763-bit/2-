"""2단계 전처리.

원자료를 분석용 표로 만든다. 모두 data/processed 에 저장한다.
- daily_panel.csv   : 일별. 환율, 국제 제품가(달러·원), 상한, 유류세, 판매가
- weekly_panel.csv  : 주별(일~토). 일별 표의 주평균 + 정유사 세전·세후 공급가. 모형 학습에 쓴다
- monthly_panel.csv : 월별. 일별 표의 월평균 + 유종별 물가지수 + 총지수
- rounds.csv        : 차수별. 상한, 결정 직전 2주 국제가, 차수 중 평균 판매가·공급가
- latest.csv        : 유종별 최신값과 그 기준일. 계기판에 쓴다
"""
import pandas as pd

import config as cfg

TODAY = cfg.TODAY


# ── 직접 입력 표 읽기 ───────────────────────────────────────────────────
def load_caps():
    """차수별 최고가격 표를 읽는다."""
    caps = pd.read_csv(cfg.CAPS_FILE, parse_dates=["start_date"])
    return caps.sort_values("start_date").reset_index(drop=True)


# ── 도구 ────────────────────────────────────────────────────────────────
def to_daily(df, index, column="value", max_fill=7):
    """date 와 값 열이 있는 표를 일별 시계열로 바꾼다.

    주말·휴일은 직전 값으로 채우되 최대 max_fill 일까지만 채운다.
    그 이상 비어 있으면 그대로 비워 두어 자료가 늦는다는 사실이 드러나게 한다.
    """
    if df is None or df.empty:
        return pd.Series(index=index, dtype=float)
    s = df.drop_duplicates("date", keep="last").set_index("date")[column].sort_index()
    return s.reindex(index).ffill(limit=max_fill)


def step_daily(table, column, index):
    """시작일부터 다음 시작일 전까지 값이 유지되는 계단형 시계열을 만든다. (상한)"""
    s = table.set_index("start_date")[column].sort_index()
    return s.reindex(index.union(s.index)).ffill().reindex(index)


def one_fuel(df, fuel):
    """긴 표(date, fuel, 값)에서 한 유종만 고른다. 표가 없으면 None."""
    if df is None or df.empty:
        return None
    return df[df["fuel"] == fuel]


def last_obs(df, column="value"):
    """표에서 실제로 관측된 마지막 값과 그 날짜를 돌려준다. 없으면 (None, None)."""
    if df is None or df.empty:
        return None, None
    row = df.dropna(subset=[column]).sort_values("date").iloc[-1]
    return float(row[column]), pd.Timestamp(row["date"]).date()


# ── 유류세 역산 ─────────────────────────────────────────────────────────
def weekly_supply(raw):
    """정유사 주간 공급가와, 거기서 역산한 유류세를 한 표로 만든다.

    세후 공급가 = 세전 공급가 × 1.1 + 유류세(부가세 포함) 이므로
    유류세 = 세후 − 1.1 × 세전 이다. 유류세 인하율이 바뀐 주마다 이 값이 계단처럼 바뀐다.
    반환: 주 종료일(토요일)이 인덱스이고 유종별 supply_pretax, supply_aftertax, tax 열을 가진 표
    """
    pre, aft = raw.get("opinet_supply_pretax"), raw.get("opinet_supply_aftertax")
    if pre is None or aft is None or pre.empty or aft.empty:
        return pd.DataFrame()
    pre_w = pre.pivot_table(index="date", columns="fuel", values="price")
    aft_w = aft.pivot_table(index="date", columns="fuel", values="price")
    out = pd.DataFrame(index=pre_w.index.union(aft_w.index))
    for fuel in cfg.FUELS:
        out[f"{fuel}_supply_pretax"] = pre_w.get(fuel)
        out[f"{fuel}_supply_aftertax"] = aft_w.get(fuel)
        out[f"{fuel}_tax"] = out[f"{fuel}_supply_aftertax"] - (1 + cfg.VAT) * out[f"{fuel}_supply_pretax"]
    return out


# ── 일별 표 ─────────────────────────────────────────────────────────────
def build_daily_panel(raw, supply):
    """모든 자료를 날짜 기준으로 묶는다. supply 는 weekly_supply() 의 결과(유류세 역산에 쓴다)."""
    index = pd.date_range(cfg.START_DATE, TODAY, freq="D")
    panel = pd.DataFrame(index=index)
    panel.index.name = "date"

    # 환율: ECOS 를 우선 쓰고, 비어 있는 날은 FRED 값으로 메운다
    fx = to_daily(raw.get("ecos_fx_krw_usd"), index)
    panel["fx"] = fx.fillna(to_daily(raw.get("fred_fx_krw_usd_fred"), index))

    # 브렌트유: 달러/배럴 → 원/리터
    panel["brent_usd_bbl"] = to_daily(raw.get("fred_brent_usd_bbl"), index)
    panel["brent_krw_l"] = panel["brent_usd_bbl"] / cfg.BARREL_L * panel["fx"]

    caps = load_caps()
    for fuel in cfg.FUELS:
        # 국제 제품가(싱가포르): 달러/배럴 → 원/리터
        panel[f"{fuel}_intl_usd_bbl"] = to_daily(one_fuel(raw.get("opinet_intl"), fuel), index, "usd_bbl")
        panel[f"{fuel}_intl_krw_l"] = panel[f"{fuel}_intl_usd_bbl"] / cfg.BARREL_L * panel["fx"]
        # 상한 (시행 전은 빈칸)
        panel[f"{fuel}_cap"] = step_daily(caps, fuel, index)
        # 유류세: 주별 값을 그 주의 일~토에 넣고, 마지막 주 이후는 직전 값을 유지한다.
        # 주 단위 자료라서 세율이 주중에 바뀐 경우 변경일이 며칠 어긋날 수 있다.
        if not supply.empty:
            tax = supply[f"{fuel}_tax"].reindex(index.union(supply.index))
            panel[f"{fuel}_tax"] = tax.bfill(limit=6).ffill().reindex(index)
        # 주유소 판매가
        panel[f"{fuel}_retail"] = to_daily(one_fuel(raw.get("opinet_retail"), fuel), index, "price", max_fill=3)

    panel["round"] = step_daily(caps, "round", index)       # 몇 차 기간인지
    panel["in_policy"] = panel.index >= cfg.POLICY_START
    return panel


# ── 주별 표 ─────────────────────────────────────────────────────────────
def build_weekly_panel(daily, supply):
    """일별 표를 일~토 주평균으로 줄이고 정유사 공급가를 붙인다. 인덱스는 주의 마지막 날(토요일)."""
    numeric = daily.drop(columns=["round", "in_policy"])
    weekly = numeric.resample("W-SAT").mean()
    weekly.index.name = "week_end"
    weekly.insert(0, "week_start", weekly.index - pd.Timedelta(days=6))
    weekly.insert(1, "days", numeric["fx"].resample("W-SAT").size())      # 7 이 아니면 덜 찬 주
    for fuel in cfg.FUELS:
        weekly[f"{fuel}_cap_days"] = daily[f"{fuel}_cap"].resample("W-SAT").count()   # 상한이 걸린 날 수
        for name in ("supply_pretax", "supply_aftertax"):
            col = f"{fuel}_{name}"
            if col in supply:
                weekly[col] = supply[col].reindex(weekly.index)       # 주별 원래 값 (아직 안 나온 주는 빈칸)
        if f"{fuel}_tax" in supply:
            # 유류세: 공급가가 있는 주는 역산한 값, 아직 없는 최근 주는 직전 값을 이어 쓴 일별 평균
            weekly[f"{fuel}_tax"] = supply[f"{fuel}_tax"].reindex(weekly.index).combine_first(weekly[f"{fuel}_tax"])
    weekly["in_policy"] = weekly.index >= cfg.POLICY_START
    return weekly


# ── 월별 표 ─────────────────────────────────────────────────────────────
def build_monthly_panel(daily, raw):
    """일별 표를 월평균으로 줄이고 물가지수를 붙인다."""
    numeric = daily.drop(columns=["round", "in_policy"])
    monthly = numeric.resample("MS").mean()
    monthly["in_policy"] = monthly.index >= pd.Timestamp(cfg.POLICY_START).replace(day=1)

    cpi = raw.get("kosis_fuel_cpi")
    if cpi is not None and not cpi.empty:
        wide = cpi.pivot_table(index="date", columns="fuel", values="value")
        wide.columns = [f"{c}_cpi" for c in wide.columns]
        monthly = monthly.join(wide)
    for name in ("cpi_total", "base_rate", "dubai_usd_bbl"):
        df = raw.get(f"ecos_{name}")
        if df is not None and not df.empty:
            monthly[name] = df.set_index("date")["value"]
    return monthly


# ── 차수별 표 ───────────────────────────────────────────────────────────
def build_round_table(daily, weekly):
    """차수마다 결정 시점의 정보와 결과를 한 줄로 정리한다. (과거 결정 재현에 쓴다)"""
    caps = load_caps()
    rows = []
    for i, cap in caps.iterrows():
        start = cap["start_date"]
        end = caps["start_date"].iloc[i + 1] - pd.Timedelta(days=1) if i + 1 < len(caps) else TODAY
        before = daily.loc[start - pd.Timedelta(days=14): start - pd.Timedelta(days=1)]   # 결정 직전 2주
        during = daily.loc[start:end]
        full_weeks = weekly[(weekly["week_start"] >= start) & (weekly.index <= end)]      # 차수 안에 온전히 든 주
        row = {"round": int(cap["round"]), "start": start.date(), "end": pd.Timestamp(end).date(),
               "days": len(during), "date_verified": cap["date_verified"],
               "fx_before": before["fx"].mean()}
        for fuel in cfg.FUELS:
            row[f"{fuel}_cap"] = cap[fuel]
            row[f"{fuel}_cap_change"] = cap[fuel] - caps[fuel].iloc[i - 1] if i > 0 else 0
            row[f"{fuel}_intl_usd_before"] = before[f"{fuel}_intl_usd_bbl"].mean()
            row[f"{fuel}_intl_krw_before"] = before[f"{fuel}_intl_krw_l"].mean()
            row[f"{fuel}_retail_avg"] = during[f"{fuel}_retail"].mean()
            if f"{fuel}_supply_aftertax" in full_weeks:
                row[f"{fuel}_supply_aftertax_avg"] = full_weeks[f"{fuel}_supply_aftertax"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# ── 최신값 표 ───────────────────────────────────────────────────────────
def build_latest(daily, raw):
    """유종별로 가장 최근 값과 그 기준일을 정리한다. 계기판에 쓴다.

    자료마다 갱신 시점이 다르므로(판매가는 매일, 국제가는 하루 뒤, 공급가는 1~2주 뒤)
    기준일은 일별 표가 아니라 원자료의 실제 마지막 관측일에서 가져온다.
    """
    fx, fx_date = last_obs(raw.get("ecos_fx_krw_usd"))
    if fx is None:                                         # ECOS 가 없으면 FRED 환율
        fx, fx_date = last_obs(raw.get("fred_fx_krw_usd_fred"))
    rows = []
    for fuel, info in cfg.FUELS.items():
        cap = daily[f"{fuel}_cap"].dropna()
        cap_now = None if cap.empty else float(cap.iloc[-1])
        price, price_date = last_obs(one_fuel(raw.get("opinet_retail"), fuel), "price")
        usd, intl_date = last_obs(one_fuel(raw.get("opinet_intl"), fuel), "usd_bbl")
        supply, supply_date = last_obs(one_fuel(raw.get("opinet_supply_aftertax"), fuel), "price")
        tax = daily[f"{fuel}_tax"].dropna() if f"{fuel}_tax" in daily else pd.Series(dtype=float)
        rows.append({
            "fuel": fuel, "name": info["ko"], "cap": cap_now,
            "retail": price, "retail_date": price_date,
            "retail_minus_cap": None if price is None or cap_now is None else round(price - cap_now, 1),
            "supply_aftertax": supply, "supply_week_end": supply_date,
            "intl_usd_bbl": usd, "intl_date": intl_date,
            "fx": fx, "fx_date": fx_date,
            "intl_krw_l": None if usd is None or fx is None else round(usd / cfg.BARREL_L * fx, 1),
            "tax": None if tax.empty else round(float(tax.iloc[-1]), 1)})
    return pd.DataFrame(rows)


def prepare_all(raw):
    """모든 표를 만들어 data/processed 에 저장한다."""
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    supply = weekly_supply(raw)                  # 주간 공급가와 역산한 유류세. 일별·주별 표가 함께 쓴다
    daily = build_daily_panel(raw, supply)
    weekly = build_weekly_panel(daily, supply)
    monthly = build_monthly_panel(daily, raw)
    rounds = build_round_table(daily, weekly)
    latest = build_latest(daily, raw)
    daily.to_csv(cfg.OUT_DIR / "daily_panel.csv", encoding="utf-8-sig")
    weekly.to_csv(cfg.OUT_DIR / "weekly_panel.csv", encoding="utf-8-sig")
    monthly.to_csv(cfg.OUT_DIR / "monthly_panel.csv", encoding="utf-8-sig")
    rounds.to_csv(cfg.OUT_DIR / "rounds.csv", index=False, encoding="utf-8-sig")
    latest.to_csv(cfg.OUT_DIR / "latest.csv", index=False, encoding="utf-8-sig")
    return {"daily": daily, "weekly": weekly, "monthly": monthly, "rounds": rounds, "latest": latest}
