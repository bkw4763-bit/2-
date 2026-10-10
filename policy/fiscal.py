# 2단계: 재정 부담. 상한제 기간(3/13~) 주별 격차로 정부가 메워야 하는 정유사 손실을 계산합니다.
# 일별 부담 = max(격차, 0) × 하루 판매량,  격차 = "없었다면" 세후 공급가 − 실제 세후 공급가(≈ 상한)
# 실행: python policy/fiscal.py [--volume 휘발유,경유,등유 (리터/일)]
import argparse

import pandas as pd

import config as C




# 1. 주마다 상한제가 적용된 일수입니다. 첫 주(3/14 종료)는 3/13~3/14 이틀만 셉니다.
def policy_days(index):
    start = (index - pd.Timedelta(days=6)).where(index - pd.Timedelta(days=6) > C.POLICY_START, C.POLICY_START)
    return ((index - start).days + 1).to_numpy().clip(0, 7)


# 2. 주별 "리터당 부담"(원·일/리터) = max(격차, 0) × 적용 일수. 판매량(리터/일)을 곱하면 원이 됩니다.
# 90% 구간은 1단계 격차 구간(gap_low, gap_high)으로 같은 계산을 합니다.
# 판매량을 몰라도 이 값은 확정되므로, 판매량이 들어오면 곱하기만 하면 됩니다.
def weekly_burden(cf, volume=None):
    pol = cf[cf["in_policy"]].copy()
    days = policy_days(pol.index)
    out = pd.DataFrame({"주차": pol["주차"], "days": days}, index=pol.index)
    for f in C.FUELS:
        for key in ("gap", "gap_low", "gap_high"):
            out[f"{f}_{key}"] = pol[f"{f}_{key}"]
            out[f"{f}_{key}_perL"] = pol[f"{f}_{key}"].clip(lower=0) * days
        if volume:
            for key in ("gap", "gap_low", "gap_high"):
                out[f"{f}_{key}_won"] = out[f"{f}_{key}_perL"] * volume[f]
    return out


# 3. 유종별 누적과 비중을 요약합니다. 판매량이 있으면 억 원 단위로도 보여 줍니다.
def summary(burden, volume=None):
    rows = []
    for f in C.FUELS:
        row = {"유종": C.KO[f],
               "누적 리터당 부담(원·일/L)": burden[f"{f}_gap_perL"].sum(),
               "90% 하한": burden[f"{f}_gap_low_perL"].sum(),
               "90% 상한": burden[f"{f}_gap_high_perL"].sum(),
               "격차>0 주 수": int((burden[f"{f}_gap"] > 0).sum())}
        if volume:
            row["누적 부담(억 원)"] = burden[f"{f}_gap_won"].sum() / 1e8
            row["하한(억 원)"] = burden[f"{f}_gap_low_won"].sum() / 1e8
            row["상한(억 원)"] = burden[f"{f}_gap_high_won"].sum() / 1e8
        rows.append(row)
    table = pd.DataFrame(rows)
    if volume:
        table["비중(%)"] = table["누적 부담(억 원)"] / table["누적 부담(억 원)"].sum() * 100
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", help="하루 판매량 리터/일: 휘발유,경유,등유")
    args = ap.parse_args()
    volume = C.parse_triplet(args.volume, C.DAILY_VOLUME_L)

    burden = weekly_burden(C.load_cf(), volume)
    table = summary(burden, volume)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    burden.to_csv(C.OUT_DIR / "fiscal_weekly.csv", encoding="utf-8-sig", float_format="%.2f")
    table.to_csv(C.OUT_DIR / "fiscal_summary.csv", index=False, encoding="utf-8-sig", float_format="%.2f")

    pd.set_option("display.width", 200)
    print(f"[2단계 재정 부담] {burden.index[0].date()} 종료 주 ~ {burden.index[-1].date()} 종료 주, "
          f"적용 {int(burden['days'].sum())}일")
    print(table.round(1).to_string(index=False))
    if not volume:
        print("\n하루 판매량(DAILY_VOLUME_L)이 미확인이라 원 단위 합계는 계산하지 않았습니다."
              "\n누적 부담(원) = 누적 리터당 부담 × 하루 판매량(리터/일). --volume 으로 넣으면 바로 계산합니다.")
    else:
        total = table["누적 부담(억 원)"].sum() / 1e4
        print(f"\n3유종 누적 재정 부담: {total:.2f}조 원 (업계 추산 손실 4~5조 원과 대조, 출처 미확인)")
    print(f"저장: {C.OUT_DIR}")


if __name__ == "__main__":
    main()
