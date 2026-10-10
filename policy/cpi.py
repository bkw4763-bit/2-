# 3단계: 물가 영향. 상한 조정이 판매가와 소비자물가에 주는 영향을 계산합니다.
# ① 반영률: 실제 2차(+210원)·7차(−150원) 조정 뒤 판매가 반응 ÷ 상한 변화
# ② 실효 인상폭: min(새 상한, "없었다면" 가격) − min(기존 상한, "없었다면" 가격)  (시장 가격 위로 올린 부분은 효과 없음)
# ③ 물가 기여도(%p) = Σ (가중치/1000) × (판매가 변화/판매가) × 100
# 실행: python policy/cpi.py [--weights 휘발유,경유,등유 (1000 기준)]
import argparse

import numpy as np
import pandas as pd

import config as C




# 1. 유종별 반영률 경로(1~4주)를 만듭니다. 두 차례 조정의 비율을 평균하고, 자료가 없는 3주는 2주와 4주의 중간값입니다.
# 반환: {유종: array([1주, 2주, 3주, 4주])}
def pass_through_path():
    out = {}
    for f in C.FUELS:
        ratios = np.mean([np.array(resp[f]) / change for change, resp in C.RESPONSE.values()], axis=0)
        out[f] = np.array([ratios[0], ratios[1], (ratios[1] + ratios[2]) / 2, ratios[2]])
    return out


# 2. 한 달(4주) 동안의 평균 반영률입니다. 물가 기여도는 월평균이므로 반영 속도까지 넣습니다.
# mode="4w" 이면 4주 값만 씁니다(단순화안). scale 은 민감도(75~100%)용 배율입니다.
def monthly_pass_through(mode=C.PASS_MODE, scale=1.0):
    path = pass_through_path()
    return {f: (p.mean() if mode == "path" else p[-1]) * scale for f, p in path.items()}


# 3. 실효 인상폭(원/리터). 배열도 받습니다(4단계에서 경로 전체를 한 번에 계산).
def effective_raise(old_cap, new_cap, cf_price):
    return np.minimum(new_cap, cf_price) - np.minimum(old_cap, cf_price)


# 4. 상한 조정의 물가 기여도(%p). 입력은 유종별 실효 인상폭(원/리터, 배열 가능)입니다.
def cpi_contribution(eff_raise, retail, weights, pt):
    return sum(weights[f] / 1000 * eff_raise[f] * pt[f] / retail[f] * 100 for f in C.FUELS)


# 5. 검증: 상한제가 없었다면 판매가 = 실제 + 격차 × 반영률. 월별로 판매가가 얼마나 낮아졌는지(%)와
# 가중치가 있으면 물가를 얼마나 낮췄는지(%p)를 계산해 정부 설명(3~8월 평균 0.6%p)과 대조합니다.
def policy_effect(cf, retail_weekly, weights=None, scale=1.0):
    pt = {f: pass_through_path()[f][-1] * scale for f in C.FUELS}   # 상한이 오래 유지된 효과이므로 4주 값
    pol = cf[cf["in_policy"]]
    out = pd.DataFrame(index=pol.index)
    for f in C.FUELS:
        retail = retail_weekly[f].reindex(pol.index)
        out[f"{f}_retail_cut_pct"] = pol[f"{f}_gap"].clip(lower=0) * pt[f] / retail * 100
    if weights:
        out["cpi_cut_pp"] = sum(weights[f] / 1000 * out[f"{f}_retail_cut_pct"] for f in C.FUELS)
    return out.groupby(out.index.to_period("M")).mean()


# 6. 주유소 주간 판매가 전체를 주 끝 날짜 기준 표로 읽습니다.
def retail_weekly():
    raw = pd.read_csv(C.RETAIL_PATH, encoding="cp949")
    idx = pd.DatetimeIndex([C.to_week_end(x) for x in raw.iloc[:, 0]])
    return pd.DataFrame({"gasoline": raw.iloc[:, 2].to_numpy(float), "diesel": raw.iloc[:, 3].to_numpy(float),
                         "kerosene": raw.iloc[:, 4].to_numpy(float)}, index=idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", help="소비자물가 가중치(1000 기준): 휘발유,경유,등유")
    args = ap.parse_args()
    weights = C.parse_triplet(args.weights, C.CPI_WEIGHTS)

    path = pass_through_path()
    table = pd.DataFrame({C.KO[f]: np.round(p * 100, 1) for f, p in path.items()}, index=["1주", "2주", "3주", "4주"]).T
    table["월평균"] = [round(v * 100, 1) for v in monthly_pass_through("path").values()]
    print("[3단계 ① 반영률, %] 2차·7차 평균 (3주는 2·4주 중간값)")
    print(table.to_string())

    effect = policy_effect(C.load_cf(), retail_weekly(), weights)
    C.OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(C.OUT_DIR / "cpi_pass_through.csv", encoding="utf-8-sig")
    effect.to_csv(C.OUT_DIR / "cpi_policy_effect.csv", encoding="utf-8-sig", float_format="%.3f")
    print("\n[3단계 검증] 상한제로 낮아진 판매가(%), 월평균")
    print(effect.round(2).to_string())
    if weights:
        mar_aug = effect.loc["2026-03":"2026-08", "cpi_cut_pp"].mean()
        print(f"\n3~8월 평균 물가 인하 효과: {mar_aug:.2f}%p (정부 설명 0.6%p)")
    else:
        print("\n물가 가중치(CPI_WEIGHTS)가 미확인이라 %p 는 계산하지 않았습니다. --weights 로 넣으면 바로 계산합니다.")
    print(f"저장: {C.OUT_DIR}")


if __name__ == "__main__":
    main()
