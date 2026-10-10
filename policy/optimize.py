# 4단계: 11차 상한 결정 최적화.
# 미래 국제가 경로 500개를 만들고, 11차의 유종별 조정 폭 216가지(6³)마다 6차수(24주)를 모의 실행해
# "물가 한도를 지키면서 평균 재정 부담이 가장 작은 첫 행동"을 권고합니다.
#   - 경로: 상한제 이전 주간 국제가 변화율(로그)을 4주 묶음으로 뽑아 이어 붙임(블록 부트스트랩, 세 유종 같은 주 → 상관 유지)
#   - "없었다면" 가격: 1단계 식 그대로 (m + w·국제가 시차) × 1.1 + 유류세
#   - 12~16차: 단순 규칙(격차가 큰 유종부터, 물가 한도 안에서 격차까지 인상)
#   - 모든 행동이 같은 경로를 씁니다(공통 난수). 행동 간 차이가 경로 운이 아니라 행동 때문이 되도록.
# 실행: python policy/optimize.py --volume 휘발유,경유,등유 --weights 휘발유,경유,등유 --cpi-limit 0.1 [--sensitivity]
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from cpi import effective_raise, monthly_pass_through




# 1. 블록 부트스트랩으로 미래 국제가 경로를 만듭니다. 반환 (경로, 유종, 주) 원/리터.
# 마지막 실제 값에 뽑은 로그 변화율을 누적해 곱하므로 가격이 음수가 되지 않습니다.
def simulate_intl(intl, weeks, n_paths=C.N_PATHS, block=C.BLOCK, seed=C.SEED, start=C.BOOT_START):
    hist = intl[(intl.index >= start) & (intl.index < C.POLICY_START)][C.FUELS]
    full = pd.date_range(hist.index[0], hist.index[-1], freq="W-SAT")
    r = np.log(hist.reindex(full)).diff().to_numpy()[1:]           # (주, 유종). 빠진 주 앞뒤는 NaN
    starts = [i for i in range(len(r) - block + 1) if np.isfinite(r[i:i + block]).all()]
    rng = np.random.default_rng(seed)
    n_blocks = -(-weeks // block)
    picks = rng.choice(starts, size=(n_paths, n_blocks))
    steps = np.stack([np.concatenate([r[s:s + block] for s in row])[:weeks] for row in picks])  # (경로, 주, 유종)
    last = intl[C.FUELS].iloc[-1].to_numpy()
    return (last * np.exp(np.cumsum(steps, axis=1))).transpose(0, 2, 1)


# 2. 경로별 주간 "없었다면" 세후 공급가 (경로, 유종, 주)와 결정 시점에 알고 있는 최신 값 (유종,)을 만듭니다.
# 앞 두 주는 실제 국제가로 시차를 채웁니다.
def counterfactual_paths(intl, paths, params):
    hist = intl[C.FUELS].iloc[-3:].to_numpy().T                     # (유종, 3주) 실제 최근 3주
    x = np.concatenate([np.broadcast_to(hist[:, 1:], (len(paths), 3, 2)), paths], axis=2)
    cf = np.stack([C.cf_aftertax(x[:, i], *params[f], C.FUEL_TAX[f]) for i, f in enumerate(C.FUELS)], axis=1)
    now = np.array([C.cf_aftertax(hist[i], *params[f], C.FUEL_TAX[f])[0] for i, f in enumerate(C.FUELS)])
    return cf, now


# 3. 조정 폭 하나의 물가 기여도(%p). cap·cf 는 (경로, 유종), act 는 (경로, 유종) 또는 (유종,).
def contribution(cap, act, cf_known, ctx):
    eff = effective_raise(cap, cap + act, cf_known)
    return (eff * ctx["k"]).sum(axis=-1)


# 4. 12~16차 단순 규칙. 격차(최신 "없었다면" 가격 − 상한)가 큰 유종부터,
# '격차 이하에서 가장 큰 조정 폭'을 남은 물가 한도 안에서 고릅니다. 격차가 음수면 인하(−60원 또는 동결)합니다.
def rule(cap, cf_known, ctx):
    P = len(cap)
    gap = cf_known - cap
    target = np.where(gap >= 0, gap, np.where(gap <= -60, -60, 0))[..., None]   # (경로, 유종, 1)
    act = np.zeros_like(cap)
    remaining = np.full(P, ctx["limit"])
    rows = np.arange(P)
    for f in np.argsort(-gap, axis=1).T:                               # 격차 큰 유종부터
        c, k, cf = cap[rows, f][:, None], ctx["k"][f][:, None], cf_known[rows, f][:, None]
        cost = (np.minimum(c + C.ACTIONS, cf) - np.minimum(c, cf)) * k   # (경로, 행동) 물가 기여도
        ok = (C.ACTIONS <= target[rows, f]) & (cost <= remaining[:, None] + 1e-12)
        choice = np.where(ok, C.ACTIONS, -np.inf).max(axis=1)
        act[rows, f] = choice
        remaining -= cost[rows, np.searchsorted(C.ACTIONS, choice)]
    return act


# 5. 첫 행동 하나로 6차수를 모의 실행합니다. 반환: 경로별 24주 재정 부담(원), 첫 4주 재정 부담(원), 첫 차수 물가 기여도(%p).
# 재정 부담 = Σ 주 Σ 유종 max("없었다면" 가격 − 상한, 0) × 하루 판매량 × 7일
def run(first, cf, now, ctx):
    P = cf.shape[0]
    cap = np.tile(ctx["cap0"], (P, 1))
    total, first4, cpi1 = np.zeros(P), None, None
    for r in range(C.ROUNDS):
        known = np.tile(now, (P, 1)) if r == 0 else cf[:, :, r * C.ROUND_WEEKS - 1]
        act = np.tile(first, (P, 1)) if r == 0 else rule(cap, known, ctx)
        if r == 0:
            cpi1 = contribution(cap, act, known, ctx)
        cap = cap + act
        week = cf[:, :, r * C.ROUND_WEEKS:(r + 1) * C.ROUND_WEEKS]
        cost = (np.clip(week - cap[:, :, None], 0, None).sum(axis=2) * ctx["vol"] * 7).sum(axis=1)
        total += cost
        if r == 0:
            first4 = cost
    return total, first4, cpi1


# 6. 216가지 첫 행동을 모두 평가하고, 한도를 지키는 것 중 평균 재정 부담이 가장 작은 행동을 고릅니다.
# 같으면 물가 기여도가 작은 쪽을 고릅니다.
# 인하 조건: 격차가 인하 폭(60원)보다 작을 때만 인하할 수 있습니다. (상한이 시장가에 가까울 때. 7차 −150원 때 격차 48원)
# 이 조건이 없으면 격차가 큰 유종을 내려 물가 한도를 비우고 다른 유종을 올리는, 재정 부담을 늘리는 인하가 뽑힙니다.
def optimize(cf, now, ctx):
    rows, totals = [], {}
    freeze_total, freeze4, _ = run(np.zeros(3), cf, now, ctx)
    for a in itertools.product(C.ACTIONS, repeat=3):
        a = np.array(a)
        total, first4, cpi1 = run(a, cf, now, ctx)
        totals[tuple(a)] = total
        cut_ok = bool(np.all((a >= 0) | (now - ctx["cap0"] < -C.ACTIONS[0])))
        rows.append({**{C.KO[f]: a[i] for i, f in enumerate(C.FUELS)},
                     "물가 기여도(%p)": cpi1[0], "한도 충족": cpi1[0] <= ctx["limit"] + 1e-12 and cut_ok,
                     "24주 평균 부담(억 원)": total.mean() / 1e8,
                     "24주 부담 5%(억 원)": np.quantile(total, 0.05) / 1e8,
                     "24주 부담 95%(억 원)": np.quantile(total, 0.95) / 1e8,
                     "4주 절감(억 원)": (freeze4 - first4).mean() / 1e8,
                     "24주 절감(억 원)": (freeze_total - total).mean() / 1e8})
    table = pd.DataFrame(rows).sort_values(["한도 충족", "24주 평균 부담(억 원)", "물가 기여도(%p)"],
                                           ascending=[False, True, True]).reset_index(drop=True)
    if not table["한도 충족"].iloc[0]:
        raise SystemExit("물가 한도를 지키는 행동이 없습니다. 한도를 확인하세요.")
    return table, totals, freeze_total


# 7. 계산에 쓰는 값 묶음. k = 유종별 "실효 인상 1원당 물가 기여도(%p)" = 가중치/1000 × 반영률 / 판매가 × 100
def context(volume, weights, limit, retail, scale=1.0, mode=C.PASS_MODE):
    pt = monthly_pass_through(mode, scale)
    return {"cap0": np.array([C.CURRENT_CAP[f] for f in C.FUELS]),
            "vol": np.array([volume[f] for f in C.FUELS]),
            "k": np.array([weights[f] / 1000 * pt[f] / retail[f] * 100 for f in C.FUELS]),
            "limit": limit}


def best(table):
    row = table.iloc[0]
    return tuple(int(row[C.KO[f]]) for f in C.FUELS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--volume", help="하루 판매량 리터/일: 휘발유,경유,등유")
    ap.add_argument("--weights", help="소비자물가 가중치(1000 기준): 휘발유,경유,등유")
    ap.add_argument("--cpi-limit", type=float, default=C.CPI_LIMIT_PP, help="월 물가 기여도 한도(%%p)")
    ap.add_argument("--sensitivity", action="store_true", help="반영률·한도·추출 기간을 바꿔 권고 유지 여부 확인")
    ap.add_argument("--out", default=str(C.OUT_DIR))
    args = ap.parse_args()
    volume = C.parse_triplet(args.volume, C.DAILY_VOLUME_L)
    weights = C.parse_triplet(args.weights, C.CPI_WEIGHTS)
    missing = [n for n, v in [("하루 판매량(--volume)", volume), ("물가 가중치(--weights)", weights),
                              ("월 물가 기여도 한도(--cpi-limit)", args.cpi_limit)] if v is None]
    if missing:
        raise SystemExit("미확인 값이 있어 최적화를 멈춥니다: " + ", ".join(missing))

    intl = C.load_intl()
    params = C.load_params()
    retail, retail_week = C.latest_retail()
    weeks = C.ROUNDS * C.ROUND_WEEKS
    cf, now = counterfactual_paths(intl, simulate_intl(intl, weeks), params)
    ctx = context(volume, weights, args.cpi_limit, retail)
    table, totals, freeze_total = optimize(cf, now, ctx)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "optimize_actions.csv", index=False, encoding="utf-8-sig", float_format="%.4f")
    rec = best(table)
    pd.DataFrame({"권고": totals[tuple(float(v) for v in rec)], "동결": freeze_total}).div(1e8).to_csv(
        out / "optimize_paths.csv", index_label="path", encoding="utf-8-sig", float_format="%.2f")

    pd.set_option("display.width", 200)
    print(f"[4단계 상태] 국제가 기준 주 {intl.index[-1].date()}, 판매가 기준 {retail_week}")
    for i, f in enumerate(C.FUELS):
        print(f"  {C.KO[f]}: 상한 {C.CURRENT_CAP[f]:.0f}, \"없었다면\" {now[i]:.1f}, 격차 {now[i] - C.CURRENT_CAP[f]:.1f}원/리터")
    r = table.iloc[0]
    print(f"\n11차 권고: " + ", ".join(f"{C.KO[f]} {rec[i]:+d}원" for i, f in enumerate(C.FUELS))
          + f" / 이번 달 물가 {r['물가 기여도(%p)']:+.3f}%p (한도 {args.cpi_limit}%p)"
          + f" / 4주 재정 절감 {r['4주 절감(억 원)']:,.0f}억 원 (동결 대비, 경로 평균)")
    print("\n[상위 10개 첫 행동]")
    print(table.head(10).round(3).to_string())

    if args.sensitivity:
        print("\n[민감도] 권고 유지 여부")
        cases = [("반영률 75%", dict(scale=0.75)), ("반영률 100%", dict(scale=1 / 0.9)),
                 ("반영률 4주 값만", dict(mode="4w")),
                 ("한도 ×0.5", dict(limit=args.cpi_limit * 0.5)), ("한도 ×1.5", dict(limit=args.cpi_limit * 1.5))]
        for name, kw in cases:
            c = context(volume, weights, kw.get("limit", args.cpi_limit), retail, kw.get("scale", 1.0), kw.get("mode", C.PASS_MODE))
            print(f"  {name}: {best(optimize(cf, now, c)[0])}")
        cf3, now3 = counterfactual_paths(intl, simulate_intl(intl, weeks, start=pd.Timestamp("2023-03-01")), params)
        print(f"  추출 기간 최근 3년: {best(optimize(cf3, now3, ctx)[0])}")
        cf2, now2 = counterfactual_paths(intl, simulate_intl(intl, weeks, seed=C.SEED + 1), params)
        print(f"  다른 난수 시드: {best(optimize(cf2, now2, ctx)[0])}")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
